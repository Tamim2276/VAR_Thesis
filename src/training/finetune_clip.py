import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.append('.')
from src.models.clip_extractor import get_patch_tokens, get_visual_token_dim, load_clip_model
from src.dataset.annotation_loader import load_annotations, get_split_samples


class FoulVideoDataset(Dataset):
    """Loads raw frames for CLIP fine-tuning.
    
    Uses 4 key frames spread across the clip for better temporal context.
    CLIP just needs to learn football-relevant features —
    the downstream classifier heads (train_fast.py) will
    train on all 16 frames' features.
    """

    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = np.load(s["clip_path"])  # Usually (16, 224, 224, 3)

        # 4 key frames spread across the clip
        T = frames.shape[0]
        key_indices = [
            int(T * 0.25),   # before contact
            int(T * 0.50),   # contact moment
            int(T * 0.75),   # after contact
            int(T * 0.90),   # end of clip
        ]
        tensors = []
        for fi in key_indices:
            t = self.preprocess(Image.fromarray(frames[fi]))
            tensors.append(t)

        tensor = torch.stack(tensors)  # (4, 3, 224, 224)

        foul_label = torch.tensor(s["foul_label"], dtype=torch.long)
        sev_label  = torch.tensor(s["sev_label"],  dtype=torch.long)

        return tensor, foul_label, sev_label


def inverse_frequency_weights(samples, label_key, num_classes, device):
    """Give each class equal total loss contribution despite imbalance."""
    counts = Counter(sample[label_key] for sample in samples)
    total = len(samples)
    weights = [total / (num_classes * counts.get(class_id, 1))
               for class_id in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


if __name__ == "__main__":

    DEVICE      = "xpu"
    BATCH_SIZE = 4           # Tiny physical batch to fit in 12GB VRAM
    ACCUMULATION_STEPS = 8   # 4 * 8 = 32 effective batch size
    NUM_EPOCHS  = 15         # more epochs needed with lower learning rates
    SAVE_PATH   = "models/clip_finetuned.pt"
    BACKBONE_LR = 5e-6       # very conservative — prevent gradient explosion
    HEAD_LR     = 3e-4       # lowered from 1e-3 to prevent NaN

    if DEVICE == "xpu" and not torch.xpu.is_available():
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    print("Loading CLIP...")
    clip_model, preprocess = load_clip_model()
    clip_model = clip_model.to(DEVICE)

    # ── Partial Freeze Strategy ──────────────────────────────────
    # Freeze first 20/24 transformer layers
    # Only fine-tune last 4 layers + ln_post
    for param in clip_model.parameters():
        param.requires_grad = False

    for layer in clip_model.visual.transformer.resblocks[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in clip_model.visual.ln_post.parameters():
        param.requires_grad = True

    trainable_backbone = sum(
        p.numel() for p in clip_model.visual.parameters() if p.requires_grad
    )
    total_backbone = sum(p.numel() for p in clip_model.visual.parameters())
    print(f"CLIP backbone: {trainable_backbone:,} / {total_backbone:,} params trainable "
          f"({trainable_backbone/total_backbone*100:.1f}%)")

    visual_dim = get_visual_token_dim(clip_model)
    print(f"ViT CLS-token dimension: {visual_dim}")
    foul_head  = nn.Linear(visual_dim, 2).to(DEVICE)
    sev_head   = nn.Linear(visual_dim, 4).to(DEVICE)

    print("Loading data...")
    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")

    foul_weights, foul_counts = inverse_frequency_weights(
        train_samples, "foul_label", num_classes=2, device=DEVICE
    )
    sev_weights, sev_counts = inverse_frequency_weights(
        train_samples, "sev_label", num_classes=4, device=DEVICE
    )
    print(f"Foul labels: {dict(sorted(foul_counts.items()))}")
    print(f"Foul loss weights: {foul_weights.tolist()}")
    print(f"Severity labels: {dict(sorted(sev_counts.items()))}")
    print(f"Severity loss weights: {sev_weights.tolist()}")

    train_dataset = FoulVideoDataset(train_samples, preprocess)
    valid_dataset = FoulVideoDataset(valid_samples, preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    # Only trainable backbone params go into optimizer
    backbone_params = [p for p in clip_model.visual.parameters() if p.requires_grad]
    head_params = list(foul_head.parameters()) + list(sev_head.parameters())
    params = backbone_params + head_params

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=1e-4)

    # Warmup scheduler — prevents gradient explosion in early training
    # 10% warmup then cosine decay to zero
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_LR, HEAD_LR],
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos'
    )

    foul_criterion = nn.CrossEntropyLoss(weight=foul_weights)
    sev_criterion  = nn.CrossEntropyLoss(weight=sev_weights)

    print(f"Learning rates: backbone={BACKBONE_LR:g} (last 4 layers), heads={HEAD_LR:g}")
    print(f"Scheduler: OneCycleLR with 10% warmup + cosine decay")
    print(f"Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} x {ACCUMULATION_STEPS} = "
          f"{BATCH_SIZE * ACCUMULATION_STEPS} effective")
    print(f"Training batches per epoch: {len(train_loader)}")

    best_val_foul_balanced = float("-inf")
    nan_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        clip_model.train()
        foul_head.train()
        sev_head.train()

        total_loss   = 0.0
        foul_correct = 0
        sev_correct  = 0
        total        = 0
        epoch_nan    = 0

        optimizer.zero_grad()

        for i, (imgs, foul_labels, sev_labels) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"
        )):
            imgs        = imgs.to(DEVICE)
            foul_labels = foul_labels.to(DEVICE)
            sev_labels  = sev_labels.to(DEVICE)

            autocast_device = "xpu" if DEVICE == "xpu" else "cpu"
            autocast_dtype  = torch.bfloat16 if DEVICE == "xpu" else torch.float32

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                B, N, C, H, W = imgs.shape
                imgs_flat  = imgs.view(B * N, C, H, W)
                feats_flat = get_patch_tokens(clip_model, imgs_flat)[:, 0, :]
                features   = feats_flat.view(B, N, -1).mean(dim=1)
                features = features.float()

                foul_logits = foul_head(features)
                sev_logits  = sev_head(features)

                loss = foul_criterion(foul_logits, foul_labels) + \
                       sev_criterion(sev_logits, sev_labels)
                loss = loss / ACCUMULATION_STEPS

            # ── NaN Guard ────────────────────────────────────────
            # If loss is NaN, skip this batch entirely
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                epoch_nan += 1
                nan_count += 1
                scheduler.step()  # still step scheduler to keep it in sync
                continue

            loss.backward()

            # Track metrics
            total        += imgs.shape[0]
            foul_correct += (foul_logits.detach().argmax(-1) == foul_labels).sum().item()
            sev_correct  += (sev_logits.detach().argmax(-1)  == sev_labels).sum().item()
            total_loss   += loss.item() * ACCUMULATION_STEPS

            # Update weights every ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # Check for NaN in gradients before stepping
                has_nan_grad = False
                for p in params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    optimizer.zero_grad()
                    epoch_nan += 1
                    nan_count += 1
                else:
                    # Tight gradient clipping
                    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()

                if DEVICE == "xpu":
                    torch.xpu.empty_cache()

            scheduler.step()

        # ── Validation ──────────────────────────────────────────
        clip_model.eval()
        foul_head.eval()
        sev_head.eval()

        val_foul_correct = 0
        val_sev_correct  = 0
        val_total        = 0
        val_foul_support = [0, 0]
        val_foul_hits    = [0, 0]
        val_foul_preds   = [0, 0]

        with torch.no_grad():
            for imgs, foul_labels, sev_labels in valid_loader:
                imgs        = imgs.to(DEVICE)
                foul_labels = foul_labels.to(DEVICE)
                sev_labels  = sev_labels.to(DEVICE)

                autocast_device = "xpu" if DEVICE == "xpu" else "cpu"
                autocast_dtype  = torch.bfloat16 if DEVICE == "xpu" else torch.float32

                with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                    B, N, C, H, W = imgs.shape
                    imgs_flat  = imgs.view(B * N, C, H, W)
                    feats_flat = get_patch_tokens(clip_model, imgs_flat)[:, 0, :]
                    features   = feats_flat.view(B, N, -1).mean(dim=1)
                    features = features.float()

                    foul_logits = foul_head(features)
                    sev_logits  = sev_head(features)

                foul_preds = foul_logits.argmax(-1)
                sev_preds  = sev_logits.argmax(-1)
                val_foul_correct += (foul_preds == foul_labels).sum().item()
                val_sev_correct  += (sev_preds == sev_labels).sum().item()
                val_total        += imgs.shape[0]

                for class_id in range(2):
                    class_mask = foul_labels == class_id
                    val_foul_support[class_id] += class_mask.sum().item()
                    val_foul_hits[class_id] += (
                        (foul_preds == class_id) & class_mask
                    ).sum().item()
                    val_foul_preds[class_id] += (foul_preds == class_id).sum().item()

        train_foul = foul_correct  / max(total, 1) * 100
        train_sev  = sev_correct   / max(total, 1) * 100
        val_foul   = val_foul_correct / val_total * 100
        val_sev    = val_sev_correct / val_total * 100
        val_foul_recall = [
            100 * hits / support if support else 0.0
            for hits, support in zip(val_foul_hits, val_foul_support)
        ]
        val_foul_balanced = sum(val_foul_recall) / len(val_foul_recall)

        print(f"Epoch {epoch}: train_loss={total_loss / max(len(train_loader), 1):.4f} "
              f"train_foul={train_foul:.1f}% train_sev={train_sev:.1f}% val_foul={val_foul:.1f}%")
        print(f"  Val severity={val_sev:.1f}% | balanced foul="
              f"{val_foul_balanced:.1f}% | recall [No foul, Foul]: "
              f"[{val_foul_recall[0]:.1f}%, {val_foul_recall[1]:.1f}%] | "
              f"predictions: {val_foul_preds}")
        if epoch_nan > 0:
            print(f"  ⚠ NaN batches skipped this epoch: {epoch_nan}")

        if val_foul_balanced > best_val_foul_balanced:
            best_val_foul_balanced = val_foul_balanced
            os.makedirs("models", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "clip_model_state": clip_model.state_dict(),
                "foul_head_state": foul_head.state_dict(),
                "severity_head_state": sev_head.state_dict(),
                "visual_token_dim": visual_dim,
                "val_foul_acc": val_foul,
                "val_foul_balanced_acc": val_foul_balanced,
                "val_sev_acc": val_sev,
            }, SAVE_PATH)
            print(f"   Saved checkpoint → {SAVE_PATH}")

    print(f"\nCLIP fine-tuning done.")
    print(f"  Best balanced val foul: {best_val_foul_balanced:.1f}%")
    print(f"  Total NaN batches skipped: {nan_count}")