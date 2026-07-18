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
    """Loads raw frames for CLIP fine-tuning."""

    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = np.load(s["clip_path"])  # (16, 224, 224, 3)

        # Use center frame (frame 8) as representative frame
        # This keeps it simple for Stage 1
        frame = frames[8]
        tensor = self.preprocess(Image.fromarray(frame))

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
    NUM_EPOCHS  = 5          # just 5 epochs to adapt CLIP to football
    LR          = 5e-6       # very small LR — CLIP already knows a lot
    SAVE_PATH   = "models/clip_finetuned.pt"
    BACKBONE_LR = 1e-6       # preserve the pretrained ViT-L representation
    HEAD_LR     = 1e-3       # new linear heads need to learn much faster

    if DEVICE == "xpu" and not torch.xpu.is_available():
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    print("Loading CLIP...")
    clip_model, preprocess = load_clip_model()
    clip_model = clip_model.to(DEVICE)

    # Use the unprojected ViT CLS token. ViT-L/14's CLIP projection is
    # 1024 -> 768, but the rest of this project uses 1024-wide token features.
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

    # This is image-only fine-tuning, so the unused text encoder is excluded.
    backbone_params = list(clip_model.visual.parameters())
    head_params = list(foul_head.parameters()) + list(sev_head.parameters())
    params = backbone_params + head_params

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=1e-4)
    foul_criterion = nn.CrossEntropyLoss(weight=foul_weights)
    sev_criterion  = nn.CrossEntropyLoss(weight=sev_weights)
    print(f"Learning rates: backbone={BACKBONE_LR:g}, heads={HEAD_LR:g}")

    best_val_foul_balanced = float("-inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        clip_model.train()
        foul_head.train()
        sev_head.train()

        total_loss   = 0.0
        foul_correct = 0
        sev_correct  = 0
        total        = 0

        # Zero gradients before starting the batch loop
        optimizer.zero_grad()

        for i, (imgs, foul_labels, sev_labels) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"
        )):
            imgs        = imgs.to(DEVICE)
            foul_labels = foul_labels.to(DEVICE)
            sev_labels  = sev_labels.to(DEVICE)

            # 1. Forward Pass with Mixed Precision context manager
            # Using bfloat16 for optimal acceleration and minimal VRAM on XPU
            autocast_device = "xpu" if DEVICE == "xpu" else "cpu"
            autocast_dtype  = torch.bfloat16 if DEVICE == "xpu" else torch.float32

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                # Keep the 1024-wide, pre-projection CLS token. This exactly
                # matches precompute_features.py and downstream classifiers.
                features = get_patch_tokens(clip_model, imgs)[:, 0, :]
                features = features.float()

                foul_logits = foul_head(features)
                sev_logits  = sev_head(features)

                loss = foul_criterion(foul_logits, foul_labels) + \
                       sev_criterion(sev_logits, sev_labels)
                
                # Scale the loss by accumulation steps
                loss = loss / ACCUMULATION_STEPS

            # 2. Backward Pass (Accumulates gradients)
            loss.backward()

            # Track training metrics using detached tensors to save memory
            total        += imgs.shape[0]
            foul_correct += (foul_logits.detach().argmax(-1) == foul_labels).sum().item()
            sev_correct  += (sev_logits.detach().argmax(-1)  == sev_labels).sum().item()
            total_loss   += loss.item() * ACCUMULATION_STEPS

            # 3. Update Weights (Only every ACCUMULATION_STEPS or at the end of the data loader)
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # Gradient clipping — prevents exploding gradients during fine-tuning
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear XPU cache periodically to combat VRAM fragmentation
                if DEVICE == "xpu":
                    torch.xpu.empty_cache()

        # Validation
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
                    features    = get_patch_tokens(clip_model, imgs)[:, 0, :]
                    features    = features.float()
                    foul_logits = foul_head(features)
                    sev_logits  = sev_head(features)
                
                foul_preds = foul_logits.argmax(-1)
                sev_preds  = sev_logits.argmax(-1)
                val_foul_correct += (foul_preds == foul_labels).sum().item()
                val_sev_correct += (sev_preds == sev_labels).sum().item()
                val_total        += imgs.shape[0]

                for class_id in range(2):
                    class_mask = foul_labels == class_id
                    val_foul_support[class_id] += class_mask.sum().item()
                    val_foul_hits[class_id] += (
                        (foul_preds == class_id) & class_mask
                    ).sum().item()
                    val_foul_preds[class_id] += (foul_preds == class_id).sum().item()

        train_foul = foul_correct  / total        * 100
        train_sev  = sev_correct   / total        * 100
        val_foul   = val_foul_correct / val_total * 100
        val_sev    = val_sev_correct / val_total * 100
        val_foul_recall = [
            100 * hits / support if support else 0.0
            for hits, support in zip(val_foul_hits, val_foul_support)
        ]
        val_foul_balanced = sum(val_foul_recall) / len(val_foul_recall)

        print(f"Epoch {epoch}: train_loss={total_loss / len(train_loader):.4f} "
              f"train_foul={train_foul:.1f}% train_sev={train_sev:.1f}% val_foul={val_foul:.1f}%")
        print(f"  Val severity={val_sev:.1f}% | balanced foul="
              f"{val_foul_balanced:.1f}% | recall [No foul, Foul]: "
              f"[{val_foul_recall[0]:.1f}%, {val_foul_recall[1]:.1f}%] | "
              f"predictions: {val_foul_preds}")

        # Raw accuracy rewards the majority-class baseline. Select checkpoints
        # by mean per-class recall instead.
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
            print(f"   Saved fine-tuning checkpoint to {SAVE_PATH}")

    print(f"\nCLIP fine-tuning done. Best balanced val foul: "
          f"{best_val_foul_balanced:.1f}%")
