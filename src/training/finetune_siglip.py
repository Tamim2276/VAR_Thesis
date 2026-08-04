"""
Fine-Tune SigLIP Vision Backbone for Soccer Foul Detection:

This script unfreezes the last 4 transformer layers of SigLIP SO400M
and trains them end-to-end with the FoulClassifier.

Phase A (Warm-up):  Freeze SigLIP, train classifier only (5 epochs)
Phase B (Fine-tune): Unfreeze last 4 layers, train both (15 epochs)

Memory tricks: gradient checkpointing, gradient accumulation, 8-frame sampling
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter
from PIL import Image

sys.path.append('.')
from src.models.hierarchical_classifiers import FoulClassifier
from src.dataset.annotation_loader import load_annotations, get_split_samples

# SigLIP imports
from transformers import SiglipVisionModel, AutoImageProcessor

SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"


# 1. DATASET — Loads raw .npy frames (NOT pre-computed .pt)
class RawFrameDataset(Dataset):
    """
    Loads raw video frames from .npy files for end-to-end training.
    Samples N frames per clip to control VRAM usage.
    """
    def __init__(self, samples, processor, num_frames=8, split="train"):
        self.processor = processor
        self.num_frames = num_frames
        self.split = split
        self.items = []
        skipped = 0

        for s in samples:
            if os.path.exists(s["clip_path"]):
                self.items.append({
                    "clip_path":  s["clip_path"],
                    "foul_label": s["foul_label"],
                    "split":      s.get("split", split),
                })
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Warning: {skipped} clips missing .npy file — skipped")
        print(f"  {split}: {len(self.items)} clips loaded for end-to-end training")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load all 16 frames: (16, 224, 224, 3)
        frames = np.load(item["clip_path"])

        # Sample N evenly-spaced frames to save VRAM
        total = frames.shape[0]
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        sampled = frames[indices]  # (N, 224, 224, 3)

        # Convert to PIL images for the SigLIP processor
        pil_images = [Image.fromarray(frame) for frame in sampled]

        # Apply SigLIP preprocessing (resize to 384x384, normalize)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # (N, 3, 384, 384)

        label = torch.tensor(item["foul_label"], dtype=torch.long)
        return pixel_values, label


# 2. END-TO-END MODEL — SigLIP + FoulClassifier
class EndToEndFoulModel(nn.Module):
    """
    Connects SigLIP vision encoder to FoulClassifier.
    SigLIP extracts per-frame features, FoulClassifier applies
    temporal attention and classifies.
    """
    def __init__(self, siglip_model, classifier):
        super().__init__()
        self.siglip = siglip_model
        self.classifier = classifier

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch, num_frames, 3, 384, 384)
        Returns:
            logits: (batch, 2)
        """
        batch_size, num_frames, C, H, W = pixel_values.shape

        # Reshape to process all frames at once: (batch * num_frames, 3, 384, 384)
        flat_pixels = pixel_values.view(batch_size * num_frames, C, H, W)

        # Extract SigLIP features
        outputs = self.siglip(pixel_values=flat_pixels)
        frame_features = outputs.pooler_output  # (batch * num_frames, 1152)

        # Reshape back: (batch, num_frames, 1152)
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # L2 normalize each frame (same as in train_foul.py)
        frame_features = F.normalize(frame_features, p=2, dim=-1)

        # Pass through the classifier with temporal attention
        logits = self.classifier(frame_features)
        return logits


# 3. LAYER FREEZING / UNFREEZING UTILITIES
def freeze_siglip(model):
    """Freeze ALL SigLIP parameters."""
    for param in model.siglip.parameters():
        param.requires_grad = False

def unfreeze_last_n_layers(model, n=4):
    """
    Unfreeze the last N transformer layers of SigLIP.
    Also unfreezes the post_layernorm and head (if they exist).
    """
    # First, make sure everything is frozen
    freeze_siglip(model)

    # Get encoder layers
    encoder_layers = model.siglip.vision_model.encoder.layers
    total_layers = len(encoder_layers)
    start_idx = total_layers - n

    print(f"  SigLIP has {total_layers} encoder layers")
    print(f"  Unfreezing layers {start_idx} to {total_layers - 1}")

    # Unfreeze last N layers
    for i in range(start_idx, total_layers):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True

    # Also unfreeze post_layernorm and head if they exist
    if hasattr(model.siglip.vision_model, 'post_layernorm'):
        for param in model.siglip.vision_model.post_layernorm.parameters():
            param.requires_grad = True

    if hasattr(model.siglip.vision_model, 'head'):
        for param in model.siglip.vision_model.head.parameters():
            param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def get_parameter_groups(model, lr_backbone=1e-5, lr_head=1e-4):
    """
    Create parameter groups with discriminative learning rates.
    SigLIP layers get a small LR, classifier head gets normal LR.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("siglip"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"  Backbone params (lr={lr_backbone}): {sum(p.numel() for p in backbone_params):,}")
    print(f"  Head params     (lr={lr_head}): {sum(p.numel() for p in head_params):,}")

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ]


# 4. TRAINING AND EVALUATION
def make_balanced_sampler(dataset):
    labels = [1 if item["foul_label"] > 0 else 0 for item in dataset.items]
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    accumulation_steps=16, use_amp=True):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]

    optimizer.zero_grad()
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for step, (pixel_values, raw_labels) in enumerate(progress):
        pixel_values = pixel_values.to(device)
        raw_labels = raw_labels.to(device)
        binary_labels = (raw_labels > 0).long()

        # Forward pass (with optional mixed precision)
        if use_amp:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(pixel_values)
                loss = criterion(logits, binary_labels)
        else:
            logits = model(pixel_values)
            loss = criterion(logits, binary_labels)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_samples += pixel_values.shape[0]

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] >= 0.60).long()
            correct += (preds == binary_labels).sum().item()

            for c in range(2):
                mask = binary_labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()

        progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.3f}"})

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(2)]
    balanced_acc = sum(recall) / 2.0
    return {"loss": total_loss / len(loader), "acc": acc, "recall": recall, "balanced_acc": balanced_acc}


def evaluate(model, loader, criterion, device, split_name, use_amp=True):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for pixel_values, raw_labels in tqdm(loader, desc=f"[{split_name}]"):
            pixel_values = pixel_values.to(device)
            raw_labels = raw_labels.to(device)
            binary_labels = (raw_labels > 0).long()

            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(pixel_values)
                    loss = criterion(logits, binary_labels)
            else:
                logits = model(pixel_values)
                loss = criterion(logits, binary_labels)

            total_loss += loss.item()
            total_samples += pixel_values.shape[0]
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] >= 0.60).long()
            correct += (preds == binary_labels).sum().item()

            for c in range(2):
                mask = binary_labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(2)]
    balanced_acc = sum(recall) / 2.0
    return {"loss": total_loss / len(loader), "acc": acc, "recall": recall, "balanced_acc": balanced_acc}



# 5. MAIN
if __name__ == "__main__":

    #Configuration
    NUM_FRAMES         = 8        # Frames per clip (8 to save VRAM, 16 for full)
    BATCH_SIZE         = 2        # Tiny batch to fit in VRAM
    ACCUMULATION_STEPS = 32       # Effective batch = 2 * 32 = 64
    PHASE_A_EPOCHS     = 5        # Warm-up: classifier only
    PHASE_B_EPOCHS     = 15       # Fine-tune: SigLIP + classifier
    LR_BACKBONE        = 1e-5     # Gentle learning rate for SigLIP
    LR_HEAD            = 1e-4     # Normal learning rate for classifier
    UNFREEZE_LAYERS    = 4        # Number of SigLIP layers to unfreeze
    SAVE_DIR           = "models"
    LOG_DIR            = "logs"
    USE_AMP            = True     # Mixed precision (bfloat16 on XPU)
    CHECKPOINT_PATH    = "models/finetune_checkpoint.pt"  # Resume point

    #Device
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"\n{'='*70}")
    print(f"  SigLIP Fine-Tuning for Foul Detection")
    print(f"  Device: {DEVICE} | Frames: {NUM_FRAMES} | Batch: {BATCH_SIZE}")
    print(f"  Effective Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"{'='*70}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    #Load SigLIP
    print("\nLoading SigLIP SO400M...")
    siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)

    # Enable gradient checkpointing to save VRAM
    siglip_model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ENABLED")

    #Build End-to-End Model
    classifier = FoulClassifier(feat_dim=1152, hidden_dim=128)
    model = EndToEndFoulModel(siglip_model, classifier).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total model parameters: {total_params:,}")

    #Load Data
    print("\nLoading annotations...")
    all_samples = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")

    train_dataset = RawFrameDataset(train_samples, processor, NUM_FRAMES, "train")
    valid_dataset = RawFrameDataset(valid_samples, processor, NUM_FRAMES, "valid")

    train_sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=0,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    #Checkpoint resume logic
    start_phase = "A"
    start_epoch = 1
    history = []
    best_balanced_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n  ✓ Found checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(DEVICE)
        start_phase = ckpt["phase"]
        start_epoch = ckpt["epoch"] + 1  # Resume from NEXT epoch
        best_balanced_acc = ckpt["best_balanced_acc"]
        history = ckpt.get("history", [])
        print(f"  Resuming from Phase {start_phase}, Epoch {start_epoch}")
        print(f"  Previous best balanced acc: {best_balanced_acc:.1f}%")
    else:
        print(f"\n  No checkpoint found — starting fresh")

    def save_checkpoint(phase, epoch, model, best_acc, history):
        """Save checkpoint after every epoch for crash recovery."""
        torch.save({
            "phase": phase,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_balanced_acc": best_acc,
            "history": history,
        }, CHECKPOINT_PATH)
        print(f"  💾 Checkpoint saved (Phase {phase}, Epoch {epoch})")

    # PHASE A: Warm-up — Freeze SigLIP, train classifier only
    if start_phase == "A":
        print(f"\n{'='*70}")
        print(f"  PHASE A: Warm-up ({PHASE_A_EPOCHS} epochs)")
        print(f"  SigLIP: FROZEN | Training: Classifier only")
        print(f"{'='*70}")

        freeze_siglip(model)
        trainable_a = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable_a:,}")

        optimizer_a = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR_HEAD, weight_decay=5e-3
        )

        for epoch in range(start_epoch, PHASE_A_EPOCHS + 1):
            train_m = train_one_epoch(model, train_loader, optimizer_a, criterion,
                                      DEVICE, epoch, ACCUMULATION_STEPS, USE_AMP)
            val_m = evaluate(model, valid_loader, criterion, DEVICE, "valid", USE_AMP)

            print(f"\n  Train — loss: {train_m['loss']:.4f} | balanced: {train_m['balanced_acc']:.1f}%")
            print(f"           recall [No Foul]: {train_m['recall'][0]:.1f}% | [Foul]: {train_m['recall'][1]:.1f}%")
            print(f"  Valid — loss: {val_m['loss']:.4f} | balanced: {val_m['balanced_acc']:.1f}%")
            print(f"           recall [No Foul]: {val_m['recall'][0]:.1f}% | [Foul]: {val_m['recall'][1]:.1f}%")

            history.append({
                "phase": "A", "epoch": epoch,
                "train_balanced_acc": train_m["balanced_acc"],
                "val_balanced_acc": val_m["balanced_acc"],
                "val_recall": val_m["recall"],
            })

            if val_m["balanced_acc"] > best_balanced_acc:
                best_balanced_acc = val_m["balanced_acc"]
                print(f"  ✓ New best: {best_balanced_acc:.1f}%")

            save_checkpoint("A", epoch, model, best_balanced_acc, history)

        print(f"\n  Phase A complete — Best balanced acc: {best_balanced_acc:.1f}%")
        # Reset start_epoch for Phase B
        start_epoch = 1

    # PHASE B: Fine-tune — Unfreeze last N layers of SigLIP
    print(f"\n{'='*70}")
    print(f"  PHASE B: Fine-Tuning ({PHASE_B_EPOCHS} epochs)")
    print(f"  SigLIP: Last {UNFREEZE_LAYERS} layers UNFROZEN")
    print(f"{'='*70}")

    unfreeze_last_n_layers(model, n=UNFREEZE_LAYERS)
    param_groups = get_parameter_groups(model, LR_BACKBONE, LR_HEAD)

    optimizer_b = torch.optim.AdamW(param_groups, weight_decay=5e-3)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=PHASE_B_EPOCHS, eta_min=1e-6
    )
    # Fast-forward scheduler if resuming mid-Phase B
    if start_phase == "B" and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler_b.step()

    no_improve = 0
    patience = 8

    for epoch in range(start_epoch, PHASE_B_EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, optimizer_b, criterion,
                                  DEVICE, epoch, ACCUMULATION_STEPS, USE_AMP)
        val_m = evaluate(model, valid_loader, criterion, DEVICE, "valid", USE_AMP)
        scheduler_b.step()

        current_lr_bb = optimizer_b.param_groups[0]["lr"]
        current_lr_hd = optimizer_b.param_groups[1]["lr"]
        print(f"\n  Train — loss: {train_m['loss']:.4f} | balanced: {train_m['balanced_acc']:.1f}%")
        print(f"           recall [No Foul]: {train_m['recall'][0]:.1f}% | [Foul]: {train_m['recall'][1]:.1f}%")
        print(f"  Valid — loss: {val_m['loss']:.4f} | balanced: {val_m['balanced_acc']:.1f}%")
        print(f"           recall [No Foul]: {val_m['recall'][0]:.1f}% | [Foul]: {val_m['recall'][1]:.1f}%")
        print(f"  LR: backbone={current_lr_bb:.2e} | head={current_lr_hd:.2e}")

        history.append({
            "phase": "B", "epoch": epoch,
            "train_balanced_acc": train_m["balanced_acc"],
            "val_balanced_acc": val_m["balanced_acc"],
            "val_recall": val_m["recall"],
        })

        if val_m["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = val_m["balanced_acc"]
            no_improve = 0

            # Save fine-tuned SigLIP weights
            torch.save(
                model.siglip.state_dict(),
                os.path.join(SAVE_DIR, "siglip_finetuned.pt")
            )
            # Save the full model (SigLIP + classifier)
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "finetuned_full_model.pt")
            )
            print(f"  ✓ New best: {best_balanced_acc:.1f}% — saved weights")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping — no improvement for {patience} epochs")
                break

        save_checkpoint("B", epoch, model, best_balanced_acc, history)

    #Save training history
    with open(os.path.join(LOG_DIR, "finetune_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    #Final summary
    print(f"\n{'='*70}")
    print(f"  FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best Validation Balanced Accuracy: {best_balanced_acc:.1f}%")
    print(f"  SigLIP weights saved to: {SAVE_DIR}/siglip_finetuned.pt")
    print(f"  Full model saved to:     {SAVE_DIR}/finetuned_full_model.pt")
    print(f"")
    print(f"  NEXT STEP:")
    print(f"  Run precompute_features.py to re-extract features with the")
    print(f"  fine-tuned SigLIP. Then re-train foul and severity models.")
    print(f"{'='*70}")
