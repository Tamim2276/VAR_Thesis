import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('.')
from src.models.classifier_heads import XVARSClassifiers
from src.dataset.annotation_loader import load_annotations, get_split_samples


class FeatureDataset(Dataset):
    """
    Loads pre-computed CLIP features instead of raw frames.
    Each __getitem__ just loads a tiny .pt file — instant.
    No CLIP inference during training at all.
    """

    def __init__(self, samples, features_root="data/features"):
        self.features_root = features_root

        # Build feature path for each sample
        # replace data/frames with data/features and .npy with .pt
        self.items = []
        skipped = 0

        for s in samples:
            feat_path = s["clip_path"] \
                .replace("data/frames", features_root) \
                .replace("data\\frames", features_root) \
                .replace(".npy", ".pt")
            feat_path = feat_path.replace("\\", "/")

            if os.path.exists(feat_path):
                self.items.append({
                    "feat_path"  : feat_path,
                    "foul_label" : s["foul_label"],
                    "sev_label"  : s["sev_label"],
                })
            else:
                skipped += 1

        if skipped > 0:
            print(f"Warning: {skipped} clips have no feature file — skipped")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load pre-computed features — instant
        data = torch.load(item["feat_path"], map_location="cpu")
        cls_tokens = data["cls"]  # (16, 1024)

        # Average across 16 frames → video vector
        video_vector = cls_tokens.mean(dim=0)  # (1024,)

        foul_label = torch.tensor(item["foul_label"], dtype=torch.long)
        sev_label  = torch.tensor(item["sev_label"],  dtype=torch.long)

        return video_vector, foul_label, sev_label


def train_one_epoch(classifiers, loader, optimizer,
                    foul_criterion, sev_criterion, device, epoch):
    classifiers.train()

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for video_vectors, foul_labels, sev_labels in progress:
        video_vectors = video_vectors.to(device)
        foul_labels   = foul_labels.to(device)
        sev_labels    = sev_labels.to(device)

        foul_logits, sev_logits = classifiers(video_vectors)
        loss = foul_criterion(foul_logits, foul_labels) + \
               sev_criterion(sev_logits,  sev_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_samples += video_vectors.shape[0]

        foul_preds = torch.argmax(foul_logits, dim=-1)
        sev_preds  = torch.argmax(sev_logits,  dim=-1)
        foul_correct += (foul_preds == foul_labels).sum().item()
        sev_correct  += (sev_preds  == sev_labels).sum().item()

        progress.set_postfix({
            "loss"    : f"{loss.item():.3f}",
            "foul_acc": f"{foul_correct/total_samples*100:.1f}%",
            "sev_acc" : f"{sev_correct/total_samples*100:.1f}%",
        })

    return (
        total_loss    / len(loader),
        foul_correct  / total_samples * 100,
        sev_correct   / total_samples * 100,
    )


def evaluate(classifiers, loader, foul_criterion,
             sev_criterion, device, split_name):
    classifiers.eval()

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0

    with torch.no_grad():
        for video_vectors, foul_labels, sev_labels in tqdm(loader, desc=f"[{split_name}]"):
            video_vectors = video_vectors.to(device)
            foul_labels   = foul_labels.to(device)
            sev_labels    = sev_labels.to(device)

            foul_logits, sev_logits = classifiers(video_vectors)
            loss = foul_criterion(foul_logits, foul_labels) + \
                   sev_criterion(sev_logits, sev_labels)

            total_loss    += loss.item()
            total_samples += video_vectors.shape[0]

            foul_preds = torch.argmax(foul_logits, dim=-1)
            sev_preds  = torch.argmax(sev_logits,  dim=-1)
            foul_correct += (foul_preds == foul_labels).sum().item()
            sev_correct  += (sev_preds  == sev_labels).sum().item()

    return (
        total_loss   / len(loader),
        foul_correct / total_samples * 100,
        sev_correct  / total_samples * 100,
    )


if __name__ == "__main__":

    # Config
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64    # much larger now — no CLIP bottleneck
    NUM_EPOCHS    = 20
    LR            = 1e-4
    DEVICE        = "xpu"
    SAVE_DIR      = "models"
    LOG_DIR       = "logs"
    #

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Verify XPU
    if DEVICE == "xpu" and not torch.xpu.is_available():
        print("XPU not available, falling back to CPU")
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    # Load annotations
    print("\nLoading annotations...")
    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")

    # Build feature datasets
    print("\nBuilding feature datasets...")
    train_dataset = FeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = FeatureDataset(valid_samples, FEATURES_ROOT)
    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    # Build model
    print("\nBuilding classifiers...")
    classifiers = XVARSClassifiers(input_dim=1024, hidden_dim=512)
    classifiers = classifiers.to(DEVICE)
    total_params = sum(p.numel() for p in classifiers.parameters()
                       if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Loss and optimizer
    # Compute class weights from training data 
    from collections import Counter
    import numpy as np

    # Count foul labels in training set
    foul_counts = Counter(s["foul_label"] for s in train_samples)
    total = sum(foul_counts.values())

    # Weight = total / (num_classes * count)
    # Minority class gets higher weight
    num_classes = 2
    foul_weights = torch.tensor([
        total / (num_classes * foul_counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=torch.float32).to(DEVICE)

    # Same for severity
    sev_counts = Counter(s["sev_label"] for s in train_samples)
    num_sev_classes = 4
    sev_weights = torch.tensor([
        total / (num_sev_classes * sev_counts.get(i, 1))
        for i in range(num_sev_classes)
    ], dtype=torch.float32).to(DEVICE)

    print(f"Foul class weights: {foul_weights.tolist()}")
    print(f"Sev  class weights: {sev_weights.tolist()}")

    # Weighted loss — minority class penalized more when wrong
    foul_criterion = nn.CrossEntropyLoss(weight=foul_weights)
    sev_criterion  = nn.CrossEntropyLoss(weight=sev_weights)

    # Lower learning rate with scheduler
    optimizer = torch.optim.Adam(classifiers.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    # Training loop
    print("\nStarting fast training...")
    print("=" * 55)

    history       = []
    best_foul_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_foul, train_sev = train_one_epoch(
            classifiers, train_loader, optimizer,
            foul_criterion, sev_criterion, DEVICE, epoch
        )

        val_loss, val_foul, val_sev = evaluate(
            classifiers, valid_loader,
            foul_criterion, sev_criterion, DEVICE, "valid"
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  LR: {current_lr:.6f}")

        print(f"  Train — loss: {train_loss:.3f} | "
              f"foul: {train_foul:.1f}% | sev: {train_sev:.1f}%")
        print(f"  Valid — loss: {val_loss:.3f}   | "
              f"foul: {val_foul:.1f}%   | sev: {val_sev:.1f}%")

        history.append({
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_foul_acc" : train_foul,
            "train_sev_acc"  : train_sev,
            "val_loss"       : val_loss,
            "val_foul_acc"   : val_foul,
            "val_sev_acc"    : val_sev,
        })

        # Save best
        if val_foul > best_foul_acc:
            best_foul_acc = val_foul
            os.makedirs(f"{SAVE_DIR}/best", exist_ok=True)
            torch.save({
                "epoch"       : epoch,
                "model_state" : classifiers.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_foul_acc": val_foul,
                "val_sev_acc" : val_sev,
            }, f"{SAVE_DIR}/best/checkpoint_epoch_{epoch}.pt")
            print(f"  New best foul accuracy: {best_foul_acc:.1f}%")

        # Save every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch"       : epoch,
                "model_state" : classifiers.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_foul_acc": val_foul,
                "val_sev_acc" : val_sev,
            }, f"{SAVE_DIR}/checkpoint_epoch_{epoch}.pt")

    # Save history
    with open(f"{LOG_DIR}/training_history_fast.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print(f"Best validation foul accuracy: {best_foul_acc:.1f}%")
    print("=" * 55)