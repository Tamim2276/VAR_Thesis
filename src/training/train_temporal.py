import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter

sys.path.append('.')
from src.models.temporal_transformer import XVARSTemporalModel
from src.dataset.annotation_loader import load_annotations, get_split_samples


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce_loss = -log_pt

        if self.alpha is not None:
            ce_loss = self.alpha[targets] * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def inverse_frequency_weights(samples, label_key, num_classes, device):
    counts = Counter(sample[label_key] for sample in samples)
    total = len(samples)
    weights = [total / (num_classes * counts.get(class_id, 1))
               for class_id in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


class TemporalFeatureDataset(Dataset):
    """
    Loads full (T, 1024) sequence of frame features for Temporal Transformer.
    No mean pooling is performed beforehand!
    """

    def __init__(self, samples, features_root="data/features"):
        self.items = []
        skipped = 0

        for s in samples:
            feat_path = (
                s["clip_path"]
                .replace("data/frames",  features_root)
                .replace("data\\frames", features_root)
                .replace(".npy", ".pt")
                .replace("\\", "/")
            )

            if os.path.exists(feat_path):
                self.items.append({
                    "feat_path" : feat_path,
                    "foul_label": s["foul_label"],
                    "sev_label" : s["sev_label"],
                })
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Warning: {skipped} clips missing feature file — skipped")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        data = torch.load(item["feat_path"], map_location="cpu", weights_only=True)
        cls_tokens = data["cls"]  # (16, 1024) full frame sequence

        foul_label = torch.tensor(item["foul_label"], dtype=torch.long)
        sev_label  = torch.tensor(item["sev_label"],  dtype=torch.long)

        return cls_tokens, foul_label, sev_label


def train_one_epoch(model, loader, optimizer, foul_criterion, sev_criterion, scheduler, device, epoch):
    model.train()

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for frame_seqs, foul_labels, sev_labels in progress:
        frame_seqs  = frame_seqs.to(device)
        foul_labels = foul_labels.to(device)
        sev_labels  = sev_labels.to(device)

        foul_logits, sev_logits, _ = model(frame_seqs)

        foul_loss = foul_criterion(foul_logits, foul_labels)
        sev_loss  = sev_criterion(sev_logits,  sev_labels)
        loss      = foul_loss + sev_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()
        total_samples += frame_seqs.shape[0]

        foul_preds = torch.argmax(foul_logits, dim=-1)
        sev_preds  = torch.argmax(sev_logits,  dim=-1)
        foul_correct += (foul_preds == foul_labels).sum().item()
        sev_correct  += (sev_preds  == sev_labels).sum().item()

        progress.set_postfix({
            "loss" : f"{loss.item():.3f}",
            "foul" : f"{foul_correct / total_samples * 100:.1f}%",
            "sev"  : f"{sev_correct  / total_samples * 100:.1f}%",
        })

    return (
        total_loss   / len(loader),
        foul_correct / total_samples * 100,
        sev_correct  / total_samples * 100,
    )


def evaluate(model, loader, foul_criterion, sev_criterion, device, split_name):
    model.eval()

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0
    foul_support  = [0, 0]
    foul_hits     = [0, 0]
    foul_predicted = [0, 0]

    with torch.no_grad():
        for frame_seqs, foul_labels, sev_labels in tqdm(loader, desc=f"[{split_name}]"):
            frame_seqs  = frame_seqs.to(device)
            foul_labels = foul_labels.to(device)
            sev_labels  = sev_labels.to(device)

            foul_logits, sev_logits, _ = model(frame_seqs)

            loss = foul_criterion(foul_logits, foul_labels) + sev_criterion(sev_logits, sev_labels)

            total_loss    += loss.item()
            total_samples += frame_seqs.shape[0]

            foul_preds = torch.argmax(foul_logits, dim=-1)
            sev_preds  = torch.argmax(sev_logits,  dim=-1)
            foul_correct += (foul_preds == foul_labels).sum().item()
            sev_correct  += (sev_preds  == sev_labels).sum().item()

            for class_id in range(2):
                class_mask = foul_labels == class_id
                foul_support[class_id] += class_mask.sum().item()
                foul_hits[class_id] += ((foul_preds == class_id) & class_mask).sum().item()
                foul_predicted[class_id] += (foul_preds == class_id).sum().item()

    foul_recall = [
        100 * hits / support if support else 0.0
        for hits, support in zip(foul_hits, foul_support)
    ]
    return {
        "loss": total_loss / len(loader),
        "foul_acc": foul_correct / total_samples * 100,
        "sev_acc": sev_correct / total_samples * 100,
        "foul_balanced_acc": sum(foul_recall) / len(foul_recall),
        "foul_recall": foul_recall,
        "foul_predictions": foul_predicted,
    }


def make_balanced_sampler(dataset):
    foul_labels = [item["foul_label"] for item in dataset.items]
    counts = Counter(foul_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in foul_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    print(f"  Balanced sampler: class weights = {class_weights}")
    return sampler


if __name__ == "__main__":
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 50
    LR            = 5e-5
    DEVICE        = "xpu"
    SAVE_DIR      = "models/temporal"
    LOG_DIR       = "logs"
    FOCAL_GAMMA   = 3.0

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    if DEVICE == "xpu" and not torch.xpu.is_available():
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    print("\nLoading annotations...")
    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")

    print("\nBuilding temporal feature datasets...")
    train_dataset = TemporalFeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = TemporalFeatureDataset(valid_samples, FEATURES_ROOT)
    print(f"  Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    train_sampler = make_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    print(f"  Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    sev_weights, _ = inverse_frequency_weights(
        train_dataset.items, "sev_label", num_classes=4, device=DEVICE
    )

    print("\nBuilding Temporal Transformer Model...")
    model = XVARSTemporalModel(
        embed_dim=1024,
        num_heads=8,
        num_layers=2,
        hidden_dim=512,
        max_frames=64,
        dropout=0.1
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    foul_criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=None)
    sev_criterion  = FocalLoss(gamma=FOCAL_GAMMA, alpha=sev_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=5e-3
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    print(f"\nStarting Temporal Transformer Training...")
    print("=" * 60)

    history = []
    best_foul_balanced_acc = float("-inf")
    best_foul_acc = 0.0
    best_sev_acc  = 0.0
    no_improve    = 0
    patience      = 15

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_foul, train_sev = train_one_epoch(
            model, train_loader, optimizer,
            foul_criterion, sev_criterion, scheduler, DEVICE, epoch
        )

        val_metrics = evaluate(
            model, valid_loader,
            foul_criterion, sev_criterion, DEVICE, "valid"
        )
        val_loss = val_metrics["loss"]
        val_foul = val_metrics["foul_acc"]
        val_sev = val_metrics["sev_acc"]
        val_foul_balanced = val_metrics["foul_balanced_acc"]

        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train — loss: {train_loss:.4f} | foul: {train_foul:.1f}% | sev: {train_sev:.1f}%")
        print(f"  Valid — loss: {val_loss:.4f}   | foul: {val_foul:.1f}%   | sev: {val_sev:.1f}%")
        print(f"  Foul balanced: {val_foul_balanced:.1f}% | recall [No foul, Foul]: {val_metrics['foul_recall']} | predictions: {val_metrics['foul_predictions']}")

        history.append({
            "epoch"                : epoch,
            "train_loss"           : train_loss,
            "train_foul_acc"       : train_foul,
            "train_sev_acc"        : train_sev,
            "val_loss"             : val_loss,
            "val_foul_acc"         : val_foul,
            "val_foul_balanced_acc": val_foul_balanced,
            "val_foul_recall"      : val_metrics["foul_recall"],
            "val_sev_acc"          : val_sev,
            "lr"                   : current_lr,
        })

        improved = False
        if val_foul_balanced > best_foul_balanced_acc:
            best_foul_balanced_acc = val_foul_balanced
            best_foul_acc = val_foul
            best_sev_acc  = val_sev
            torch.save({
                "epoch"                : epoch,
                "model_state"          : model.state_dict(),
                "optim_state"          : optimizer.state_dict(),
                "val_foul_acc"         : val_foul,
                "val_foul_balanced_acc": val_foul_balanced,
                "val_sev_acc"          : val_sev,
            }, f"{SAVE_DIR}/best_temporal_model.pt")
            print(f"  ✓ New best temporal model — foul: {best_foul_acc:.1f}% | sev: {best_sev_acc:.1f}% | balanced: {best_foul_balanced_acc:.1f}%")
            improved = True
            no_improve = 0

        if not improved:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping — no improvement for {patience} epochs")
                break

    with open(f"{LOG_DIR}/training_history_temporal.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TEMPORAL TRANSFORMER TRAINING COMPLETE")
    print(f"Best validation balanced foul accuracy: {best_foul_balanced_acc:.1f}%")
    print(f"Best validation foul accuracy         : {best_foul_acc:.1f}%")
    print(f"Best validation sev  accuracy          : {best_sev_acc:.1f}%")
    print(f"Checkpoint saved to                  : {SAVE_DIR}/best_temporal_model.pt")
    print("=" * 60)
