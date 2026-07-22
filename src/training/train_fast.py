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
from src.models.classifier_heads import XVARSClassifiers
from src.dataset.annotation_loader import load_annotations, get_split_samples


# Focal Loss
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Designed specifically for datasets where one class dominates.

    How it works:
    - Easy examples (model already confident) get DOWN-weighted
    - Hard examples (model uncertain) get UP-weighted
    - gamma=2 is the standard value from the original paper

    Compare to CrossEntropyLoss:
    - CrossEntropy treats all examples equally
    - alpha additionally increases the loss from minority classes
    """
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

        # alpha upweights rare classes; the focal term downweights easy ones.
        if self.alpha is not None:
            ce_loss = self.alpha[targets] * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# Dataset and training functions are below. These are used in train_fast.py to train the classifier heads on pre-computed CLIP features.
def inverse_frequency_weights(samples, label_key, num_classes, device):
    """Give each class equal total contribution to the training loss."""
    counts = Counter(sample[label_key] for sample in samples)
    total = len(samples)
    weights = [total / (num_classes * counts.get(class_id, 1))
               for class_id in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


class FeatureDataset(Dataset):
    """
    Loads pre-computed CLIP features from data/features/.
    Each file is a tiny .pt tensor — loads instantly.
    No CLIP inference happens during training at all.
    """

    def __init__(self, samples, features_root="data/features"):
        self.items   = []
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

        data         = torch.load(item["feat_path"], map_location="cpu",
                                  weights_only=True)
        cls_tokens   = data["cls"]              # (16, 1024)
        video_vector = cls_tokens.mean(dim=0)   # (1024,)  average over frames

        foul_label = torch.tensor(item["foul_label"], dtype=torch.long)
        sev_label  = torch.tensor(item["sev_label"],  dtype=torch.long)

        return video_vector, foul_label, sev_label


# Training functions 
def train_one_epoch(classifiers, loader, optimizer,
                    foul_criterion, sev_criterion, scheduler, device, epoch):
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

        foul_loss = foul_criterion(foul_logits, foul_labels)
        sev_loss  = sev_criterion(sev_logits,  sev_labels)
        loss      = foul_loss + sev_loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents loss explosion
        torch.nn.utils.clip_grad_norm_(classifiers.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()
        total_samples += video_vectors.shape[0]

        foul_preds = torch.argmax(foul_logits, dim=-1)
        sev_preds  = torch.argmax(sev_logits,  dim=-1)
        foul_correct += (foul_preds == foul_labels).sum().item()
        sev_correct  += (sev_preds  == sev_labels).sum().item()

        progress.set_postfix({
            "loss"    : f"{loss.item():.3f}",
            "foul"    : f"{foul_correct / total_samples * 100:.1f}%",
            "sev"     : f"{sev_correct  / total_samples * 100:.1f}%",
        })

    return (
        total_loss   / len(loader),
        foul_correct / total_samples * 100,
        sev_correct  / total_samples * 100,
    )


def evaluate(classifiers, loader, foul_criterion,
             sev_criterion, device, split_name):
    classifiers.eval()

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0
    foul_support  = [0, 0]
    foul_hits     = [0, 0]
    foul_predicted = [0, 0]

    with torch.no_grad():
        for video_vectors, foul_labels, sev_labels in tqdm(
            loader, desc=f"[{split_name}]"
        ):
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

            for class_id in range(2):
                class_mask = foul_labels == class_id
                foul_support[class_id] += class_mask.sum().item()
                foul_hits[class_id] += (
                    (foul_preds == class_id) & class_mask
                ).sum().item()
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


def print_class_distribution(train_samples):
    """Print label distribution so we know how imbalanced the data is."""
    foul_counts = Counter(s["foul_label"] for s in train_samples)
    sev_counts  = Counter(s["sev_label"]  for s in train_samples)
    total = len(train_samples)

    foul_names = {0: "No foul", 1: "Foul"}
    sev_names  = {0: "No card", 1: "No card+", 2: "Yellow", 3: "Red"}

    print("\n  Foul label distribution:")
    for k, v in sorted(foul_counts.items()):
        print(f"    {foul_names[k]:12s}: {v:5d}  ({v/total*100:.1f}%)")

    print("  Severity label distribution:")
    for k, v in sorted(sev_counts.items()):
        print(f"    {sev_names[k]:12s}: {v:5d}  ({v/total*100:.1f}%)")


def make_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler that oversamples the minority class.
    
    Without this: each batch is ~86% foul, ~14% no-foul (matches dataset ratio)
    With this:    each batch is ~50% foul, ~50% no-foul (balanced)
    
    The model is FORCED to see equal numbers of both classes,
    so it can't take the shortcut of always predicting "foul".
    """
    foul_labels = [item["foul_label"] for item in dataset.items]
    counts = Counter(foul_labels)
    
    # Weight for each sample = 1 / (number of samples in its class)
    # This makes the total weight of each class equal
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in foul_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),  # sample same number as dataset size
        replacement=True           # must be True for oversampling minority
    )
    
    print(f"  Balanced sampler: class weights = {class_weights}")
    return sampler


# Main
if __name__ == "__main__":

    # Config 
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 50
    LR            = 5e-5        # lowered from 1e-4 for more stable training
    DEVICE        = "xpu"
    SAVE_DIR      = "models"
    LOG_DIR       = "logs"
    FOCAL_GAMMA   = 3.0         # increased from 2.0 — more aggressive on easy examples


    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    # Device check
    if DEVICE == "xpu" and not torch.xpu.is_available():
        print("XPU not available, falling back to CPU")
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    # Annotations
    print("\nLoading annotations...")
    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")
    print_class_distribution(train_samples)

    # Datasets
    print("\nBuilding feature datasets...")
    train_dataset = FeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = FeatureDataset(valid_samples, FEATURES_ROOT)
    print(f"  Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    # Balanced sampler — forces 50/50 foul/no-foul in each batch
    train_sampler = make_balanced_sampler(train_dataset)

    # Note: when using a sampler, shuffle must be False
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    print(f"  Train batches: {len(train_loader)} | "
          f"Valid batches: {len(valid_loader)}")

    foul_weights, _ = inverse_frequency_weights(
        train_dataset.items, "foul_label", num_classes=2, device=DEVICE
    )
    sev_weights, _ = inverse_frequency_weights(
        train_dataset.items, "sev_label", num_classes=4, device=DEVICE
    )
    print(f"  Cached-train foul loss weights: {foul_weights.tolist()}")
    print(f"  Cached-train severity loss weights: {sev_weights.tolist()}")

    # Model
    print("\nBuilding classifiers...")
    classifiers = XVARSClassifiers(input_dim=1024, hidden_dim=512)
    classifiers = classifiers.to(DEVICE)
    total_params = sum(
        p.numel() for p in classifiers.parameters() if p.requires_grad
    )
    print(f"  Trainable parameters: {total_params:,}")

    # Loss: Focal Loss with higher gamma for more aggressive downweighting
    foul_criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=foul_weights)
    sev_criterion  = FocalLoss(gamma=FOCAL_GAMMA, alpha=sev_weights)

    # Optimizer with stronger weight decay for regularization
    optimizer = torch.optim.AdamW(
        classifiers.parameters(),
        lr=LR,
        weight_decay=5e-3   # increased from 1e-4 — fights overfitting
    )

    # Warmup for first 10% then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1,      # 10% warmup
        anneal_strategy='cos'
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"  LR: {LR} | Weight decay: 5e-3 | Focal gamma: {FOCAL_GAMMA}")
    print(f"  Balanced sampling: ON")
    print("=" * 60)

    history       = []
    best_foul_balanced_acc = float("-inf")
    best_foul_acc = 0.0
    best_sev_acc  = 0.0
    no_improve    = 0
    patience      = 15   # stop if no improvement for 15 epochs

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_foul, train_sev = train_one_epoch(
            classifiers, train_loader, optimizer,
            foul_criterion, sev_criterion, scheduler, DEVICE, epoch
        )

        val_metrics = evaluate(
            classifiers, valid_loader,
            foul_criterion, sev_criterion, DEVICE, "valid"
        )
        val_loss = val_metrics["loss"]
        val_foul = val_metrics["foul_acc"]
        val_sev = val_metrics["sev_acc"]
        val_foul_balanced = val_metrics["foul_balanced_acc"]

        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train — loss: {train_loss:.4f} | "
              f"foul: {train_foul:.1f}% | sev: {train_sev:.1f}%")
        print(f"  Valid — loss: {val_loss:.4f}   | "
              f"foul: {val_foul:.1f}%   | sev: {val_sev:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Foul balanced: {val_foul_balanced:.1f}% | "
              f"recall [No foul, Foul]: {val_metrics['foul_recall']} | "
              f"predictions: {val_metrics['foul_predictions']}")

        history.append({
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_foul_acc" : train_foul,
            "train_sev_acc"  : train_sev,
            "val_loss"       : val_loss,
            "val_foul_acc"   : val_foul,
            "val_foul_balanced_acc": val_foul_balanced,
            "val_foul_recall": val_metrics["foul_recall"],
            "val_sev_acc"    : val_sev,
            "lr"             : current_lr,
        })

        # Raw accuracy rewards the all-majority-class baseline. Select and
        # early-stop using mean per-class foul recall instead.
        improved = False
        if val_foul_balanced > best_foul_balanced_acc:
            best_foul_balanced_acc = val_foul_balanced
            best_foul_acc = val_foul
            best_sev_acc  = val_sev
            os.makedirs(f"{SAVE_DIR}/best", exist_ok=True)
            torch.save({
                "epoch"       : epoch,
                "model_state" : classifiers.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_foul_acc": val_foul,
                "val_foul_balanced_acc": val_foul_balanced,
                "val_sev_acc" : val_sev,
            }, f"{SAVE_DIR}/best/checkpoint_epoch_{epoch}.pt")
            print(f"  ✓ New best — foul: {best_foul_acc:.1f}% | "
                  f"sev: {best_sev_acc:.1f}%")
            improved = True
            no_improve = 0
            print(f"  Selection metric — balanced foul: "
                  f"{best_foul_balanced_acc:.1f}%")

        if not improved:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping — no improvement for {patience} epochs")
                break

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch"       : epoch,
                "model_state" : classifiers.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_foul_acc": val_foul,
                "val_foul_balanced_acc": val_foul_balanced,
                "val_sev_acc" : val_sev,
            }, f"{SAVE_DIR}/checkpoint_epoch_{epoch}.pt")

    # Save history
    with open(f"{LOG_DIR}/training_history_fast.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation balanced foul accuracy: "
          f"{best_foul_balanced_acc:.1f}%")
    print(f"Best validation foul accuracy : {best_foul_acc:.1f}%")
    print(f"Best validation sev  accuracy : {best_sev_acc:.1f}%")
    print(f"Checkpoint saved to           : {SAVE_DIR}/best/")
    print("=" * 60)