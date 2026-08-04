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

sys.path.append('.')
from src.models.hierarchical_classifiers import SeverityClassifier
from src.dataset.annotation_loader import load_annotations, get_split_samples

SEV_NAMES = ["No Card", "Yellow", "Red"]  # After label shift (0, 1, 2)

class FeatureDataset(Dataset):
    """Loads pre-computed SigLIP features from data/features/"""
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
                    "sev_label" : s["sev_label"],
                })
                # Check for flipped version (data augmentation)
                flip_path = feat_path.replace(".pt", "_flip.pt")
                if s["split"] == "train" and os.path.exists(flip_path):
                    self.items.append({
                        "feat_path" : flip_path,
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
        cls_tokens = data["cls"]               # (16, 1152) — KEEP ALL 16 FRAMES!
        
        # L2 normalize each frame independently
        cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
        
        raw_label = torch.tensor(item["sev_label"], dtype=torch.long)
        return cls_tokens, raw_label

def make_balanced_sampler(dataset):
    """Balances No Card, Yellow, and Red so they appear equally in batches"""
    sev_labels = [item["sev_label"] for item in dataset.items]
    counts = Counter(sev_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in sev_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    print(f"  Balanced sampler class weights: {class_weights}")
    return sampler

def mixup_data(x, y, alpha=0.4, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    # Per-class tracking for recall (3 classes: No Card, Yellow, Red)
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]
    
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for video_vectors, raw_labels in progress:
        video_vectors = video_vectors.to(device)
        raw_labels = raw_labels.to(device)
        
        # Labels are already 0, 1, 2 (No Card, Yellow, Red)
        labels = raw_labels

        # Apply Mixup
        mixed_features, y_a, y_b, lam = mixup_data(video_vectors, labels, alpha=0.4, device=device)

        logits = model(mixed_features)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_samples += video_vectors.shape[0]
        
        with torch.no_grad():
            clean_logits = model(video_vectors)
            preds = torch.argmax(clean_logits, dim=-1)
            
        correct += (preds == labels).sum().item()
        
        # Track per-class recall
        for c in range(3):
            mask = labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((preds == c) & mask).sum().item()

        progress.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / total_samples * 100:.1f}%"})

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(3)]
    balanced_acc = sum(recall) / 3.0
    
    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "recall": recall,
        "balanced_acc": balanced_acc,
    }

def evaluate(model, loader, criterion, device, split_name):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    class_correct   = [0, 0, 0]
    class_total     = [0, 0, 0]
    class_predicted = [0, 0, 0]
    
    with torch.no_grad():
        for video_vectors, raw_labels in tqdm(loader, desc=f"[{split_name}]"):
            video_vectors = video_vectors.to(device)
            raw_labels = raw_labels.to(device)
            
            labels = raw_labels

            logits = model(video_vectors)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_samples += video_vectors.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            
            for c in range(3):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()
                class_predicted[c] += (preds == c).sum().item()

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(3)]
    balanced_acc = sum(recall) / 3.0
    
    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "recall": recall,
        "balanced_acc": balanced_acc,
        "predictions": class_predicted,
        "support": class_total,
    }

if __name__ == "__main__":
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 30
    LR            = 1e-4
    SAVE_DIR      = "models/hierarchical"
    LOG_DIR       = "logs"
    
    # Device check for Intel Arc XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    print(f"\n--- Starting Severity Detector (3-Class) Training on {DEVICE.upper()} ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_samples   = load_annotations()
    raw_train = get_split_samples(all_samples, "train")
    raw_valid = get_split_samples(all_samples, "valid")
    
    # Only keep clips that ARE fouls (foul_label > 0)
    train_samples = [s for s in raw_train if s["foul_label"] > 0]
    valid_samples = [s for s in raw_valid if s["foul_label"] > 0]
    
    print(f"  Filtered Train: {len(train_samples)} foul clips (removed {len(raw_train)-len(train_samples)} clean plays)")
    print(f"  Filtered Valid: {len(valid_samples)} foul clips (removed {len(raw_valid)-len(valid_samples)} clean plays)")
    
    # Print the 3-class distribution after filtering
    train_sev_counts = Counter(s["sev_label"] for s in train_samples)
    print(f"  Train distribution — No Card(0): {train_sev_counts.get(0,0)}, Yellow(1): {train_sev_counts.get(1,0)}, Red(2): {train_sev_counts.get(2,0)}")

    train_dataset = FeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = FeatureDataset(valid_samples, FEATURES_ROOT)

    train_sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SeverityClassifier(feat_dim=1152, hidden_dim=128).to(DEVICE)
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS, pct_start=0.1
    )

    print(f"\n  Config: LR={LR} | BatchSize={BATCH_SIZE} | WeightDecay=5e-3 | Epochs={NUM_EPOCHS}")
    print("=" * 70)

    history = []
    best_balanced_acc = 0.0
    no_improve = 0
    patience = 15

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, DEVICE, epoch)
        val_metrics = evaluate(model, valid_loader, criterion, DEVICE, "valid")
        
        # Print BOTH train and validation side by side!
        print(f"\n  Train — loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.1f}% | balanced: {train_metrics['balanced_acc']:.1f}%")
        print(f"           recall [No Card]: {train_metrics['recall'][0]:.1f}% | [Yellow]: {train_metrics['recall'][1]:.1f}% | [Red]: {train_metrics['recall'][2]:.1f}%")
        print(f"  Valid — loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.1f}% | balanced: {val_metrics['balanced_acc']:.1f}%")
        print(f"           recall [No Card]: {val_metrics['recall'][0]:.1f}% | [Yellow]: {val_metrics['recall'][1]:.1f}% | [Red]: {val_metrics['recall'][2]:.1f}%")
        print(f"           predictions: No Card={val_metrics['predictions'][0]}, Yellow={val_metrics['predictions'][1]}, Red={val_metrics['predictions'][2]}")
        print(f"           support:     No Card={val_metrics['support'][0]}, Yellow={val_metrics['support'][1]}, Red={val_metrics['support'][2]}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_recall": val_metrics["recall"],
        })
        
        # Use balanced accuracy to decide the best model
        if val_metrics["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = val_metrics["balanced_acc"]
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_severity_model.pt")
            print(f"  ✓ Saved new best Severity Model (balanced acc: {best_balanced_acc:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping — no improvement for {patience} epochs")
                break

    # Save training history for plotting later
    with open(f"{LOG_DIR}/severity_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("SEVERITY DETECTOR TRAINING COMPLETE")
    print(f"  Best validation balanced accuracy: {best_balanced_acc:.1f}%")
    print(f"  Model saved to: {SAVE_DIR}/best_severity_model.pt")
    print(f"  History saved to: {LOG_DIR}/severity_training_history.json")
    print("=" * 70)