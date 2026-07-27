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
from src.models.hierarchical_classifiers import FoulClassifier
from src.dataset.annotation_loader import load_annotations, get_split_samples

class FeatureDataset(Dataset):
    """Loads pre-computed SigLIP features — keeps all 16 frames!"""
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
        cls_tokens = data["cls"]  # (16, 1152) — KEEP ALL 16 FRAMES!
        
        # L2 normalize each frame independently
        cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
        
        raw_label = torch.tensor(item["foul_label"], dtype=torch.long)
        return cls_tokens, raw_label

def make_balanced_sampler(dataset):
    binary_labels = [1 if item["foul_label"] > 0 else 0 for item in dataset.items]
    counts = Counter(binary_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in binary_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
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
    class_correct = [0, 0]
    class_total   = [0, 0]
    
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for frame_features, raw_labels in progress:
        frame_features = frame_features.to(device)  # (batch, 16, 1152)
        raw_labels = raw_labels.to(device)
        binary_labels = (raw_labels > 0).long()
        
        # Apply Mixup
        mixed_features, y_a, y_b, lam = mixup_data(frame_features, binary_labels, alpha=0.4, device=device)

        logits = model(mixed_features)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_samples += frame_features.shape[0]
        
        with torch.no_grad():
            clean_logits = model(frame_features)
            probs = torch.softmax(clean_logits, dim=-1)
            preds = (probs[:, 1] >= 0.60).long()
            
        correct += (preds == binary_labels).sum().item()
        
        for c in range(2):
            mask = binary_labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((preds == c) & mask).sum().item()

        progress.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / total_samples * 100:.1f}%"})

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(2)]
    balanced_acc = sum(recall) / 2.0
    return {"loss": total_loss / len(loader), "acc": acc, "recall": recall, "balanced_acc": balanced_acc}

def evaluate(model, loader, criterion, device, split_name):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    class_correct   = [0, 0]
    class_total     = [0, 0]
    class_predicted = [0, 0]
    
    with torch.no_grad():
        for frame_features, raw_labels in tqdm(loader, desc=f"[{split_name}]"):
            frame_features = frame_features.to(device)
            raw_labels = raw_labels.to(device)
            binary_labels = (raw_labels > 0).long()

            logits = model(frame_features)
            loss = criterion(logits, binary_labels)

            total_loss += loss.item()
            total_samples += frame_features.shape[0]
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] >= 0.60).long()
            correct += (preds == binary_labels).sum().item()
            
            for c in range(2):
                mask = binary_labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()
                class_predicted[c] += (preds == c).sum().item()

    acc = correct / total_samples * 100
    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(2)]
    balanced_acc = sum(recall) / 2.0
    return {"loss": total_loss / len(loader), "acc": acc, "recall": recall, "balanced_acc": balanced_acc,
            "predictions": class_predicted, "support": class_total}

if __name__ == "__main__":
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 30
    LR            = 1e-4
    SAVE_DIR      = "models/hierarchical"
    LOG_DIR       = "logs"
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    print(f"\n--- Starting Foul Detector (Temporal Attention) on {DEVICE.upper()} ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")
    
    train_foul_counts = Counter(1 if s["foul_label"] > 0 else 0 for s in train_samples)
    print(f"  Train distribution — No Foul: {train_foul_counts[0]}, Foul: {train_foul_counts[1]}")

    train_dataset = FeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = FeatureDataset(valid_samples, FEATURES_ROOT)

    train_sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FoulClassifier(feat_dim=1152, hidden_dim=128).to(DEVICE)
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS, pct_start=0.1
    )

    print(f"\n  Config: LR={LR} | BatchSize={BATCH_SIZE} | WeightDecay=5e-3")
    print(f"  Strategy: Temporal Attn + Frame Masking + Mixup + Balanced Sampler + 0.60 Threshold")
    print("=" * 70)

    history = []
    best_balanced_acc = 0.0
    best_metrics = {}
    no_improve = 0
    patience = 15

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, DEVICE, epoch)
        val_metrics = evaluate(model, valid_loader, criterion, DEVICE, "valid")
        
        print(f"\n  Train — loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.1f}% | balanced: {train_metrics['balanced_acc']:.1f}%")
        print(f"           recall [No Foul]: {train_metrics['recall'][0]:.1f}% | [Foul]: {train_metrics['recall'][1]:.1f}%")
        print(f"  Valid — loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.1f}% | balanced: {val_metrics['balanced_acc']:.1f}%")
        print(f"           recall [No Foul]: {val_metrics['recall'][0]:.1f}% | [Foul]: {val_metrics['recall'][1]:.1f}%")
        print(f"           predictions: No Foul={val_metrics['predictions'][0]}, Foul={val_metrics['predictions'][1]}")
        print(f"           support:     No Foul={val_metrics['support'][0]}, Foul={val_metrics['support'][1]}")
        
        history.append({
            "epoch": epoch, "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"], "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"], "val_recall": val_metrics["recall"],
        })
        
        if val_metrics["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = val_metrics["balanced_acc"]
            best_metrics = {
                "epoch": epoch,
                "train_acc": train_metrics["acc"],
                "train_recall": train_metrics["recall"],
                "val_acc": val_metrics["acc"],
                "val_recall": val_metrics["recall"],
            }
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_foul_model.pt")
            print(f"  ✓ Saved new best Foul Model (balanced acc: {best_balanced_acc:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping — no improvement for {patience} epochs")
                break

    with open(f"{LOG_DIR}/foul_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("FOUL DETECTOR TRAINING COMPLETE")
    print(f"  Best Model (Epoch {best_metrics.get('epoch', 'N/A')}):")
    print(f"  Validation Balanced Accuracy: {best_balanced_acc:.1f}%")
    if best_metrics:
        print(f"  Train Acc: {best_metrics['train_acc']:.1f}% | Valid Acc: {best_metrics['val_acc']:.1f}%")
        print(f"  Train Recall [No Foul]: {best_metrics['train_recall'][0]:.1f}% | [Foul]: {best_metrics['train_recall'][1]:.1f}%")
        print(f"  Valid Recall [No Foul]: {best_metrics['val_recall'][0]:.1f}% | [Foul]: {best_metrics['val_recall'][1]:.1f}%")
    print(f"  Model saved to: {SAVE_DIR}/best_foul_model.pt")
    print("=" * 70)