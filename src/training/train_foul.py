import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter

sys.path.append('.')
from src.models.hierarchical_classifiers import FoulClassifier
from src.dataset.annotation_loader import load_annotations, get_split_samples

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
        cls_tokens = data["cls"]               # (16, 1152)
        video_vector = cls_tokens.mean(dim=0)  # Average over 16 frames -> (1152,)
        
        raw_label = torch.tensor(item["foul_label"], dtype=torch.long)
        return video_vector, raw_label

def make_balanced_sampler(dataset):
    """Forces the DataLoader to pick exactly 50% Fouls and 50% No Fouls every batch"""
    # Temporarily convert 4-class labels to binary to count them
    binary_labels = [1 if item["foul_label"] > 0 else 0 for item in dataset.items]
    counts = Counter(binary_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in binary_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for video_vectors, raw_labels in progress:
        video_vectors = video_vectors.to(device)
        raw_labels = raw_labels.to(device)
        
        # MATH MAGIC: Convert 4-class labels into Binary (0 or 1)
        binary_labels = (raw_labels > 0).long()

        logits = model(video_vectors)
        loss = criterion(logits, binary_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_samples += video_vectors.shape[0]
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == binary_labels).sum().item()

        progress.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / total_samples * 100:.1f}%"})

    return total_loss / len(loader), correct / total_samples * 100

def evaluate(model, loader, criterion, device, split_name):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for video_vectors, raw_labels in tqdm(loader, desc=f"[{split_name}]"):
            video_vectors = video_vectors.to(device)
            raw_labels = raw_labels.to(device)
            
            #Same conversion for validation data
            binary_labels = (raw_labels > 0).long()

            logits = model(video_vectors)
            loss = criterion(logits, binary_labels)

            total_loss += loss.item()
            total_samples += video_vectors.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == binary_labels).sum().item()

    return {"loss": total_loss / len(loader), "acc": correct / total_samples * 100}

if __name__ == "__main__":
    FEATURES_ROOT = "data/features"
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 50
    LR            = 1e-4
    SAVE_DIR      = "models/hierarchical"
    
    # Device check for Intel Arc XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    print(f"\n--- Starting Foul Detector (Binary) Training on {DEVICE.upper()} ---")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_samples   = load_annotations()
    train_samples = get_split_samples(all_samples, "train")
    valid_samples = get_split_samples(all_samples, "valid")

    train_dataset = FeatureDataset(train_samples, FEATURES_ROOT)
    valid_dataset = FeatureDataset(valid_samples, FEATURES_ROOT)

    # Use the sampler for training so batches are balanced, but do NOT shuffle validation!
    train_sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load the Model 1 Architecture
    model = FoulClassifier(input_dim=1152, hidden_dim=512).to(DEVICE)
    
    # Because batches are strictly 50/50, we can use the incredibly fast CrossEntropyLoss!
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS, pct_start=0.1
    )

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, DEVICE, epoch)
        val_metrics = evaluate(model, valid_loader, criterion, DEVICE, "valid")
        
        print(f"  Valid — loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.1f}%")
        
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_foul_model.pt")
            print(f"  ✓ Saved new best Foul Model ({best_acc:.1f}% accuracy)")