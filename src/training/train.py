import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
sys.path.append('.')

from src.models.clip_extractor import load_clip_model, extract_spatial_tokens
from src.models.classifier_heads import XVARSClassifiers
from src.dataset.annotation_loader import load_annotations, get_split_samples



# DATASET

class FoulDataset(Dataset):
    """
    PyTorch Dataset that loads pre-extracted frames and their labels.
    DataLoader calls __len__ and __getitem__ automatically
    to feed batches of data to the model during training.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = np.load(sample["clip_path"])           # (16, 224, 224, 3) uint8
        frames = torch.from_numpy(frames).float()        # convert to float tensor
        foul_label = torch.tensor(sample["foul_label"], dtype=torch.long)
        sev_label = torch.tensor(sample["sev_label"],  dtype=torch.long)
        return frames, foul_label, sev_label


# TRAINING FUNCTIONS

def train_one_epoch(clip_model, preprocess, classifiers, dataloader,
                    optimizer, foul_criterion, sev_criterion, device, epoch):
    
    classifiers.train()  # enable dropout

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [train]")

    for batch_idx, (frames_batch, foul_labels, sev_labels) in enumerate(progress):

        batch_size = frames_batch.shape[0]
        video_vectors = []

        for i in range(batch_size):
            # (16, 224, 224, 3) float → uint8 numpy for CLIP preprocessor
            frames_numpy = frames_batch[i].numpy().astype(np.uint8)

            with torch.no_grad():  # CLIP is frozen
                cls_tokens, _ = extract_spatial_tokens(
                    clip_model, preprocess, frames_numpy
                )

            # average 16 frame tokens → one video vector (1024,)
            video_vector = cls_tokens.mean(dim=0)
            video_vectors.append(video_vector)

        # stack into batch (batch_size, 1024)
        video_vectors = torch.stack(video_vectors, dim=0).to(device)

        foul_labels = foul_labels.to(device)
        sev_labels  = sev_labels.to(device)

        # forward pass through classifier heads
        foul_logits, sev_logits = classifiers(video_vectors)

        # compute losses
        foul_loss = foul_criterion(foul_logits, foul_labels)
        sev_loss  = sev_criterion(sev_logits,  sev_labels)
        loss      = foul_loss + sev_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track metrics
        total_loss    += loss.item()
        total_samples += batch_size

        foul_preds = torch.argmax(foul_logits, dim=-1)
        sev_preds  = torch.argmax(sev_logits,  dim=-1)
        foul_correct += (foul_preds == foul_labels).sum().item()
        sev_correct  += (sev_preds  == sev_labels).sum().item()

        progress.set_postfix({
            "loss"    : f"{loss.item():.3f}",
            "foul_acc": f"{foul_correct/total_samples*100:.1f}%",
            "sev_acc" : f"{sev_correct/total_samples*100:.1f}%",
        })

    avg_loss = total_loss    / len(dataloader)
    foul_acc = foul_correct  / total_samples * 100
    sev_acc  = sev_correct   / total_samples * 100

    return avg_loss, foul_acc, sev_acc


def evaluate(clip_model, preprocess, classifiers, dataloader,
             foul_criterion, sev_criterion, device, split_name):
    """
    Evaluate on validation or test set.
    Same as training but no gradient updates.
    """
    classifiers.eval()  # disable dropout

    total_loss    = 0.0
    foul_correct  = 0
    sev_correct   = 0
    total_samples = 0

    progress = tqdm(dataloader, desc=f"[{split_name}]")

    with torch.no_grad():
        for frames_batch, foul_labels, sev_labels in progress:

            batch_size = frames_batch.shape[0]
            video_vectors = []

            for i in range(batch_size):
                frames_numpy = frames_batch[i].numpy().astype(np.uint8)
                cls_tokens, _ = extract_spatial_tokens(
                    clip_model, preprocess, frames_numpy
                )
                video_vectors.append(cls_tokens.mean(dim=0))

            video_vectors = torch.stack(video_vectors, dim=0).to(device)
            foul_labels   = foul_labels.to(device)
            sev_labels    = sev_labels.to(device)

            foul_logits, sev_logits = classifiers(video_vectors)
            foul_loss = foul_criterion(foul_logits, foul_labels)
            sev_loss  = sev_criterion(sev_logits,  sev_labels)
            loss      = foul_loss + sev_loss

            total_loss    += loss.item()
            total_samples += batch_size

            foul_preds = torch.argmax(foul_logits, dim=-1)
            sev_preds  = torch.argmax(sev_logits,  dim=-1)
            foul_correct += (foul_preds == foul_labels).sum().item()
            sev_correct  += (sev_preds  == sev_labels).sum().item()

    avg_loss = total_loss   / len(dataloader)
    foul_acc = foul_correct / total_samples * 100
    sev_acc  = sev_correct  / total_samples * 100

    return avg_loss, foul_acc, sev_acc


def save_checkpoint(classifiers, optimizer, epoch,
                    val_foul_acc, val_sev_acc, save_dir="models"):
    """Save model weights to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch"        : epoch,
        "model_state"  : classifiers.state_dict(),
        "optim_state"  : optimizer.state_dict(),
        "val_foul_acc" : val_foul_acc,
        "val_sev_acc"  : val_sev_acc,
    }, path)
    print(f"  Saved: {path}")
    return path



# MAIN


if __name__ == "__main__":

    # Config
    BATCH_SIZE  = 4      # small batch fits in 12GB VRAM
    NUM_EPOCHS  = 10
    LR          = 1e-4
    SAVE_EVERY  = 2      # save checkpoint every N epochs
    DEVICE      = "xpu"  # confirmed working on XPU first

    print("="*50)
    print("X-VARS TRAINING")
    print("="*50)

    # Verify XPU
    if DEVICE == "xpu":
        if not torch.xpu.is_available():
            print("WARNING: XPU not available, falling back to CPU")
            DEVICE = "cpu"
        else:
            print(f"Using Intel Arc B580 (XPU)")

    print(f"Device: {DEVICE}")

    # Step 1: Load CLIP (frozen)
    print("\nStep 1: Loading CLIP...")
    clip_model, preprocess = load_clip_model()
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("  CLIP loaded and frozen.")

    # Step 2: Build classifiers
    print("\nStep 2: Building classifiers...")
    classifiers = XVARSClassifiers(input_dim=1024, hidden_dim=512)
    classifiers = classifiers.to(DEVICE)
    total_params = sum(p.numel() for p in classifiers.parameters()
                       if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Step 3: Load annotations
    print("\nStep 3: Loading annotations...")
    all_samples    = load_annotations()
    train_samples  = get_split_samples(all_samples, "train")
    valid_samples  = get_split_samples(all_samples, "valid")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Valid: {len(valid_samples)} samples")

    # Step 4: Build datasets and dataloaders
    print("\nStep 4: Building dataloaders...")
    train_dataset = FoulDataset(train_samples)
    valid_dataset = FoulDataset(valid_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")

    # Step 5: Loss functions and optimizer
    print("\nStep 5: Setting up loss and optimizer...")
    foul_criterion = nn.CrossEntropyLoss()
    sev_criterion  = nn.CrossEntropyLoss()
    optimizer      = torch.optim.Adam(
        classifiers.parameters(),
        lr=LR
    )
    print(f"  Loss     : CrossEntropyLoss")
    print(f"  Optimizer: Adam, lr={LR}")

    # Step 6: Training loop
    print("\nStep 6: Starting training...")
    print("="*50)

    history          = []
    best_val_foul_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # train
        train_loss, train_foul_acc, train_sev_acc = train_one_epoch(
            clip_model, preprocess, classifiers,
            train_loader, optimizer,
            foul_criterion, sev_criterion,
            DEVICE, epoch
        )

        # validate
        val_loss, val_foul_acc, val_sev_acc = evaluate(
            clip_model, preprocess, classifiers,
            valid_loader, foul_criterion, sev_criterion,
            DEVICE, "valid"
        )

        # print results
        print(f"\n  Train — loss: {train_loss:.3f} | "
              f"foul: {train_foul_acc:.1f}% | sev: {train_sev_acc:.1f}%")
        print(f"  Valid — loss: {val_loss:.3f} | "
              f"foul: {val_foul_acc:.1f}% | sev: {val_sev_acc:.1f}%")

        # save history
        history.append({
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_foul_acc" : train_foul_acc,
            "train_sev_acc"  : train_sev_acc,
            "val_loss"       : val_loss,
            "val_foul_acc"   : val_foul_acc,
            "val_sev_acc"    : val_sev_acc,
        })

        # save checkpoint every N epochs
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(
                classifiers, optimizer, epoch,
                val_foul_acc, val_sev_acc
            )

        # save best model
        if val_foul_acc > best_val_foul_acc:
            best_val_foul_acc = val_foul_acc
            save_checkpoint(
                classifiers, optimizer, epoch,
                val_foul_acc, val_sev_acc,
                save_dir="models/best"
            )
            print(f"  New best foul accuracy: {best_val_foul_acc:.1f}%")

    # save training history to disk
    os.makedirs("logs", exist_ok=True)
    with open("logs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nHistory saved to logs/training_history.json")

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Best validation foul accuracy: {best_val_foul_acc:.1f}%")
    print("="*50)