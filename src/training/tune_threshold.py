import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append('.')
from src.models.hierarchical_classifiers import FoulClassifier
from src.training.train_foul import FeatureDataset
from src.dataset.annotation_loader import load_annotations, get_split_samples

def evaluate_thresholds(model_path="models/hierarchical/best_foul_model.pt", features_root="data/features"):
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Loading model on {device.upper()}...")
    model = FoulClassifier(feat_dim=1152, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_samples = load_annotations()
    valid_samples = get_split_samples(all_samples, "valid")
    valid_dataset = FeatureDataset(valid_samples, features_root)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    all_probs = []
    all_labels = []

    print("Extracting validation probabilities...")
    with torch.no_grad():
        for frame_features, raw_labels in tqdm(valid_loader):
            frame_features = frame_features.to(device)
            binary_labels = (raw_labels > 0).long()
            
            logits = model(frame_features)
            probs = torch.softmax(logits, dim=-1)[:, 1] # Probability of Foul
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(binary_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n--- Threshold Tuning Results ---")
    print(f"{'Threshold':<10} | {'No Foul Recall':<15} | {'Foul Recall':<15} | {'Balanced Acc':<15}")
    print("-" * 65)

    best_thresh = 0.5
    best_bal_acc = 0.0

    for thresh in np.arange(0.3, 0.85, 0.05):
        preds = (all_probs >= thresh).astype(int)
        
        correct_no_foul = np.sum((preds == 0) & (all_labels == 0))
        total_no_foul = np.sum(all_labels == 0)
        recall_no_foul = correct_no_foul / total_no_foul * 100 if total_no_foul > 0 else 0

        correct_foul = np.sum((preds == 1) & (all_labels == 1))
        total_foul = np.sum(all_labels == 1)
        recall_foul = correct_foul / total_foul * 100 if total_foul > 0 else 0

        bal_acc = (recall_no_foul + recall_foul) / 2.0

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh

        print(f"{thresh:<10.2f} | {recall_no_foul:<15.1f} | {recall_foul:<15.1f} | {bal_acc:<15.1f}")

    print("-" * 65)
    print(f"Optimal Threshold: {best_thresh:.2f} (Balanced Acc: {best_bal_acc:.1f}%)")

if __name__ == "__main__":
    evaluate_thresholds()
