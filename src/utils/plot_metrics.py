import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataset.annotation_loader import load_annotations

# Ensure plots directory exists in the project
SCRATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../plots'))
os.makedirs(SCRATCH_DIR, exist_ok=True)

def plot_distribution():
    print("Plotting dataset distribution...")
    try:
        samples = load_annotations()
    except Exception as e:
        print(f"Could not load annotations: {e}")
        return

    # Count distributions
    foul_labels = []
    sev_labels = []
    
    # Mapping back to text for plotting
    foul_map = {0: "No Foul", 1: "Foul"}
    sev_map = {0: "No Card", 1: "No card+", 2: "Yellow", 3: "Red"}
    
    for s in samples:
        foul_labels.append(foul_map[s["foul_label"]])
        # Only plot severity for actual fouls, or all? Let's plot for all
        sev_labels.append(sev_map[s["sev_label"]])
                
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Foul Distribution
    foul_counts = Counter(foul_labels)
    foul_df = pd.DataFrame(list(foul_counts.items()), columns=["Label", "Count"])
    sns.barplot(data=foul_df, x="Label", y="Count", ax=ax1, palette=["#3498db", "#e74c3c"])
    ax1.set_title("Foul vs No Foul Distribution", fontsize=14, weight='bold')
    for i, v in enumerate(foul_df["Count"]):
        ax1.text(i, v + 20, str(v), ha='center', fontsize=12)

    # Plot Severity Distribution
    sev_counts = Counter(sev_labels)
    # Ensure correct order
    order = ["No Card", "No card+", "Yellow", "Red"]
    sev_df = pd.DataFrame([{"Label": k, "Count": sev_counts[k]} for k in order if k in sev_counts])
    sns.barplot(data=sev_df, x="Label", y="Count", ax=ax2, palette=["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
    ax2.set_title("Severity Distribution (All actions)", fontsize=14, weight='bold')
    for i, v in enumerate(sev_df["Count"]):
        ax2.text(i, v + 10, str(v), ha='center', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(SCRATCH_DIR, "dataset_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved distribution plot to {save_path}")

def plot_training():
    print("Plotting training history...")
    try:
        with open("logs/training_history_fast.json", "r") as f:
            history = json.load(f)
    except Exception as e:
        print(f"Could not load history: {e}")
        return

    df = pd.DataFrame(history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=ax1, label="Train Loss", linewidth=2)
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=ax1, label="Valid Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, weight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Accuracy
    sns.lineplot(data=df, x="epoch", y="train_foul_acc", ax=ax2, label="Train Foul (Raw)", linewidth=2, color="#3498db")
    sns.lineplot(data=df, x="epoch", y="val_foul_acc", ax=ax2, label="Valid Foul (Raw)", linewidth=2, linestyle='--', color="#3498db")
    sns.lineplot(data=df, x="epoch", y="val_foul_balanced_acc", ax=ax2, label="Valid Foul (Balanced)", linewidth=3, color="#2ecc71")
    
    ax2.set_title("Model Accuracy over Time", fontsize=14, weight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(SCRATCH_DIR, "training_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {save_path}")

if __name__ == "__main__":
    plot_distribution()
    plot_training()
