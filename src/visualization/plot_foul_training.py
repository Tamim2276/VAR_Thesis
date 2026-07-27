import os
import json
import matplotlib.pyplot as plt

def plot_training_history(log_path="logs/foul_training_history.json", save_path="logs/foul_training_plot.png"):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    with open(log_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_bal_acc = [h["train_balanced_acc"] for h in history]
    val_bal_acc = [h["val_balanced_acc"] for h in history]
    
    val_recall_no_foul = [h["val_recall"][0] for h in history]
    val_recall_foul = [h["val_recall"][1] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Foul Detector (Temporal Attention) Training History", fontsize=16)

    # Plot 1: Loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Cross Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Accuracy
    ax2.plot(epochs, train_acc, label='Train Acc', color='blue', linestyle='--')
    ax2.plot(epochs, val_acc, label='Valid Acc', color='red', linestyle='--')
    ax2.plot(epochs, train_bal_acc, label='Train Balanced Acc', color='blue')
    ax2.plot(epochs, val_bal_acc, label='Valid Balanced Acc', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy & Balanced Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Recall
    ax3.plot(epochs, val_recall_no_foul, label='Valid Recall (No Foul)', color='green')
    ax3.plot(epochs, val_recall_foul, label='Valid Recall (Foul)', color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall (%)')
    ax3.set_title('Validation Class-wise Recall')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully generated and saved to {save_path}")

if __name__ == "__main__":
    os.makedirs("src/visualization", exist_ok=True)
    plot_training_history()
