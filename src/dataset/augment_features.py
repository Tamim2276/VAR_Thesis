import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from src.models.siglip_extractor import load_siglip_model, extract_spatial_tokens

def augment_horizontal_flip(
    frames_root="data/frames/train",
    save_root="data/features/train",
    device="xpu",
):
    """
    Load raw frames from the train split, flip them horizontally,
    extract SigLIP features, and save as `_flip.pt`.
    """
    print("Loading SigLIP for Data Augmentation...")
    model, processor = load_siglip_model()
    model = model.to(device)
    model.eval()
    print(f"SigLIP loaded on {device}")

    if not os.path.exists(frames_root):
        print(f"Frames directory not found: {frames_root}")
        return

    action_names = sorted([
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ])

    print(f"\nProcessing horizontal flips for {len(action_names)} actions in train split...")

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for action_name in tqdm(action_names, desc="Augmenting"):
        action_path = os.path.join(frames_root, action_name)
        save_dir = os.path.join(save_root, action_name)
        os.makedirs(save_dir, exist_ok=True)

        clip_files = sorted([f for f in os.listdir(action_path) if f.endswith(".npy")])

        for clip_file in clip_files:
            clip_path = os.path.join(action_path, clip_file)
            save_name = clip_file.replace(".npy", "_flip.pt")
            save_path = os.path.join(save_dir, save_name)

            if os.path.exists(save_path):
                total_skipped += 1
                continue

            try:
                # Load original frames (16, 224, 224, 3)
                frames = np.load(clip_path)
                
                # Flip horizontally (axis 2 is width for H, W, C format)
                # Use .copy() to ensure memory is contiguous after flipping
                flipped_frames = np.flip(frames, axis=2).copy()

                # Extract SigLIP features
                global_tokens, _ = extract_spatial_tokens(model, processor, flipped_frames, batch_size=8)

                # Save to disk
                torch.save(
                    {"cls": global_tokens.cpu()},
                    save_path
                )
                total_saved += 1

            except Exception as e:
                print(f"\nFailed: {clip_path} — {e}")
                total_failed += 1

    print(f"\nAugmentation Complete!")
    print(f"  New Flipped Clips Saved : {total_saved}")
    print(f"  Skipped (Already exists): {total_skipped}")
    print(f"  Failed to Process       : {total_failed}")

if __name__ == "__main__":
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    augment_horizontal_flip(device=DEVICE)
