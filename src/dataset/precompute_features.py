import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

sys.path.append('.')
from src.models.siglip_extractor import load_siglip_model, extract_spatial_tokens


def precompute_siglip_features(
    frames_root="data/frames",
    save_root="data/features",
    device="xpu",
    finetuned_weights_path="models/siglip_finetuned.pt",
    overwrite=False,
):
    """
    Run every clip through SigLIP once and save the global pooler token.
    
    During training, DataLoader loads these tiny files
    instead of running SigLIP every batch.
    """

    print("Loading SigLIP...")
    model, processor = load_siglip_model()

    if finetuned_weights_path and os.path.exists(finetuned_weights_path):
        checkpoint = torch.load(
            finetuned_weights_path, map_location="cpu", weights_only=True
        )
        state_dict = checkpoint.get("siglip_model_state", checkpoint)
        model.load_state_dict(state_dict)
        overwrite = True
        print(f"Loaded fine-tuned SigLIP weights: {finetuned_weights_path}")
        print("Existing feature files will be regenerated.")

    model = model.to(device)
    model.eval()
    print(f"SigLIP loaded on {device}")

    total_saved   = 0
    total_skipped = 0
    total_failed  = 0

    for split in ["train", "valid", "test"]:
        split_path = os.path.join(frames_root, split)

        if not os.path.exists(split_path):
            print(f"Skipping {split} — not found")
            continue

        action_names = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])

        print(f"\nProcessing {split}: {len(action_names)} actions")

        for action_name in tqdm(action_names, desc=split):
            action_path = os.path.join(split_path, action_name)

            clip_files = sorted([
                f for f in os.listdir(action_path)
                if f.endswith(".npy")
            ])

            for clip_file in clip_files:
                clip_path = os.path.join(action_path, clip_file)

                # Save path mirrors frames structure
                save_dir = os.path.join(save_root, split, action_name)
                os.makedirs(save_dir, exist_ok=True)

                save_name = clip_file.replace(".npy", ".pt")
                save_path = os.path.join(save_dir, save_name)

                # Recompute cached features when a fine-tuned encoder is used.
                if os.path.exists(save_path) and not overwrite:
                    total_skipped += 1
                    continue

                try:
                    # Load frames
                    frames = np.load(clip_path)  # (16, 224, 224, 3)

                    # Extract SigLIP features in chunks of 8 to prevent OOM
                    global_tokens, _ = extract_spatial_tokens(model, processor, frames, batch_size=8)

                    # Save to CPU before writing to disk
                    torch.save(
                        {"cls": global_tokens.cpu()},
                        save_path
                    )
                    total_saved += 1

                except Exception as e:
                    print(f"\nFailed: {clip_path} — {e}")
                    total_failed += 1

    print(f"\nDone!")
    print(f"  Saved   : {total_saved}")
    print(f"  Skipped : {total_skipped} (already existed)")
    print(f"  Failed  : {total_failed}")

    # Verify one file
    print("\nVerifying one saved file...")
    for split in ["train", "valid", "test"]:
        sample_dir = os.path.join(save_root, split)
        if os.path.exists(sample_dir):
            for action in os.listdir(sample_dir):
                for f in os.listdir(os.path.join(sample_dir, action)):
                    sample = os.path.join(sample_dir, action, f)
                    data = torch.load(sample, map_location="cpu", weights_only=True)
                    print(f"  File    : {sample}")
                    print(f"  CLS shape: {data['cls'].shape} (Expected 16, 1152)")
                    size_kb = os.path.getsize(sample) / 1024
                    print(f"  Size    : {size_kb:.1f} KB per clip")
                    break
                break
            break


if __name__ == "__main__":
    precompute_siglip_features(overwrite=True)
