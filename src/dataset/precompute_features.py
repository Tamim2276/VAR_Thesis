import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

sys.path.append('.')
from src.models.clip_extractor import load_clip_model


def precompute_clip_features(
    frames_root="data/frames",
    save_root="data/features",
    device="xpu"
):
    """
    Run every clip through CLIP once and save the CLS token.
    
    During training, DataLoader loads these tiny files
    instead of running CLIP every batch.
    """

    print("Loading CLIP...")
    model, preprocess = load_clip_model()
    model = model.to(device)
    model.eval()
    print(f"CLIP loaded on {device}")

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

                # Skip if already computed
                if os.path.exists(save_path):
                    total_skipped += 1
                    continue

                try:
                    # Load frames
                    frames = np.load(clip_path)  # (16, 224, 224, 3)
                    T = frames.shape[0]

                    # Preprocess all frames
                    frame_tensors = []
                    for i in range(T):
                        pil = Image.fromarray(frames[i])
                        tensor = preprocess(pil)
                        frame_tensors.append(tensor)

                    batch = torch.stack(frame_tensors).to(device)  # (16, 3, 224, 224)

                    # Run through CLIP — extract CLS tokens
                    with torch.no_grad():
                        visual = model.visual.to(device)

                        x = visual.conv1(batch)
                        x = x.reshape(x.shape[0], x.shape[1], -1)
                        x = x.permute(0, 2, 1)

                        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0)
                        cls = cls.expand(T, -1, -1)
                        x = torch.cat([cls, x], dim=1)
                        x = x + visual.positional_embedding.unsqueeze(0)

                        if hasattr(visual, 'patch_dropout'):
                            x = visual.patch_dropout(x)
                        x = visual.ln_pre(x)

                        x = x.permute(1, 0, 2)
                        x = visual.transformer(x)
                        x = x.permute(1, 0, 2)
                        x = visual.ln_post(x)

                    cls_tokens = x[:, 0, :]  # (16, 1024)

                    # Save to CPU before writing to disk
                    torch.save(
                        {"cls": cls_tokens.cpu()},
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
                    data = torch.load(sample)
                    print(f"  File    : {sample}")
                    print(f"  CLS shape: {data['cls'].shape}")
                    size_kb = os.path.getsize(sample) / 1024
                    print(f"  Size    : {size_kb:.1f} KB per clip")
                    break
                break
            break


if __name__ == "__main__":
    precompute_clip_features()