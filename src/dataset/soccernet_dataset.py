import os
import numpy as np
from collections import Counter


def find_actions(frames_root="data/frames"):
    actions = []

    for split in ["train", "valid", "test", "challenge"]:
        split_path = os.path.join(frames_root, split)

        if not os.path.exists(split_path):
            print(f"Skipping {split} — not found in {frames_root}")
            continue

        for action_name in sorted(os.listdir(split_path)):
            action_path = os.path.join(split_path, action_name)

            if not os.path.isdir(action_path):
                continue

            # Look for .npy files instead of .mp4 now
            clips = [
                os.path.join(action_path, f)
                for f in sorted(os.listdir(action_path))
                if f.endswith(".npy")
            ]

            if len(clips) == 0:
                continue

            actions.append({
                "action_id": action_name,
                "split": split,
                "clips": clips
            })

    return actions


def load_clip(npy_path):
    """
    Load a pre-extracted clip from disk.
    Returns numpy array of shape (16, 224, 224, 3)
    
    This replaces the old extract_frames_from_clip() —
    instead of decoding video, we just load the saved array.
    """
    if not os.path.exists(npy_path):
        raise ValueError(f"File not found: {npy_path}")

    frames = np.load(npy_path)

    # Quick sanity check on shape
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D array (T,H,W,C), got shape: {frames.shape}")

    return frames


if __name__ == "__main__":
    print("Scanning pre-extracted frames...")
    actions = find_actions()

    if len(actions) == 0:
        print("Nothing found in data/frames/ yet.")
        print("Run preprocess.py first to extract frames from videos.")
    else:
        print(f"Found {len(actions)} actions total\n")

        # Count per split
        splits = Counter(a["split"] for a in actions)
        for split, count in splits.items():
            print(f"  {split:12s}: {count} actions")

        # Count total clips
        total_clips = sum(len(a["clips"]) for a in actions)
        print(f"\n  Total clips : {total_clips}")
        print(f"  Avg views   : {total_clips / len(actions):.1f} per action")

        # Test loading the first clip
        first = actions[0]
        print(f"\nTest loading: {first['clips'][0]}")

        try:
            frames = load_clip(first["clips"][0])
            print(f"  Shape  : {frames.shape}")
            print(f"  Dtype  : {frames.dtype}")
            print(f"  Range  : [{frames.min()}, {frames.max()}]")
            print("\nDataset loader ready!")
        except ValueError as e:
            print(f"Error: {e}")