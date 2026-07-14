import os
import cv2
import numpy as np
from tqdm import tqdm


def preprocess_and_save(
    data_root="data/soccernet/mvfouls",
    save_root="data/frames",
    num_frames=16,
    output_size=(224, 224)
):

    splits = ["train", "valid", "test", "challenge"]
    total_saved = 0
    total_skipped = 0

    for split in splits:
        split_path = os.path.join(data_root, split)

        if not os.path.exists(split_path):
            print(f"Skipping {split} — folder not found")
            continue

        action_names = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])

        print(f"\nProcessing {split}: {len(action_names)} actions")

        for action_name in tqdm(action_names, desc=split):
            action_path = os.path.join(split_path, action_name)

            # Find all clips in this action folder
            clip_files = sorted([
                f for f in os.listdir(action_path)
                if f.endswith(".mp4")
            ])

            for clip_file in clip_files:
                clip_path = os.path.join(action_path, clip_file)

                # Build the save path — mirrors the original structure
                # e.g. data/frames/train/action_1/clip1.npy
                save_dir = os.path.join(save_root, split, action_name)
                os.makedirs(save_dir, exist_ok=True)

                clip_name = clip_file.replace(".mp4", ".npy")
                save_path = os.path.join(save_dir, clip_name)

                # Skip if already processed — safe to re-run
                if os.path.exists(save_path):
                    total_skipped += 1
                    continue

                # Extract frames
                try:
                    frames = extract_frames(clip_path, num_frames, output_size)
                    np.save(save_path, frames)
                    total_saved += 1

                except Exception as e:
                    print(f"\nFailed: {clip_path} — {e}")

    print(f"\nDone!")
    print(f"  Saved   : {total_saved} clips")
    print(f"  Skipped : {total_skipped} clips (already existed)")
    print(f"  Frames saved to: {save_root}")


def extract_frames(clip_path, num_frames=16, output_size=(224, 224)):
    """Extract evenly spaced frames from one clip."""

    cap = cv2.VideoCapture(clip_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open: {clip_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        raise ValueError(f"Empty video: {clip_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, output_size)

        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)


def verify_saved_frames(save_root="data/frames"):
    """Quick check — count saved files and load one to verify shape."""

    print("\nVerifying saved frames...")
    total = 0
    sample_path = None

    for root, dirs, files in os.walk(save_root):
        for f in files:
            if f.endswith(".npy"):
                total += 1
                if sample_path is None:
                    sample_path = os.path.join(root, f)

    print(f"  Total .npy files: {total}")

    if sample_path:
        frames = np.load(sample_path)
        print(f"  Sample file     : {sample_path}")
        print(f"  Sample shape    : {frames.shape}")
        print(f"  Pixel range     : [{frames.min()}, {frames.max()}]")
        print(f"  Size on disk    : {os.path.getsize(sample_path) / 1024:.1f} KB per clip")


if __name__ == "__main__":
    preprocess_and_save()
    verify_saved_frames()