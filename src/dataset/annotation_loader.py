import json
import os


FOUL_LABELS = {
    "No offence" : 0,
    "Offence"    : 1,
    "Between"    : 0,
    ""           : 0,
}

SEVERITY_LABELS = {
    1.0 : 0,  # No card
    2.0 : 1,  # No card borderline
    3.0 : 2,  # Yellow card
    4.0 : 3,  # Red card
    5.0 : 3,  # treat as Red card (most severe)
}

ACTION_CLASS_LABELS = {
    "Tackling"         : 0,
    "Holding"          : 1,
    "Pushing"          : 2,
    "Standing tackling": 3,
    "Elbowing"         : 4,
    "Dive"             : 5,
    "Challenge"        : 6,
    "High leg"         : 7,
    "Dont know"        : 0, 
    ""                 : 0,
}


def load_annotations(data_root="data/soccernet/mvfouls"):
    """
    Load annotations from all splits and build a flat list of samples.
    Each sample is one clip with its labels.

    """
    all_samples = []

    for split in ["train", "valid", "test", "challenge"]:
        ann_path = os.path.join(data_root, split, "annotations.json")

        if not os.path.exists(ann_path):
            print(f"Skipping {split} — annotations.json not found")
            continue

        with open(ann_path) as f:
            data = json.load(f)

        actions = data["Actions"]
        print(f"{split}: {len(actions)} actions in annotations")

        for action_key, action_data in actions.items():
            action_id = f"action_{action_key}"

            # --- foul label ---
            offence_str = action_data.get("Offence", "")
            foul_label = FOUL_LABELS.get(offence_str, 0)

            # --- severity label ---
            severity_str = action_data.get("Severity", "")
            try:
                severity_float = float(severity_str) if severity_str != "" else 1.0
            except:
                severity_float = 1.0
            sev_label = SEVERITY_LABELS.get(severity_float, 0)

            # --- action class label ---
            action_str = action_data.get("Action class", "")
            action_class = ACTION_CLASS_LABELS.get(action_str, 0)

            # --- clips ---
            clips = action_data.get("Clips", [])

            for clip_idx, clip_info in enumerate(clips):
                clip_path = os.path.join(
                    "data/frames",
                    split,
                    action_id,
                    f"clip_{clip_idx}.npy"
                )

                if not os.path.exists(clip_path):
                    continue

                all_samples.append({
                    "action_id"   : action_id,
                    "split"       : split,
                    "clip_path"   : clip_path,
                    "foul_label"  : foul_label,
                    "sev_label"   : sev_label,
                    "action_class": action_class,
                    "offence_str" : offence_str,
                    "severity_str": severity_str,
                    "action_str"  : action_str,
                    "camera_type" : clip_info.get("Camera type", "Unknown"),
                })

    return all_samples


def get_split_samples(all_samples, split):
    """Filter samples by split name."""
    return [s for s in all_samples if s["split"] == split]


def print_dataset_stats(all_samples):
    """Print a summary of the dataset labels."""

    from collections import Counter

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)

    # per split
    splits = Counter(s["split"] for s in all_samples)
    print("\nSamples per split:")
    for split, count in splits.items():
        print(f"  {split:12s}: {count}")

    # foul label
    foul_counts = Counter(s["offence_str"] for s in all_samples)
    print("\nFoul label distribution:")
    for label, count in foul_counts.items():
        pct = count / len(all_samples) * 100
        print(f"  {str(label):20s}: {count} ({pct:.1f}%)")

    # severity
    sev_counts = Counter(s["severity_str"] for s in all_samples)
    print("\nSeverity distribution:")
    sev_names = {
        1.0: "No card",
        2.0: "No card+",
        3.0: "Yellow card",
        4.0: "Red card"
    }
    for sev, count in sorted(sev_counts.items(), key=lambda x: str(x[0])):
        try:
            name = sev_names.get(float(sev), str(sev))
        except:
            name = "Unknown"
        pct = count / len(all_samples) * 100
        print(f"  {str(sev):6s} ({name:12s}): {count} ({pct:.1f}%)")

    # action class
    action_counts = Counter(s["action_str"] for s in all_samples)
    print("\nAction class distribution:")
    for action, count in action_counts.most_common():
        pct = count / len(all_samples) * 100
        print(f"  {str(action):25s}: {count} ({pct:.1f}%)")

    print("="*50)
    print(f"Total samples: {len(all_samples)}")
    print("="*50)


if __name__ == "__main__":
    print("Loading annotations...")
    samples = load_annotations()

    print_dataset_stats(samples)

    # show one sample in detail
    print("\nExample sample:")
    s = samples[0]
    for k, v in s.items():
        print(f"  {k:15s}: {v}")

    # verify split filtering
    train_samples = get_split_samples(samples, "train")
    print(f"\nTrain samples ready for training: {len(train_samples)}")