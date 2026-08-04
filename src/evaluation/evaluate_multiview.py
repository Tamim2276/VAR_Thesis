"""
Multi-View Fusion Evaluation:

Mimics the real-life VAR process: instead of judging each camera clip independently,
this script groups all camera views for the same action and fuses their predictions
into a single, more robust decision.

Phase 1: Late Fusion (Prediction Averaging)
- No retraining needed — uses the existing best models
- Groups clips by action_id
- Averages softmax probabilities across all camera views
- Makes one final prediction per action
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.append('.')
from src.models.hierarchical_classifiers import FoulClassifier, SeverityClassifier
from src.dataset.annotation_loader import load_annotations, get_split_samples


def load_features(sample, features_root="data/features"):
    """Load pre-computed SigLIP features for a single clip."""
    feat_path = (
        sample["clip_path"]
        .replace("data/frames",  features_root)
        .replace("data\\frames", features_root)
        .replace(".npy", ".pt")
        .replace("\\", "/")
    )
    if not os.path.exists(feat_path):
        return None
    data = torch.load(feat_path, map_location="cpu", weights_only=True)
    cls_tokens = data["cls"]  # (16, 1152)
    cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
    return cls_tokens


def evaluate_per_clip(model, samples, features_root, device, num_classes=2, threshold=0.60):
    """Standard per-clip evaluation (baseline)."""
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for sample in tqdm(samples, desc="Per-clip eval"):
            features = load_features(sample, features_root)
            if features is None:
                continue

            features = features.unsqueeze(0).to(device)  # (1, 16, 1152)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)  # (num_classes,)

            if num_classes == 2:
                label = 1 if sample["foul_label"] > 0 else 0
                pred = 1 if probs[1] >= threshold else 0
            else:
                label = sample["sev_label"]
                pred = probs.argmax().item()

            total += 1
            if pred == label:
                correct += 1
                class_correct[label] += 1
            class_total[label] += 1

    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
              for c in range(num_classes)]
    balanced_acc = sum(recall) / num_classes
    return {"balanced_acc": balanced_acc, "recall": recall, "support": class_total,
            "total_samples": total}


def evaluate_multiview_fusion(model, samples, features_root, device,
                               num_classes=2, threshold=0.60, fusion="average"):
    """
    Multi-View Fusion evaluation.
    Groups clips by action_id, fuses predictions, reports per-ACTION accuracy.
    """
    model.eval()

    # Step 1: Collect per-clip probabilities, grouped by action_id
    action_probs = defaultdict(list)    # {action_id: [prob_tensor, ...]}
    action_labels = {}                  # {action_id: label}
    skipped = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Multi-view eval"):
            features = load_features(sample, features_root)
            if features is None:
                skipped += 1
                continue

            features = features.unsqueeze(0).to(device)  # (1, 16, 1152)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)  # (num_classes,)

            action_id = sample["action_id"]
            action_probs[action_id].append(probs)

            if num_classes == 2:
                action_labels[action_id] = 1 if sample["foul_label"] > 0 else 0
            else:
                action_labels[action_id] = sample["sev_label"]

    # Step 2: Fuse predictions per action
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # For detailed analysis
    agreement_stats = {"all_agree": 0, "disagree": 0, "fusion_correct_disagree": 0}

    for action_id, prob_list in action_probs.items():
        label = action_labels[action_id]
        num_views = len(prob_list)

        # Stack all view probabilities: (num_views, num_classes)
        stacked = torch.stack(prob_list, dim=0)

        if fusion == "average":
            fused_probs = stacked.mean(dim=0)  # (num_classes,)
        elif fusion == "max":
            fused_probs = stacked.max(dim=0).values
        elif fusion == "vote":
            # Majority voting
            if num_classes == 2:
                votes = [(p[1] >= threshold).long().item() for p in prob_list]
            else:
                votes = [p.argmax().item() for p in prob_list]
            vote_count = Counter(votes)
            pred = vote_count.most_common(1)[0][0]
            # Skip the normal pred logic below
            total += 1
            if pred == label:
                correct += 1
                class_correct[label] += 1
            class_total[label] += 1
            continue

        # Final prediction from fused probabilities
        if num_classes == 2:
            pred = 1 if fused_probs[1] >= threshold else 0
        else:
            pred = fused_probs.argmax().item()

        # Check if individual views agreed
        if num_classes == 2:
            individual_preds = [(p[1] >= threshold).long().item() for p in prob_list]
        else:
            individual_preds = [p.argmax().item() for p in prob_list]

        all_same = len(set(individual_preds)) == 1
        if all_same:
            agreement_stats["all_agree"] += 1
        else:
            agreement_stats["disagree"] += 1
            if pred == label:
                agreement_stats["fusion_correct_disagree"] += 1

        total += 1
        if pred == label:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

    recall = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
              for c in range(num_classes)]
    balanced_acc = sum(recall) / num_classes

    return {
        "balanced_acc": balanced_acc,
        "recall": recall,
        "support": class_total,
        "total_actions": total,
        "skipped_clips": skipped,
        "agreement": agreement_stats,
        "views_distribution": Counter(len(v) for v in action_probs.values()),
    }


def print_results(title, results, class_names):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Balanced Accuracy: {results['balanced_acc']:.1f}%")
    for i, name in enumerate(class_names):
        support = results['support'][i]
        recall = results['recall'][i]
        print(f"  {name:12s}: recall={recall:.1f}%  (support={support})")

    if "total_actions" in results:
        print(f"\n  Total actions evaluated: {results['total_actions']}")
        print(f"  Views distribution: {dict(results['views_distribution'])}")
        ag = results["agreement"]
        print(f"  Camera agreement: {ag['all_agree']} actions all cameras agreed")
        print(f"  Camera disagreement: {ag['disagree']} actions cameras disagreed")
        if ag["disagree"] > 0:
            pct = 100 * ag["fusion_correct_disagree"] / ag["disagree"]
            print(f"  Fusion resolved correctly: {ag['fusion_correct_disagree']}/{ag['disagree']} ({pct:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    FEATURES_ROOT = "data/features"
    FOUL_MODEL_PATH = "models/hierarchical/best_foul_model.pt"
    SEV_MODEL_PATH = "models/hierarchical/best_severity_model.pt"

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Device: {DEVICE}")

    # Load annotations
    all_samples = load_annotations()
    valid_samples = get_split_samples(all_samples, "valid")
    valid_foul_only = [s for s in valid_samples if s["foul_label"] > 0]

    print(f"\nValidation set: {len(valid_samples)} total clips")
    print(f"Foul-only clips (for severity): {len(valid_foul_only)}")


    # FOUL DETECTION
    print("\n" + "="*60)
    print("  FOUL DETECTION EVALUATION")
    print("="*60)

    foul_model = FoulClassifier(feat_dim=1152, hidden_dim=128).to(DEVICE)
    foul_model.load_state_dict(torch.load(FOUL_MODEL_PATH, map_location=DEVICE, weights_only=True))
    foul_model.eval()

    # Baseline: per-clip
    baseline_foul = evaluate_per_clip(
        foul_model, valid_samples, FEATURES_ROOT, DEVICE,
        num_classes=2, threshold=0.60
    )
    print_results("FOUL — Per-Clip Baseline", baseline_foul, ["No Foul", "Foul"])

    # Multi-View: Average fusion
    for fusion_method in ["average", "vote", "max"]:
        mv_foul = evaluate_multiview_fusion(
            foul_model, valid_samples, FEATURES_ROOT, DEVICE,
            num_classes=2, threshold=0.60, fusion=fusion_method
        )
        print_results(f"FOUL — Multi-View ({fusion_method.upper()})", mv_foul, ["No Foul", "Foul"])

    # SEVERITY CLASSIFICATION
    print("\n" + "="*60)
    print("  SEVERITY CLASSIFICATION EVALUATION")
    print("="*60)

    sev_model = SeverityClassifier(feat_dim=1152, hidden_dim=128).to(DEVICE)
    sev_model.load_state_dict(torch.load(SEV_MODEL_PATH, map_location=DEVICE, weights_only=True))
    sev_model.eval()

    # Baseline: per-clip
    baseline_sev = evaluate_per_clip(
        sev_model, valid_foul_only, FEATURES_ROOT, DEVICE,
        num_classes=3, threshold=0.50
    )
    print_results("SEVERITY — Per-Clip Baseline", baseline_sev, ["No Card", "Yellow", "Red"])

    # Multi-View: Average fusion
    for fusion_method in ["average", "vote", "max"]:
        mv_sev = evaluate_multiview_fusion(
            sev_model, valid_foul_only, FEATURES_ROOT, DEVICE,
            num_classes=3, threshold=0.50, fusion=fusion_method
        )
        print_results(f"SEVERITY — Multi-View ({fusion_method.upper()})", mv_sev, ["No Card", "Yellow", "Red"])

    # SUMMARY
    print("\n" + "="*60)
    print("  SUMMARY: Per-Clip vs Multi-View Fusion")
    print("="*60)
    print(f"  Foul Detection:")
    print(f"    Per-Clip:       {baseline_foul['balanced_acc']:.1f}%")
    print(f"    Multi-View Avg: Check results above")
    print(f"\n  Severity:")
    print(f"    Per-Clip:       {baseline_sev['balanced_acc']:.1f}%")
    print(f"    Multi-View Avg: Check results above")
    print("="*60)
