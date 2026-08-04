# Multi-View Fusion: VAR-Inspired Camera Consensus

## Goal

Currently, each camera clip is classified independently. An action with 3 camera views produces 3 separate predictions that are never combined. Multi-View Fusion groups all camera clips belonging to the same action and fuses their predictions into a single, more robust decision — exactly like a real Video Assistant Referee.

## Background

- **Foul Detection:** 56.5% ➔ **59.7%** (+3.2% with Fine-Tuned SigLIP)
- **Severity Detection:** 42.4% ➔ **45.0%** (+2.6% with Fine-Tuned SigLIP + MAX Fusion)
- **2 clips per action:** 292 actions (71%)
- **3 clips per action:** 90 actions (22%)
- **4 clips per action:** 29 actions (7%)
- **Average:** 2.4 camera views per action

This means every single action has at least 2 camera angles available. The current evaluation wastes this by treating each clip as an independent sample.

## Current Performance (Baseline)

| Task | Balanced Accuracy | Method |
|------|------------------|--------|
| Foul Detection | **60.3%** | Per-clip evaluation |
| Severity (3-Class) | **43.0%** | Per-clip evaluation |

## Proposed Changes

### Phase 1: Late Fusion (Prediction Averaging) — Zero Retraining

> [!IMPORTANT]
> This phase requires **zero retraining**. We only change how we evaluate the model. The trained weights stay exactly the same.

#### [NEW] [evaluate_multiview.py](file:///D:/VAR%20Thesis/xvars_thesis/src/evaluation/evaluate_multiview.py)

A standalone evaluation script that:
1. Loads the best saved foul model and severity model
2. Runs inference on every clip in the validation set
3. Groups predictions by `action_id`
4. Averages the softmax probabilities across all camera views for each action
5. Makes the final prediction from the averaged probabilities
6. Reports per-action balanced accuracy (instead of per-clip)

```python
# Pseudocode for the core logic:
action_predictions = {}  # {action_id: [prob_array_cam1, prob_array_cam2, ...]}

for clip in validation_set:
    probs = model(clip.features)  # softmax probabilities
    action_predictions[clip.action_id].append(probs)

for action_id, prob_list in action_predictions.items():
    fused_probs = average(prob_list)       # Average across cameras
    final_prediction = argmax(fused_probs)  # Single decision per action
```

### Phase 2: Learned Attention Fusion — Small Trainable Module

> [!NOTE]
> This phase adds a tiny trainable module (~2K parameters) that learns which camera view is most informative per action.

#### [MODIFY] [hierarchical_classifiers.py](file:///D:/VAR%20Thesis/xvars_thesis/src/models/hierarchical_classifiers.py)

Add a `ViewAttentionFusion` module:
```python
class ViewAttentionFusion(nn.Module):
    """Learns which camera view to trust most for each action."""
    def __init__(self, feat_dim=2):  # 2 for foul, 3 for severity
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, view_probs):
        # view_probs: (num_views, num_classes)
        weights = softmax(self.gate(view_probs), dim=0)  # (num_views, 1)
        fused = (view_probs * weights).sum(dim=0)         # (num_classes,)
        return fused
```

## Open Questions

> [!IMPORTANT]
> **Q1: Should we report BOTH per-clip and per-action accuracy?**
> For the paper, we should report both. Per-clip accuracy is the standard benchmark metric. Per-action accuracy (with multi-view fusion) is our novel contribution. Showing both demonstrates the improvement from fusion.

> [!IMPORTANT]
> **Q2: Should we implement Phase 1 first, measure the gain, and then decide if Phase 2 (Learned Attention) is needed?**
> I recommend this incremental approach. Phase 1 (simple averaging) is risk-free and often captures 80% of the possible fusion gain. Phase 2 adds complexity that may or may not be worth it depending on the Phase 1 results.

## Verification Plan

### Automated Tests
1. Run `evaluate_multiview.py` on the validation set for both foul and severity models
2. Compare per-action balanced accuracy against the per-clip baseline
3. Print per-class recall breakdown to ensure no class is harmed by fusion

### Expected Results
- Foul Detection: 60.3% → **65–68%** balanced accuracy
- Severity (3-Class): 43.0% → **47–50%** balanced accuracy

### Manual Verification
- Print a few example actions showing how individual camera predictions differ and how fusion resolves disagreements
