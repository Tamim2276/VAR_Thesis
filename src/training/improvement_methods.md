# Methods for Improving Classification Accuracy
### Towards 70–75% Balanced Accuracy on Foul Detection and Severity Classification

---

## 1. Current Baseline Performance

The current X-VARS pipeline uses a **frozen SigLIP SO400M** vision encoder to extract 1152-dimensional frame-level features from 16 sampled video frames. Lightweight classifier heads operate on these frozen features.

| Task | Balanced Accuracy | Architecture | Parameters |
|------|------------------|--------------|------------|
| Foul Detection (Binary) | **58.6%** | Temporal Attention + Mixup | 295,811 |
| Severity (3-Class) | **38.5%** | Average Pooling + Mixup | 148,227 |

### Identified Bottlenecks

1. **Frozen features:** SigLIP was pre-trained on general web images, not soccer tackles. Its feature space doesn't capture domain-specific patterns like stud height, angle of approach, or body contact dynamics.

2. **Single-view prediction:** Each prediction uses 1 camera clip, but the dataset has 4–5 synchronised views per action. A real VAR referee always checks multiple angles.

3. **Image-level features:** SigLIP encodes each frame independently — it has no concept of *motion* (tackle speed, leg trajectory, deceleration after contact), which are critical cues.

---

## 2. Tier 1: Highest Impact Methods (+5–12% Each)

> [!IMPORTANT]
> These methods address the fundamental bottlenecks and offer the largest accuracy gains. Expected combined improvement: **+15–25%**.

---

### Method 1: Fine-Tuning the SigLIP Vision Backbone

**Motivation:**
The frozen SigLIP encoder produces general-purpose embeddings. Fine-tuning the last few transformer layers allows it to learn football-specific visual patterns.

**Approach:**
1. Unfreeze the **last 4 transformer layers** of SigLIP (out of 27 total)
2. Train end-to-end: SigLIP (partial) → Temporal Attention → Classifier
3. **Discriminative learning rate:** $10^{-5}$ for SigLIP layers, $10^{-4}$ for classifier head
4. Apply **gradient checkpointing** to fit within 16 GB VRAM

```
Currently:  SigLIP (FROZEN) → features → classifier
Proposed:   SigLIP (last 4 layers UNFROZEN) → features → classifier
```

**Expected Improvement:**
- Foul Detection: 58.6% → **68–72%**
- Severity (3-Class): 38.5% → **50–55%**

**Hardware:** Feasible on Arc A770 with gradient checkpointing and batch size 16–32.

---

### Method 2: Multi-View Fusion

**Motivation:**
The dataset provides **4–5 synchronised camera views** per action. A foul that's ambiguous from one angle may be clearly visible from another — exactly why VAR uses multiple cameras.

**Three Fusion Strategies:**

**Strategy A — Late Fusion (Prediction Averaging):**
```
Camera 1 → model → P(foul) = 0.72
Camera 2 → model → P(foul) = 0.85
Camera 3 → model → P(foul) = 0.41
Camera 4 → model → P(foul) = 0.68

Average: 0.665 → Final: "Foul" ✅ (more robust than any single view)
```

**Strategy B — Attention-Based Fusion:**
A learned attention mechanism assigns higher weight to the most informative camera view (e.g., the close-up showing the point of contact).

**Strategy C — Cross-View Transformer:**
Multi-head self-attention across all view embeddings, allowing each camera view to attend to information from other views.

**Expected Improvement:**
- Foul Detection: **+5–8%**
- Severity: **+4–6%**

**Hardware:** Fully feasible on Arc A770 (operates on pre-computed features).

---

### Method 3: Video-Native Transformer Backbone

**Motivation:**
SigLIP processes each frame independently, discarding all temporal information. A video transformer captures **motion dynamics** — tackle speed, leg trajectory, timing of contact.

**Candidate Architectures:**

| Model | Parameters | Input | Pre-training |
|-------|-----------|-------|-------------|
| TimeSFormer-Base | 121M | 8 × 224² | Kinetics-400 |
| VideoMAE v2-Base | 87M | 16 × 224² | Kinetics-710 |
| Video Swin-Base | 88M | 32 × 224² | Kinetics-600 |
| InternVideo2 | 1B | 16 × 224² | Multi-dataset |

**Approach:**
```
Currently:  16 separate images → SigLIP → 16 vectors → attention → predict
Proposed:   16 frames as VIDEO → VideoMAE → 1 video vector → predict
```

**Expected Improvement:**
- Foul Detection: **+7–10%**
- Severity: **+5–8%**

**Hardware:** ⚠️ **Challenging** on Arc A770. Requires 20–24 GB VRAM. Possible with aggressive gradient checkpointing or cloud GPU.

---

## 3. Tier 2: Medium Impact Methods (+3–5% Each)

> [!NOTE]
> These provide meaningful gains with moderate effort. Most effective when combined with Tier 1 methods. Expected combined improvement: **+8–15%**.

---

### Method 4: Cross-Attention Text-Video Fusion (Node N4)

**Motivation:**
This is **Node N4** in the X-VARS architecture diagram. By encoding class descriptions as text embeddings and computing cross-attention with video features, the model learns to align visual patterns with semantic descriptions.

**Approach:**
```python
# Text embeddings for each class
text_features = SigLIP_text([
    "A clean play with no physical contact",
    "A cautionable foul deserving a yellow card",
    "A violent foul deserving a red card"
])

# Cross-attention: video attends to text descriptions
fused = CrossAttention(Q=video_features, K=text_features, V=text_features)
# The model learns which text description matches the video
```

**Expected Improvement:** +3–5% on both tasks.

---

### Method 5: Horizontal Flip Augmentation (Instant 2× Data)

**Motivation:**
A tackle from the left is physically identical to a tackle from the right. Flipping all videos **doubles the dataset** with zero manual labelling.

**Approach:**
1. For each training video, create a horizontally flipped copy
2. Extract 16 frames from the flipped video
3. Run through frozen SigLIP encoder
4. Save as a new `.pt` feature file with the same labels

**Impact on Dataset Size:**

| Class | Before | After (2×) |
|-------|--------|-----------|
| No Foul | 934 | **1,868** |
| Foul | 5,687 | **11,374** |
| No Card | 3,901 | **7,802** |
| Yellow Card | 1,601 | **3,202** |
| Red Card | 185 | **370** |

**Expected Improvement:** +3–5%. The minority class benefits most.

**Implementation Time:** ~1 hour.

---

### Method 6: Model Ensemble (5 Models)

**Motivation:**
Individual models are sensitive to random initialisation. An ensemble of independently trained models reduces prediction variance.

**Approach:**
```
Model_1 (seed=42)  → P(foul) = 0.72
Model_2 (seed=123) → P(foul) = 0.65
Model_3 (seed=7)   → P(foul) = 0.51
Model_4 (seed=999) → P(foul) = 0.68
Model_5 (seed=314) → P(foul) = 0.59

Average: 0.63 → Final: "Foul" ✅
```

**Expected Improvement:** +3–4% balanced accuracy. Guaranteed improvement.

---

### Method 7: Contrastive Pre-Training (SupCon Loss)

**Motivation:**
Before training the classifier, pre-train the feature space using Supervised Contrastive Loss to pull same-class samples together and push different-class samples apart.

**How it works:**
- "Foul" features cluster together in feature space
- "No Foul" features cluster together
- The gap between clusters becomes larger → easier to classify

**Expected Improvement:** +3–5%, particularly beneficial for the minority Red Card class.

---

## 4. Tier 3: Smaller but Accessible Gains (+1–3% Each)

> [!TIP]
> Easy to implement with minimal risk. Expected combined improvement: **+3–7%**.

---

### Method 8: Temporal Jittering

Instead of extracting exactly 16 evenly-spaced frames, create additional training samples by shifting the extraction window by ±1–2 frames.

```
Original:  frames [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
Jittered:  frames [8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
Jittered:  frames [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
```

**Expected Improvement:** +1–2%.

---

### Method 9: Test-Time Augmentation (TTA)

At inference, run the model on both the original clip AND a horizontally flipped version, then average:

```
P_original = model(clip)        = 0.62
P_flipped  = model(flip(clip))  = 0.58

P_final = (0.62 + 0.58) / 2 = 0.60
```

**Free improvement** — requires zero retraining.

**Expected Improvement:** +1–2%.

---

### Method 10: Knowledge Distillation

Train a larger "teacher" model (unfrozen SigLIP), then train the lightweight "student" to mimic its soft probability outputs.

The student learns from the teacher's confidence distribution (e.g., "this is 70% foul, 25% borderline, 5% clean") rather than just hard labels (0 or 1).

**Expected Improvement:** +2–3%.

---

### Method 11: Better Backbone — InternVideo2

Replace SigLIP entirely with **InternVideo2**, a video-language model pre-trained on 2 billion video-text pairs. Unlike SigLIP (image-only), InternVideo2 natively encodes spatio-temporal features.

**Expected Improvement:** +5–8%, but requires 24 GB+ VRAM.

---

## 5. Projected Cumulative Accuracy

| Configuration | Foul Detection | Severity (3-Class) |
|--------------|---------------|-------------------|
| **Baseline** (frozen SigLIP) | 58.6% | 38.5% |
| + Horizontal Flip (Method 5) | ~62% | ~42% |
| + Temporal Jittering (Method 8) | ~63% | ~43% |
| + Multi-View Fusion (Method 2) | ~67% | ~48% |
| + Fine-Tune SigLIP (Method 1) | ~72% | ~55% |
| + Cross-Attention Fusion (Method 4) | ~74% | ~58% |
| + Ensemble of 5 (Method 6) | **~76%** | **~61%** |
| Full Replacement: Video Transformer | **~78%** | **~65%** |

---

## 6. Hardware Feasibility

| Method | Arc A770 (16 GB) | RTX 4090 (24 GB) | A100 (80 GB) |
|--------|:---:|:---:|:---:|
| Horizontal Flip | ✅ | ✅ | ✅ |
| Temporal Jittering | ✅ | ✅ | ✅ |
| Multi-View Fusion | ✅ | ✅ | ✅ |
| Cross-Attention Fusion | ✅ | ✅ | ✅ |
| Model Ensemble | ✅ | ✅ | ✅ |
| Test-Time Augmentation | ✅ | ✅ | ✅ |
| Contrastive Pre-Training | ✅ | ✅ | ✅ |
| Fine-Tune SigLIP (4 layers) | ⚠️* | ✅ | ✅ |
| Knowledge Distillation | ⚠️* | ✅ | ✅ |
| VideoMAE v2 Fine-Tuning | ❌ | ⚠️ | ✅ |
| InternVideo2 | ❌ | ❌ | ✅ |

> *⚠️ = Requires gradient checkpointing and reduced batch size (16–32)*

---

## 7. Recommended Implementation Order

Sorted by **impact-to-effort ratio** (best bang for your buck first):

| Priority | Method | Time | Expected Gain |
|----------|--------|------|--------------|
| 1️⃣ | Horizontal Flip Augmentation | 1 hour | +3–5% |
| 2️⃣ | Multi-View Fusion | 1 day | +5–8% |
| 3️⃣ | Cross-Attention Text-Video Fusion | 1 day | +3–5% |
| 4️⃣ | Fine-Tune SigLIP (4 layers) | 1–2 days | +8–12% |
| 5️⃣ | Ensemble of 5 Models | 1 day | +3–4% |
| 6️⃣ | Test-Time Augmentation | 30 min | +1–2% |

> [!IMPORTANT]
> Implementing steps 1–4 should bring foul detection to approximately **72–75% balanced accuracy**, achieving the target performance within the constraints of the Intel Arc A770 GPU.
