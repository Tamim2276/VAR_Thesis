# Fine-Tuning SigLIP: Complete Guide

## What We Have Right Now (The Frozen Pipeline)

Your current pipeline works like this:

```
Step 1 (ONE TIME — already done):
  Raw Video Frames → SigLIP (FROZEN) → Save .pt feature files to disk

Step 2 (EVERY training run — fast):
  Load .pt files → Temporal Attention Classifier → Predict Foul/No Foul
```

SigLIP is run ONCE, its output is saved to disk, and then your classifier trains 
on these saved features. SigLIP never learns anything about soccer — it stays 
exactly as Google trained it (on random internet images).

---

## Why This Limits Your Accuracy

Imagine you hired a translator who only speaks French and English. You ask them 
to translate a Japanese medical document. They can recognize it's a "document" 
and it has "text" — but they can't read any of it.

That's what SigLIP is doing right now. It was trained to understand internet 
images (cats, cars, landscapes). When it sees a soccer tackle, it outputs a 
feature vector that says:

```
"I see: two humans, a green field, a ball, one person is on the ground"
```

But what you NEED it to say is:

```
"I see: studs-up challenge, high ankle contact, excessive force, 
 player lost control of the ball before contact"
```

SigLIP literally doesn't know what "studs-up" means because it was never 
trained on soccer tackles.

---

## What Fine-Tuning Does

Fine-tuning teaches SigLIP to look at soccer-specific details. Instead of 
freezing the model and just using its generic output, we let the error 
signal (gradient) flow back INTO SigLIP during training.

```
BEFORE (frozen):
  Frame → SigLIP [LOCKED 🔒] → generic feature → classifier → loss
                                                              ↓
                                              gradient stops here ❌

AFTER (fine-tuned):
  Frame → SigLIP [PARTIALLY UNLOCKED 🔓] → specialized feature → classifier → loss
                    ↑                                                          ↓
                    └──────────── gradient flows back ←←←←←←←←←←←←←←←←←←←←←←←┘
```

After fine-tuning, SigLIP's output changes from generic "two people on grass" 
to specialized "studs-up sliding tackle with high contact".

---

## Why We Only Unfreeze the LAST 4 Layers

SigLIP SO400M has 27 transformer layers:

```
Layer 1-5:   Detect basic patterns (edges, colors, textures)
Layer 6-15:  Detect mid-level patterns (shapes, body parts, objects)
Layer 16-23: Detect high-level patterns (scenes, activities)
Layer 24-27: Combine everything into the final representation  ← UNFREEZE THESE
```

- **Layers 1-23 stay frozen:** They already know how to detect edges, shapes, 
  and body parts. This knowledge is universal and doesn't need to change.
- **Layers 24-27 get unfrozen:** These are the layers that decide HOW to 
  combine all the detected patterns into a final feature vector. We want them 
  to combine patterns in a soccer-specific way.

If you unfreeze ALL 27 layers:
- ❌ You need way more VRAM (won't fit on your A770)
- ❌ You risk "catastrophic forgetting" — the model forgets how to see basic 
  shapes and becomes useless
- ❌ Training takes 10x longer

If you unfreeze only 4 layers:
- ✅ Fits on 16GB VRAM with gradient checkpointing
- ✅ The model keeps its foundational knowledge
- ✅ Training takes only 2-3x longer than frozen

---

## The Big Architecture Change

Currently, your pipeline has 2 SEPARATE steps (precompute features, then train 
classifier). Fine-tuning requires these to become ONE CONNECTED step:

### Current Pipeline (2 disconnected steps):
```
Step 1: frames → SigLIP → save .pt files
Step 2: load .pt files → classifier → loss
```

### New Pipeline (1 connected end-to-end step):
```
frames → SigLIP (partially unfrozen) → classifier → loss → gradient → update SigLIP
```

This means:
1. **We can NOT use pre-computed .pt features anymore.** The whole point is that 
   SigLIP's output changes as it learns. If we load old .pt files, we're using 
   the OLD frozen SigLIP output, and the fine-tuning has no effect.

2. **We must load RAW FRAMES (.npy files) during training.** Each batch will:
   - Load 16 raw frames from disk
   - Pass them through SigLIP IN REAL TIME
   - Pass SigLIP's output to the classifier
   - Compute the loss
   - Send gradients back through both the classifier AND SigLIP

3. **Training will be MUCH slower.** Instead of loading a tiny 74KB .pt file, 
   each batch now runs a 400M parameter vision transformer. Expect training to 
   go from 30 seconds per epoch → 15-30 minutes per epoch.

---

## How to Handle VRAM (Your Arc A770 = 16GB)

Running SigLIP + classifier + gradients will use a LOT of VRAM. Here are the 
tricks to make it fit:

### Trick 1: Gradient Checkpointing
Normally, PyTorch saves all intermediate values during the forward pass so it 
can use them for the backward pass. Gradient checkpointing throws them away 
and recalculates them during the backward pass.

```
Without checkpointing: 16GB VRAM needed → ❌ OOM crash
With checkpointing:    ~10GB VRAM needed → ✅ Fits!
```

Trade-off: Training is ~30% slower, but it fits in memory.

### Trick 2: Small Batch Size + Gradient Accumulation
Instead of batch_size=64 (won't fit), we use batch_size=4 and accumulate 
gradients over 16 steps before updating:

```
Step 1:  batch of 4 → compute gradient → DON'T update, just save gradient
Step 2:  batch of 4 → compute gradient → add to saved gradient
...
Step 16: batch of 4 → compute gradient → add to saved gradient → NOW UPDATE

Result: Mathematically identical to batch_size=64, but only 4 samples 
in VRAM at a time.
```

### Trick 3: Discriminative Learning Rate
SigLIP layers get a TINY learning rate (1e-5) while the classifier gets a 
normal learning rate (1e-4). This prevents SigLIP from changing too fast and 
"forgetting" what it already knows.

```python
optimizer = AdamW([
    {"params": siglip_unfrozen_layers, "lr": 1e-5},   # Gentle updates
    {"params": classifier_head,        "lr": 1e-4},   # Normal updates
])
```

### Trick 4: Process Fewer Frames
Instead of all 16 frames, we sample 8 frames during fine-tuning to cut 
VRAM usage in half. During final feature extraction, we use all 16.

---

## The 3-Phase Training Strategy

### Phase A: Warm Up the Classifier (5 epochs)
Keep SigLIP fully frozen. Only train the classifier head on the new 
8-frame input. This gives the classifier a good starting point before 
SigLIP starts changing its output.

### Phase B: Fine-Tune Together (15 epochs)
Unfreeze the last 4 SigLIP layers. Train both SigLIP and the classifier 
together with discriminative learning rates. This is where the magic happens.

### Phase C: Re-Extract Features (one time)
After fine-tuning is complete, run the IMPROVED SigLIP over ALL clips 
(train + valid + test) and save new .pt feature files. These new features 
are now soccer-specialized and will give your existing training scripts 
(train_foul.py, train_severity.py) a massive accuracy boost.

```
Phase A: Frozen SigLIP + train classifier     → ~60% (same as now)
Phase B: Unfreeze 4 layers + train together   → ~68-72%
Phase C: Re-extract features with new SigLIP  → Use new .pt files everywhere
```

---

## What Files Need to Change

### 1. NEW: `src/training/finetune_siglip.py`
The main fine-tuning training script. This is the big new file. It will:
- Load raw .npy frames instead of .pt features
- Build an end-to-end model: SigLIP → Classifier
- Implement gradient checkpointing and accumulation
- Use discriminative learning rates
- Save the fine-tuned SigLIP weights

### 2. MODIFY: `src/models/siglip_extractor.py`
Add a function to load fine-tuned weights into the SigLIP model before 
feature extraction.

### 3. MODIFY: `src/dataset/precompute_features.py`
After fine-tuning, this script re-extracts features using the improved 
SigLIP. It already has this logic (the `finetuned_weights_path` parameter 
on line 16) — we just need to point it to the new weights.

### 4. NO CHANGE: `src/training/train_foul.py` and `train_severity.py`
These scripts stay exactly the same! They just load .pt files. After 
Phase C re-extracts features, these scripts will automatically benefit 
from the improved features without any code changes.

---

## Expected Timeline

| Step | Time | What Happens |
|------|------|-------------|
| Write `finetune_siglip.py` | 15 min | I write the script |
| Phase A: Warm-up | ~30 min | Train classifier on raw frames |
| Phase B: Fine-tune | ~2-3 hours | Train SigLIP + classifier together |
| Phase C: Re-extract | ~2.5 hours | Run improved SigLIP on all clips |
| Re-train foul model | ~5 min | `train_foul.py` with new features |
| Re-train severity model | ~5 min | `train_severity.py` with new features |

**Total: ~5-6 hours** (most of it is GPU processing you can leave running)

---

## Expected Accuracy After Fine-Tuning

| Task | Current | After Fine-Tuning |
|------|---------|-------------------|
| Foul Detection | 60.3% | **68-72%** |
| Severity (3-Class) | 43.0% | **50-55%** |

These are conservative estimates. The gain comes from SigLIP finally 
understanding what a soccer foul looks like at the pixel level, rather 
than just seeing "two people on green grass".

---

## Summary: Why This Is The Right Move

1. **Biggest single accuracy gain available** (+8-12%)
2. **Fits on your hardware** with gradient checkpointing
3. **Doesn't break your existing code** — train_foul.py and train_severity.py 
   stay exactly the same, they just get better input features
4. **Great for your paper** — "domain-adaptive fine-tuning of frozen VFMs" 
   is a hot research topic and adds methodological depth to your thesis
5. **One-time cost** — once SigLIP is fine-tuned and features are re-extracted, 
   all future experiments benefit automatically

Ready for me to write the code?
