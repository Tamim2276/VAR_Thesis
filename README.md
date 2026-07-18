# VAR Thesis — Grounded Multimodal Explainability for Football Officiating

> Extending [X-VARS (CVPR 2024)](https://arxiv.org/abs/2404.06332) with frame-level attention heatmaps, rule-grounded explanations, and an agentic LangGraph referee system.

---

## What This Thesis Is About

When an AI system like X-VARS decides a football action is a foul, it produces a text explanation like:

> *"The defender pulled the attacker's jersey backwards with medium intensity."*

But there is no visual evidence. Which frame? Which body region? Which specific contact triggered that decision?

This thesis adds what X-VARS is missing:

```
X-VARS output:
  Decision + Text Explanation

This thesis output:
  Decision + Text Explanation + Frame-Level Heatmap + Rule Citation
```

The heatmap shows exactly which spatial regions of which frames the model paid attention to when making the decision. This is called **visual grounding** — and no paper in sports XAI has done it before this work.

---

## The Four Research Contributions

| # | Contribution | What it solves |
|---|---|---|
| 1 | **Visually grounded heatmaps** from CLIP spatial tokens | X-VARS throws away spatial information — this keeps it |
| 2 | **Rule-retrieval augmented XAI** — grounds retrieved Laws of the Game to specific frames | SoccerRef-Agents retrieves rules but cannot show which frame activates them |
| 3 | **Contrastive bias-resistant training** — model maintains decisions under leading questions | RefereeBench found sycophancy is a major unsolved problem |
| 4 | **FaithScore metric** — automatic alignment score between text and visual attention | No sport-specific faithfulness metric exists anywhere |

---

## Dataset

**SoccerNet MVFoul** — requires NDA agreement at [soccer-net.org](https://www.soccer-net.org/tasks/mvfoul)

| Split | Actions | Clips |
|---|---|---|
| Train | 2,916 | 6,621 |
| Valid | 411 | 970 |
| Test | 301 | 706 |
| **Total** | **3,628** | **8,297** |

Each action has 2 camera views of the same foul incident. Each clip is 5 seconds. Labels include foul type, severity (no card / yellow / red), and referee explanations.

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | Intel Arc B580 12GB VRAM |
| CPU | AMD Ryzen 5 7500F |
| RAM | 16GB |
| OS | Windows 11 |
| Python | 3.10.11 |
| PyTorch | 2.13.0+xpu |

---

## Project Structure

```
xvars_thesis/
│
├── data/                          ← not in git (too large)
│   ├── soccernet/mvfouls/         ← raw downloaded videos
│   │   ├── train/
│   │   │   ├── action_0/
│   │   │   │   ├── clip_0.mp4
│   │   │   │   └── clip_1.mp4
│   │   │   └── annotations.json
│   │   ├── valid/
│   │   └── test/
│   └── frames/                    ← pre-extracted numpy arrays
│       ├── train/action_0/clip_0.npy   ← shape (16, 224, 224, 3)
│       ├── valid/
│       └── test/
│
├── src/
│   ├── dataset/
│   │   ├── preprocess.py          ← extract frames from videos, save as .npy
│   │   ├── soccernet_dataset.py   ← load .npy files, build clip catalogue
│   │   └── annotation_loader.py  ← load labels from annotations.json
│   │
│   ├── models/
│   │   ├── clip_extractor.py      ← CLIP ViT-L/14, extract spatial tokens
│   │   ├── classifier_heads.py    ← C_foul and C_sev classifier heads
│   │   └── pipeline.py            ← end-to-end: frames → predictions + heatmaps
│   │
│   ├── training/
│   │   └── train.py               ← training loop, evaluation, checkpointing
│   │
│   └── visualization/
│       └── heatmap.py             ← spatial tokens → attention heatmap overlay
│
├── models/                        ← not in git (large files)
│   └── best/
│       └── checkpoint_epoch_1.pt
│
├── outputs/                       ← saved heatmap visualizations
│   ├── heatmap_test.png
│   └── pipeline_test.png
│
├── logs/
│   └── training_history.json
│
├── download_data.py               ← download SoccerNet MVFoul dataset
├── requirements.txt
└── README.md
```

---

## What Each File Does

### `download_data.py`
Downloads the SoccerNet MVFoul dataset using the SoccerNet Python API. Requires NDA access and the password from the SoccerNet team. Downloads train, valid, test, and challenge splits as zip files and extracts them.

---

### `src/dataset/preprocess.py`
**Run once before anything else.**

Opens every `.mp4` clip in `data/soccernet/mvfouls/`, extracts 16 evenly-spaced frames, resizes each to 224×224 pixels, converts from BGR to RGB, and saves as a `.npy` file in `data/frames/`. This is done once because reading numpy arrays during training is instant, while decoding video every epoch would be very slow.

Key function: `preprocess_and_save()` — processes all splits with a progress bar and skips already-processed clips so it is safe to re-run.

---

### `src/dataset/soccernet_dataset.py`
Scans `data/frames/` and builds a catalogue of all clips. Returns a list of dictionaries, one per action, each containing the action ID, split name, and paths to the `.npy` clip files.

Key functions:
- `find_actions()` — walks the frames folder and builds the catalogue
- `load_clip()` — loads one `.npy` file and returns a `(16, 224, 224, 3)` array

---

### `src/dataset/annotation_loader.py`
Reads `annotations.json` from each split folder and maps each clip to its ground truth labels. Handles edge cases found in the real data: the `"Between"` foul label, empty severity strings, severity value `5.0`, and the `"Dont know"` action class.

Key function: `load_annotations()` — returns a flat list of samples, each with `clip_path`, `foul_label` (0 or 1), `sev_label` (0–3), and `action_class` (0–7).

Label mappings:
```
Foul:     No offence=0, Offence=1, Between=0
Severity: 1.0=0 (no card), 2.0=1, 3.0=2 (yellow), 4.0/5.0=3 (red)
```

---

### `src/models/clip_extractor.py`
The core of the thesis contribution. Loads CLIP ViT-L/14 and extracts **both** the CLS token and the spatial patch tokens — something standard CLIP usage does not do.

**Why this matters:**

```
Standard CLIP:  image → average pool → 1 vector (loses location)
This code:      image → keep all tokens → 256 vectors (keeps location)
```

CLIP ViT-L/14 cuts each 224×224 frame into 14×14 pixel patches, producing a 16×16 grid of 256 patch tokens. Each token is a 1024-dimensional vector describing what CLIP saw in that 14×14 region. By keeping these tokens instead of averaging them away, you can see spatially where the model was paying attention.

Key functions:
- `load_clip_model()` — loads ViT-L/14 pretrained on OpenAI data
- `get_patch_tokens()` — manually steps through the ViT to capture all 257 tokens (1 CLS + 256 spatial)
- `extract_spatial_tokens()` — runs all 16 frames and returns `cls_tokens (16, 1024)` and `spatial_tokens (16, 256, 1024)`

---

### `src/models/classifier_heads.py`
Two lightweight neural networks that sit on top of CLIP's CLS token:

```
CLS token (1024,)
    ↓
FoulClassifier     → (2,)  — No foul / Foul
SeverityClassifier → (4,)  — No card / No card+ / Yellow / Red
```

Each classifier is a two-layer MLP: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→classes). The `XVARSClassifiers` wrapper runs both heads in one forward pass.

Total trainable parameters: **1,052,678**

---

### `src/visualization/heatmap.py`
Converts CLIP spatial tokens into a visible heatmap and overlays it on the original frame.

Pipeline per frame:
```
spatial_tokens (256, 1024)
    → L2 norm of each token → attention scores (256,)
    → normalize to [0, 1]
    → reshape to 16×16 grid
    → upsample to 224×224 (bicubic interpolation)
    → apply JET colormap (blue=low, red=high attention)
    → blend with original frame at alpha=0.5
```

Key functions:
- `tokens_to_heatmap()` — converts one frame's tokens to a (224, 224) attention map
- `overlay_heatmap_on_frame()` — blends heatmap with original RGB frame
- `visualize_clip_heatmaps()` — saves a grid image of 8 frames with their heatmaps

---

### `src/models/pipeline.py`
Connects everything into one end-to-end forward pass. Given a path to a `.npy` clip file, returns all outputs: CLS tokens, spatial tokens, video vector, foul logits, severity logits, predictions, heatmaps, and overlay images.

Also has `predict_readable()` which prints a human-readable summary of the decision.

---

### `src/training/train.py`
The full training loop. Uses PyTorch `Dataset` and `DataLoader` to batch clips and labels. In each batch:

1. Load frames from `.npy` file
2. Run frames through CLIP to get CLS tokens (CLIP is **frozen** — no gradient)
3. Average CLS tokens across 16 frames to get one video vector
4. Pass video vector through classifier heads
5. Compute CrossEntropyLoss for foul + severity
6. Backpropagate and update only the classifier head weights

Saves checkpoints every N epochs and keeps the best model by validation foul accuracy.

---

## Code Flow — How Everything Connects

```
Step 1: Download raw data
  download_data.py
  → data/soccernet/mvfouls/{split}/{action}/clip_N.mp4

Step 2: Pre-extract frames (run once)
  src/dataset/preprocess.py
  → data/frames/{split}/{action}/clip_N.npy  shape (16, 224, 224, 3)

Step 3: Build clip catalogue
  src/dataset/soccernet_dataset.py
  → list of {action_id, split, clips: [path1, path2]}

Step 4: Load labels
  src/dataset/annotation_loader.py
  → list of {clip_path, foul_label, sev_label, action_class}

Step 5: Extract features
  src/models/clip_extractor.py
  → cls_tokens     shape (16, 1024)
  → spatial_tokens shape (16, 256, 1024)

Step 6: Generate heatmaps                    ← thesis contribution
  src/visualization/heatmap.py
  → heatmaps shape (16, 224, 224)
  → overlays shape (16, 224, 224, 3)

Step 7: Classify
  src/models/classifier_heads.py
  → foul_pred  0 or 1
  → sev_pred   0, 1, 2, or 3

Step 8: Train
  src/training/train.py
  → models/best/checkpoint_epoch_N.pt

Step 9: Full pipeline test
  src/models/pipeline.py
  → outputs/pipeline_test.png
```

---

## How to Run Everything

### 0. Environment setup

```bash
# Create virtual environment with Python 3.10
py -3.10 -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch for Intel Arc GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu

# Install all other dependencies
pip install transformers==4.40.0 open-clip-torch opencv-python Pillow
pip install numpy pandas matplotlib seaborn tqdm scikit-learn
pip install jupyter ipykernel SoccerNet
```

Verify your GPU is detected:
```bash
python -c "import torch; print('XPU:', torch.xpu.is_available())"
# Should print: XPU: True
```

---

### 1. Download the dataset

You must first sign the NDA at [soccer-net.org](https://www.soccer-net.org) and receive the password by email.

```bash
python download_data.py
```

Then extract the downloaded zips:
```bash
python -c "
import zipfile, os
for split in ['train', 'valid', 'test', 'challenge']:
    zip_path = f'data/soccernet/mvfouls/{split}.zip'
    extract_to = f'data/soccernet/mvfouls/{split}'
    os.makedirs(extract_to, exist_ok=True)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f'{split} extracted')
"
```

---

### 2. Pre-extract frames (run once)

```bash
python src/dataset/preprocess.py
```

This takes 10–20 minutes and saves 20GB of `.npy` files to `data/frames/`. You only need to do this once.

Expected output:
```
Processing train: 2916 actions
Processing valid: 411 actions
Processing test:  301 actions

Done!
  Saved   : 8923 clips
  Total .npy files : 8923
  Sample shape     : (16, 224, 224, 3)
```

---

### 3. Verify the dataset loader

```bash
python src/dataset/soccernet_dataset.py
```

Expected output:
```
Found 3901 actions total
  train       : 2916 actions
  valid       : 411 actions
  test        : 301 actions
  Total clips : 8923
Dataset loader ready!
```

---

### 4. Verify annotations loaded correctly

```bash
python src/dataset/annotation_loader.py
```

Expected output:
```
train: 2916 actions in annotations
valid: 411 actions in annotations
Total samples: 8297
Train samples ready for training: 6621
```

---

### 5. Test CLIP feature extraction

```bash
python src/models/clip_extractor.py
```

Expected output:
```
CLIP loaded.
Frames shape: (16, 224, 224, 3)
CLS tokens     : torch.Size([16, 1024])
Spatial tokens : torch.Size([16, 256, 1024])
All shapes correct!
```

---

### 6. Test classifier heads

```bash
python src/models/classifier_heads.py
```

Expected output:
```
Foul logits shape   : torch.Size([1, 2])
Severity logits shape: torch.Size([1, 4])
All shapes correct!
```

---

### 7. Generate heatmaps

```bash
python src/visualization/heatmap.py
```

Opens `outputs/heatmap_test.png` showing 8 frames with attention heatmaps overlaid. Before training, heatmaps will highlight high-contrast regions like scoreboards. After training they will focus on player contact regions.

---

### 8. Run the full pipeline

```bash
python src/models/pipeline.py
```

Expected output:
```
==================================================
X-VARS PREDICTION
==================================================
Foul decision : Foul
Severity      : Yellow card
Heatmaps      : generated for 16 frames
==================================================
Saved to: outputs/pipeline_test.png
```

---

### 9. Train the model

```bash
python src/training/train.py
```

Training runs on your Intel Arc B580 XPU. Expected speed: ~1.7 batches/second. Each epoch takes about 45 minutes locally.

After 2 epochs locally (to confirm training works), move to Kaggle GPU for full training.

Current local results after 2 epochs:
```
Epoch 1: foul 85.9% train | 88.6% valid
Epoch 2: foul 85.9% train | 88.6% valid (stable)
Severity: 59.3% train | 55.7% valid
```

---

## Current Results

| Metric | Value | Notes |
|---|---|---|
| Val foul accuracy | 88.6% | After 2 epochs locally |
| Val severity accuracy | 55.7% | After 2 epochs locally |
| Heatmap status | Generating | Focuses on players after full training |
| Training device | Intel Arc B580 XPU | |
| Trainable params | 1,052,678 | Classifier heads only, CLIP frozen |

---

## Key Design Decisions

**Why freeze CLIP?**
CLIP ViT-L/14 has 427M parameters. Fine-tuning all of them requires significantly more VRAM and training time. In Stage 1, only the 1M classifier head parameters are trained. In a later stage, the top layers of CLIP will be unfrozen for domain-specific fine-tuning.

**Why save frames as .npy instead of decoding video during training?**
Reading a `.npy` file is instantaneous. Decoding a video requires opening the file, seeking to frames, and decompressing — which adds ~2 seconds per clip. Over 6621 training clips per epoch this becomes the bottleneck. Pre-extracting saves hours of training time.

**Why keep spatial tokens instead of the CLS token only?**
The CLS token is a single vector summarizing the whole frame. Keeping all 256 spatial tokens preserves the spatial location of each 14×14 pixel patch. This is what enables heatmap generation — you can see which patch the model attended to.

**Why 16 frames per clip?**
X-VARS uses 16 frames — 8 frames before the foul and 8 after. This is enough to capture the full incident without being computationally prohibitive.

---

## Dependencies

```
torch==2.13.0+xpu
transformers==4.40.0
open-clip-torch
opencv-python==5.0.0
Pillow
numpy==2.2.6
pandas
matplotlib
seaborn
tqdm
scikit-learn
SoccerNet
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- **X-VARS** — Held et al., CVPR Workshop 2024. [arxiv.org/abs/2404.06332](https://arxiv.org/abs/2404.06332)
- **SoccerNet MVFoul** — Held et al., CVPR Workshop 2023. [soccer-net.org](https://www.soccer-net.org/tasks/mvfoul)
- **CLIP** — Radford et al., ICML 2021. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- **SPORTU** — ICLR 2025. [arxiv.org/abs/2410.08474](https://arxiv.org/abs/2410.08474)
- **RefereeBench** — April 2026. [arxiv.org/abs/2604.15736](https://arxiv.org/abs/2604.15736)
- **DeepSport** — November 2025. [arxiv.org/abs/2511.12908](https://arxiv.org/abs/2511.12908)
- **SoccerRef-Agents** — 2025. [arxiv.org/abs/2604.23392](https://arxiv.org/abs/2604.23392)

---

## License

Code: MIT License

Dataset: SoccerNet MVFoul is subject to the SoccerNet NDA. Do not redistribute.
