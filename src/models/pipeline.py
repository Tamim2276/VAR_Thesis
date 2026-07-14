import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('.')

from src.models.clip_extractor import load_clip_model, extract_spatial_tokens
from src.models.classifier_heads import XVARSClassifiers
from src.visualization.heatmap import tokens_to_heatmap, overlay_heatmap_on_frame


class XVARSPipeline(nn.Module):
    """
    The full X-VARS pipeline.

    Given a video clip it produces:
        1. Foul prediction     (foul / no foul)
        2. Severity prediction (no card / yellow / red)
        3. Attention heatmaps  (which regions mattered)

    This is Contribution 1
    """

    def __init__(self):
        super(XVARSPipeline, self).__init__()

        # Load CLIP
        print("  Loading CLIP ViT-L/14...")
        self.clip_model, self.preprocess = load_clip_model()

        # Freeze CLIP weights for now
        # we don't want to change CLIP during early testing
        # only the classifier heads will be trained first
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Build classifier heads
        print("  Building classifier heads...")
        self.classifiers = XVARSClassifiers(
            input_dim=1024,
            hidden_dim=512
        )

        # Labels for readable output
        self.foul_labels = ["No foul", "Foul"]
        self.sev_labels = [
            "No offence",
            "Offence - No card",
            "Offence - Yellow card",
            "Offence - Red card"
        ]

    def forward(self, frames_numpy):
        """
        Full forward pass — from raw frames to predictions + heatmaps.

        Args:
            frames_numpy: shape (T, 224, 224, 3) numpy array
                          T = number of frames (16)

        Returns:
            results: dict containing
                'cls_tokens'     : (T, 1024)
                'spatial_tokens' : (T, 256, 1024)
                'video_vector'   : (1, 1024) averaged across frames
                'foul_logits'    : (1, 2)
                'sev_logits'     : (1, 4)
                'foul_pred'      : int — 0 or 1
                'sev_pred'       : int — 0, 1, 2, or 3
                'heatmaps'       : (T, 224, 224) attention maps
                'overlays'       : (T, 224, 224, 3) frames with heatmap
        """

        # Step 1 — extract features from CLIP
        cls_tokens, spatial_tokens = extract_spatial_tokens(
            self.clip_model,
            self.preprocess,
            frames_numpy
        )
        # cls_tokens     : (T, 1024)
        # spatial_tokens : (T, 256, 1024)

        # Step 2 — average CLS tokens across all frames
        # to get one single video-level representation
        video_vector = cls_tokens.mean(dim=0).unsqueeze(0)
        # (T, 1024) → (1024,) → (1, 1024)

        # Step 3 — run classifier heads
        foul_logits, sev_logits = self.classifiers(video_vector)
        foul_pred, sev_pred = self.classifiers.predict(video_vector)

        # Step 4 — generate heatmaps for every frame
        T = frames_numpy.shape[0]
        heatmaps = []
        overlays = []

        for i in range(T):
            # convert spatial tokens to 2D attention map
            heatmap = tokens_to_heatmap(spatial_tokens[i])  # (224, 224)

            # overlay on original frame
            overlay = overlay_heatmap_on_frame(
                frames_numpy[i],
                heatmap,
                alpha=0.5
            )  # (224, 224, 3)

            heatmaps.append(heatmap)
            overlays.append(overlay)

        heatmaps = np.stack(heatmaps, axis=0)  # (T, 224, 224)
        overlays = np.stack(overlays, axis=0)  # (T, 224, 224, 3)

        return {
            'cls_tokens'     : cls_tokens,
            'spatial_tokens' : spatial_tokens,
            'video_vector'   : video_vector,
            'foul_logits'    : foul_logits,
            'sev_logits'     : sev_logits,
            'foul_pred'      : foul_pred.item(),
            'sev_pred'       : sev_pred.item(),
            'heatmaps'       : heatmaps,
            'overlays'       : overlays,
        }

    def predict_readable(self, frames_numpy):
        """
        Same as forward but prints a human readable summary.
        """
        results = self.forward(frames_numpy)

        print("\n" + "="*50)
        print("X-VARS PREDICTION")
        print("="*50)
        print(f"Foul decision : {self.foul_labels[results['foul_pred']]}")
        print(f"Severity      : {self.sev_labels[results['sev_pred']]}")
        print(f"Heatmaps      : generated for {len(results['heatmaps'])} frames")
        print("="*50)

        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Building X-VARS pipeline...")
    pipeline = XVARSPipeline()

    print("\nLoading a real foul clip...")
    frames = np.load("data/frames/test/action_0/clip_0.npy")
    print(f"Frames shape: {frames.shape}")

    print("\nRunning full pipeline...")
    results = pipeline.predict_readable(frames)

    print("\nOutput shapes:")
    print(f"  cls_tokens     : {results['cls_tokens'].shape}")
    print(f"  spatial_tokens : {results['spatial_tokens'].shape}")
    print(f"  video_vector   : {results['video_vector'].shape}")
    print(f"  foul_logits    : {results['foul_logits'].shape}")
    print(f"  sev_logits     : {results['sev_logits'].shape}")
    print(f"  heatmaps       : {results['heatmaps'].shape}")
    print(f"  overlays       : {results['overlays'].shape}")

    # Save a visualization of the full pipeline output
    print("\nSaving pipeline visualization...")
    os.makedirs("outputs", exist_ok=True)

    T = frames.shape[0]
    selected = list(range(0, T, 2))  # every other frame

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f"X-VARS Full Pipeline Output\n"
        f"Decision: {pipeline.foul_labels[results['foul_pred']]} | "
        f"Severity: {pipeline.sev_labels[results['sev_pred']]}",
        fontsize=13,
        fontweight='bold'
    )

    gs = gridspec.GridSpec(2, len(selected), hspace=0.3, wspace=0.05)

    for col, idx in enumerate(selected):
        # top row — original frame
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[idx])
        ax.axis('off')
        ax.set_title(f'Frame {idx}', fontsize=8)

        # bottom row — heatmap overlay
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(results['overlays'][idx])
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Attention', fontsize=8)

    save_path = "outputs/pipeline_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to: {save_path}")
    print("\nPipeline complete and working!")
    print("All three outputs produced:")
    print("  1. Foul decision")
    print("  2. Severity level")
    print("  3. Attention heatmaps")