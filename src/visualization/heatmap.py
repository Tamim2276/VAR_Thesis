import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def tokens_to_heatmap(spatial_tokens):
    """
    Convert spatial tokens from CLIP into a 2D attention heatmap.

    Args:
        spatial_tokens: shape (256, 1024)
                        256 patch tokens for one single frame
                        1024 numbers per token

    Returns:
        heatmap: shape (224, 224)
                 float values between 0 and 1
                 0 = low attention, 1 = high attention
    """

    # Step 1 — compute attention score for each patch
    # We use the L2 norm (magnitude) of each token vector
    # A token with large magnitude = CLIP paid more attention there
    # spatial_tokens shape: (256, 1024)
    # norm shape after: (256,) — one score per patch
    scores = torch.norm(spatial_tokens, dim=-1)  # (256,)

    # Step 2 — normalize scores to range [0, 1]
    # so the least attended patch = 0, most attended = 1
    scores = scores - scores.min()
    scores = scores / (scores.max() + 1e-8)  # 1e-8 prevents division by zero

    # Step 3 — reshape from flat list to 2D grid
    # 256 tokens → 16×16 grid
    grid = scores.reshape(16, 16)  # (16, 16)

    # Step 4 — convert to numpy for OpenCV
    grid_numpy = grid.detach().numpy()  # (16, 16) float

    # Step 5 — resize from 16×16 to 224×224
    # so it matches the original frame size
    # INTER_CUBIC gives smooth upscaling
    heatmap = cv2.resize(
        grid_numpy,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # Step 6 — normalize again after resize
    # resizing can slightly change the range
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap  # (224, 224) float values 0 to 1


def overlay_heatmap_on_frame(frame, heatmap, alpha=0.5):
    """
    Overlay a heatmap on top of the original video frame.

    Args:
        frame  : shape (224, 224, 3)
        heatmap: shape (224, 224) float 0-1 — attention map
        alpha  : how transparent the heatmap is
    Returns:
        blended: shape (224, 224, 3) frame with heatmap overlay
    """

    # Step 1 — convert heatmap float [0,1] to uint8 [0,255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Step 2 — apply colormap
    # COLORMAP_JET: blue=low attention, green=medium, red=high attention
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Step 3 — convert from BGR to RGB
    # OpenCV uses BGR, but our frames are RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Step 4 — blend original frame with colored heatmap
    # result = frame * (1-alpha) + heatmap * alpha
    # alpha=0.5 means 50% original frame, 50% heatmap
    frame_float = frame.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    blended = (frame_float * (1 - alpha) + heatmap_float * alpha)

    # Step 5 — convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended  # (224, 224, 3)


def visualize_clip_heatmaps(frames, spatial_tokens, save_path="outputs/heatmap_test.png"):
    """
    For a full clip (16 frames), generate heatmaps and save
    a grid image showing original frames + heatmap overlays.

    Args:
        frames        : shape (16, 224, 224, 3) — original frames
        spatial_tokens: shape (16, 256, 1024) — tokens from CLIP
        save_path     : where to save the output image
    """

    T = frames.shape[0]  # number of frames, 16

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # We'll show 8 frames (every other one) to keep the image readable
    selected = list(range(0, T, 2))

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle("X-VARS Attention Heatmaps",
                 fontsize=14, fontweight='bold')

    # Two rows: top = original frames, bottom = heatmap overlays
    gs = gridspec.GridSpec(2, len(selected), hspace=0.3, wspace=0.05)

    for col, frame_idx in enumerate(selected):

        # Get this frame and its tokens
        frame = frames[frame_idx]                          # (224, 224, 3)
        tokens = spatial_tokens[frame_idx]                 # (256, 1024)

        # Generate heatmap
        heatmap = tokens_to_heatmap(tokens)               # (224, 224)

        # Overlay heatmap on frame
        overlay = overlay_heatmap_on_frame(frame, heatmap)  # (224, 224, 3)

        # Top row — original frame
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(frame)
        ax_top.axis('off')
        ax_top.set_title(f'Frame {frame_idx}', fontsize=8)

        # Bottom row — heatmap overlay
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.imshow(overlay)
        ax_bot.axis('off')
        if col == 0:
            ax_bot.set_ylabel('+ Heatmap', fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap visualization to: {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')

    from src.models.clip_extractor import load_clip_model, extract_spatial_tokens

    print("Loading CLIP...")
    model, preprocess = load_clip_model()

    print("Loading clip from data/frames...")
    frames = np.load("data/frames/test/action_0/clip_0.npy")
    print(f"Frames shape: {frames.shape}")

    print("Extracting spatial tokens...")
    cls_tokens, spatial_tokens = extract_spatial_tokens(model, preprocess, frames)
    print(f"Spatial tokens shape: {spatial_tokens.shape}")

    print("Generating heatmaps...")
    visualize_clip_heatmaps(
        frames,
        spatial_tokens,
        save_path="outputs/heatmap_test.png"
    )

    print("\nDone! Open outputs/heatmap_test.png to see your heatmaps.")