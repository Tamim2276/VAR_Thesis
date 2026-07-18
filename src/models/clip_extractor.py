import torch
import numpy as np
import open_clip
from PIL import Image

CLIP_MODEL_NAME = "ViT-L-14-quickgelu"
CLIP_PRETRAINED_TAG = "openai"


def load_clip_model():
    """
    Load CLIP ViT-L/14 and return model + preprocessor.
    Called once at the start of your pipeline.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED_TAG,
    )
    model.eval()  # evaluation mode — disables dropout
    return model, preprocess


def get_visual_token_dim(model):
    """Return the width of the unprojected CLS and patch tokens."""
    return model.visual.class_embedding.shape[-1]


def extract_spatial_tokens(model, preprocess, frames_numpy):
    """
    Given a numpy array of frames, run each through CLIP and
    extract the spatial patch tokens (not just the CLS token).

    ViT-L/14 patch math:
        Image size   : 224 x 224 pixels
        Patch size   : 14 x 14 pixels
        Grid         : 224 / 14 = 16 patches per side
        Total patches: 16 x 16 = 256 patches per frame
        Total tokens : 1 CLS + 256 patches = 257 tokens per frame

    Returns:
        cls_tokens     : shape (T, 1024)
                         one summary vector per frame
        spatial_tokens : shape (T, 256, 1024)
                         256 patch vectors per frame
                         arranged as a 16x16 grid spatially
                         1024 = token dimension for ViT-L/14
    """

    T = frames_numpy.shape[0]  # number of frames,16
    cls_list = []
    spatial_list = []
    device = next(model.parameters()).device
    
    with torch.no_grad():  # no gradients needed for feature extraction
        for i in range(T):

            # Step 1 — convert numpy frame to PIL Image
            # PIL is what CLIP's preprocessor expects
            frame_pil = Image.fromarray(frames_numpy[i])

            # Step 2 — preprocess: normalize pixel values, resize, etc.
            frame_tensor = preprocess(frame_pil)

            # Step 3 — add batch dimension
            # (3, 224, 224) → (1, 3, 224, 224)
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            # Step 4 — run through CLIP visual encoder
            # returns ALL tokens including spatial patches
            tokens = get_patch_tokens(model, frame_tensor)

            # tokens shape: (1, 257, 1024)
            # 257 = 1 CLS token + 256 patch tokens (16x16 grid)
            # 1024 = hidden dimension of ViT-L/14

            cls_token = tokens[:, 0, :]      # (1, 1024) — summary token
            patch_tokens = tokens[:, 1:, :]  # (1, 256, 1024) — spatial tokens

            # Move back to CPU for list collection to save VRAM, or leave if VRAM is abundant
            cls_list.append(cls_token.squeeze(0).cpu())     # (1024,)
            spatial_list.append(patch_tokens.squeeze(0).cpu())  # (256, 1024)

    # Stack all frames together
    cls_tokens = torch.stack(cls_list, dim=0)       # (T, 1024)
    spatial_tokens = torch.stack(spatial_list, dim=0)  # (T, 256, 1024)

    return cls_tokens, spatial_tokens


def get_patch_tokens(model, image_tensor):
    """
    Run image through CLIP's visual transformer and return
    ALL tokens including spatial patches — not just the final CLS.

    This is the key difference from standard CLIP usage:
        Standard CLIP : returns only CLS token (throws away spatial info)
        My version   : returns ALL 257 tokens (keeps spatial info)
        
    Returns:
        tokens: shape (1, 257, 1024)
    """
    visual = model.visual  # the vision transformer part of CLIP

    x = image_tensor

    # Patch embedding
    # conv1 splits the 224x224 image into 14x14 pixel patches
    # with stride 14, producing a 16x16 grid of patch embeddings
    # output: (1, 1024, 16, 16)
    x = visual.conv1(x)

    # Flatten spatial dimensions
    # (1, 1024, 16, 16) → (1, 1024, 256)
    x = x.reshape(x.shape[0], x.shape[1], -1)

    # Transpose to token format
    # (1, 1024, 256) → (1, 256, 1024)
    x = x.permute(0, 2, 1)

    # Add the CLS token at position 0
    # class_embedding shape: (1024,)
    # after unsqueeze twice: (1, 1, 1024)
    cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0)
    cls_token = cls_token.expand(x.shape[0], -1, -1)  # (1, 1, 1024)

    # Concatenate CLS + patch tokens
    # (1, 1, 1024) + (1, 256, 1024) → (1, 257, 1024)
    x = torch.cat([cls_token, x], dim=1)

    # Add positional embeddings
    # tells the model where each patch is located in the image
    x = x + visual.positional_embedding.unsqueeze(0)

    # Patch dropout (only active during training, skipped in eval mode)
    x = visual.patch_dropout(x) if hasattr(visual, 'patch_dropout') else x

    # Layer norm before transformer
    x = visual.ln_pre(x)

    # open_clip 3.x transformers are batch-first (B, tokens, width). Passing
    # sequence-first tokens to these models makes the CLS token ignore image
    # patches and yields identical features for different frames.
    if getattr(visual.transformer, "batch_first", False):
        return visual.ln_post(visual.transformer(x))

    # Transformer expects sequence first
    # (1, 257, 1024) → (257, 1, 1024)
    x = x.permute(1, 0, 2)

    # Run through all transformer attention layers
    x = visual.transformer(x)

    # Back to batch first
    # (257, 1, 1024) → (1, 257, 1024)
    x = x.permute(1, 0, 2)

    # Final layer norm
    x = visual.ln_post(x)

    return x  # (1, 257, 1024)


if __name__ == "__main__":

    print("Loading CLIP ViT-L/14...")
    model, preprocess = load_clip_model()
    print("CLIP loaded.")

    print("\nLoading one real clip from data/frames...")
    frames = np.load("data/frames/test/action_0/clip_0.npy")
    print(f"Frames shape: {frames.shape}")  # expect (16, 224, 224, 3)

    print("\nExtracting spatial tokens...")
    cls_tokens, spatial_tokens = extract_spatial_tokens(model, preprocess, frames)

    print(f"\nResults:")
    print(f"  CLS tokens     : {cls_tokens.shape}")
    print(f"  Spatial tokens : {spatial_tokens.shape}")
    print(f"  Expected CLS   : torch.Size([16, 1024])")
    print(f"  Expected spatial: torch.Size([16, 256, 1024])")

    # Verify shapes are correct
    assert cls_tokens.shape == torch.Size([16, 1024]), \
        f"CLS shape wrong! Got {cls_tokens.shape}"
    assert spatial_tokens.shape == torch.Size([16, 256, 1024]), \
        f"Spatial shape wrong! Got {spatial_tokens.shape}"

    print("\nAll shapes correct!")
    print("Each frame has a 16x16 grid of spatial tokens.")
    print("These will become the attention heatmap in the next step.")
