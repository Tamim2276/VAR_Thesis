import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any

class SpatioTemporalGroundingNode:
    def __init__(self):
        """
        Node 2: Generates 2D Spatial Heatmaps and 1D Temporal Attention Curves
        based on raw SigLIP feature tensors.
        """
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads from state: 'spatial_features' (tensor of shape [16, 729, 1152])
        Writes to state: 'temporal_curve', 'peak_frame_idx', 'spatial_heatmap'
        """
        print("--- Executing Node 2: Spatio-Temporal Grounding ---")
        
        # 1. Load the spatial features (T=16, Patches=729, Dim=1152)
        features = state.get("spatial_features")
        if features is None:
            raise ValueError("No spatial features found in state. Did Node 1 run?")
            
        # Ensure it's a CPU tensor for processing
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        # ---------------------------------------------------------
        # Step 1: Temporal Grounding (Contribution 5)
        # ---------------------------------------------------------
        # Calculate L2 norm along the embedding dimension (axis=2) -> shape (16, 729)
        patch_magnitudes = np.linalg.norm(features, axis=2)
        
        # Average the activation across all 729 patches to get the overall frame activation -> shape (16,)
        temporal_curve = np.mean(patch_magnitudes, axis=1)
        
        # Identify the exact frame index where the action peaks
        peak_frame_idx = int(np.argmax(temporal_curve))
        print(f"Temporal Grounding: Highest activation detected at Frame {peak_frame_idx}")

        # ---------------------------------------------------------
        # Step 2: Spatial Grounding (Contribution 1)
        # ---------------------------------------------------------
        # Isolate the 729 patch activations for the peak frame -> shape (729,)
        peak_frame_patches = patch_magnitudes[peak_frame_idx]
        
        # Reshape into the 27x27 grid (because your SigLIP model uses 384 resolution / 14 patch size = 27.4)
        grid_27x27 = peak_frame_patches.reshape(27, 27)
        
        # Normalize the grid between 0 and 1 so it draws cleanly
        grid_normalized = (grid_27x27 - np.min(grid_27x27)) / (np.max(grid_27x27) - np.min(grid_27x27) + 1e-8)
        
        # Write the results back to the LangGraph memory state
        state["temporal_curve"] = temporal_curve
        state["peak_frame_idx"] = peak_frame_idx
        state["spatial_heatmap"] = grid_normalized
        
        return state

    def visualize(self, temporal_curve, spatial_heatmap, original_frame=None, save_path="grounding_output.png"):
        """
        Helper function to actually draw and save the graphs and heatmaps!
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: The 1D Temporal Spike
        ax1.plot(temporal_curve, marker='o', color='blue', linewidth=2)
        ax1.set_title("1D Temporal Attention (The 'When')")
        ax1.set_xlabel("Frame Number (0 to 15)")
        ax1.set_ylabel("Activation Magnitude")
        ax1.axvline(x=np.argmax(temporal_curve), color='red', linestyle='--', label='Impact Frame')
        ax1.legend()
        
        # Plot 2: The 2D Spatial Heatmap
        # Scale the 27x27 grid up to the standard image size (224x224)
        heatmap_resized = cv2.resize(spatial_heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        if original_frame is not None:
            # If you pass in the real video frame, we blend the glowing heatmap over it!
            # Ensure original frame is also 224x224
            original_resized = cv2.resize(original_frame, (224, 224))
            
            # Apply the JET colormap (glows Red on high spots, Blue on low spots)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            # Blend the original image (50%) and the heatmap (50%)
            overlay = cv2.addWeighted(original_resized, 0.5, heatmap_color, 0.5, 0)
            
            # OpenCV uses BGR, but Matplotlib uses RGB. We must convert it before plotting.
            ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            # If no original frame is provided, just show the raw heatmap
            im = ax2.imshow(heatmap_resized, cmap='jet')
            plt.colorbar(im, ax=ax2)
            
        ax2.set_title("2D Spatial Heatmap (The 'Where')")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved visualizations to {save_path}")