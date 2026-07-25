import torch
import numpy as np
import cv2

from src.nodes.node2_grounding import SpatioTemporalGroundingNode
from src.models.siglip_extractor import load_siglip_model, extract_spatial_tokens

def main():
    print("Loading Real Data...")
    
    # 1. Load your actual video frames! (Shape: 16, 224, 224, 3)
    frames_path = "data/frames/valid/action_0/clip_0.npy"
    frames = np.load(frames_path)
    
    # 2. Load SigLIP to get the spatial features on the fly
    print("Loading SigLIP...")
    siglip_model, siglip_processor = load_siglip_model()
    
    # Move model to your Intel Arc GPU (XPU) if available!
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using device: {device.upper()}")
    siglip_model = siglip_model.to(device)
    
    print("Extracting Spatial Features...")
    _, spatial_features = extract_spatial_tokens(siglip_model, siglip_processor, frames)
    
    # 3. Create the LangGraph memory state
    state = {
        "spatial_features": spatial_features
    }
    
    # 4. Initialize and Run Node 2!
    node = SpatioTemporalGroundingNode()
    new_state = node(state)
    
    temporal_curve = new_state["temporal_curve"]
    spatial_heatmap = new_state["spatial_heatmap"]
    peak_frame_idx = new_state["peak_frame_idx"]
    
    print(f"\nReal Temporal Curve Shape: {temporal_curve.shape}")
    print(f"Real Spatial Heatmap Shape: {spatial_heatmap.shape}")
    
    # 5. Visualize it! We pass in the actual Peak Frame so it blends together!
    real_frame = frames[peak_frame_idx]
    
    node.visualize(
        temporal_curve=temporal_curve, 
        spatial_heatmap=spatial_heatmap, 
        original_frame=real_frame,  # Pass the real image in!
        save_path="real_grounding_output.png"
    )

if __name__ == "__main__":
    main()