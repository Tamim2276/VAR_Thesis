import torch
import numpy as np
from PIL import Image
from transformers import SiglipVisionModel, AutoImageProcessor

SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"

def load_siglip_model():
    """
    Load SigLIP Vision Model and return model + processor.
    """
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model.eval()
    return model, processor

def extract_spatial_tokens(model, processor, frames_numpy, batch_size=8):
    """
    Extract global and spatial tokens using SigLIP.
    Returns:
        global_tokens : (T, 1152)
        spatial_tokens: (T, 729, 1152)
    """
    T = frames_numpy.shape[0]
    device = next(model.parameters()).device
    
    global_list = []
    spatial_list = []
    
    with torch.no_grad():
        # Process in chunks to save VRAM
        for i in range(0, T, batch_size):
            chunk = frames_numpy[i:i+batch_size]
            pil_images = [Image.fromarray(frame) for frame in chunk]
            
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            global_list.append(outputs.pooler_output.cpu())
            spatial_list.append(outputs.last_hidden_state.cpu())
            
    global_tokens = torch.cat(global_list, dim=0)       # (T, 1152)
    spatial_tokens = torch.cat(spatial_list, dim=0)     # (T, 729, 1152)
    
    return global_tokens, spatial_tokens
