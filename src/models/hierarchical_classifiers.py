import torch
import torch.nn as nn
import torch.nn.functional as F

class FoulClassifier(nn.Module):
    """
    Model 1: Binary Classifier with Temporal Attention.
    Instead of averaging all 16 frames, it LEARNS which frames contain the foul.
    Input: (Batch, 16, 1152) — all 16 frame features
    Output: 2 logits
    """
    def __init__(self, feat_dim=1152, hidden_dim=128):
        super().__init__()
        # Temporal Attention: learns which of the 16 frames matters most
        self.frame_attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Classifier on the attention-weighted feature
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        # x: (batch, 16, 1152)
        
        # Advanced Regularization: Temporal Frame Masking
        # Randomly zero out 2 to 3 frames during training to prevent relying on a single memorized frame.
        if self.training:
            batch_size, seq_len, _ = x.shape
            mask = torch.ones(batch_size, seq_len, 1, device=x.device)
            for i in range(batch_size):
                num_drop = torch.randint(2, 4, (1,)).item()
                drop_indices = torch.randperm(seq_len)[:num_drop]
                mask[i, drop_indices, 0] = 0.0
            x = x * mask
        # Step 1: Compute attention score for each frame
        attn_scores = self.frame_attention(x)          # (batch, 16, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, 16, 1)
        
        # Step 2: Weighted sum — the "impact frame" gets highest weight
        weighted = (x * attn_weights).sum(dim=1)       # (batch, 1152)
        
        # Step 3: Classify
        return self.classifier(weighted)

class SeverityClassifier(nn.Module):
    """
    Model 2: Simple 3-Class Classifier with Average Pooling.
    Uses average pooling over 16 frames instead of Temporal Attention 
    to prevent overfitting on the tiny Red Card class (185 samples).
    Input: (Batch, 16, 1152) or (Batch, 1152)
    Output: 3 logits
    """
    def __init__(self, feat_dim=1152, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x):
        # x: (batch, 16, 1152) or (batch, 1152)
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average pool over 16 frames -> (batch, 1152)
        return self.classifier(x)

if __name__ == "__main__":
    print("Testing Hierarchical Classifiers with Temporal Attention...")
    
    # Now input is (batch, 16_frames, 1152_features) instead of (batch, 1152)
    dummy_input = torch.rand(8, 16, 1152)
    
    foul_model = FoulClassifier()
    sev_model = SeverityClassifier()
    
    foul_logits = foul_model(dummy_input)
    sev_logits = sev_model(dummy_input)
    
    print(f"Foul Model Output Shape: {foul_logits.shape} (Expected: 8, 2)")
    print(f"Severity Model Output Shape: {sev_logits.shape} (Expected: 8, 3)")
    print(f"Foul Model Params: {sum(p.numel() for p in foul_model.parameters()):,}")
    print(f"Severity Model Params: {sum(p.numel() for p in sev_model.parameters()):,}")