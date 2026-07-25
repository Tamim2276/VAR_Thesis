import torch
import torch.nn as nn

class XVARSTemporalModel(nn.Module):
    """
    Temporal Transformer model for video sequence classification.
    Processes a sequence of frame embeddings (T, D) using Transformer Encoder layers
    with positional encodings to capture motion and temporal dynamics.
    """
    def __init__(self, embed_dim=1152, num_heads=8, num_layers=2, hidden_dim=512, max_frames=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        
        # Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Heads
        self.foul_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.sev_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        B, T, D = x.shape
        
        # Add positional embedding
        x = x + self.pos_embedding[:, :T, :]
        
        # Pass through transformer encoder
        feat_seq = self.transformer(x)  # (B, T, D)
        
        # Pool across temporal dimension (mean pooling after temporal attention interaction)
        video_feat = feat_seq.mean(dim=1)  # (B, D)
        
        foul_logits = self.foul_head(video_feat)
        sev_logits  = self.sev_head(video_feat)
        
        attn_weights = feat_seq.mean(dim=-1) # (B, T) frame importance score
        
        return foul_logits, sev_logits, attn_weights
