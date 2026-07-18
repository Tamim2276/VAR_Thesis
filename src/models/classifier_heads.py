import torch
import torch.nn as nn


class FoulClassifier(nn.Module):
    """
    Takes CLIP's CLS token and predicts whether it is a foul or not.

    Input : (batch_size, 1024) — CLS token from CLIP
    Output: (batch_size, 2)    — scores for [No foul, Foul]
    """

    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=2):
        """
        Args:
            input_dim  : size of CLIP's output vector (1024 for ViT-L/14)
            hidden_dim : size of the middle layer (512 is a good default)
            num_classes: 2 — No foul or Foul
        """
        super(FoulClassifier, self).__init__()

        # New deeper with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class SeverityClassifier(nn.Module):
    """
    Takes CLIP's CLS token and predicts the severity of the foul.

    Severity levels (from SoccerNet MVFoul):
        0 = No offence
        1 = Offence + No card
        2 = Offence + Yellow card
        3 = Offence + Red card

    Input : (batch_size, 1024) — CLS token from CLIP
    Output: (batch_size, 4)    — scores for each severity level
    """

    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=4):
        """
        Args:
            input_dim  : size of CLIP's output vector (1024 for ViT-L/14)
            hidden_dim : size of the middle layer
            num_classes: 4 severity levels
        """
        super(SeverityClassifier, self).__init__()

        # New deeper with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class XVARSClassifiers(nn.Module):
    """
    Combines both classifiers into one module.
    This mirrors X-VARS's C_foul and C_sev heads.

    Input : (batch_size, 1024) — CLS token from CLIP
    Output: 
        foul_logits: (batch_size, 2) — foul or not
        sev_logits : (batch_size, 4) — severity level
    """

    def __init__(self, input_dim=1024, hidden_dim=512):
        super(XVARSClassifiers, self).__init__()

        self.foul_head = FoulClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2
        )

        self.severity_head = SeverityClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=4
        )

    def forward(self, cls_token):
        foul_logits = self.foul_head(cls_token)
        sev_logits = self.severity_head(cls_token)

        return foul_logits, sev_logits

    def predict(self, cls_token):
        """
        Same as forward but returns the predicted class index directly.
        Used during inference (not training).

        Returns:
            foul_pred: shape (batch_size,) — 0=No foul, 1=Foul
            sev_pred : shape (batch_size,) — 0,1,2,3 severity level
        """
        foul_logits, sev_logits = self.forward(cls_token)

        # argmax picks the index with the highest score
        foul_pred = torch.argmax(foul_logits, dim=-1)
        sev_pred = torch.argmax(sev_logits, dim=-1)

        return foul_pred, sev_pred


if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('.')

    from src.models.clip_extractor import load_clip_model, extract_spatial_tokens

    print("Step 1 — Load CLIP...")
    model, preprocess = load_clip_model()

    print("Step 2 — Load one real clip...")
    frames = np.load("data/frames/test/action_0/clip_0.npy")
    print(f"  Frames shape: {frames.shape}")

    print("Step 3 — Extract CLS tokens from CLIP...")
    cls_tokens, spatial_tokens = extract_spatial_tokens(model, preprocess, frames)
    print(f"  CLS tokens shape: {cls_tokens.shape}")

    # Average CLS tokens across all 16 frames
    # to get one single video-level representation
    # (16, 1024) → (1024,) → (1, 1024) for batch dimension
    video_vector = cls_tokens.mean(dim=0).unsqueeze(0)
    print(f"  Video vector shape: {video_vector.shape}")

    print("\nStep 4 — Build classifiers...")
    classifiers = XVARSClassifiers(input_dim=1024, hidden_dim=512)
    print(f"  Foul head params    : {sum(p.numel() for p in classifiers.foul_head.parameters()):,}")
    print(f"  Severity head params: {sum(p.numel() for p in classifiers.severity_head.parameters()):,}")

    print("\nStep 5 — Run forward pass...")
    foul_logits, sev_logits = classifiers(video_vector)
    print(f"  Foul logits shape   : {foul_logits.shape}")
    print(f"  Severity logits shape: {sev_logits.shape}")

    print("\nStep 6 — Get predictions...")
    foul_pred, sev_pred = classifiers.predict(video_vector)

    foul_labels = ["No foul", "Foul"]
    sev_labels = [
        "No offence",
        "Offence - No card",
        "Offence - Yellow card",
        "Offence - Red card",
    ]

    print(f"  Foul prediction    : {foul_labels[foul_pred.item()]}")
    print(f"  Severity prediction: {sev_labels[sev_pred.item()]}")
    print()
