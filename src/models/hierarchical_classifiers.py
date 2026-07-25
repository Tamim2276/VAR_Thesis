import torch
import torch.nn as nn

class FoulClassifier(nn.Module):
    """
    Model 1: Binary Classifier (No Foul vs Foul)
    Output: 2 logits
    """
    def __init__(self, input_dim=1152, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

class SeverityClassifier(nn.Module):
    """
    Model 2: 3-Class Classifier (No Card, Yellow, Red)
    Output: 3 logits
    """
    def __init__(self, input_dim=1152, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 3) # Only 3 classes now!
        )

    def forward(self, x):
        return self.net(x)