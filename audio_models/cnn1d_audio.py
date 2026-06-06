import torch
import torch.nn as nn

class AudioCNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)
