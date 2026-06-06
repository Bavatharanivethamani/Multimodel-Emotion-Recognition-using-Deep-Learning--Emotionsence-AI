import torch
import torch.nn as nn

class AudioCNNLSTM(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU()
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
