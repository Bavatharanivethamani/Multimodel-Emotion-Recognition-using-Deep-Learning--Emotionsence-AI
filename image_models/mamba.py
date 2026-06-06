import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class VisionMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48*48*3, 256)
        self.mamba = MambaBlock(256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.mamba(x)
        return self.fc2(x)

def train_mamba(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = VisionMamba().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = loss_fn(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    return evaluate_model(y_test.numpy(), preds, "Mamba Vision")