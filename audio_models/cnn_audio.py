import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*63*63,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def main():
    # Dummy data for demonstration; replace with your real data loading
    X = np.random.rand(100, 128, 128)
    y = np.random.randint(0, 7, 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = AudioCNN(len(set(y.tolist()))).to(device)
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        opt.zero_grad()
        out = model(X_train.to(device))
        loss = loss_fn(out, y_train.to(device))
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    preds = model(X_test.to(device)).argmax(1).cpu().numpy()
    acc, prec, rec, f1 = evaluate(y_test.numpy(), preds)
    print("\n=== AUDIO CNN FINAL RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

if __name__ == "__main__":
    main()