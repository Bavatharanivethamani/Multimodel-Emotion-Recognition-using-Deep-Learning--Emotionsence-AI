import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=32*63, hidden_size=64, batch_first=True)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, 32, H, W)

        b, c, h, w = x.size()
        x = x.view(b, h, c*w)  # reshape for LSTM

        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def train_cnn_lstm(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = CNN_LSTM(len(set(y.tolist()))).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train.to(device))
        loss = loss_fn(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    return evaluate_model(y_test.numpy(), preds, "CNN + LSTM Audio")