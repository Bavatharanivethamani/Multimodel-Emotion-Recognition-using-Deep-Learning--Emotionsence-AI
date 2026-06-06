import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def train_lstm_audio(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reshape (batch, time, features)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X = X.view(X.shape[0], X.shape[1], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = AudioLSTM(input_size=X.shape[2], hidden_size=64, num_classes=len(set(y.tolist())))
    model.to(device)

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

    return evaluate_model(y_test.numpy(), preds, "LSTM Audio")