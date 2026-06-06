import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global pooling
        return self.fc(x)


def train_transformer_audio(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # reshape to (batch, seq_len, features)
    X = X.view(X.shape[0], X.shape[1], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = AudioTransformer(input_dim=X.shape[2], num_classes=len(set(y.tolist())))
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

    return evaluate_model(y_test.numpy(), preds, "Transformer Audio")