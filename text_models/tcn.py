import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from utils.metrics import evaluate_model

class TCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x is originally (batch_size, 1, sequence_length)
        # nn.Conv1d expects (batch_size, channels, sequence_length)
        x = self.network(x)
        x = x.squeeze(-1)
        return self.fc(x)


def train_tcn(texts, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert text → vectors
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, seq, feature)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = TCN(input_size=1, num_classes=len(set(labels))).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    for epoch in range(3):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train.to(device))
        loss = loss_fn(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    return evaluate_model(y_test.numpy(), preds, "TCN")

if __name__ == '__main__':
    import numpy as np
    print("\n" + "="*60)
    print("🚀 TCN TEXT EMOTION RECOGNITION TRAINING")
    print("="*60)
    
    # Generate dummy data for testing
    texts = [f"sample text {i}" for i in range(100)]
    labels = np.random.randint(0, 5, 100).tolist()
    
    print(f"📁 Text samples: {len(texts)}")
    print(f"📊 Classes: {len(set(labels))}")
    print("\n🔥 STARTING TRAINING AND EVALUATION...\n")
    
    train_tcn(texts, labels)
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)