import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model
from sklearn.feature_extraction.text import CountVectorizer

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def train_lstm_text(texts, labels):
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = LSTMModel(X.shape[2],64,len(set(labels)))
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(3):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out,y_train)
        loss.backward()
        opt.step()

    preds = model(X_test).argmax(1).detach().numpy()
    return evaluate_model(y_test.numpy(), preds, "LSTM Text")

if __name__ == '__main__':
    import numpy as np
    print("\n" + "="*60)
    print("🚀 LSTM TEXT EMOTION RECOGNITION TRAINING")
    print("="*60)
    
    # Generate dummy data for testing
    texts = [f"sample text {i}" for i in range(100)]
    labels = np.random.randint(0, 5, 100).tolist()
    
    print(f"📁 Text samples: {len(texts)}")
    print(f"📊 Classes: {len(set(labels))}")
    print("\n🔥 STARTING TRAINING AND EVALUATION...\n")
    
    train_lstm_text(texts, labels)
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)