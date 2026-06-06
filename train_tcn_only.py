import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Model definition (matching text_models.tcn)
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
        x = self.network(x)
        x = x.squeeze(-1)
        return self.fc(x)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def load_text_data(max_samples=1000, num_classes=5):
    # Dummy data: replace with real text and labels
    texts = [f"Sample text for TCN emotion analysis {i}" for i in range(max_samples)]
    y = np.random.randint(0, num_classes, max_samples).tolist()
    return texts, y

def train_tcn_model(texts, labels, device, num_epochs=25, batch_size=8, lr=1e-3):
    print("\n" + "="*60)
    print("🚀 TCN TEXT EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print("="*60)

    # Vectorization
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()
    
    # Preprocessing
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (batch, channels=1, features)
    y = torch.tensor(labels, dtype=torch.long)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model init
    model = TCN(input_size=1, num_classes=len(set(labels)))
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print("\n🔥 STARTING TRAINING...\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Avg Loss: {total_loss/len(train_loader):.4f}")

    print("\n🔍 STARTING EVALUATION...\n")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    
    print("\n" + "="*60)
    print("📊 TCN TEXT FINAL RESULTS")
    print("="*60)
    print(f"🎯 ACCURACY:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🎯 PRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 RECALL:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"🎯 F1-SCORE:  {f1:.4f}    ({f1*100:.2f}%)")
    print("="*60)
    return accuracy, precision, recall, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texts, y = load_text_data(max_samples=1000, num_classes=5)
    
    print(f"📁 Text samples: {len(texts)}")
    print(f"📊 Classes: {len(set(y))}")
    
    metrics = train_tcn_model(texts, y, device, num_epochs=100, batch_size=8)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("📈 Overall Performance (Decimal):")
    print(f"   • Accuracy:  {metrics[0]:.4f}")
    print(f"   • Precision: {metrics[1]:.4f}")
    print(f"   • Recall:    {metrics[2]:.4f}")
    print(f"   • F1-Score:  {metrics[3]:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
