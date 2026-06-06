import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from audio_models.transformer_audio import AudioTransformer

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def load_audio_data(max_samples=1000, num_classes=7):
    # Dummy data: replace with real audio feature loading
    X = np.random.rand(max_samples, 128, 40).astype(np.float32)  # (batch, time, features)
    y = np.random.randint(0, num_classes, max_samples).astype(np.int64)
    return X, y

def train_transformer_audio(X, y, device, num_epochs=25, lr=1e-4):
    print("\n" + "="*60)
    print("🚀 TRANSFORMER AUDIO EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Learning Rate: {lr}")
    print("="*60)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    X = X.view(X.shape[0], X.shape[1], -1)
    dataset = TensorDataset(X, y)
    val_split = 0.2
    val_count = int(len(dataset) * val_split)
    train_count = len(dataset) - val_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = AudioTransformer(input_dim=X.shape[2], num_classes=len(set(y.tolist())))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print("\n🔥 STARTING TRAINING...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            batch_count += inputs.size(0)
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1:2d} - Loss: {loss.item():.4f}")
        epoch_loss = running_loss / batch_count
        print(f"  ✅ Epoch {epoch+1} Completed - Average Loss: {epoch_loss:.4f}\n")
    print("🔍 STARTING EVALUATION...\n")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if (batch_idx + 1) % 2 == 0:
                print(f"  Evaluating batch {batch_idx+1}...")
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    print("\n" + "="*60)
    print("📊 TRANSFORMER AUDIO FINAL RESULTS")
    print("="*60)
    print(f"🎯 ACCURACY:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🎯 PRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 RECALL:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"🎯 F1-SCORE:  {f1:.4f}    ({f1*100:.2f}%)")
    print("="*60)
    return accuracy, precision, recall, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = load_audio_data(max_samples=1000, num_classes=7)
    print(f"📁 Audio samples: {X.shape[0]}")
    print(f"📊 Classes: {len(set(y.tolist()))}")
    metrics = train_transformer_audio(X, y, device, num_epochs=25, lr=1e-4)
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("📈 Show these metrics to your mentor:")
    print(f"   • Accuracy:  {metrics[0]:.4f}")
    print(f"   • Precision: {metrics[1]:.4f}")
    print(f"   • Recall:    {metrics[2]:.4f}")
    print(f"   • F1-Score:  {metrics[3]:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
