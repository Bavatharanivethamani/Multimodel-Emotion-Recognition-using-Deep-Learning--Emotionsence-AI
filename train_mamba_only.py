import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np

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

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def build_dataloaders(root_dir, image_size=48, batch_size=32, val_split=0.2, max_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = Subset(dataset, indices)
    total = len(dataset)
    val_count = int(total * val_split)
    train_count = total - val_count
    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42)
    )
    return {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0),
        'test': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    }, train_count, val_count

def train_mamba(dataloaders, device, num_epochs=25, lr=1e-4):
    print("\n" + "="*60)
    print("🚀 MAMBA EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Learning Rate: {lr}")
    print("="*60)
    model = VisionMamba().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("\n🔥 STARTING TRAINING...\n")
    for epoch in range(num_epochs):
        print(f"📚 EPOCH {epoch+1}/{num_epochs}")
        print("-" * 30)
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            batch_count += inputs.size(0)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:2d} - Loss: {loss.item():.4f}")
        epoch_loss = running_loss / batch_count
        print(f"  ✅ Epoch {epoch+1} Completed - Average Loss: {epoch_loss:.4f}\n")
    print("🔍 STARTING EVALUATION...\n")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if (batch_idx + 1) % 5 == 0:
                print(f"  Evaluating batch {batch_idx+1}...")
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    print("\n" + "="*60)
    print("📊 MAMBA FINAL RESULTS")
    print("="*60)
    print(f"🎯 ACCURACY:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🎯 PRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 RECALL:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"🎯 F1-SCORE:  {f1:.4f}    ({f1*100:.2f}%)")
    print("="*60)
    return accuracy, precision, recall, f1

def main():
    root_dir = 'data/image'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders, train_size, val_size = build_dataloaders(
        root_dir,
        image_size=48,
        batch_size=32,
        val_split=0.2,
        max_samples=1000
    )
    print(f"📁 Dataset: {root_dir}")
    print(f"📊 Training samples: {train_size}")
    print(f"📊 Validation samples: {val_size}")
    print(f"📊 Total samples: {train_size + val_size}")
    metrics = train_mamba(dataloaders, device, num_epochs=25, lr=1e-4)
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
