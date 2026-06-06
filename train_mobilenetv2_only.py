import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader
import numpy as np


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1


def train_mobilenetv2(dataloaders, device, num_epochs=25, lr=1e-4):
    print("\n" + "="*60)
    print("MOBILENETV2 EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Batch Size: 8")
    print("="*60)

    # Load MobileNetV2 with pretrained weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nSTARTING TRAINING...\n")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_y_true = []
        train_y_pred = []
        train_running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(preds.cpu().numpy())

        train_loss = train_running_loss / len(dataloaders['train'].dataset)
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(train_y_true, train_y_pred)

        # --- VALIDATION PHASE ---
        model.eval()
        val_y_true = []
        val_y_pred = []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(dataloaders['test'].dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(val_y_true, val_y_pred)

        # --- OUTPUT METRICS ---
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"  [Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print("-" * 30)

    print("\n" + "="*60)
    print("FINAL MOBILENETV2 RESULTS")
    print("="*60)
    print(f"ACCURACY:  {val_acc:.4f}")
    print(f"PRECISION: {val_prec:.4f}")
    print(f"RECALL:    {val_rec:.4f}")
    print(f"F1-SCORE:  {val_f1:.4f}")
    print("="*60)

    # Save model
    torch.save(model.state_dict(), 'best_mobilenetv2_emotion.pth')
    print("Model saved as 'best_mobilenetv2_emotion.pth'")

    return val_acc, val_prec, val_rec, val_f1


def build_dataloaders(root_dir, image_size=224, batch_size=8, val_split=0.2, max_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root_dir, transform=transform)

    # Limit dataset for faster training if specified (highly recommended for CPU)
    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    total = len(dataset)
    val_count = int(total * val_split)
    train_count = total - val_count

    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 to minimize heat/overhead
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }, train_count, val_count


def main():
    root_dir = 'data/image'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Using a subset of 1000 samples to prevent overheating on CPU training
    dataloaders, train_size, val_size = build_dataloaders(
        root_dir,
        image_size=224,
        batch_size=8,
        val_split=0.2,
        max_samples=1000
    )

    print(f"Dataset: {root_dir}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    metrics = train_mobilenetv2(dataloaders, device, num_epochs=25, lr=1e-4)

    print("\nTRAINING COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    main()
