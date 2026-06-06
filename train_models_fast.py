import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1


def train_model(model, dataloaders, device, model_name="Model", num_epochs=2, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    for epoch in range(num_epochs):
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
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} Completed - Avg Loss: {epoch_loss:.4f}\n")

    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
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
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluating batch {batch_idx+1}...")

    metrics = evaluate(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall:    {metrics[2]:.4f}")
    print(f"F1-Score:  {metrics[3]:.4f}")
    print(f"{'='*50}\n")
    
    return metrics


def build_dataloaders(root_dir, image_size=224, batch_size=32, val_split=0.2, max_samples=None):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root_dir, transform=transform)
    
    # Limit dataset for faster training if specified
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
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }, train_count, val_count


def main():
    root_dir = 'data/image'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🚀 EMOTION RECOGNITION MODEL TRAINING")
    print(f"Device: {device}")
    print(f"Using CPU: {'Yes' if device.type == 'cpu' else 'No'}\n")

    # Build dataloaders with limited samples for faster training on CPU
    dataloaders, train_size, val_size = build_dataloaders(
        root_dir, 
        image_size=224, 
        batch_size=32, 
        val_split=0.2,
        max_samples=2000  # Limit to 2000 samples for faster training
    )
    
    print(f'Dataset Loaded:')
    print(f'  Training samples: {train_size}')
    print(f'  Validation samples: {val_size}')
    print(f'  Total: {train_size + val_size}\n')

    results = {}

    # Train ResNet18
    print("\n🔴 ResNet18 Training Starting...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 7)
    resnet = resnet.to(device)
    resnet_metrics = train_model(resnet, dataloaders, device, "ResNet18", num_epochs=2, lr=1e-4)
    results['ResNet18'] = resnet_metrics

    # Train EfficientNet-B0
    print("\n🟢 EfficientNet-B0 Training Starting...")
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 7)
    efficientnet = efficientnet.to(device)
    efficientnet_metrics = train_model(efficientnet, dataloaders, device, "EfficientNet-B0", num_epochs=2, lr=1e-4)
    results['EfficientNet-B0'] = efficientnet_metrics

    # Summary
    print(f"\n{'='*60}")
    print(f"{'FINAL COMPARISON':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics[0]:<12.4f} {metrics[1]:<12.4f} {metrics[2]:<12.4f} {metrics[3]:<12.4f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
