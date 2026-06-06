import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def train_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing metrics script on {device}...")
    
    # Tiny dataset
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    full_dataset = datasets.ImageFolder('data/image', transform=transform)
    indices = np.random.choice(len(full_dataset), 50, replace=False)
    dataset = Subset(full_dataset, indices)
    
    train_size = 40
    val_size = 10
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
        'test': DataLoader(val_dataset, batch_size=8, shuffle=False)
    }
    
    model = models.mobilenet_v2(weights=None) # No weights for speed
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(1):
        model.train()
        train_y_true, train_y_pred = [], []
        train_running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
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
        
        model.eval()
        val_y_true, val_y_pred = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())
        
        val_loss = val_running_loss / len(dataloaders['test'].dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(val_y_true, val_y_pred)
        
        print(f"Epoch 1/1")
        print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"  [Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

if __name__ == '__main__':
    train_test()
