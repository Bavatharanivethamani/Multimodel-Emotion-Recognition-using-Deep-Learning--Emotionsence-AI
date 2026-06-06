import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1


def train_model(model, dataloaders, device, num_epochs=2, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return evaluate(y_true, y_pred)


def build_dataloaders(root_dir, image_size=224, batch_size=64, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root_dir, transform=transform)
    total = len(dataset)
    val_count = int(total * val_split)
    train_count = total - val_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42))

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }


def main():
    root_dir = 'data/image'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataloaders = build_dataloaders(root_dir, image_size=224, batch_size=64, val_split=0.2)
    print('Dataset sizes - train:', len(dataloaders['train'].dataset), 'test:', len(dataloaders['test'].dataset))

    print('\n=== Training ResNet18 ===')
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 7)
    resnet = resnet.to(device)
    resnet_metrics = train_model(resnet, dataloaders, device, num_epochs=2, lr=1e-4)
    print('\nResNet18 metrics: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1={:.4f}'.format(*resnet_metrics))

    print('\n=== Training EfficientNet-B0 ===')
    efficientnet = models.efficientnet_b0(pretrained=True)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 7)
    efficientnet = efficientnet.to(device)
    efficientnet_metrics = train_model(efficientnet, dataloaders, device, num_epochs=2, lr=1e-4)
    print('\nEfficientNet-B0 metrics: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1={:.4f}'.format(*efficientnet_metrics))

    print('\n=== Summary ===')
    print('ResNet18 | Acc={:.4f} | Prec={:.4f} | Rec={:.4f} | F1={:.4f}'.format(*resnet_metrics))
    print('EfficientNet-B0 | Acc={:.4f} | Prec={:.4f} | Rec={:.4f} | F1={:.4f}'.format(*efficientnet_metrics))


if __name__ == '__main__':
    main()
