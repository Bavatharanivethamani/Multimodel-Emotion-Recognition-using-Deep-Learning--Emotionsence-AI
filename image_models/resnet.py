import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

def train_resnet(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = loss_fn(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    return evaluate_model(y_test.numpy(), preds, "ResNet")