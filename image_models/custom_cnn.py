import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*10*10,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

def train_custom_cnn(X,y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    y = torch.tensor(y, dtype=torch.long)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = CNN().to(device)
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(3):
        opt.zero_grad()
        out = model(X_train.to(device))
        loss = loss_fn(out,y_train.to(device))
        loss.backward()
        opt.step()

    preds = model(X_test.to(device)).argmax(1).cpu().numpy()
    return evaluate_model(y_test.numpy(), preds, "Custom CNN")