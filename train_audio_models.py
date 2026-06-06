import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from audio_models.audio_utils import load_wav, extract_features
from audio_models.cnn1d_audio import AudioCNN1D
from audio_models.lstm_audio import AudioLSTM
from audio_models.cnn_lstm_audio import AudioCNNLSTM
from audio_models.transformer_audio import AudioTransformer

class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=16000, use_features=False):
        self.samples = []
        self.labels = []
        # Standardized 7 Emotions Mapping (FER-style)
        fer_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.label_map = {cls: i for i, cls in enumerate(fer_classes)}
        
        # Only include folders that are in our standard list
        for folder in fer_classes:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_path} not found.")
                continue
                
            label_idx = self.label_map[folder]
            for fname in os.listdir(folder_path):
                if fname.endswith('.wav'):
                    self.samples.append(os.path.join(folder_path, fname))
                    self.labels.append(label_idx)
        self.target_length = target_length
        self.use_features = use_features
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        audio = load_wav(self.samples[idx], self.target_length)
        if self.use_features:
            audio = extract_features(audio)
        return torch.tensor(audio, dtype=torch.float32), self.labels[idx]

def train_and_eval(model, train_loader, test_loader, device, model_name, num_epochs=50):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f"\nTraining {model_name} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if len(xb.shape) == 2: xb = xb.unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if len(xb.shape) == 2: xb = xb.unsqueeze(1)
            preds = model(xb).argmax(1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n=== {model_name} RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    return acc, prec, rec, f1

if __name__ == "__main__":
    root = 'data/audio'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioDataset(root, target_length=16000, use_features=False)
    n = len(dataset)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*n), n-int(0.8*n)])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)
    results = {}
    # 1D CNN
    model = AudioCNN1D(input_length=16000, num_classes=len(dataset.label_map))
    results["1D CNN"] = train_and_eval(model, train_loader, test_loader, device, "1D CNN", num_epochs=25)

    # LSTM
    model = AudioLSTM(input_size=16000, hidden_size=64, num_classes=len(dataset.label_map))
    results["LSTM"] = train_and_eval(model, train_loader, test_loader, device, "LSTM", num_epochs=25)

    # # CNN+LSTM
    # model = AudioCNNLSTM(input_length=16000, num_classes=len(dataset.label_map))
    # results["CNN+LSTM"] = train_and_eval(model, train_loader, test_loader, device, "CNN+LSTM", num_epochs=25)
    
    # # Transformer
    # model = AudioTransformer(input_size=16000, num_classes=len(dataset.label_map))
    # results["Transformer"] = train_and_eval(model, train_loader, test_loader, device, "Transformer", num_epochs=25)

    # FINAL SUMMARY TABLE
    print("\n" + "="*65)
    print(f"{'FINAL AUDIO MODEL COMPARISON':^65}")
    print("="*65)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for name, metrics in results.items():
        if metrics:
            acc, prec, rec, f1 = metrics
            print(f"{name:<15} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
    print("="*65)
