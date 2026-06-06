import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import librosa
import urllib.request
import soundfile as sf
import pandas as pd

# ==========================================
# 1. DATA HANDLING (Raw Waveform Only)
# ==========================================
class RawAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, duration=2.0):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.file_paths = []
        self.labels = []
        self.classes = []
        
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_dir) or len(os.listdir(self.data_dir)) == 0:
            print(f"Directory {self.data_dir} not found or empty. Generating Dummy Data...")
            self._generate_dummy_data()

        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.file_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(class_to_idx[cls_name])
                    
    def _generate_dummy_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for cls_name in ['happy', 'sad', 'angry']:
            cls_dir = os.path.join(self.data_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(10): # 10 dummy files per class
                dummy_audio = np.random.randn(self.sample_rate * 2) # 2 seconds of noise
                sf.write(os.path.join(cls_dir, f'dummy_{i}.wav'), dummy_audio, self.sample_rate)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load raw audio (NO mfcc or spectrograms)
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        
        # Preprocessing: Normalize audio signal
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        # Pad or truncate to fixed length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')

        # Convert to Tensor format: Shape (1, sequence_length) -> 1 Channel
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        return audio_tensor, label


# ==========================================
# 2. MODELS IMPLEMENTATION
# ==========================================

# A) 1D CNN Model
class RawAudioCNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16, stride=4, padding=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Flattens over time
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(-1) # Remove time dimension
        return self.fc_layers(x)


# B) LSTM Model
class RawAudioLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # To avoid massive memory issues from feeding 32,000 raw timesteps statically,
        # we reshape to pseudo-frames (e.g. sequence of small chunks).
        # We will dynamically chunk the audio inside forward pass: 
        # e.g., 32000 length -> 250 sequence steps of 128 features each.
        self.hidden_dim = 128
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x is (batch, 1, time)
        batch_size, _, seq_len = x.shape
        chunk_size = 128
        num_chunks = seq_len // chunk_size
        
        # Reshape into sequence of small raw chunks -> (batch, num_chunks, chunk_size)
        x = x.view(batch_size, num_chunks, chunk_size)
        
        _, (hn, _) = self.lstm(x)
        # Take the hidden state of the top layer
        return self.fc(hn[-1])


# C) CNN-LSTM Hybrid Model
class RawAudioCNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # After CNN, time dimension is reduced, channel becomes "features"
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x) # -> (batch, features=64, reduced_time)
        x = x.permute(0, 2, 1)  # -> (batch, reduced_time, features=64), required for LSTM
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# D) Transformer-based Model
class RawAudioTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use a preliminary linear embedding to map raw chunks to d_model
        self.chunk_size = 256
        self.d_model = 64
        self.embedding = nn.Linear(self.chunk_size, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x is (batch, 1, time)
        batch_size, _, seq_len = x.shape
        num_chunks = seq_len // self.chunk_size
        
        x = x.view(batch_size, num_chunks, self.chunk_size)
        x = self.embedding(x) # -> (batch, num_chunks, d_model)
        
        x = self.transformer_encoder(x) # -> (batch, num_chunks, d_model)
        # Global average pooling over time chunks
        x = x.mean(dim=1) 
        return self.fc(x)


# ==========================================
# 3. & 4. TRAINING & EVALUATION PIPELINE
# ==========================================

def evaluate_metrics(y_true, y_pred, model_name):
    # Compute the following metrics: Accuracy, Precision, Recall, F1-score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n[{model_name}] Evaluation Results:")
    print(f" -> Accuracy:  {acc:.4f}")
    print(f" -> Precision: {prec:.4f}")
    print(f" -> Recall:    {rec:.4f}")
    print(f" -> F1-score:  {f1:.4f}")
    
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

def train_and_test(model, model_name, train_loader, test_loader, device, epochs=3):
    print(f"\n" + "="*50)
    print(f"🧠 Training {model_name}...")
    print("="*50)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Step
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Eval Step
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return evaluate_metrics(all_labels, all_preds, model_name)


# ==========================================
# 5. & 6. COMPARISON & FINAL OUTPUT
# ==========================================

def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device}")
    
    dataset_dir = "data/raw_audio"
    target_duration = 2.0 # Fixed length 2.0s
    
    print("Loading Raw Waveform Dataset...")
    dataset = RawAudioDataset(dataset_dir, duration=target_duration)
    num_classes = len(dataset.classes)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize Models
    models = [
        (RawAudioCNN1D(num_classes), "1D CNN Model"),
        (RawAudioLSTM(num_classes), "LSTM Model"),
        (RawAudioCNNLSTM(num_classes), "CNN-LSTM Hybrid Model"),
        (RawAudioTransformer(num_classes), "Transformer-based Model")
    ]
    
    results = []
    
    # Train each model for multiple epochs and collect metrics
    for model, name in models:
        metrics = train_and_test(model, name, train_loader, test_loader, device, epochs=3)
        results.append(metrics)
        
    print("\n" + "="*60)
    print("🏆 FINAL COMPARISON TABLE")
    print("="*60)
    df = pd.DataFrame(results)
    
    # Format and print the table clearly
    print(df.to_string(index=False))
    print("="*60)
    
    # Identify the best model based on highest F1-score & balanced Precision/Recall
    
    # Optional logic: we check F1 strictly, although we can average F1+Prec+Rec to show balance.
    df['Balance_Score'] = (df['F1'] * 2 + df['Precision'] + df['Recall']) / 4
    best_row = df.loc[df['Balance_Score'].idxmax()]
    best_model_name = best_row['Model']
    
    print("\n⭐ CONCLUSION ⭐")
    print(f"Best Model: {best_model_name}")
    print(f"(Chosen based on F1-Score {best_row['F1']:.4f} and optimally balanced precision/recall metric)")

if __name__ == '__main__':
    run_pipeline()
