import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def load_text_data(max_samples=100, num_classes=5):
    # Dummy data: replace with real text and labels
    texts = [f"Sample text {i}" for i in range(max_samples)]
    y = np.random.randint(0, num_classes, max_samples).tolist()
    return texts, y

def train_bert(texts, labels, device, num_epochs=25, lr=2e-5):
    print("\n" + "="*60)
    print("🚀 BERT TEXT EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Learning Rate: {lr}")
    print("="*60)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("\n🔥 STARTING TRAINING...\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    print("🔍 STARTING EVALUATION...\n")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    print("\n" + "="*60)
    print("📊 BERT FINAL RESULTS")
    print("="*60)
    print(f"🎯 ACCURACY:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🎯 PRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 RECALL:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"🎯 F1-SCORE:  {f1:.4f}    ({f1*100:.2f}%)")
    print("="*60)
    return accuracy, precision, recall, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texts, y = load_text_data(max_samples=100, num_classes=5)
    print(f"📁 Text samples: {len(texts)}")
    print(f"📊 Classes: {len(set(y))}")
    metrics = train_bert(texts, y, device, num_epochs=100, lr=2e-5)
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
