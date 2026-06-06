# ==========================================
# 🔷 Visualization Utilities
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix


# ==========================================
# 🔷 1. Plot Metrics (Bar Graph)
# ==========================================
def plot_metrics(model_names, accuracy, precision, recall, f1):
    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5*width, precision, width, label='Precision')
    plt.bar(x + 0.5*width, recall, width, label='Recall')
    plt.bar(x + 1.5*width, f1, width, label='F1-score')

    plt.xticks(x, model_names, rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title("Model Comparison Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 2. Confusion Matrix
# ==========================================
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 3. Show Sample Images
# ==========================================
def show_sample_images(X, y, num=5):
    plt.figure(figsize=(12, 4))

    for i in range(num):
        plt.subplot(1, num, i + 1)

        # Handle grayscale or RGB
        if len(X[i].shape) == 2:
            plt.imshow(X[i], cmap='gray')
        else:
            plt.imshow(X[i].astype('uint8'))

        plt.title(f"Label: {y[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 4. Audio Waveform
# ==========================================
def plot_audio_waveform(file_path):
    audio, sr = librosa.load(file_path)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)

    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 5. MFCC Visualization
# ==========================================
def plot_mfcc(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')

    plt.colorbar()
    plt.title("MFCC Features")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 6. Accuracy Comparison Line Plot
# ==========================================
def plot_accuracy_curve(model_names, accuracy):
    plt.figure(figsize=(8, 5))
    plt.plot(model_names, accuracy, marker='o')

    plt.title("Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ==========================================
# 🔷 7. Distribution of Classes
# ==========================================
def plot_class_distribution(labels, title="Class Distribution"):
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique, counts)

    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()