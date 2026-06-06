import numpy as np
import torch
import sys
from io import StringIO

# Image models
from image_models.vgg16 import train_vgg16
from image_models.resnet import train_resnet
from image_models.efficientnet import train_efficientnet
from image_models.mobilenetv2 import train_mobilenetv2
from image_models.mamba import train_mamba
from image_models.custom_cnn import train_custom_cnn

# Audio models
from audio_models.cnn_audio import train_audio_cnn
from audio_models.lstm_audio import train_lstm_audio
from audio_models.cnn_lstm import train_cnn_lstm
from audio_models.transformer_audio import train_transformer_audio

# Text models
from text_models.bert import train_bert
from text_models.lstm_text import train_lstm_text
from text_models.tcn import train_tcn

def main():
    print("\n>>> QUICK TRAINING - ALL MODELS\n")
    
    # ============================
    # DUMMY DATA
    # ============================
    X_img = np.random.rand(100, 48, 48, 3)
    y_img = np.random.randint(0, 7, 100)

    X_audio = np.random.rand(100, 128, 128)
    y_audio = np.random.randint(0, 7, 100)

    texts = ["I am happy", "I am sad"] * 50
    y_text = np.random.randint(0, 5, 100)

    results = {}

    # ============================
    # IMAGE MODELS
    # ============================
    print("\n=========== IMAGE MODELS ===========\n")
    
    try:
        print("Training VGG16...")
        result = train_vgg16(X_img, y_img)
        results['VGG16'] = result
    except Exception as e:
        print(f"VGG16 Error: {e}")
    
    try:
        print("\nTraining ResNet...")
        result = train_resnet(X_img, y_img)
        results['ResNet'] = result
    except Exception as e:
        print(f"ResNet Error: {e}")

    try:
        print("\nTraining EfficientNet...")
        result = train_efficientnet(X_img, y_img)
        results['EfficientNet'] = result
    except Exception as e:
        print(f"EfficientNet Error: {e}")

    try:
        print("\nTraining MobileNetV2...")
        result = train_mobilenetv2(X_img, y_img)
        results['MobileNetV2'] = result
    except Exception as e:
        print(f"MobileNetV2 Error: {e}")

    try:
        print("\nTraining Mamba...")
        result = train_mamba(X_img, y_img)
        results['Mamba'] = result
    except Exception as e:
        print(f"Mamba Error: {e}")

    try:
        print("\nTraining Custom CNN...")
        result = train_custom_cnn(X_img, y_img)
        results['CustomCNN'] = result
    except Exception as e:
        print(f"CustomCNN Error: {e}")

    # ============================
    # AUDIO MODELS
    # ============================
    print("\n=========== AUDIO MODELS ===========\n")

    try:
        print("Training Audio CNN...")
        result = train_audio_cnn(X_audio, y_audio)
        results['AudioCNN'] = result
    except Exception as e:
        print(f"AudioCNN Error: {e}")

    try:
        print("\nTraining LSTM Audio...")
        result = train_lstm_audio(X_audio, y_audio)
        results['LSTMAudio'] = result
    except Exception as e:
        print(f"LSTMAudio Error: {e}")

    try:
        print("\nTraining CNN-LSTM...")
        result = train_cnn_lstm(X_audio, y_audio)
        results['CNNLSTM'] = result
    except Exception as e:
        print(f"CNNLSTM Error: {e}")

    try:
        print("\nTraining Transformer Audio...")
        result = train_transformer_audio(X_audio, y_audio)
        results['TransformerAudio'] = result
    except Exception as e:
        print(f"TransformerAudio Error: {e}")

    # ============================
    # TEXT MODELS
    # ============================
    print("\n=========== TEXT MODELS ===========\n")

    try:
        print("Training BERT...")
        result = train_bert(texts, y_text)
        results['BERT'] = result
    except Exception as e:
        print(f"BERT Error: {e}")

    try:
        print("\nTraining LSTM Text...")
        result = train_lstm_text(texts, y_text)
        results['LSTMText'] = result
    except Exception as e:
        print(f"LSTMText Error: {e}")

    try:
        print("\nTraining TCN...")
        result = train_tcn(texts, y_text)
        results['TCN'] = result
    except Exception as e:
        print(f"TCN Error: {e}")

    # ============================
    # SUMMARY
    # ============================
    print("\n" + "="*70)
    print(">>> PERFORMANCE METRICS SUMMARY")
    print("="*70 + "\n")
    
    summary_data = []
    for model_name, metrics in results.items():
        if metrics:
            accuracy, precision, recall, f1 = metrics
            summary_data.append((model_name, accuracy, precision, recall, f1))
            print(f"{model_name:20} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    
    print("\n" + "="*70)
    if summary_data:
        best_model = max(summary_data, key=lambda x: x[1])
        print(f"\n>>> Best Model (by Accuracy): {best_model[0]} with {best_model[1]:.4f} accuracy\n")
    
    print("[COMPLETED] ALL MODELS COMPLETED")

if __name__ == "__main__":
    main()
