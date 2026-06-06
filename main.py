import numpy as np

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
    print("\n🚀 PROJECT STARTED\n")

    # ============================
    # 🔷 DUMMY DATA (REPLACE LATER)
    # ============================
    X_img = np.random.rand(100, 48, 48, 3)
    y_img = np.random.randint(0, 7, 100)

    X_audio = np.random.rand(100, 128, 128)
    y_audio = np.random.randint(0, 7, 100)

    texts = ["I am happy", "I am sad"] * 50
    y_text = np.random.randint(0, 5, 100)

    # ============================
    # 🔷 IMAGE MODELS
    # ============================
    print("\n=========== IMAGE MODELS ===========")

    train_vgg16(X_img, y_img)
    train_resnet(X_img, y_img)
    train_efficientnet(X_img, y_img)
    train_mobilenetv2(X_img, y_img)
    train_mamba(X_img, y_img)
    train_custom_cnn(X_img, y_img)

    # ============================
    # 🔷 AUDIO MODELS
    # ============================
    print("\n=========== AUDIO MODELS ===========")

    train_audio_cnn(X_audio, y_audio)
    train_lstm_audio(X_audio, y_audio)
    train_cnn_lstm(X_audio, y_audio)
    train_transformer_audio(X_audio, y_audio)

    # ============================
    # 🔷 TEXT MODELS
    # ============================
    print("\n=========== TEXT MODELS ===========")

    train_bert(texts, y_text)
    train_lstm_text(texts, y_text)
    train_tcn(texts, y_text)

    print("\n✅ ALL MODELS COMPLETED")


if __name__ == "__main__":
    main()