import pandas as pd
import numpy as np
import cv2

def load_fer2013(csv_path):
    data = pd.read_csv(csv_path)

    pixels = data['pixels'].tolist()
    images = []

    for pixel_seq in pixels:
        img = np.array(pixel_seq.split(), dtype='float32')
        img = img.reshape(48, 48)

        # Convert grayscale → 3 channel
        img = np.stack([img]*3, axis=-1)
        images.append(img)

    X = np.array(images)
    y = data['emotion'].values

    return X, y