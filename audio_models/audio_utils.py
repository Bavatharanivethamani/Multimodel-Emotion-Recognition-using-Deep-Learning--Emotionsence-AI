import os
import numpy as np
import torch
import wave

def load_wav(filepath, target_length=16000):
    with wave.open(filepath, 'rb') as wf:
        sr = wf.getframerate()
        samples = wf.readframes(wf.getnframes())
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
        audio = audio / np.max(np.abs(audio))
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio

def compute_fft(audio):
    fft = np.abs(np.fft.rfft(audio))
    return fft / np.max(fft)

def zero_crossing_rate(audio):
    return ((audio[:-1] * audio[1:]) < 0).sum() / len(audio)

def energy(audio):
    return np.sum(audio ** 2) / len(audio)

def extract_features(audio):
    fft_feat = compute_fft(audio)
    zcr = np.array([zero_crossing_rate(audio)])
    eng = np.array([energy(audio)])
    return np.concatenate([audio, fft_feat, zcr, eng])
