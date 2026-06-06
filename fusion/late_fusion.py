import numpy as np
from utils.metrics import evaluate_model

# ---------------------------------------------------
# 🔷 1. Majority Voting Fusion
# ---------------------------------------------------
def majority_voting(image_pred, audio_pred, text_pred):
    final_pred = []

    for i in range(len(image_pred)):
        votes = [image_pred[i], audio_pred[i], text_pred[i]]

        # Pick most frequent label
        final = max(set(votes), key=votes.count)
        final_pred.append(final)

    return np.array(final_pred)


# ---------------------------------------------------
# 🔷 2. Probability Averaging Fusion (BEST METHOD)
# ---------------------------------------------------
def probability_fusion(image_probs, audio_probs, text_probs):
    """
    Each input shape: (num_samples, num_classes)
    """

    # Ensure same shape
    min_len = min(len(image_probs), len(audio_probs), len(text_probs))

    image_probs = image_probs[:min_len]
    audio_probs = audio_probs[:min_len]
    text_probs = text_probs[:min_len]

    # Average probabilities
    avg_probs = (image_probs + audio_probs + text_probs) / 3

    final_pred = np.argmax(avg_probs, axis=1)

    return final_pred


# ---------------------------------------------------
# 🔷 3. Weighted Fusion (ADVANCED)
# ---------------------------------------------------
def weighted_fusion(image_probs, audio_probs, text_probs,
                    w_img=0.4, w_audio=0.3, w_text=0.3):

    min_len = min(len(image_probs), len(audio_probs), len(text_probs))

    image_probs = image_probs[:min_len]
    audio_probs = audio_probs[:min_len]
    text_probs = text_probs[:min_len]

    fused = (w_img * image_probs +
             w_audio * audio_probs +
             w_text * text_probs)

    final_pred = np.argmax(fused, axis=1)

    return final_pred


# ---------------------------------------------------
# 🔷 4. Evaluate Fusion
# ---------------------------------------------------
def evaluate_fusion(y_true, y_pred, method="Fusion Model"):
    return evaluate_model(y_true, y_pred, method)