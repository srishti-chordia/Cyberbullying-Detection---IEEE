import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

MAX_SEQUENCE_LENGTH = 100  # use what you had during training


def load_model_and_tokenizer(model_path):
    model = load_model(model_path)
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_texts(model, tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    preds = model.predict(padded)
    labels = [
        "Cyberstalking",
        "Doxing",
        "Sexual Harassment",
        "Revenge Porn",
        "Slut Shaming",
    ]

    results = []
    for i, pred in enumerate(preds):
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        results.append(
            {
                "text": texts[i],
                "class": labels[idx],
                "is_bullying": confidence > 0.7,
                "confidence": confidence,
            }
        )
    return results
