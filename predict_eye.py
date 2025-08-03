import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import time

# === Paths ===
model_path = os.path.join("model", "eye_model.keras")
class_file = os.path.join("model", "eye_classes.json")

# === Load Model & Classes ===
model = tf.keras.models.load_model(model_path)
with open(class_file, "r") as f:
    class_labels = json.load(f)

# === CLAHE Preprocessing ===
def enhance_with_clahe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(enhanced_rgb, (224, 224)) / 255.0
    return np.expand_dims(resized, axis=0)

# === Standard Preprocessing ===
def preprocess_standard(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# === Confidence Level Mapping ===
def map_confidence_level(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.3:
        return "Medium"
    elif score >= 0.1:
        return "Low"
    else:
        return "Very Low"

# === Confidence Filtering ===
def filter_predictions(preds, labels, threshold=0.3):
    filtered = {
        labels[i]: float(preds[i])
        for i in preds.argsort()[::-1]
        if preds[i] >= threshold
    }
    if not filtered:
        top_i = preds.argmax()
        filtered = {labels[top_i]: float(preds[top_i])}
    return filtered

# === Prediction Function ===
def predict_eye(img_path, threshold=0.3, use_clahe=True):  # ✅ Renamed here
    start = time.time()
    img_array = enhance_with_clahe(img_path) if use_clahe else preprocess_standard(img_path)
    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[::-1]
    predicted_classes = [class_labels[i] for i in top_indices]

    filtered = filter_predictions(preds, class_labels, threshold)
    top_prediction = list(filtered.keys())[0]
    top_confidence = filtered[top_prediction]

    confidences = {
        label: round(score * 100, 2)
        for label, score in filtered.items()
    }
    mapped = {
        label: {
            "score": round(score * 100, 2),
            "level": map_confidence_level(score)
        }
        for label, score in filtered.items()
    }

    return {
        "region": "eye",
        "prediction": top_prediction,
        "confidence": round(top_confidence * 100, 2),
        "confidence_level": map_confidence_level(top_confidence),
        "predicted_classes": predicted_classes[:4],
        "confidences": confidences,
        "top_predictions": list(mapped.items())[:3],
        "all_probabilities": mapped,
        "inference_time_sec": round(time.time() - start, 3)
    }

# === Example Usage ===
if __name__ == "__main__":
    test_image = "datasets/eye/normal/normal eyes_1.jpg"  # Update with a valid eye image path
    result = predict_eye(test_image, threshold=0.3)  # ✅ Updated usage
    print(json.dumps(result, indent=2))