import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import time
import random

# === Paths ===
model_path = os.path.join("model", "face_model.keras")
class_file = os.path.join("model", "face_classes.json")

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

# === Simulated Vitals Generator ===
import random

def generate_fake_vitals():
    return {
        "blood_pressure": f"{random.randint(110, 130)}/{random.randint(70, 90)}",
        "heart_rate": random.randint(60, 100),
        "hr_variability": random.choice(["Low", "Moderate", "High"]),
        "breathing_rate": random.randint(12, 20),
        "oxygen_saturation": f"{random.randint(95, 100)}%",
        "sympathetic_stress": random.choice(["Low", "Moderate", "High"]),
        "parasympathetic_activity": random.choice(["Low", "Normal", "High"]),
        "prq_quality": random.choice(["Normal", "Low", "Good"]),
        "body_temperature": f"{random.uniform(97.0, 99.5):.1f}°F",
        "stroke_risk": random.choice(["Low", "Medium", "High"]),
        "parkinsons_indicators": random.choice(["None", "Mild", "Possible"]),
        "ecg_estimation": random.choice(["Normal", "Irregular"]),
        "eeg_brainwaves": random.choice(["Alpha Dominant", "Beta Dominant", "Theta Mixed"])
    }


# === Prediction Function ===
def predict_face(img_path, threshold=0.3, use_clahe=True):
    start = time.time()

    # Preprocess
    img_array = enhance_with_clahe(img_path) if use_clahe else preprocess_standard(img_path)
    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[::-1]
    predicted_classes = [class_labels[i] for i in top_indices]

    # Map predictions
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

    # Output
    return {
        "region": "face",
        "prediction": top_prediction,
        "confidence": round(top_confidence * 100, 2),
        "confidence_level": map_confidence_level(top_confidence),
        "predicted_classes": predicted_classes[:4],
        "confidences": confidences,
        "top_predictions": list(mapped.items())[:3],
        "all_probabilities": mapped,
        "inference_time_sec": round(time.time() - start, 3),
        "face_metrics": generate_fake_vitals()
    }

# === Example Run ===
if __name__ == "__main__":
    test_image = "datasets/facescan/acne/sample.jpg"
    result = predict_face(test_image, threshold=0.3)
    print(json.dumps(result, indent=2))

