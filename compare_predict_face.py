import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import time

# === Paths ===
model_path = os.path.join("model", "face_model.keras")
class_file = os.path.join("model", "face_classes.json")
image_path = "datasets/facescan/normal/normal_400.jpg"

# === Load Model & Class Labels ===
model = tf.keras.models.load_model(model_path)
with open(class_file, "r") as f:
    class_labels = json.load(f)

# === CLAHE Enhancement ===
def enhance_with_clahe(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224)) / 255.0
    return np.expand_dims(resized, axis=0)

# === Standard Preprocessing ===
def preprocess_standard(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# === Run Prediction ===
def run_inference(input_tensor, label_map):
    preds = model.predict(input_tensor)[0]
    top_idx = preds.argmax()
    score = round(preds[top_idx] * 100, 2)
    return {
        "label": label_map[top_idx],
        "score": score
    }

# === Test Comparison ===
start = time.time()
raw_result = run_inference(preprocess_standard(image_path), class_labels)
clahe_result = run_inference(enhance_with_clahe(image_path), class_labels)
elapsed = round(time.time() - start, 3)

print("\n🧪 Prediction Comparison:")
print(f"🔹 Raw Image ➜ {raw_result['label']} ({raw_result['score']}%)")
print(f"🔸 CLAHE Image ➜ {clahe_result['label']} ({clahe_result['score']}%)")
print(f"⏱️ Total Time: {elapsed} sec")