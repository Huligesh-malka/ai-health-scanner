import os
import csv
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# === Paths ===
dataset_dir = "datasets/eye"
model_path = os.path.join("model", "eye_model.keras")
class_file = os.path.join("model", "eye_classes.json")
output_csv = "eye_predictions.csv"

# === Load Model ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# === Load Class Labels ===
if not os.path.exists(class_file):
    raise FileNotFoundError(f"❌ Class file not found: {class_file}")
with open(class_file, "r") as f:
    class_labels = json.load(f)

# === Open CSV for writing ===
with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "true_label", "predicted_label", "confidence(%)", "correct"])

    # === Loop through subfolders and images ===
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                preds = model.predict(img_array, verbose=0)[0]
                top_index = np.argmax(preds)
                predicted_label = class_labels[top_index]
                confidence = preds[top_index] * 100

                # True label from folder name
                true_label = folder
                correct = "yes" if predicted_label == true_label else "no"

                # Write row
                writer.writerow([img_path, true_label, predicted_label, f"{confidence:.2f}", correct])
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

print(f"✅ Prediction CSV saved to {output_csv}")
