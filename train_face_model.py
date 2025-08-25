# === Imports ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import cv2

# === Utilities ===
def apply_clahe(image_path):
    img = cv2.imread(image_path, 0)
    clahe = cv2.createCLAHE()
    enhanced_img = clahe.apply(img)
    return enhanced_img

def filter_predictions(predictions, threshold=10):
    return {label: conf for label, conf in predictions.items() if conf >= threshold}

def map_confidence_levels(confidence):
    if confidence >= 70:
        return "High confidence"
    elif confidence >= 30:
        return "Moderate confidence"
    elif confidence >= 10:
        return "Low confidence"
    else:
        return "Very low confidence"

# === Paths ===
dataset_dir = "datasets/facescan"
model_dir = "model"

os.makedirs(model_dir, exist_ok=True)

# === Load Classes ===
eye_classes = sorted(os.listdir(dataset_dir))
print(f"📂 Found classes: {eye_classes}")

# === Image Data Generator ===
batch_size = 16
img_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# === Compute Class Weights ===
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights_array))

# === Build Model ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(len(eye_classes), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Callbacks ===
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", verbose=1)

# === Train Model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# === Fine-Tuning Top Layers ===
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# === Save Final Model ===
model.save(model_path)
print(f"✅ Model saved to {model_path}")

# === Plot Training Curves ===
def plot_history(hist):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["accuracy"], label="Train Acc")
    plt.plot(hist.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.show()

plot_history(history)

# === Evaluate Model ===
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=eye_classes))

print("\n🔁 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# === Confidence Display Example ===
for i, sample_conf in enumerate(y_pred):
    pred_dict = {eye_classes[j]: round(conf * 100, 2) for j, conf in enumerate(sample_conf)}
    filtered = filter_predictions(pred_dict)
    print(f"\nImage {i+1}:")
    for label, score in filtered.items():

        print(f"🔹 {label}: {map_confidence_levels(score)} ({score}%)")

