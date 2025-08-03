import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# === Enable GPU if available ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU is available and memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU memory setup failed:", e)
else:
    print("⚠️ GPU not available. Using CPU.")

# === Paths ===
dataset_dir = "datasets"
csv_path = os.path.join(dataset_dir, "labels.csv")
model_path = "model/health_model.keras"

# === Load and preprocess CSV ===
df = pd.read_csv(csv_path)

# Remove rows with missing image files
df = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(dataset_dir, x)))]

# Convert labels to strings
df['label'] = df['label'].astype(str)

# Get all unique labels (comma-separated multi-labels)
all_labels = sorted(set(label.strip() for sublist in df['label'].str.split(',') for label in sublist))

# Remove "normal" if present
if "normal" in all_labels:
    all_labels.remove("normal")

# One-hot encode labels for multi-label classification
for label in all_labels:
    df[label] = df['label'].apply(lambda x: int(label in x.split(',')))

# === Train-validation split ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# === Image settings ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1  # Train 1 photo per step

# === Image generators ===
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=dataset_dir,
    x_col='filename',
    y_col=all_labels,
    target_size=IMAGE_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=dataset_dir,
    x_col='filename',
    y_col=all_labels,
    target_size=IMAGE_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# === Build Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(len(all_labels), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# === Compile ===
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train ===
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen)
)

# === Save Model ===
os.makedirs("model", exist_ok=True)
model.save(model_path)
print(f"✅ Model trained and saved to {model_path}")
