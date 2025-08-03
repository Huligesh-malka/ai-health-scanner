import os
import csv

# === Path settings ===
dataset_dir = 'datasets'
labels_csv = os.path.join(dataset_dir, 'labels.csv')

# === Store deleted filenames ===
deleted_files = []

# === Read the CSV and delete images labeled 'normal' ===
with open(labels_csv, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        label = row['label'].strip().lower()
        if label == 'normal':
            image_path = os.path.join(dataset_dir, row['filename'])
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_files.append(row['filename'])

# === Output ===
print(f"\n✅ Deleted {len(deleted_files)} 'normal' images:")
for name in deleted_files:
    print(f" - {name}")
