import os
import csv

# Path where images and labels.csv are located
folder_path = r"C:\Users\hulig\health-scanner-app\backend\datasets"
labels_csv = os.path.join(folder_path, "labels.csv")

# List all non_vigilant_*.jpg images
non_vigilant_files = [f for f in os.listdir(folder_path) if f.startswith("non_vigilant_") and f.endswith(".jpg")]

# Append labels to CSV
with open(labels_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    for filename in sorted(non_vigilant_files):
        writer.writerow([filename, "non_vigilant"])

print("✅ non_vigilant labels appended to labels.csv successfully!")
