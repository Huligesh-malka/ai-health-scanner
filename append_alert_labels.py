import os
import csv

# Folder where alert images are stored
alert_folder = "C:/Users/hulig/health-scanner-app/backend/datasets"

# Path to existing labels.csv
labels_csv_path = os.path.join(alert_folder, "labels.csv")

# Find all alert_*.jpg files
alert_files = [f for f in os.listdir(alert_folder) if f.startswith("alert_") and f.endswith(".jpg")]

# Append mode
with open(labels_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    for filename in sorted(alert_files):
        writer.writerow([filename, 'alert'])

print("✅ Alert labels appended to labels.csv successfully!")
    