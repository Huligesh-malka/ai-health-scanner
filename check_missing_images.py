import os
import pandas as pd

# === Set dataset directory ===
image_dir = "datasets"
csv_path = os.path.join(image_dir, "labels.csv")

# === Check image files in dataset folder ===
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print("✅ Total image files in folder:", len(image_files))

# === Check labels.csv file ===
if not os.path.exists(csv_path):
    print("❌ Error: labels.csv not found!")
else:
    df = pd.read_csv(csv_path)
    print("✅ Total rows in labels.csv:", len(df))

    # === Check how many CSV filenames match real image files ===
    matched = df['filename'].isin(image_files)
    valid_rows = df[matched]
    print("✅ Matched image entries in CSV:", len(valid_rows))

    # === Check if any mismatched filenames exist ===
    unmatched = df[~matched]
    if not unmatched.empty:
        print("\n⚠️ Unmatched CSV entries (files not found in folder):")
        print(unmatched.head(10))  # Show top 10 invalid rows
    else:
        print("🎉 All filenames in CSV match actual image files.")
