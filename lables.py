import os
import pandas as pd

# === Base Dataset Directory ===
base_dir = r'C:\Users\hulig\health-scanner-app\backend\datasets'
subfolders = ['facescan', 'eye']

# Labels you want to EXCLUDE from CSV
exclude_labels = []  # ✅ Add more to skip

# List to hold [filename, label]
rows = []

# === Label Extraction Function ===
def convert_label(text):
    name = text.lower()
    if 'normal eyes' in name:
        return 'normal eyes'
    elif 'normal' in name:
        return 'normal'
    elif 'acne' in name:
        return 'acne'
    elif 'fatigue' in name:
        return 'fatigue'
    elif 'wrinkle' in name:
        return 'wrinkle'
    elif 'bacterialconjunctivitisuveitis' in name:
        return 'BacterialConjunctivitisUveitis'
    elif 'bacterialconjunctivitis' in name:
        return 'BacterialConjunctivitis'
    elif 'cataractpterygium' in name:
        return 'CataractPterygium'
    elif 'cataracts' in name:
        return 'Cataracts'
    elif 'cataract' in name:
        return 'Cataract'
    elif 'glaucomapterygium' in name:
        return 'GlaucomaPterygium'
    elif 'glaucoma' in name:
        return 'Glaucoma'
    elif 'pterygium' in name:
        return 'Pterygium'
    elif 'uveitis' in name:
        return 'Uveitis'
    elif 'healthyviralconjunctivitis' in name:
        return 'HealthyViralConjunctivitis'
    elif 'viralconjunctivitis' in name:
        return 'ViralConjunctivitis'
    elif 'allergicconjunctivitis' in name:
        return 'AllergicConjunctivitis'
    
    else:
        return 'unknown'

# === Loop through folders ===
for subfolder in subfolders:
    full_path = os.path.join(base_dir, subfolder)
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                label = convert_label(rel_path)
                if label == 'unknown' or label in exclude_labels:
                    continue  # Skip unknown or excluded labels
                rows.append([rel_path.replace("\\", "/"), label])

# === Save CSV ===
csv_path = os.path.join(base_dir, 'labels.csv')
df = pd.DataFrame(rows, columns=['filename', 'label'])
df.to_csv(csv_path, index=False)

print(f"✅ labels.csv created at: {csv_path}")
print(f"📸 Total images (excluding {exclude_labels}): {len(rows)}")