import os
import glob
import shutil
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

# 1. Download dataset automatically via kagglehub
print("Downloading HAM10000 dataset via kagglehub...")
download_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Dataset downloaded to:", download_path)

# 2. Define Output Paths and Class Mappings
OUTPUT_DIR = "dataset"
LABEL_MAPPING = {
    'akiec': 'Actinic keratosis',
    'bcc': 'Basal cell carcinoma',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'bkl': 'Seborrheic keratosis',
    'vasc': 'Vascular lesion'
}

# Find metadata CSV
csv_file = os.path.join(download_path, "HAM10000_metadata.csv")
if not os.path.exists(csv_file):
    # Search recursively if file is in a subfolder
    csv_matches = glob.glob(os.path.join(download_path, "**", "HAM10000_metadata.csv"), recursive=True)
    if csv_matches:
        csv_file = csv_matches[0]
    else:
        raise FileNotFoundError("HAM10000_metadata.csv not found in downloaded dataset.")

print(f"Reading metadata from: {csv_file}")
df = pd.read_csv(csv_file)
df['class_name'] = df['dx'].map(LABEL_MAPPING)

# Map all downloaded .jpg files into a lookup dictionary
print("Indexing image files...")
image_lookup = {}
for root, _, files in os.walk(download_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_lookup[file] = os.path.join(root, file)

# 3. Perform Train (70%), Val (15%), Test (15%) Stratified Split
print("Splitting dataset into train (70%), val (15%), and test (15%)...")
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['class_name'])
val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df['class_name'])

splits = {'train': train_df, 'val': val_df, 'test': test_df}

# 4. Copy Images to 'dataset/train', 'dataset/val', and 'dataset/test'
for split_name, split_df in splits.items():
    print(f"Organizing {split_name} split...")
    copied_count = 0
    for _, row in split_df.iterrows():
        img_filename = row['image_id'] + '.jpg'
        label = row['class_name']
        
        dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        
        if img_filename in image_lookup:
            src_path = image_lookup[img_filename]
            dest_path = os.path.join(dest_dir, img_filename)
            if not os.path.exists(dest_path):
                shutil.copy(src_path, dest_path)
            copied_count += 1

    print(f"Successfully processed {copied_count} images for {split_name}.")

print("\nDataset preparation complete! All folders under 'dataset/' are ready.")