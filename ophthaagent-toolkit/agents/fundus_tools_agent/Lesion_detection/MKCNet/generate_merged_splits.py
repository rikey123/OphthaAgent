import os
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_merged_splits():
    # Paths
    deepdrid_root = "<ANON_ABS_PATH>"
    new_dataset_root = "<ANON_ABS_PATH>"
    
    output_dir = "dataset_splits/DeepDRiD_5Class_Merged"
    os.makedirs(output_dir, exist_ok=True)
    
    data_entries = []
    
    # 1. Process DeepDRiD (Old)
    # We can reuse existing splits or re-scan. Let's re-scan to be safe and consistent.
    # Assuming DeepDRiD structure: /regular_fundus_images/Validation/1_0.jpg etc?
    # Or we can read from existing split files if available.
    # Let's try to read existing DeepDRiD splits to preserve train/test separation if possible.
    
    deepdrid_split_dir = "dataset_splits/DeepDRiD_5Class"
    if os.path.exists(deepdrid_split_dir):
        for split in ['train', 'test', 'val']:
            split_file = os.path.join(deepdrid_split_dir, f"{split}.txt")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    next(f) # skip header
                    for line in f:
                        parts = line.strip().split(',')
                        # parts[0] is relative path to deepdrid_root
                        abs_path = os.path.join(deepdrid_root, parts[0])
                        label = parts[1]
                        iq = parts[2]
                        data_entries.append({
                            'path': abs_path,
                            'label': label,
                            'iq': iq,
                            'split': split # Keep original split
                        })
    else:
        print("DeepDRiD splits not found, skipping DeepDRiD (or implement raw scan)")
        
    # 2. Process New Dataset (59_Diabetic_Retinopathy_Level_Detection)
    # We need to know the structure. Let's assume it has a CSV or subfolders.
    # Let's scan the directory first to see structure.
    # Since I cannot see it interactively in this script, I will assume a standard structure or CSV.
    # If there is a labels.csv, read it.
    
    # Heuristic: Look for CSV
    csv_files = glob.glob(os.path.join(new_dataset_root, "*.csv"))
    new_data = []
    
    # Check for training/testing subfolders structure (Common in ImageFolder datasets)
    # Structure: root/training/0/img.jpg
    for split_folder in ['training', 'testing', 'train', 'test', 'val', 'validation']:
        split_path = os.path.join(new_dataset_root, split_folder)
        if os.path.exists(split_path):
            print(f"Found split folder: {split_folder}")
            # Look for class folders 0-4
            for i in range(5):
                class_folder = os.path.join(split_path, str(i))
                if os.path.exists(class_folder):
                    imgs = glob.glob(os.path.join(class_folder, "*"))
                    print(f"  Found {len(imgs)} images in class {i}")
                    for img in imgs:
                        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            new_data.append({
                                'path': img,
                                'label': i,
                                'iq': 0
                            })

    if not new_data and csv_files:
        # Assume first CSV is the label file
        df = pd.read_csv(csv_files[0])
        # Assume columns: image, level (0-4)
        # Adjust column names as needed based on common datasets
        # Common names: 'image', 'level', 'diagnosis', 'DR_grade'
        
        img_col = None
        lbl_col = None
        
        for col in df.columns:
            if 'image' in col.lower() or 'id' in col.lower():
                img_col = col
            if 'level' in col.lower() or 'grade' in col.lower() or 'label' in col.lower():
                lbl_col = col
                
        if img_col and lbl_col:
            for _, row in df.iterrows():
                img_name = str(row[img_col])
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_name += '.jpg' # Try adding extension if missing
                    
                # Check existence
                # Try direct in root or in 'train'/'test' subfolders?
                # Let's try root first
                abs_path = os.path.join(new_dataset_root, img_name)
                if not os.path.exists(abs_path):
                    # Try recursive search? Or just skip
                    # Let's try adding .jpeg
                    abs_path = os.path.join(new_dataset_root, img_name.replace('.jpg', '.jpeg'))
                
                if os.path.exists(abs_path):
                    label = int(row[lbl_col])
                    # Map label to 0-4 if needed. Assuming it is 0-4.
                    new_data.append({
                        'path': abs_path,
                        'label': label,
                        'iq': 0 # Assume good quality or unknown
                    })
    if not new_data and not csv_files:
        # Fallback: Subfolders 0, 1, 2, 3, 4 in root?
        for i in range(5):
            subfolder = os.path.join(new_dataset_root, str(i))
            if os.path.exists(subfolder):
                imgs = glob.glob(os.path.join(subfolder, "*"))
                for img in imgs:
                    new_data.append({
                        'path': img,
                        'label': i,
                        'iq': 0
                    })
                    
    # Split new data
    if new_data:
        train_new, test_new = train_test_split(new_data, test_size=0.2, random_state=42, stratify=[d['label'] for d in new_data])
        
        for d in train_new:
            d['split'] = 'train'
            data_entries.append(d)
            
        for d in test_new:
            d['split'] = 'test' # or val
            data_entries.append(d)
            
    # Write merged splits
    for split_name in ['train', 'test', 'val']:
        entries = [d for d in data_entries if d.get('split') == split_name]
        if not entries and split_name == 'val':
             # If no val, use test as val or empty
             pass
             
        with open(os.path.join(output_dir, f"{split_name}.txt"), 'w') as f:
            f.write("image_path,label,iq\n")
            for entry in entries:
                f.write(f"{entry['path']},{entry['label']},{entry['iq']}\n")
                
    print(f"Generated merged splits in {output_dir}")
    print(f"Total images: {len(data_entries)}")

if __name__ == "__main__":
    generate_merged_splits()
