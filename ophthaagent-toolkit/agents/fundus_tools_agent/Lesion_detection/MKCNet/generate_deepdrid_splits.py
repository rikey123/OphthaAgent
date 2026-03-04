import csv
import os
import random

def generate_splits():
    base_path = "<ANON_ABS_PATH>"
    train_csv = os.path.join(base_path, "regular-fundus-training/regular-fundus-training.csv")
    val_csv = os.path.join(base_path, "regular-fundus-validation/regular-fundus-validation.csv")
    
    output_dir = "<ANON_ABS_PATH>"
    
    # Process Training Data
    train_lines = []
    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id']
            patient_id = row['patient_id']
            quality = int(row['Overall quality']) # 0 or 1
            
            label = -1
            if row['left_eye_DR_Level']:
                label = int(row['left_eye_DR_Level'])
            elif row['right_eye_DR_Level']:
                label = int(row['right_eye_DR_Level'])
                
            if label == -1:
                continue
                
            rel_path = f"regular-fundus-training/Images/{patient_id}/{img_id}.jpg"
            train_lines.append(f"{rel_path},{label},{quality}")

    # Process Validation Data
    val_lines = []
    with open(val_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id']
            patient_id = row['patient_id']
            quality = int(row['Overall quality'])
            
            label = -1
            if row['left_eye_DR_Level']:
                label = int(row['left_eye_DR_Level'])
            elif row['right_eye_DR_Level']:
                label = int(row['right_eye_DR_Level'])
                
            if label == -1:
                continue
                
            rel_path = f"regular-fundus-validation/Images/{patient_id}/{img_id}.jpg"
            val_lines.append(f"{rel_path},{label},{quality}")

    # Write to files
    # MKCNet expects train.txt, val.txt, test.txt
    # We will use train_lines for train.txt
    # We will split val_lines into val.txt and test.txt (50/50)
    
    random.seed(42)
    random.shuffle(val_lines)
    split_idx = len(val_lines) // 2
    real_val_lines = val_lines[:split_idx]
    test_lines = val_lines[split_idx:]
    
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        f.write("path,label_T,label_IQ\n")
        f.write("\n".join(train_lines))
        
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        f.write("path,label_T,label_IQ\n")
        f.write("\n".join(real_val_lines))
        
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        f.write("path,label_T,label_IQ\n")
        f.write("\n".join(test_lines))
        
    print(f"Generated splits: Train={len(train_lines)}, Val={len(real_val_lines)}, Test={len(test_lines)}")

if __name__ == "__main__":
    generate_splits()
