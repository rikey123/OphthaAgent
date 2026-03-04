import os
import sys
import argparse
import torch
import csv
import time
from PIL import Image
# import pandas as pd

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_single_image import load_model, preprocess_image, find_model_files
from configs.defaults import _C as cfg_default
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

def get_deepdrid_data(csv_path, images_dir):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id']
            patient_id = row['patient_id']
            
            # Determine label
            label = -1
            if row['left_eye_DR_Level']:
                label = int(row['left_eye_DR_Level'])
            elif row['right_eye_DR_Level']:
                label = int(row['right_eye_DR_Level'])
                
            if label == -1:
                continue
                
            # Construct image path
            img_filename = f"{img_id}.jpg"
            img_path = os.path.join(images_dir, patient_id, img_filename)
            
            if os.path.exists(img_path):
                data.append((img_path, label))
            else:
                print(f"Warning: Image not found: {img_path}")
            
    return data

def map_label(label):
    # Training Standard Mapping (based on dataset.py):
    # 0 -> 0
    # 1, 2, 3 -> 1
    # 4 -> 2
    
    if label == 0:
        return 0
    elif label in [1, 2, 3]:
        return 1
    elif label == 4:
        return 2
    else:
        return -1

def calculate_metrics(y_true, y_pred):
    # 3 classes: 0, 1, 2
    classes = [0, 1, 2]
    metrics = {}
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, target_names=['Class 0', 'Class 1', 'Class 2']))

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    # F1 per class
    f1_scores = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        metrics[f'f1_class_{cls}'] = f1
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    return accuracy, macro_f1

class DeepDRiDDataset(Dataset):
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self.means = cfg.DATASET.NORMALIZATION_MEAN
        self.stds = cfg.DATASET.NORMALIZATION_STD
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.means, self.stds)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            if self.cfg.DATASET.NAME in ['DEEPDR', 'EYEQ']:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('L')
            
            image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 256, 256)), -1, img_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Change working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load Model (DEEPDR config)
    dataset = 'DEEPDR'
    model_type = 'MKCNet'
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset}.yaml")
    cfg.merge_from_file(config_file)
    
    model_path, meta_model_path, model_dir = find_model_files(model_type, dataset)
    print(f"Loading model from {model_path}")
    
    model, metalearner = load_model(cfg, model_path, meta_model_path, device)
    model = model.to(device)
    model.eval()
    if metalearner:
        metalearner = metalearner.to(device)
        metalearner.eval()

    # Load Data
    data = get_deepdrid_data(args.csv_path, args.images_dir)
    print(f"Found {len(data)} images.")

    # Create DataLoader
    deepdrid_dataset = DeepDRiDDataset(data, cfg)
    dataloader = DataLoader(deepdrid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    y_true = []
    y_pred = []

    start_time = time.time()
    print("Starting inference...")
    
    try:
        with torch.no_grad():
            for i, (images, labels, paths) in enumerate(dataloader):
                if i == 0:
                    print(f"\n--- Diagnostic: Batch {i} ---")
                    print(f"Input Tensor Shape: {images.shape}")
                    print(f"Input Tensor Stats: Min={images.min():.4f}, Max={images.max():.4f}, Mean={images.mean():.4f}, Std={images.std():.4f}")
                
                if i % 10 == 0:
                    print(f"Processing batch {i}/{len(dataloader)}...")
                
                images = images.to(device)
                
                if model_type in ['MKCNet', 'FirstOrder_MKCNet']:
                    output_T, output_M, output_IQ = model(images)
                    preds = torch.argmax(output_T, dim=1).cpu().numpy()
                else:
                    output = model(images)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                
                for label, pred, path in zip(labels, preds, paths):
                    label = label.item()
                    if label == -1: 
                        continue
                    
                    mapped_label = map_label(label)
                    y_true.append(mapped_label)
                    y_pred.append(pred)
    except KeyboardInterrupt:
        print("\nInterrupted! Calculating metrics for processed images...")
    except Exception as e:
        print(f"\nError: {e}")

    acc, f1 = calculate_metrics(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
