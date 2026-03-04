import os
import sys
import argparse
import torch
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.defaults import _C as cfg_default
from dataset.dataset_manager import get_transform
from model.model_manager import get_model

def map_label_5_to_3(label):
    """
    Map Messidor-2 5-class labels to EyeQ 3-class labels.
    0 -> 0 (Normal)
    1 (Mild), 2 (Moderate), 3 (Severe) -> 1 (Non-Proliferative DR)
    4 (Proliferative) -> 2 (Proliferative DR)
    """
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
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, target_names=['Class 0 (Normal)', 'Class 1 (NPDR)', 'Class 2 (PDR)']))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    return accuracy

class MessidorDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip ungradable if necessary, or handle them. 
                # Assuming we only want gradable ones.
                if row.get('adjudicated_gradable') == '0':
                    continue
                
                img_name = row['image_id']
                # Some CSVs might have extension, some might not. 
                # Usually Messidor-2 CSV has image names like "20051020_43808_0100_PP.png" or just ID.
                # Let's assume the CSV provided has correct filenames or we need to append .jpg/.png
                # Based on previous scripts, it seems we might need to check existence.
                
                # In evaluate_messidor2_5class.py: img_path = os.path.join(images_dir, img_name)
                
                label = int(row['adjudicated_dr_grade'])
                
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    self.data.append((img_path, label))
                else:
                    # Try appending .jpg or .png if not found
                    if os.path.exists(img_path + '.jpg'):
                         self.data.append((img_path + '.jpg', label))
                    elif os.path.exists(img_path + '.png'):
                         self.data.append((img_path + '.png', label))
                    # else:
                    #     print(f"Warning: Image {img_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 256, 256)), -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to Messidor-2 CSV')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to Messidor-2 Images directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to EyeQ model checkpoint')
    parser.add_argument('--meta_model_path', type=str, default=None, help='Path to meta learner checkpoint')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Load Config for EYEQ
    dataset_name = 'EYEQ'
    model_type = 'FirstOrder_MKCNet'
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset_name
    
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset_name}.yaml")
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    
    # Get transforms (EYEQ normalization)
    _, test_transform = get_transform(cfg)
    
    # Initialize Messidor Dataset
    print(f"Loading Messidor-2 dataset from {args.images_dir}...")
    test_dataset = MessidorDataset(csv_path=args.csv_path, images_dir=args.images_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(test_dataset)} images.")

    # Load Model
    print(f"Loading model from {args.model_path}")
    
    psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
    model, metalearner, _ = get_model(cfg, psi)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    if args.meta_model_path and os.path.exists(args.meta_model_path):
        print(f"Loading meta learner from {args.meta_model_path}")
        metalearner.load_state_dict(torch.load(args.meta_model_path, map_location=device))
        metalearner.to(device)
        metalearner.eval()
    
    # Inference
    print("Starting inference...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
                
            _, predicted = torch.max(output.data, 1)
            
            # Map Ground Truth labels (0-4) to 3 classes
            mapped_labels = [map_label_5_to_3(l.item()) for l in labels]
            
            y_true.extend(mapped_labels)
            y_pred.extend(predicted.cpu().numpy())
            
    # Calculate Metrics
    acc = calculate_metrics(y_true, y_pred)
    print(f"Accuracy (Mapped 3-class): {acc:.4f}")

if __name__ == "__main__":
    main()
