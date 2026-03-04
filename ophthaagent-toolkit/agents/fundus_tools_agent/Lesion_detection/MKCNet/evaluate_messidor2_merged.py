import os
import sys
import argparse
import torch
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from tqdm import tqdm

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.defaults import _C as cfg_default
from dataset.dataset_manager import get_transform
from model.model_manager import get_model

def calculate_metrics(y_true, y_pred):
    # 5 classes: 0, 1, 2, 3, 4
    classes = [0, 1, 2, 3, 4]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return accuracy, kappa

class MessidorDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip ungradable if necessary
                if row.get('adjudicated_gradable') == '0':
                    continue
                
                img_name = row['image_id']
                label = int(row['adjudicated_dr_grade'])
                
                img_path = os.path.join(images_dir, img_name)
                
                # Check for extensions if not present
                if not os.path.exists(img_path):
                    if os.path.exists(img_path + '.jpg'):
                        img_path += '.jpg'
                    elif os.path.exists(img_path + '.png'):
                        img_path += '.png'
                    elif os.path.exists(img_path + '.JPG'):
                        img_path += '.JPG'
                    elif os.path.exists(img_path + '.PNG'):
                        img_path += '.PNG'
                
                if os.path.exists(img_path):
                    self.data.append((img_path, label))

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
    parser.add_argument('--csv_path', type=str, default='<ANON_ABS_PATH>', help='Path to Messidor-2 CSV')
    parser.add_argument('--images_dir', type=str, default='<ANON_ABS_PATH>', help='Path to Messidor-2 Images directory')
    parser.add_argument('--model_path', type=str, default='result/test/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='6')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Load Config
    dataset_name = 'DeepDRiD_5Class_Merged'
    model_type = 'FirstOrder_MKCNet'
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset_name
    
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset_name}.yaml")
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        print(f"Config file not found: {config_file}")
        return

    # Get transforms
    _, test_transform = get_transform(cfg)
    
    # Initialize Messidor Dataset
    print(f"Loading Messidor-2 dataset from {args.images_dir}...")
    test_dataset = MessidorDataset(csv_path=args.csv_path, images_dir=args.images_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(test_dataset)} images.")

    # Load Model
    print(f"Loading model from {args.model_path}")
    
    psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
    # Note: get_model returns (model, metalearner, model_compute) for FirstOrder_MKCNet
    model_ret = get_model(cfg, psi)
    if isinstance(model_ret, tuple):
        model = model_ret[0]
    else:
        model = model_ret
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Inference
    print("Starting inference...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            
            output = model(images)
            # FirstOrder_MKCNet returns (output_T, output_M, output_IQ)
            if isinstance(output, tuple):
                output = output[0] # We only care about the task output (DR grade)
                
            _, predicted = torch.max(output.data, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Calculate Metrics
    acc, kappa = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

if __name__ == "__main__":
    main()
