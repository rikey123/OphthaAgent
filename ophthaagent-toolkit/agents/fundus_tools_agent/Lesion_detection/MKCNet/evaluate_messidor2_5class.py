import os
import sys
import argparse
import torch
import csv
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_single_image import load_model, preprocess_image, find_model_files
from configs.defaults import _C as cfg_default

def get_messidor2_data(csv_path, images_dir):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('adjudicated_gradable') == '0':
                continue
            
            img_name = row['image_id']
            label = int(row['adjudicated_dr_grade'])
            
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                data.append((img_path, label))
    return data

def calculate_metrics(y_true, y_pred):
    # 5 classes: 0, 1, 2, 3, 4
    classes = [0, 1, 2, 3, 4]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    return accuracy

class MessidorDataset(Dataset):
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
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 256, 256)), -1, img_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--meta_model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Load Config
    dataset = 'DeepDRiD_5Class'
    model_type = 'MKCNet'
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset}.yaml")
    cfg.merge_from_file(config_file)
    
    # Load Model
    print(f"Loading model from {args.model_path}")
    model, metalearner = load_model(cfg, args.model_path, args.meta_model_path, device)
    model = model.to(device)
    model.eval()

    # Load Data
    data = get_messidor2_data(args.csv_path, args.images_dir)
    print(f"Found {len(data)} images.")

    # DataLoader
    messidor_dataset = MessidorDataset(data, cfg)
    dataloader = DataLoader(messidor_dataset, batch_size=32, shuffle=False, num_workers=4)

    y_true = []
    y_pred = []

    print("Starting inference...")
    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}...")
            
            images = images.to(device)
            output_T, _, _ = model(images)
            preds = torch.argmax(output_T, dim=1).cpu().numpy()
            
            for label, pred in zip(labels, preds):
                if label.item() == -1: continue
                y_true.append(label.item())
                y_pred.append(pred)

    acc = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
