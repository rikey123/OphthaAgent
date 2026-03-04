import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.defaults import _C as cfg_default
from dataset.dataset import DeepDRiD_5Class
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
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--meta_model_path', type=str, default=None, help='Path to meta learner checkpoint (optional)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Load Config
    dataset_name = 'DeepDRiD_5Class'
    model_type = 'MKCNet'
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_T = 5 # Ensure 5 classes
    
    # Load config file if exists
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset_name}.yaml")
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    
    # Get transforms
    _, test_transform = get_transform(cfg)
    
    # Initialize Dataset
    print(f"Loading dataset from {args.root} with split 'test'...")
    test_dataset = DeepDRiD_5Class(root=args.root, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(test_dataset)} images.")

    # Load Model
    print(f"Loading model from {args.model_path}")
    
    # Initialize model structure
    psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
    model, metalearner = get_model(cfg, psi)
    
    # Load weights
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
        for images, labels_T, labels_IQ, labels_M in tqdm(test_loader):
            images = images.to(device)
            labels_T = labels_T.to(device)
            
            # Forward pass
            # MKCNet forward returns: output, features (or similar depending on implementation)
            # Let's check model_manager or MKCNet implementation.
            # Usually model(x) returns logits.
            # But MKCNet might return more.
            
            # In main.py: output = model(images)
            output = model(images)
            
            # If output is a tuple, take the first element (logits)
            if isinstance(output, tuple):
                output = output[0]
                
            _, predicted = torch.max(output.data, 1)
            
            y_true.extend(labels_T.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Calculate Metrics
    acc = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
