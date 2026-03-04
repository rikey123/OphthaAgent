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

def map_label_5_to_3(label):
    """
    Map DeepDRiD 5-class labels to EyeQ 3-class labels.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to DeepDRiD dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to EyeQ model checkpoint')
    parser.add_argument('--meta_model_path', type=str, default=None, help='Path to meta learner checkpoint')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Load Config for EYEQ (since model is trained on EYEQ)
    dataset_name = 'EYEQ'
    model_type = 'FirstOrder_MKCNet' # Based on folder name
    
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = model_type
    cfg.DATASET.NAME = dataset_name
    
    # Load EYEQ config
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/datasets/{dataset_name}.yaml")
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    
    # Override ROOT for dataset loading (though dataset class handles split path internally)
    cfg.DATASET.ROOT = args.root
    
    # Get transforms (will use EYEQ normalization)
    _, test_transform = get_transform(cfg)
    
    # Initialize DeepDRiD Dataset
    print(f"Loading DeepDRiD dataset from {args.root}...")
    # Note: DeepDRiD_5Class ignores cfg in __init__ and uses its own split logic
    test_dataset = DeepDRiD_5Class(root=args.root, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(test_dataset)} images.")

    # Load Model
    print(f"Loading model from {args.model_path}")
    
    # Initialize model structure with EYEQ settings
    # EYEQ config: NUM_M=9, META_LENGTH=5
    psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
    model, metalearner, _ = get_model(cfg, psi) # FirstOrder_MKCNet returns 3 items
    
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
            
            # Forward pass
            output = model(images)
            
            # FirstOrder_MKCNet might return (output, ...)
            if isinstance(output, tuple):
                output = output[0]
                
            _, predicted = torch.max(output.data, 1)
            
            # Map Ground Truth labels (0-4) to 3 classes
            mapped_labels = [map_label_5_to_3(l.item()) for l in labels_T]
            
            y_true.extend(mapped_labels)
            y_pred.extend(predicted.cpu().numpy())
            
    # Calculate Metrics
    acc = calculate_metrics(y_true, y_pred)
    print(f"Accuracy (Mapped 3-class): {acc:.4f}")

if __name__ == "__main__":
    main()
