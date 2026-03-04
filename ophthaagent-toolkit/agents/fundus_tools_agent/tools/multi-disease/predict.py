import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image
import os
import time

def build_model(num_classes, pretrained=False):
    """
    Builds a ResNet-50 model.
    
    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): If True, uses a model pre-trained on ImageNet.
        
    Returns:
        A ResNet-50 model.
    """
    print("Building model...")
    if pretrained:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)
    
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    print("Model built.")
    return model

def predict(image_path, model_path, classes_path, device):
    """
    Predicts the class of an image using a trained model.
    
    Args:
        image_path (str): The path to the image.
        model_path (str): The path to the trained model.
        classes_path (str): The path to the file containing the class names.
        device (str): The device to run the model on ('cpu' or 'cuda').
    """
    print(f"Using device: {device}")
    # Load class names
    print("Loading class names...")
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")

    # Load the model
    model = build_model(num_classes)
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    model.to(device)
    model.eval()

    # Preprocess the image
    print("Preprocessing image...")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    print("Image preprocessed.")

    # Make a prediction
    print("Making prediction...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
    print(f"Prediction made in {time.time() - start_time:.2f} seconds.")

    result = {
        "predicted_class": predicted_class,
        "confidence_scores": {classes[i]: prob.item() for i, prob in enumerate(probabilities)}
    }
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained ResNet-50 model.')
    parser.add_argument('image_path', type=str, help='The path to the image.')
    parser.add_argument('--model_path', type=str, default='runs/resnet50/best.pth', help='The path to the trained model.')
    parser.add_argument('--classes_path', type=str, default='runs/resnet50/classes.json', help='The path to the file containing the class names.')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu', help='The device to run the model on (e.g., cpu, cuda, cuda:0, cuda:1).')
    parser.add_argument('--output_file_path', type=str, default=None, help='If provided, the output JSON will be saved to this file.')
    args = parser.parse_args()

    prediction_result = predict(args.image_path, args.model_path, args.classes_path, args.device)

    if args.output_file_path:
        with open(args.output_file_path, 'w') as f:
            json.dump(prediction_result, f, indent=2)
        print(f"Result saved to {args.output_file_path}")
    else:
        print(json.dumps(prediction_result, indent=2))