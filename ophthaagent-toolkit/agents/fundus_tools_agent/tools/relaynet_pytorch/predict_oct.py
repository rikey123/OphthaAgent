# First, import the necessary modules to get them into sys.modules
import relaynet_pytorch
import relaynet_pytorch.relay_net
import sys

# Mock the path to match the one used during model saving
sys.modules['networks'] = sys.modules['relaynet_pytorch']
sys.modules['networks.relay_net'] = sys.modules['relaynet_pytorch.relay_net']

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import h5py
from skimage.morphology import remove_small_objects, disk, binary_opening
from skimage.measure import label

# The class is now findable via the mocked path.
from relaynet_pytorch.relay_net import ReLayNet


def convert_to_h5(image_path, output_h5_path):
    """Converts an image to an HDF5 file in the format expected by the model."""
    print(f"Converting {image_path} to {output_h5_path}...")

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Replicate the exact preprocessing from data_utils.py
    # 1. Resize so that height is 600 to prepare for slicing
    w, h = image.size
    # The original data was sliced from 61:573, which is 512 pixels high.
    # To be safe, let's resize to a height that can accommodate this slice.
    # Let's assume the original images were around 576 pixels high.
    # We will resize to a larger height and then take the slice.
    # Let's resize to a height of 600 to be safe.
    new_h = 600
    new_w = int(w * (new_h / h))
    image_resized = TF.resize(image, [new_h, new_w])

    # Convert to numpy array
    image_np = np.array(image_resized)

    # TRANSPOSE: Rotate 90 degrees counter-clockwise to make it vertical
    image_transposed = np.transpose(image_np)

    # 2. Take the exact slice [61:573]
    # Convert to numpy to slice, as PIL doesn't support this directly
    image_sliced = image_transposed[61:573, :] # Height slice

    # 3. Center crop the width to 496
    h_sliced, w_sliced = image_sliced.shape
    left = (w_sliced - 496) // 2
    right = left + 496
    image_cropped = image_sliced[:, left:right]

    # The data is now a numpy array of size 512x496.
    # NORMALIZE to [0.0, 1.0] and convert to float32.
    image_normalized = image_cropped.astype(np.float32) / 255.0

    # Create HDF5 file
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('data', data=[image_normalized])

    print("Conversion complete.")


def predict(model_path, image_path, output_path, num_classes=9, original_image_path=None):
    # Load the model
    print(f"Loading model from {model_path}...")
    # Set weights_only=False to load the full model object
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    print("Model loaded successfully.")

    # Load and preprocess the image
    print(f"Loading and preprocessing image from {image_path}...")
    
    if image_path.endswith('.h5'):
        with h5py.File(image_path, 'r') as f:
            # Data is already preprocessed (sliced, cropped, normalized)
            image_np = f['data'][0]
        # Convert to tensor, add channel and batch dimensions
        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
    else:
        # Fallback to original image processing if not an H5 file
        image = Image.open(image_path).convert('L')
        # Replicate the exact preprocessing from data_utils.py
        w, h = image.size
        new_h = 600 # Resize to a safe height
        new_w = int(w * (new_h / h))
        image_resized = TF.resize(image, [new_h, new_w])
        
        image_np = np.array(image_resized)

        # TRANSPOSE: Rotate 90 degrees counter-clockwise
        image_transposed = np.transpose(image_np)

        image_sliced = image_transposed[61:573, :] # Exact slice

        h_sliced, w_sliced = image_sliced.shape
        left = (w_sliced - 496) // 2
        right = left + 496
        image_cropped = image_sliced[:, left:right]

        # NORMALIZE to [0.0, 1.0] and convert to float32.
        image_normalized = image_cropped.astype(np.float32) / 255.0

        # Convert to float tensor, add channel and batch dim
        image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0).unsqueeze(0)

    print("Image preprocessed successfully.")

    # Predict
    print("Running prediction...")
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    print("Prediction complete.")

    # Convert prediction to numpy array for post-processing
    predicted_np = predicted.squeeze(0).cpu().numpy()

    print("Filtering segmentation to keep only the main retinal structure...")
    # --- Advanced Filtering: Keep only the largest connected component of the core retinal layers ---
    
    # 1. Define the core retinal layers (Classes 1-7)
    classes_to_keep = {1, 2, 3, 4, 5, 6, 7}
    
    # 2. Create a binary mask for all pixels belonging to these core layers
    core_retina_mask = np.isin(predicted_np, list(classes_to_keep))
    
    # 3. Find all distinct connected components (regions) in the core mask
    # The background is labeled 0, other components are labeled with integers starting from 1
    labeled_components, num_components = label(core_retina_mask, return_num=True)
    
    filtered_predicted_np = np.zeros_like(predicted_np)

    if num_components > 0:
        # 4. Find the largest connected component (excluding background)
        # We calculate the size of each component
        component_sizes = np.bincount(labeled_components.ravel())
        # The component label 0 is the background, so we ignore it by setting its size to 0
        component_sizes[0] = 0
        # The label of the largest component is the one with the maximum size
        largest_component_label = component_sizes.argmax()

        # 5. Create a mask that isolates only the largest component
        main_component_mask = (labeled_components == largest_component_label)

        # 6. Create the new filtered prediction:
        # Keep original class values, but only for pixels within the largest component
        filtered_predicted_np[main_component_mask] = predicted_np[main_component_mask]
    
    # All subsequent processing will now use the filtered prediction
    predicted_np = filtered_predicted_np

    print("Post-processing segmentation...")
    cleaned_segmentation = np.zeros_like(predicted_np)
    # We process each class separately to avoid merging small objects from different classes
    # Define a structuring element for morphological operations
    selem = disk(2) # A small disk of radius 2
    for i in range(1, num_classes + 1): # Process classes 1 to 9
        class_mask = (predicted_np == i)
        # 1. Smooth boundaries and remove thin connections with morphological opening
        opened_mask = binary_opening(class_mask, selem)
        # 2. Remove small, isolated objects that remain
        cleaned_mask = remove_small_objects(opened_mask, min_size=64)
        cleaned_segmentation[cleaned_mask] = i
    
    # The result of post-processing is our image to be visualized
    # Also keep the original background prediction (class 0)
    cleaned_segmentation[predicted_np == 0] = 0
    predicted_image = cleaned_segmentation.astype(np.uint8)
    
    # Official color map based on the provided SEG_LABELS_LIST
    # Index i in this array corresponds to the model's output class i
    color_map = np.array([
        [128, 0, 0],      # Class 0: Region above the retina (RaR)
        [0, 128, 0],      # Class 1: ILM: Inner limiting membrane
        [128, 128, 0],    # Class 2: NFL-IPL: Nerve fiber ending to Inner plexiform layer
        [0, 0, 128],      # Class 3: INL: Inner Nuclear layer
        [128, 0, 128],    # Class 4: OPL: Outer plexiform layer
        [0, 128, 128],    # Class 5: ONL-ISM: Outer Nuclear layer to Inner segment myeloid
        [128, 128, 128],  # Class 6: ISE: Inner segment ellipsoid
        [64, 0, 0],       # Class 7: OS-RPE: Outer segment to Retinal pigment epithelium
        [192, 0, 0]       # Class 8: Region below RPE (RbR)
    ])

    # Ensure color_map has enough colors
    if num_classes > len(color_map):
        # Generate random colors for extra classes
        extra_colors = np.random.randint(0, 255, size=(num_classes - len(color_map), 3))
        color_map = np.vstack([color_map, extra_colors])

    # Map the prediction to RGB colors
    rgb_segmentation = color_map[predicted_image]

    # --- New visualization logic ---
    # Load original image for background
    # Use original_image_path if provided, otherwise fallback to image_path (for JPG inputs)
    bg_image_path = original_image_path if original_image_path else image_path
    
    if bg_image_path.endswith('.h5'):
        print("Warning: Cannot use H5 file as background. Need original image path via --original-image-path. Defaulting to colored background.")
        final_image_np = rgb_segmentation
    else:
        print(f"Using {bg_image_path} as background for visualization.")
        original_image = Image.open(bg_image_path).convert('L') # Start with grayscale

        # Replicate the exact preprocessing to get the correctly oriented background
        w, h = original_image.size
        new_h = 600
        new_w = int(w * (new_h / h))
        image_resized = TF.resize(original_image, [new_h, new_w])
        image_np = np.array(image_resized)
        image_transposed = np.transpose(image_np)
        image_sliced = image_transposed[61:573, :]
        h_sliced, w_sliced = image_sliced.shape
        left = (w_sliced - 496) // 2
        right = left + 496
        image_cropped = image_sliced[:, left:right]

        # Convert grayscale background to RGB
        original_image_np = np.stack([image_cropped]*3, axis=-1)

        # Determine background class dynamically as the most frequent class
        background_class = np.bincount(predicted_image.flatten()).argmax()
        print(f"Dynamically determined background class: {background_class}")
        background_mask = predicted_image == background_class

        # Create the final image by overlaying segmentation on the original image
        final_image_np = rgb_segmentation.copy() # Important to copy
        # Where the mask is true (background), use pixels from the processed original image
        final_image_np[background_mask] = original_image_np[background_mask]

    output_image = Image.fromarray(final_image_np.astype(np.uint8))
    output_image.save(output_path)
    print(f"Segmentation map saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCT Image Segmentation using ReLayNet')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'convert'], help='Operation mode: predict or convert to H5.')
    parser.add_argument('--model-path', type=str, default='models/Exp01/relaynet_epoch20.model', help='Path to the pre-trained model.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input OCT image (JPG/PNG) or H5 file.')
    parser.add_argument('--original-image-path', type=str, default=None, help='Path to the original image for background visualization in predict mode.')
    parser.add_argument('--output-path', type=str, default='output_segmentation.png', help='Path to save the output segmentation map (for predict mode).')
    parser.add_argument('--output-h5-path', type=str, default='output_data.h5', help='Path to save the output HDF5 file (for convert mode).')
    parser.add_argument('--num-classes', type=int, default=9, help='Number of segmentation classes.')

    args = parser.parse_args()

    if args.mode == 'predict':
        predict(args.model_path, args.image_path, args.output_path, args.num_classes, args.original_image_path)
    elif args.mode == 'convert':
        convert_to_h5(args.image_path, args.output_h5_path)