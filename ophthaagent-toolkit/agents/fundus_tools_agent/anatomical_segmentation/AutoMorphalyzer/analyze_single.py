#!/usr/bin/env python3
"""
AutoMorphalyzer Single Image Analyzer

This script analyzes a single fundus image or all images in a directory,
and outputs results in both CSV and JSON formats.

Usage:
    python analyze_single.py --input <image_path> --output <output_dir>
    python analyze_single.py --input <image_dir> --output <output_dir> --batch

Examples:
    # Analyze a single image
    python analyze_single.py --input example_data/test_images/100.png --output results/

    # Analyze all images in a directory
    python analyze_single.py --input example_data/test_images/ --output results/ --batch

    # Analyze with custom JSON output name
    python analyze_single.py --input test.png --output results/ --json-output metrics.json
"""

import os
# 1. 强制指定 Hugging Face 缓存根目录
os.environ["HF_HOME"] = "<ANON_ABS_PATH>"  # 替换为你的实际路径

# 2. 强制 Hugging Face 库以离线模式运行（禁止任何网络请求）
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add automorph module to path
SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
AUTOMORPH_PATH = os.path.join(SCRIPT_PATH, 'automorph')
sys.path.append(SCRIPT_PATH)
sys.path.append(AUTOMORPH_PATH)

from automorph import utils
from automorph.preprocess import preprocess
from automorph.segment import segment
from automorph.measure import measure
import pandas as pd
import numpy as np
import torch


def setup_device(device_id=None):
    """
    Setup computation device (CPU/GPU)

    Args:
        device_id: Device specification
            - None: Auto-detect (use GPU if available)
            - 'cpu': Force CPU
            - 'cuda' or 'cuda:0': Use first GPU
            - 'cuda:1', 'cuda:2', etc: Use specific GPU
            - int (0, 1, 2, etc): GPU index

    Returns:
        device: torch.device object
        device_name: String description of the device
    """
    if device_id is None:
        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f'CUDA GPU 0 ({torch.cuda.get_device_name(0)})'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = 'Apple MPS'
        else:
            device = torch.device('cpu')
            device_name = 'CPU'

    elif isinstance(device_id, int):
        # GPU index specified as integer
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                device = torch.device(f'cuda:{device_id}')
                device_name = f'CUDA GPU {device_id} ({torch.cuda.get_device_name(device_id)})'
            else:
                print(f"Warning: GPU {device_id} not available (only {torch.cuda.device_count()} GPUs found)")
                print("Falling back to GPU 0")
                device = torch.device('cuda:0')
                device_name = f'CUDA GPU 0 ({torch.cuda.get_device_name(0)})'
        else:
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            device_name = 'CPU'

    elif isinstance(device_id, str):
        # String specification
        device_id_lower = device_id.lower()

        if device_id_lower == 'cpu':
            device = torch.device('cpu')
            device_name = 'CPU'

        elif device_id_lower == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                device_name = 'Apple MPS'
            else:
                print("Warning: MPS not available, falling back to CPU")
                device = torch.device('cpu')
                device_name = 'CPU'

        elif device_id_lower.startswith('cuda'):
            if torch.cuda.is_available():
                # Extract GPU index if specified (e.g., 'cuda:1')
                if ':' in device_id_lower:
                    gpu_idx = int(device_id_lower.split(':')[1])
                    if gpu_idx < torch.cuda.device_count():
                        device = torch.device(device_id_lower)
                        device_name = f'CUDA GPU {gpu_idx} ({torch.cuda.get_device_name(gpu_idx)})'
                    else:
                        print(f"Warning: GPU {gpu_idx} not available (only {torch.cuda.device_count()} GPUs found)")
                        print("Falling back to GPU 0")
                        device = torch.device('cuda:0')
                        device_name = f'CUDA GPU 0 ({torch.cuda.get_device_name(0)})'
                else:
                    device = torch.device('cuda')
                    device_name = f'CUDA GPU 0 ({torch.cuda.get_device_name(0)})'
            else:
                print("Warning: CUDA not available, falling back to CPU")
                device = torch.device('cpu')
                device_name = 'CPU'
        else:
            print(f"Warning: Unknown device '{device_id}', using auto-detect")
            return setup_device(None)

    else:
        print(f"Warning: Invalid device type {type(device_id)}, using auto-detect")
        return setup_device(None)

    return device, device_name


def list_available_devices():
    """Print all available computation devices"""
    print("\nAvailable devices:")
    print("-" * 60)

    # CPU
    print("  [cpu] CPU")

    # CUDA GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  [cuda:{i}] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  CUDA GPUs: Not available")

    # Apple MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  [mps] Apple Metal Performance Shaders")

    print("-" * 60)


def setup_output_directory(output_dir):
    """Create output directory structure"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subdirectories
    for subdir in ['M0/images', 'M2/binary_vessel/raw_binary',
                   'M2/artery_vein/raw_binary', 'M2<ANON_ABS_PATH>',
                   'M3/segmentations']:
        path = os.path.join(output_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def csv_to_json(csv_path, json_path=None):
    """
    Convert feature_measurements.csv to structured JSON format

    Args:
        csv_path: Path to the CSV file
        json_path: Path to save JSON file (optional)

    Returns:
        Dictionary with structured results
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Initialize result structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "analyzer": "AutoMorphalyzer",
            "version": "2025"
        },
        "images": []
    }

    # Process each image
    for idx, row in df.iterrows():
        image_data = {
            "filename": row['Filename'],
            "quality": {
                "quickqual_score": convert_to_serializable(row.get('QuickQual_quality', -1))
            },
            "optic_disc": {
                "laterality": row.get('laterality', 'Unknown'),
                "macular_centred": convert_to_serializable(row.get('macular_centred', -1)),
                "disc_height_px": convert_to_serializable(row.get('disc_height', -1)),
                "disc_width_px": convert_to_serializable(row.get('disc_width', -1)),
                "cup_height_px": convert_to_serializable(row.get('cup_height', -1)),
                "cup_width_px": convert_to_serializable(row.get('cup_width', -1)),
                "cdr_vertical": convert_to_serializable(row.get('CDR_vertical', -1)),
                "cdr_horizontal": convert_to_serializable(row.get('CDR_horizontal', -1))
            },
            "vessel_features": {
                "binary": {},
                "artery": {},
                "vein": {}
            },
            "large_vessel_metrics": {}
        }

        # Extract vessel features for each vessel type and zone
        vessel_types = ['binary', 'artery', 'vein']
        zones = ['whole', 'B', 'C']
        features = ['vessel_density', 'fractal_dimension', 'average_global_calibre',
                   'average_local_calibre', 'tortuosity_distance', 'tortuosity_density']

        for vtype in vessel_types:
            image_data["vessel_features"][vtype] = {}
            for zone in zones:
                zone_data = {}
                for feat in features:
                    col_name = f'{feat}_{vtype}_{zone}'
                    if col_name in row:
                        zone_data[feat] = convert_to_serializable(row[col_name])

                if zone_data:  # Only add if there's data
                    image_data["vessel_features"][vtype][zone] = zone_data

        # Extract large vessel metrics (CRAE, CRVE, AVR)
        large_vessel_cols = {
            'CRAE_artery_B': 'CRAE_Knudtson_artery_B',
            'CRAE_artery_C': 'CRAE_Knudtson_artery_C',
            'CRVE_vein_B': 'CRVE_Knudtson_vein_B',
            'CRVE_vein_C': 'CRVE_Knudtson_vein_C',
            'AVR_B': 'AVR_B',
            'AVR_C': 'AVR_C'
        }

        for json_key, csv_col in large_vessel_cols.items():
            if csv_col in row:
                image_data["large_vessel_metrics"][json_key] = convert_to_serializable(row[csv_col])

        results["images"].append(image_data)


    # Add absolute mask paths to each image record
    output_dir = os.path.dirname(json_path) if json_path else '.'
    for image_record in results['images']:
        basename, _ = os.path.splitext(image_record['filename'])
        image_record['vessel_mask_path'] = os.path.abspath(os.path.join(
            output_dir, 'M2', 'binary_vessel', 'raw_binary', f'{basename}.png'))
        image_record['disc_mask_path'] = os.path.abspath(os.path.join(
            output_dir, 'M2', 'optic_disc', 'raw_binary', f'{basename}.png'))

    # Save to JSON if path provided
    if json_path:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results saved to: {json_path}")

    return results


def analyze_images(input_path, output_dir, batch_mode=False, device=None):
    """
    Analyze fundus image(s) and generate results

    Args:
        input_path: Path to single image or directory
        output_dir: Directory to save results
        batch_mode: If True, process all images in directory
        device: Computation device (None=auto, 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)

    Returns:
        Path to results CSV file
    """

    # 替换第310-325行的代码:
    # Setup computation device
    # 使用绝对路径添加 device_config.py 所在的目录到 sys.path
    DEVICE_CONFIG_DIR = '<ANON_ABS_PATH>'
    sys.path.insert(0, DEVICE_CONFIG_DIR)
    from device_config import get_device

    # 获取设备配置 (get_device() 只返回 torch.device 对象)
    device_obj = get_device()

    # 根据设备对象生成设备名称
    if device_obj.type == 'cuda':
        if device_obj.index is not None:
            try:
                device_name = f'CUDA GPU {device_obj.index} ({torch.cuda.get_device_name(device_obj.index)})'
            except:
                device_name = f'CUDA GPU {device_obj.index}'
        else:
            try:
                device_name = f'CUDA GPU 0 ({torch.cuda.get_device_name(0)})'
            except:
                device_name = 'CUDA GPU 0'
    elif device_obj.type == 'cpu':
        device_name = 'CPU'
    else:
        device_name = str(device_obj)

    sys.path.pop(0)
    print(f"\n{'='*60}")
    print(f"Using device: {device_name}")
    print(f"{'='*60}")

    # Set environment variable for device (used by utils.py)
    if device_obj.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_obj.index) if device_obj.index is not None else '0'

    setup_output_directory(output_dir)

    # Determine input images
    image_list = []
    input_directory = None

    if os.path.isfile(input_path):
        # Single image mode
        input_directory = os.path.dirname(input_path)
        image_list = [os.path.basename(input_path)]
        print(f"Processing single image: {input_path}")
    elif os.path.isdir(input_path):
        # Directory mode
        if not batch_mode:
            print("ERROR: Input is a directory but --batch flag not specified.")
            print("Use --batch to process all images in the directory.")
            sys.exit(1)
        input_directory = input_path
        for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
            image_list += list(Path(input_path).glob(f'*{ext}'))
        image_list = [os.path.basename(str(p)) for p in image_list]
        print(f"Found {len(image_list)} images to process in: {input_path}")
    else:
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)

    if len(image_list) == 0:
        print("ERROR: No supported image files found.")
        print("Supported formats: .bmp, .png, .jpg, .jpeg")
        sys.exit(1)

    # Check for model weights and download if necessary
    print('\nChecking for model weights...')
    utils.check_unpack_model_weights('binary', AUTOMORPH_PATH)
    utils.check_unpack_model_weights('artery_vein', AUTOMORPH_PATH)
    utils.check_unpack_model_weights('optic_disc', AUTOMORPH_PATH)
    print('Model weights ready!')

    # M0: Preprocessing
    print('\n' + '='*50)
    print('M0: Preprocessing')
    print('='*50)
    preprocess.preprocess_dataset(input_directory, image_list, output_dir)
    print('Preprocessing complete!')

    # M2: Segmentation
    print('\n' + '='*50)
    print('M2: Segmentation')
    print('='*50)

    # Binary vessel segmentation
    print('\n[1/3] Binary vessel segmentation...')
    bin_networks = utils.get_binary_models()
    segment.binary_vessel_segmentation(bin_networks, output_dir)
    del bin_networks

    # Artery-Vein segmentation
    print('\n[2/3] Artery-Vein segmentation...')
    av_networks = utils.get_av_models()
    segment.arteryvein_vessel_segmentation(av_networks, output_dir)
    del av_networks

    # Optic disc segmentation
    print('\n[3/3] Optic disc segmentation...')
    od_networks = utils.get_od_models()
    segment.opticdisc_segmentation(od_networks, output_dir)
    del od_networks
    print('Segmentation complete!')

    # DEBUG: Verify mask saving
    print("\n[DEBUG in analyze_single.py] Verifying mask files...")
    for img_name in image_list:
        basename, _ = os.path.splitext(img_name)
        vessel_mask_path = os.path.join(output_dir, 'M2', 'binary_vessel', 'raw_binary', f'{basename}.png')
        disc_mask_path = os.path.join(output_dir, 'M2', 'optic_disc', 'raw_binary', f'{basename}.png')
        print(f"[DEBUG in analyze_single.py] Checking for {img_name}:")
        print(f"[DEBUG in analyze_single.py]   Vessel mask exists: {os.path.exists(vessel_mask_path)} at {vessel_mask_path}")
        print(f"[DEBUG in analyze_single.py]   Disc mask exists: {os.path.exists(disc_mask_path)} at {disc_mask_path}")
    print("[DEBUG in analyze_single.py] Verification complete.\n")

    # M3: Feature measurement
    print('\n' + '='*50)
    print('M3: Feature Measurement')
    print('='*50)
    measure.feature_measurement(image_list, output_dir)
    print('Feature measurement complete!')

    # Return path to results
    results_csv = os.path.join(output_dir, 'M3', 'feature_measurements.csv')
    return results_csv


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description='AutoMorphalyzer Single Image Analyzer - Analyze fundus images and output JSON results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python analyze_single.py --input example_data/test_images/100.png --output results/

  # Analyze all images in a directory
  python analyze_single.py --input example_data/test_images/ --output results/ --batch

  # Custom JSON output filename
  python analyze_single.py --input test.png --output results/ --json-output my_metrics.json

  # Skip analysis if CSV exists (only convert to JSON)
  python analyze_single.py --input test.png --output results/ --skip-if-exists

  # Specify GPU device
  python analyze_single.py --input test.png --output results/ --device cuda:0
  python analyze_single.py --input test.png --output results/ --device cuda:1
  python analyze_single.py --input test.png --output results/ --device cpu

  # List available devices
  python analyze_single.py --list-devices
        """
    )

    parser.add_argument('--input', '-i',
                       help='Path to input image or directory')

    parser.add_argument('--output', '-o',
                       help='Path to output directory')

    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all images in input directory')

    parser.add_argument('--json-output', '-j', default='measurements.json',
                       help='JSON output filename (default: measurements.json)')

    parser.add_argument('--skip-if-exists', '-s', action='store_true',
                       help='Skip analysis if results CSV already exists')

    parser.add_argument('--no-json', action='store_true',
                       help='Do not generate JSON output (CSV only)')

    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Computation device: cpu, cuda, cuda:0, cuda:1, etc. (default: auto-detect)')

    parser.add_argument('--list-devices', action='store_true',
                       help='List all available computation devices and exit')

    args = parser.parse_args()

    # Handle list-devices flag
    if args.list_devices:
        list_available_devices()
        sys.exit(0)

    # Validate required arguments
    if not args.input or not args.output:
        parser.error("--input and --output are required (unless using --list-devices)")

    # Print header
    print("\n" + "="*60)
    print("  AutoMorphalyzer Single Image Analyzer")
    print("="*60)

    # Check if results already exist
    results_csv = os.path.join(args.output, 'M3', 'feature_measurements.csv')

    if args.skip_if_exists and os.path.exists(results_csv):
        print(f"\nResults CSV already exists: {results_csv}")
        print("Skipping analysis (--skip-if-exists flag is set)")
    else:
        # Run analysis with specified device
        results_csv = analyze_images(args.input, args.output, args.batch, device=args.device)

    # Convert to JSON
    if not args.no_json:
        print('\n' + '='*50)
        print('Converting results to JSON format')
        print('='*50)

        json_path = os.path.join(args.output, args.json_output)
        results_json = csv_to_json(results_csv, json_path)

        # Print JSON results to console
        print('\n' + '='*60)
        print('JSON Results:')
        print('='*60)
        print(json.dumps(results_json, indent=2, ensure_ascii=False))
        print('='*60)

        # Print summary
        print(f"\nAnalysis complete!")
        print(f"  - Images processed: {len(results_json['images'])}")
        print(f"  - CSV results: {results_csv}")
        print(f"  - JSON results: {json_path}")
    else:
        print(f"\nAnalysis complete!")
        print(f"  - CSV results: {results_csv}")

    print("\nAll files saved to:", args.output)
    print("="*60)


if __name__ == "__main__":
    main()
