#!/usr/bin/env python3
"""
对单张眼底图像进行全面分析的脚本
结合原始DeepSeeNet和DeepSeeNet Plus的功能
"""

import os

# 禁用TensorFlow的详细日志，只显示错误，保持输出整洁
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
import keras.utils as image
from keras.utils import get_file
import tensorflow as tf
# 在TensorFlow导入前设置设备
import sys

def setup_tensorflow_device_from_config():
    """从配置文件读取设备设置，不依赖torch"""
    config_path = '<ANON_ABS_PATH>'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        device_setting = config.get("device", "auto")
        
        if device_setting == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制CPU
            print("TensorFlow configured to use CPU")
        elif device_setting.startswith("cuda"):
            # 提取GPU索引 (cuda:0 -> 0)
            gpu_idx = device_setting.split(':')[-1] if ':' in device_setting else '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx  # 强制特定GPU
            print(f"TensorFlow configured to use GPU {gpu_idx}")
        else:
            # auto 模式，让TensorFlow自己决定
            print("TensorFlow configured to auto-detect device")
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # 如果配置文件不存在或无法读取，使用默认设置
        print(f"Warning: Could not read device config ({e}), using TensorFlow auto-detection")
        # 不设置 CUDA_VISIBLE_DEVICES，让TensorFlow自己决定

# 在TensorFlow导入后立即设置设备
setup_tensorflow_device_from_config()


# 定义类别标签和完整说明
DRUSEN_LABELS = ['None/Small', 'Intermediate', 'Large']
DRUSEN_DESCRIPTION = "玻璃膜疣大小 (Drusen Size)"
PIGMENT_LABELS = ['No', 'Yes']
PIGMENT_DESCRIPTION = "色素异常 (Pigmentary Abnormality)"
AMD_LABELS = ['No', 'Yes']
AMD_DESCRIPTION = "晚期年龄相关性黄斑变性 (Advanced AMD)"
GA_LABELS = ['No', 'Yes']
GA_DESCRIPTION = "地理萎缩 (Geographic Atrophy)"
CGA_LABELS = ['No', 'Yes']
CGA_DESCRIPTION = "中央地理萎缩 (Central GA)"

# 简化严重程度评分说明
SIMPLIFIED_SCORE_DESCRIPTION = "AREDS简化严重程度评分 (0-5分)，根据玻璃膜疣大小、色素异常和晚期AMD情况综合计算"

def crop2square(img):
    """
    Crop the image to a square based on the short edge.

    Args:
        img: PIL Image instance.

    Returns:
        A PIL Image instance.
    """
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image into a Numpy array

    Args:
        image_path: Path or file object.
        target_size: Target size for the image (width, height)

    Returns:
        Numpy array
    """
    logging.debug('Processing: %s', image_path)
    img = crop2square(image.load_img(image_path)).resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 使用DeepSeeNet的预处理方法
    x = x / 127.5 - 1.
    return x

def get_model_input_shape(model):
    """
    获取模型期望的输入形状
    """
    input_shape = model.input_shape
    return input_shape[1:3]  # 返回 (height, width)

# 定义需要分析的模型及其相关信息
models_to_analyze = {
    "Drusen": {"filename": "drusen_model.h5", "labels": ["drusen", "no drusen"]},
    "Pigment": {"filename": "pigment_model.h5", "labels": ["pigment", "no pigment"]},
    "Advanced AMD": {"filename": "adv_amd_model.h5", "labels": ["amd", "no amd"]},
    "Geographic Atrophy": {"filename": "ga_model.h5", "labels": ["ga", "no ga"]},
    "Central GA": {"filename": "cga_model.h5", "labels": ["central_ga", "no central_ga"]}
}

# 全局变量，用于缓存加载的模型
loaded_models_cache = {}

def load_all_models(models_dir='models/'):
    """
    一次性加载所有需要的模型到内存中，并进行缓存。

    Args:
        models_dir (str): 存放模型文件的目录路径。
    """
    global loaded_models_cache
    if loaded_models_cache:
        return

    print("Initializing and loading all models...")

    for model_name, model_info in models_to_analyze.items():
        h5_path = os.path.join(models_dir, model_info['filename'])
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found: {h5_path}. Cannot proceed with analysis.")
        
        try:
            model = load_model(h5_path)
            loaded_models_cache[model_name] = model
            print(f"- Successfully loaded model (.h5): {model_name}")
        except Exception as e:
            raise IOError(f"Failed to load model {model_name} from {h5_path}: {e}")

    print("All models loaded successfully.")


def analyze_single_image(image_path):
    """对单张图像进行分析"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    print(f"正在分析图像: {image_path}")

    results = {}
    preprocessed_images_cache = {}  # 用于缓存不同尺寸的预处理图像

    # 按顺序遍历定义的模型
    for model_name, model_info in models_to_analyze.items():
        if model_name in loaded_models_cache:
            model = loaded_models_cache[model_name]

            # 1. 获取当前模型所需的输入尺寸
            target_shape = get_model_input_shape(model)

            # 2. 检查所需尺寸的图像是否已在缓存中，如果不在则进行预处理
            if target_shape not in preprocessed_images_cache:
                print(f"图像预处理，目标尺寸: {target_shape}")
                preprocessed_images_cache[target_shape] = preprocess_image(image_path, target_shape)

            # 3. 从缓存中获取正确尺寸的图像
            processed_img = preprocessed_images_cache[target_shape]

            # 4. 使用 model() 直接调用以获得最佳性能
            #    并确保输入是 tf.Tensor
            tensor_img = tf.convert_to_tensor(processed_img)
            pred = model(tensor_img, training=False)
            
            # model() 返回的是张量，需要转换为numpy数组
            pred_numpy = pred.numpy()
            score = np.argmax(pred_numpy, axis=1)[0]
            confidence = np.max(pred_numpy)

            # 根据模型名称，填充对应的结果
            # 注意：这里的键（'drusen', 'pigment'等）是小写的，以保持与原始输出格式一致
            if model_name == 'Drusen':
                results['drusen'] = {
                    'score': int(score),
                    'label': DRUSEN_LABELS[score],
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in pred_numpy.flatten().tolist()],
                    'description': DRUSEN_DESCRIPTION
                }
                print(f"Drusen: {results['drusen']['label']} (置信度: {results['drusen']['confidence']:.4f})")
            elif model_name == 'Pigment':
                results['pigment'] = {
                    'score': int(score),
                    'label': PIGMENT_LABELS[score],
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in pred_numpy.flatten().tolist()],
                    'description': PIGMENT_DESCRIPTION
                }
                print(f"Pigment: {results['pigment']['label']} (置信度: {results['pigment']['confidence']:.4f})")
            elif model_name == 'Advanced AMD':
                results['amd'] = {
                    'score': int(score),
                    'label': AMD_LABELS[score],
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in pred_numpy.flatten().tolist()],
                    'description': AMD_DESCRIPTION
                }
                print(f"Advanced AMD: {results['amd']['label']} (置信度: {results['amd']['confidence']:.4f})")
            elif model_name == 'Geographic Atrophy':
                results['ga'] = {
                    'score': int(score),
                    'label': GA_LABELS[score],
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in pred_numpy.flatten().tolist()],
                    'description': GA_DESCRIPTION
                }
                print(f"Geographic Atrophy: {results['ga']['label']} (置信度: {results['ga']['confidence']:.4f})")
            elif model_name == 'Central GA':
                results['cga'] = {
                    'score': int(score),
                    'label': CGA_LABELS[score],
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in pred_numpy.flatten().tolist()],
                    'description': CGA_DESCRIPTION
                }
                print(f"Central GA: {results['cga']['label']} (置信度: {results['cga']['confidence']:.4f})")

    return results

def calculate_simplified_score(results):
    """计算简化严重程度评分"""
    def has_adv_amd(score):
        return score == 1

    def has_pigment(score):
        return score == 1

    def has_large_drusen(score):
        return score == 2

    def has_intermediate_drusen(score):
        return score == 1

    score = 0
    
    # 检查是否有AMD评分结果
    if 'amd' in results:
        if has_adv_amd(results['amd']['score']):
            score += 5
            
    # 检查是否有色素异常评分结果
    if 'pigment' in results:
        if has_pigment(results['pigment']['score']):
            score += 1
            
    # 检查是否有玻璃膜疣评分结果
    if 'drusen' in results:
        if has_large_drusen(results['drusen']['score']):
            score += 1
        # 注意：简化评分需要双眼信息，这里仅作示例
            
    return min(score, 5)

def main():
    # 最终解决方案：硬编码模型的绝对路径以绕过环境问题
    # 这是最稳妥的方法，确保无论脚本从何处调用，都能找到模型
    hardcoded_models_dir = '<ANON_ABS_PATH>'

    parser = argparse.ArgumentParser(description='对单张眼底图像进行全面分析')
    parser.add_argument('image_path', help='眼底图像路径')
    parser.add_argument('--models_dir', default=hardcoded_models_dir, help=f'模型文件夹路径 (默认: {hardcoded_models_dir})')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出结果')
    parser.add_argument('--output_file_path', help='将JSON输出保存到指定文件')
    
    args = parser.parse_args()
    
    # 确认GPU是否被TensorFlow检测到
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"成功检测到 {len(gpus)} 个GPU设备。脚本将尝试使用指定的GPU。")
        # 可选：打印可见的GPU设备以供调试
        try:
            visible_gpus = tf.config.get_visible_devices('GPU')
            print(f"TensorFlow可见的GPU: {visible_gpus}")
        except RuntimeError as e:
            print(f"无法获取可见设备列表: {e}")
    else:
        print("警告: 未检测到GPU。脚本将回退到CPU执行。请检查CUDA安装和GPU驱动。")

    # 在执行任何分析之前，先加载所有模型
    try:
        load_all_models(args.models_dir)
    except (FileNotFoundError, IOError) as e:
        print(f"Error during model initialization: {e}", file=sys.stderr)
        # 如果是 JSON 输出模式，则输出错误 JSON
        if args.json:
            error_output = {
                "image_path": args.image_path,
                "error": "Model loading failed.",
                "details": str(e)
            }
            print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # 执行单张图像的分析
    try:
        analysis_result = analyze_single_image(args.image_path)
        
        # 基于分析结果计算 AREDS 简化严重程度评分
        simplified_score = calculate_simplified_score(analysis_result)
        
        # 准备最终输出
        output = {
            "image_path": args.image_path,
            "simplified_score": {
                "score": simplified_score,
                "description": SIMPLIFIED_SCORE_DESCRIPTION
            },
            "analysis_results": analysis_result
        }

        # 根据用户选择的格式输出结果
        if args.output_file_path:
            with open(args.output_file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        elif args.json:
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"\n--- Analysis Report for: {os.path.basename(args.image_path)} ---")
            print(f"\nAREDS Simplified Severity Score: {simplified_score} - {SIMPLIFIED_SCORE_DESCRIPTION}")
            print("\n--- Detailed Predictions ---")
            for model_name, result in analysis_result.items():
                if "error" in result:
                    print(f"{model_name}: Error - {result['error']}")
                else:
                    print(f"{model_name}: {result['prediction']} (Confidence: {result['confidence']:.4f})")
            print("\n---------------------------------")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": "An unexpected error occurred.", "details": str(e)}, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()