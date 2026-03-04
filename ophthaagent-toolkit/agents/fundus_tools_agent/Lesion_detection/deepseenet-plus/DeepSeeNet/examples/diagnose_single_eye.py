#!/usr/bin/env python3
"""
对单个眼底图像进行诊断并输出JSON格式结果

Usage:
    diagnose_single_eye [options] <image_path>

Options:
    --models_dir=<str>     Models directory [default: models]
    --pretty               Pretty print JSON output
"""

import json
import logging
import sys
import os

import docopt
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseenet import deepseenet_drusen, deepseenet_pigment, deepseenet_adv_amd
from deepseenet.utils import pick_device


def diagnose_single_eye(image_path, models_dir="models"):
    """
    对单个眼底图像进行诊断并返回JSON格式结果
    
    Args:
        image_path: 眼底图像路径
        models_dir: 模型文件所在目录
        
    Returns:
        dict: 包含诊断结果和预测概率的字典
    """
    # 构建模型文件路径
    drusen_model_path = os.path.join(models_dir, 'drusen_model.h5')
    pigment_model_path = os.path.join(models_dir, 'pigment_model.h5')
    adv_amd_model_path = os.path.join(models_dir, 'adv_amd_model.h5')
    
    # 检查模型文件是否存在
    for model_path in [drusen_model_path, pigment_model_path, adv_amd_model_path]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 初始化模型
    drusen_model = deepseenet_drusen.DeepSeeNetDrusen(drusen_model_path)
    pigment_model = deepseenet_pigment.DeepSeeNetPigment(pigment_model_path)
    adv_amd_model = deepseenet_adv_amd.DeepSeeNetAdvancedAMD(adv_amd_model_path)
    
    # 预处理图像
    drusen_x = deepseenet_drusen.preprocess_image(image_path)
    pigment_x = deepseenet_pigment.preprocess_image(image_path)
    adv_x = deepseenet_adv_amd.preprocess_image(image_path)
    
    # 进行预测
    # Drusen 大小预测 (0: small/none, 1: intermediate, 2: large)
    drusen_scores = drusen_model.predict(drusen_x)[0]  # 获取第一个样本的预测结果
    drusen_pred = int(np.argmax(drusen_scores))
    
    # 色素异常预测 (0: no, 1: yes)
    pigment_scores = pigment_model.predict(pigment_x)[0]
    pigment_pred = int(np.argmax(pigment_scores))
    
    # 晚期AMD预测 (0: no, 1: yes)
    adv_scores = adv_amd_model.predict(adv_x)[0]
    adv_pred = int(np.argmax(adv_scores))
    
    # 获取预测概率
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    # 组织结果
    results = {
        "input_image": image_path,
        "diagnosis": {
            "drusen_size": {
                "prediction": drusen_pred,
                "label": get_drusen_label(drusen_pred),
                "scores": {
                    "small_none": float(drusen_scores[0]),
                    "intermediate": float(drusen_scores[1]),
                    "large": float(drusen_scores[2])
                }
            },
            "pigment_abnormality": {
                "prediction": pigment_pred,
                "label": get_binary_label(pigment_pred),
                "scores": {
                    "no": float(pigment_scores[0]),
                    "yes": float(pigment_scores[1])
                }
            },
            "advanced_amd": {
                "prediction": adv_pred,
                "label": get_binary_label(adv_pred),
                "scores": {
                    "no": float(adv_scores[0]),
                    "yes": float(adv_scores[1])
                }
            }
        }
    }
    
    return results


def get_drusen_label(pred):
    """获取玻璃膜疣大小的文本标签"""
    labels = {
        0: "small/none",
        1: "intermediate",
        2: "large"
    }
    return labels.get(pred, "unknown")


def get_binary_label(pred):
    """获取二分类结果的文本标签"""
    return "yes" if pred == 1 else "no"


def main():
    argv = docopt.docopt(__doc__, argv=sys.argv[1:])
    
    try:
        # 尝试使用GPU
        pick_device()
    except Exception:
        pass  # 如果没有GPU，将继续使用CPU
    
    try:
        # 进行诊断
        results = diagnose_single_eye(
            argv['<image_path>'],
            argv['--models_dir']
        )
        
        # 输出JSON结果
        if argv['--pretty']:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))
            
    except Exception as e:
        error_result = {
            "error": str(e),
            "input_image": argv['<image_path>']
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == '__main__':
    main()