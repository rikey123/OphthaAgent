#!/usr/bin/env python3
"""
对指定的眼底图像进行自动诊断

Usage:
    diagnose_image [options] <left_eye_image> <right_eye_image>

Options:
    --models_dir=<str>     Models directory [default: models]
    -v, --verbose          Print verbose output
"""

import logging
import sys
import os

import docopt
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseenet import deepseenet_simplified, deepseenet_drusen, deepseenet_pigment, deepseenet_adv_amd
from deepseenet.utils import pick_device


def diagnose_eye(left_image_path, right_image_path, models_dir="models", verbose=False):
    """
    对左右眼图像进行诊断
    
    Args:
        left_image_path: 左眼图像路径
        right_image_path: 右眼图像路径
        models_dir: 模型文件所在目录
        verbose: 是否输出详细信息
        
    Returns:
        dict: 包含各项诊断结果的字典
    """
    # 设置日志级别
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # 构建模型文件路径
    drusen_model_path = os.path.join(models_dir, 'drusen_model.h5')
    pigment_model_path = os.path.join(models_dir, 'pigment_model.h5')
    adv_amd_model_path = os.path.join(models_dir, 'adv_amd_model.h5')
    
    # 检查模型文件是否存在
    for model_path in [drusen_model_path, pigment_model_path, adv_amd_model_path]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 初始化模型
    print("正在加载模型...")
    drusen_model = deepseenet_drusen.DeepSeeNetDrusen(drusen_model_path)
    pigment_model = deepseenet_pigment.DeepSeeNetPigment(pigment_model_path)
    adv_amd_model = deepseenet_adv_amd.DeepSeeNetAdvancedAMD(adv_amd_model_path)
    
    # 预处理图像
    print("正在处理图像...")
    left_drusen_x = deepseenet_drusen.preprocess_image(left_image_path)
    right_drusen_x = deepseenet_drusen.preprocess_image(right_image_path)
    
    left_pigment_x = deepseenet_pigment.preprocess_image(left_image_path)
    right_pigment_x = deepseenet_pigment.preprocess_image(right_image_path)
    
    left_adv_x = deepseenet_adv_amd.preprocess_image(left_image_path)
    right_adv_x = deepseenet_adv_amd.preprocess_image(right_image_path)
    
    # 进行预测
    print("正在进行诊断...")
    # Drusen 大小预测 (0: small/none, 1: intermediate, 2: large)
    left_drusen_score = drusen_model.predict(left_drusen_x)
    right_drusen_score = drusen_model.predict(right_drusen_x)
    left_drusen_pred = np.argmax(left_drusen_score, axis=1)[0]
    right_drusen_pred = np.argmax(right_drusen_score, axis=1)[0]
    
    # 色素异常预测 (0: no, 1: yes)
    left_pigment_score = pigment_model.predict(left_pigment_x)
    right_pigment_score = pigment_model.predict(right_pigment_x)
    left_pigment_pred = np.argmax(left_pigment_score, axis=1)[0]
    right_pigment_pred = np.argmax(right_pigment_score, axis=1)[0]
    
    # 晚期AMD预测 (0: no, 1: yes)
    left_adv_score = adv_amd_model.predict(left_adv_x)
    right_adv_score = adv_amd_model.predict(right_adv_x)
    left_adv_pred = np.argmax(left_adv_score, axis=1)[0]
    right_adv_pred = np.argmax(right_adv_score, axis=1)[0]
    
    # 获取drusen大小的文本描述
    def get_drusen_text(pred):
        if pred == 0:
            return "small/none"
        elif pred == 1:
            return "intermediate"
        elif pred == 2:
            return "large"
        else:
            return "unknown"
    
    # 获取二分类结果的文本描述
    def get_binary_text(pred):
        return "yes" if pred == 1 else "no"
    
    # 计算简化评分
    scores = {
        'drusen': (left_drusen_pred, right_drusen_pred),
        'pigment': (left_pigment_pred, right_pigment_pred),
        'advanced_amd': (left_adv_pred, right_adv_pred)
    }
    
    simplified_score = deepseenet_simplified.get_simplified_score(scores)
    
    # 组织结果
    results = {
        'left_eye': {
            'drusen_size': (left_drusen_pred, get_drusen_text(left_drusen_pred)),
            'pigment_abnormality': (left_pigment_pred, get_binary_text(left_pigment_pred)),
            'advanced_amd': (left_adv_pred, get_binary_text(left_adv_pred)),
            'drusen_score': left_drusen_score
        },
        'right_eye': {
            'drusen_size': (right_drusen_pred, get_drusen_text(right_drusen_pred)),
            'pigment_abnormality': (right_pigment_pred, get_binary_text(right_pigment_pred)),
            'advanced_amd': (right_adv_pred, get_binary_text(right_adv_pred)),
            'drusen_score': right_drusen_score
        },
        'simplified_score': simplified_score
    }
    
    return results


def print_results(results):
    """
    打印诊断结果
    
    Args:
        results: diagnose_eye函数返回的结果字典
    """
    print("\n" + "="*50)
    print("眼底图像自动诊断结果")
    print("="*50)
    
    print("\n左眼诊断结果:")
    print(f"  玻璃膜疣大小: {results['left_eye']['drusen_size'][1]} (等级: {results['left_eye']['drusen_size'][0]})")
    print(f"  色素异常: {results['left_eye']['pigment_abnormality'][1]}")
    print(f"  晚期AMD: {results['left_eye']['advanced_amd'][1]}")
    
    print("\n右眼诊断结果:")
    print(f"  玻璃膜疣大小: {results['right_eye']['drusen_size'][1]} (等级: {results['right_eye']['drusen_size'][0]})")
    print(f"  色素异常: {results['right_eye']['pigment_abnormality'][1]}")
    print(f"  晚期AMD: {results['right_eye']['advanced_amd'][1]}")
    
    print(f"\nAREDS简化严重程度评分: {results['simplified_score']}")
    
    # 解释评分含义
    score_explanation = {
        0: "无AMD或极轻微AMD",
        1: "轻度AMD",
        2: "轻中度AMD",
        3: "中重度AMD",
        4: "重度AMD",
        5: "晚期AMD或极重度AMD"
    }
    
    explanation = score_explanation.get(results['simplified_score'], "未知")
    print(f"  含义: {explanation}")


if __name__ == '__main__':
    argv = docopt.docopt(__doc__, argv=sys.argv[1:])
    
    # 如果启用详细模式，则设置日志级别
    verbose = argv['--verbose']
    
    try:
        # 尝试使用GPU
        pick_device()
    except Exception as e:
        if verbose:
            print(f"无法使用GPU: {e}")
        pass  # 如果没有GPU，将继续使用CPU
    
    try:
        # 进行诊断
        results = diagnose_eye(
            argv['<left_eye_image>'],
            argv['<right_eye_image>'],
            argv['--models_dir'],
            verbose
        )
        
        # 打印结果
        print_results(results)
        
    except Exception as e:
        print(f"诊断过程中发生错误: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)