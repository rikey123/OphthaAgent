#!/usr/bin/env python3
"""
改进的分类模型，结合分割模块的结果进行更准确的分类
"""

import os
import sys
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
import argparse

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def load_segmentation_model(model_path, model_module):
    """加载分割模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_module.build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def segment_image(image_path, model, device, output_size=(512, 512)):
    """使用模型对图像进行分割"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整图像大小
    image_resized = cv2.resize(image, output_size)
    
    # 预处理
    x = np.transpose(image_resized, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    
    # 预测
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)
    
    return image_resized, pred_y

def extract_features_from_segmentations(image_path):
    """从所有分割结果中提取特征"""
    features = {}
    
    # 血管分割特征
    try:
        sys.path.insert(0, os.path.join(project_root, "Blood Vessel Segmentation", "UNET"))
        import model as vessel_model
        sys.path.pop(0)
        
        model_path = os.path.join(project_root, "Blood Vessel Segmentation", "UNET", "checkpoint.pth")
        if os.path.exists(model_path):
            model, device = load_segmentation_model(model_path, vessel_model)
            _, vessel_mask = segment_image(image_path, model, device)
            # 计算血管密度
            vessel_density = np.sum(vessel_mask) / vessel_mask.size
            features['vessel_density'] = vessel_density
        else:
            features['vessel_density'] = 0
    except Exception as e:
        features['vessel_density'] = 0
    
    # 出血分割特征
    try:
        sys.path.insert(0, os.path.join(project_root, "Haemorage Segmentation"))
        import model as haem_model
        sys.path.pop(0)
        
        model_path = os.path.join(project_root, "Haemorage Segmentation", "checkpoint.pth")
        if os.path.exists(model_path):
            model, device = load_segmentation_model(model_path, haem_model)
            _, haem_mask = segment_image(image_path, model, device)
            # 计算出血区域占比
            haem_ratio = np.sum(haem_mask) / haem_mask.size
            features['haem_ratio'] = haem_ratio
        else:
            features['haem_ratio'] = 0
    except Exception as e:
        features['haem_ratio'] = 0
    
    # 硬渗出物分割特征
    try:
        sys.path.insert(0, os.path.join(project_root, "Hard Exudate Segmentation"))
        import model as exudate_model
        sys.path.pop(0)
        
        model_path = os.path.join(project_root, "Hard Exudate Segmentation", "checkpoint.pth")
        if os.path.exists(model_path):
            model, device = load_segmentation_model(model_path, exudate_model)
            _, exudate_mask = segment_image(image_path, model, device)
            # 计算硬渗出物区域占比
            exudate_ratio = np.sum(exudate_mask) / exudate_mask.size
            features['exudate_ratio'] = exudate_ratio
        else:
            features['exudate_ratio'] = 0
    except Exception as e:
        features['exudate_ratio'] = 0
    
    # 微动脉瘤分割特征
    try:
        sys.path.insert(0, os.path.join(project_root, "Microanuerism Segmentation"))
        import model as micro_model
        sys.path.pop(0)
        
        model_path = os.path.join(project_root, "Microanuerism Segmentation", "checkpoint.pth")
        if os.path.exists(model_path):
            model, device = load_segmentation_model(model_path, micro_model)
            _, micro_mask = segment_image(image_path, model, device)
            # 计算微动脉瘤区域占比
            micro_ratio = np.sum(micro_mask) / micro_mask.size
            features['micro_ratio'] = micro_ratio
        else:
            features['micro_ratio'] = 0
    except Exception as e:
        features['micro_ratio'] = 0
    
    return features

def classify_dr_with_segmentation_features(image_path, model_path):
    """使用分割特征的改进分类方法"""
    # 提取分割特征
    features = extract_features_from_segmentations(image_path)
    
    # 加载原始分类模型
    model = load_model(model_path, compile=False)
    
    # 读取并预处理图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
        
    img = cv2.resize(img, (64, 64))
    img = np.reshape(img, [1, 64, 64, 3])
    
    # 获取原始模型预测
    d = model.predict(img)
    base_prediction = d[0][0]
    base_prediction = round(base_prediction) - 2
    
    # 结合分割特征进行调整
    # 这里是一个简单的规则示例，实际应用中可以使用更复杂的融合方法
    adjusted_prediction = base_prediction
    
    # 如果检测到出血或微动脉瘤，提高严重程度评级
    if features.get('haem_ratio', 0) > 0.01 or features.get('micro_ratio', 0) > 0.01:
        adjusted_prediction = min(adjusted_prediction + 1, 4)
    
    # 如果血管密度异常低，可能表示严重病变
    if features.get('vessel_density', 0) < 0.1:
        adjusted_prediction = min(adjusted_prediction + 1, 4)
    
    # 解释结果
    if adjusted_prediction <= 0:
        result = 'No DIABETIC RETINOPATHY'  # 0
        severity = 0
    elif adjusted_prediction == 1:
        result = 'Mild DIABETIC RETINOPATHY'  # 1
        severity = 1
    elif adjusted_prediction == 2:
        result = 'Moderate DIABETIC RETINOPATHY'  # 2
        severity = 2
    elif adjusted_prediction == 3:
        result = 'Severe DIABETIC RETINOPATHY'  # 3
        severity = 3
    else:
        result = 'Proliferative DIABETIC RETINOPATHY'  # 4
        severity = 4
    
    return result, severity, features

def main():
    parser = argparse.ArgumentParser(description='改进的糖尿病视网膜病变分类工具')
    parser.add_argument('image_path', help='待分析的眼底图像路径')
    args = parser.parse_args()
    
    # 检查输入图像是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件 {args.image_path} 不存在")
        return
    
    try:
        result, severity, features = classify_dr_with_segmentation_features(
            args.image_path, 
            os.path.join(project_root, "Classification", "model", "model.h5")
        )
        
        print(f"分类结果: {result} (严重程度: {severity})")
        print("\n分割特征:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value:.4f}")
            
    except Exception as e:
        print(f"分类出错: {str(e)}")

if __name__ == "__main__":
    main()