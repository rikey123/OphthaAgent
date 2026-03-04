#!/usr/bin/env python3
"""
综合分析脚本，用于对指定的眼底图像进行全面的糖尿病视网膜病变分析
该脚本整合了项目中的所有分析工具：
1. 血管分割
2. 出血分割
3. 硬渗出物分割
4. 微动脉瘤分割
5. 视盘分割
6. 软渗出物分割
7. 整体分类诊断
"""

import os

# 彻底禁用该环境的 GPU 使用，避免 CUDA 驱动版本不匹配导致的 Graph 报错
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 减少 TensorFlow 日志干扰

import sys
import cv2
import numpy as np
import torch
# ... 后面保持不变 ...
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json
import time

import traceback

import traceback

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


def create_dir(path):
    """创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_segmentation_model(model_path, model_module):
    """加载分割模型"""
    # 在文件开头添加
    import sys
    #sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
    from device_config import get_device

    # 替换原来的设备设置
    device = get_device()
    model = model_module.build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    #print(f"Model loaded on device: {device}")
    return model, device


def segment_image(image_resized, model, device):
    """使用模型对图像进行分割"""
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
        pred_y = pred_y > 0.7
        pred_y = np.array(pred_y, dtype=np.uint8)

    return image_resized, pred_y


def classify_dr(image_path, model_path):
    """使用分类模型对DR进行分级"""
    # 加载模型
    model = load_model(model_path, compile=False)

    # 读取并预处理图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img = cv2.resize(img, (64, 64))
    img = np.reshape(img, [1, 64, 64, 3])

    # 预测
    d = model.predict(img)
    r = d[0][0]
    r = round(r) - 2

    # 解释结果
    if r <= 0:
        result = 'No DIABETIC RETINOPATHY'  # 0
        severity = 0
    elif r == 1:
        result = 'Mild DIABETIC RETINOPATHY'  # 1
        severity = 1
    elif r == 2:
        result = 'Moderate DIABETIC RETINOPATHY'  # 2
        severity = 2
    elif r == 3:
        result = 'Severe DIABETIC RETINOPATHY'  # 3
        severity = 3
    else:
        result = 'Proliferative DIABETIC RETINOPATHY'  # 4
        severity = 4

    return result, severity


def mask_parse(mask):
    """解析掩码图像"""
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def calculate_vessel_density(mask):
    """计算血管密度"""
    # 血管密度 = 血管区域像素数 / 总像素数
    vessel_pixels = np.sum(mask)
    total_pixels = mask.size
    density = vessel_pixels / total_pixels
    return density


def calculate_lesion_ratio(mask):
    """计算病变区域占比"""
    # 病变区域占比 = 病变区域像素数 / 总像素数
    lesion_pixels = np.sum(mask)
    total_pixels = mask.size
    ratio = lesion_pixels / total_pixels
    return ratio


def calculate_microaneurysm_count(mask):
    """计算微动脉瘤数量"""
    # 使用轮廓检测计算微动脉瘤数量
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤掉过小的区域（可能是噪声）
    min_area = 5  # 最小面积阈值
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            count += 1
    return count


def calculate_hemorrhage_size(mask):
    """计算出血区域总大小"""
    # 出血区域总大小 = 所有出血区域面积之和
    total_area = np.sum(mask)
    return total_area


def calculate_exudate_distribution(mask):
    """计算渗出物分布特征"""
    # 计算渗出物区域的分布特征
    if np.sum(mask) == 0:
        return 0, 0  # 如果没有渗出物，返回0

    # 计算渗出物区域的紧密度和分散度
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算所有轮廓的总面积
    total_area = sum(cv2.contourArea(contour) for contour in contours)

    # 计算所有轮廓的边界框总面积
    bounding_boxes_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes_area += w * h

    # 分布指数 = 轮廓总面积 / 边界框面积（越接近1表示越集中）
    if bounding_boxes_area > 0:
        distribution_index = total_area / bounding_boxes_area
    else:
        distribution_index = 0

    # 平均面积
    avg_area = total_area / len(contours) if contours else 0

    return distribution_index, avg_area


def analyze_blood_vessels(image_resized):
    """分析血管分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Blood Vessel Segmentation", "UNET"))
        import model as vessel_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Blood Vessel Segmentation", "UNET", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}
        model, device = load_segmentation_model(model_path, vessel_model)
        image, pred_mask = segment_image(image_resized, model, device)

        # 计算临床指标
        vessel_density = calculate_vessel_density(pred_mask)
        clinical_metrics = {
            "血管密度": float(vessel_density)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "血管分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"血管分割出错: {str(e)}"
        traceback.print_exc() # 打印完整的堆栈信息
        return None, error_msg, {}


def analyze_haemorages(image_resized):
    """分析出血分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Haemorage Segmentation"))
        import model as haem_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Haemorage Segmentation", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}

        model, device = load_segmentation_model(model_path, haem_model)
        image, pred_mask = segment_image(image_resized, model, device)

        # 计算临床指标
        hemorrhage_ratio = calculate_lesion_ratio(pred_mask)
        hemorrhage_size = calculate_hemorrhage_size(pred_mask)
        clinical_metrics = {
            "出血区域占比": float(hemorrhage_ratio),  # 确保转换为可JSON序列化的类型
            "出血区域总大小": float(hemorrhage_size)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "出血分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"出血分割出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, {}


def analyze_hard_exudates(image_resized):
    """分析硬渗出物分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Hard Exudate Segmentation"))
        import model as exudate_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Hard Exudate Segmentation", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}

        model, device = load_segmentation_model(model_path, exudate_model)
        image, pred_mask = segment_image(image_resized, model, device)

        # 计算临床指标
        exudate_ratio = calculate_lesion_ratio(pred_mask)
        distribution_index, avg_area = calculate_exudate_distribution(pred_mask)
        clinical_metrics = {
            "硬渗出物区域占比": float(exudate_ratio),  # 确保转换为可JSON序列化的类型
            "硬渗出物分布指数": float(distribution_index),  # 确保转换为可JSON序列化的类型
            "硬渗出物平均面积": float(avg_area)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "硬渗出物分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"硬渗出物分割出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, {}


def analyze_microaneurysms(image_resized):
    """分析微动脉瘤分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Microanuerism Segmentation"))
        import model as micro_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Microanuerism Segmentation", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}

        model, device = load_segmentation_model(model_path, micro_model)
        image, pred_mask = segment_image(image_resized, model, device)

        # 计算临床指标
        microaneurysm_count = calculate_microaneurysm_count(pred_mask)
        microaneurysm_ratio = calculate_lesion_ratio(pred_mask)
        clinical_metrics = {
            "微动脉瘤数量": int(microaneurysm_count),  # 确保转换为可JSON序列化的类型
            "微动脉瘤区域占比": float(microaneurysm_ratio)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "微动脉瘤分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"微动脉瘤分割出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, {}


def analyze_optical_disc(image_resized):
    """分析视盘分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Optical Disc Segmentation"))
        import model as disc_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Optical Disc Segmentation", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}
        model, device = load_segmentation_model(model_path, disc_model)
        image, pred_mask = segment_image(image_resized, model, device)
        # 计算临床指标
        disc_area_ratio = calculate_lesion_ratio(pred_mask)
        clinical_metrics = {
            "视盘区域占比": float(disc_area_ratio)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "视盘分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"视盘分割出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, {}


def analyze_soft_exudates(image_resized):
    """分析软渗出物分割"""
    try:
        # 动态导入模块
        sys.path.insert(0, os.path.join(project_root, "Soft Exudate Segmentation"))
        import model as sex_model
        sys.path.pop(0)

        model_path = os.path.join(project_root, "Soft Exudate Segmentation", "checkpoint.pth")
        if not os.path.exists(model_path):
            return None, "模型文件不存在，请先下载模型", {}

        model, device = load_segmentation_model(model_path, sex_model)
        image, pred_mask = segment_image(image_resized, model, device)

        # 计算临床指标
        soft_exudate_ratio = calculate_lesion_ratio(pred_mask)
        distribution_index, avg_area = calculate_exudate_distribution(pred_mask)
        clinical_metrics = {
            "软渗出物区域占比": float(soft_exudate_ratio),  # 确保转换为可JSON序列化的类型
            "软渗出物分布指数": float(distribution_index),  # 确保转换为可JSON序列化的类型
            "软渗出物平均面积": float(avg_area)  # 确保转换为可JSON序列化的类型
        }

        # 创建可视化结果
        pred_mask_vis = mask_parse(pred_mask)
        # 修复：确保原始图像不被修改，只对分割结果进行必要的处理
        original_image_copy = image.copy()
        separator = np.ones((512, 10, 3), dtype=np.uint8) * 128
        pred_mask_vis_uint8 = (pred_mask_vis * 255).astype(np.uint8)
        result_img = np.concatenate([original_image_copy, separator, pred_mask_vis_uint8], axis=1)
        return result_img, "软渗出物分割完成", clinical_metrics
    except Exception as e:
        error_msg = f"软渗出物分割出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, {}


def main():
    parser = argparse.ArgumentParser(description='糖尿病视网膜病变综合分析工具')
    parser.add_argument('image_path', help='待分析的眼底图像路径')
    parser.add_argument('-o', '--output', default='analysis_results', help='结果保存目录')
    args = parser.parse_args()

    start_time_total = time.time()

    # 检查输入图像是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件 {args.image_path} 不存在")
        return

    # 一次性读取和预处理图像
    try:
        original_image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            raise ValueError(f"无法读取图像: {args.image_path}")
        
        # 预处理为模型输入尺寸
        input_image_resized = cv2.resize(original_image, (512, 512))

    except Exception as e:
        print(f"错误：处理输入图像时失败 - {e}")
        return

    # 创建结果目录
    create_dir(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output, f"analysis_{timestamp}")
    create_dir(result_dir)

    print(f"开始分析图像: {args.image_path}")
    print(f"结果将保存到: {result_dir}")

    # 存储所有结果
    results = {}
    all_clinical_metrics = {}
    saved_files = {}  # 用于记录保存的文件路径

    # 1. 血管分割
    vessel_result, vessel_msg, vessel_metrics = analyze_blood_vessels(input_image_resized)
    results['blood_vessels'] = (vessel_result, vessel_msg)
    all_clinical_metrics.update(vessel_metrics)

    # 2. 出血分割
    haem_result, haem_msg, haem_metrics = analyze_haemorages(input_image_resized)
    results['haemorages'] = (haem_result, haem_msg)
    all_clinical_metrics.update(haem_metrics)

    # 3. 硬渗出物分割
    exudate_result, exudate_msg, exudate_metrics = analyze_hard_exudates(input_image_resized)
    results['hard_exudates'] = (exudate_result, exudate_msg)
    all_clinical_metrics.update(exudate_metrics)

    # 4. 微动脉瘤分割
    micro_result, micro_msg, micro_metrics = analyze_microaneurysms(input_image_resized)
    results['microaneurysms'] = (micro_result, micro_msg)
    all_clinical_metrics.update(micro_metrics)

    # 5. 视盘分割

    disc_result, disc_msg, disc_metrics = analyze_optical_disc(input_image_resized)
    results['optical_disc'] = (disc_result, disc_msg)
    all_clinical_metrics.update(disc_metrics)

    # 6. 软渗出物分割
    sex_result, sex_msg, sex_metrics = analyze_soft_exudates(input_image_resized)
    results['soft_exudates'] = (sex_result, sex_msg)
    all_clinical_metrics.update(sex_metrics)

    # 7. DR整体分类
    #print("\n7. 正在进行DR整体分类...")
    #classification_result = None
    #severity = None
    # try:
    #     classification_result, severity = classify_dr(args.image_path,
    #                                                   os.path.join(project_root, "Classification", "model", "model.h5"))
    #     results['classification'] = (classification_result, severity)
    #     print(f"分类结果: {classification_result} (严重程度: {severity})")
    # except Exception as e:
    #     results['classification'] = (None, f"分类出错: {str(e)}")
    #     print(f"分类出错: {str(e)}")

    # 保存结果
    for key, (result_img, msg) in results.items():
        if key != 'classification' and result_img is not None:
            output_path = os.path.join(result_dir, f"{key}_analysis.png")
            # 修复：使用cv2.imwrite而不是plt.imsave来保持颜色准确性
            cv2.imwrite(output_path, result_img)
            saved_files[key] = output_path  # 记录保存的文件路径

    # # 生成报告
    # report_path = os.path.join(result_dir, "analysis_report.txt")
    # with open(report_path, 'w', encoding='utf-8') as f:
    #     f.write(f"糖尿病视网膜病变综合分析报告\n")
    #     f.write(f"=========================\n")
    #     f.write(f"输入图像: {args.image_path}\n\n")

    #     f.write(f"1. 血管分割: {results['blood_vessels'][1]}\n")
    #     f.write(f"2. 出血分割: {results['haemorages'][1]}\n")
    #     f.write(f"3. 硬渗出物分割: {results['hard_exudates'][1]}\n")
    #     f.write(f"4. 微动脉瘤分割: {results['microaneurysms'][1]}\n")
    #     f.write(f"5. 视盘分割: {results['optical_disc'][1]}\n")
    #     f.write(f"6. 软渗出物分割: {results['soft_exudates'][1]}\n")
    #     # if results['classification'][0]:
    #     #     f.write(f"7. DR整体分类: {results['classification'][0]}\n")
    #     # else:
    #     #     f.write(f"7. DR整体分类: {results['classification'][1]}\n")

    # 打印临床指标
    print("\n量化分析指标:")
    print("=" * 30)
    for metric_name, value in all_clinical_metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.6f}")
        else:
            print(f"{metric_name}: {value}")

    # 构建完整的分析结果字典
    analysis_results = {
        "status": "success",
        "输入图像": os.path.abspath(args.image_path),
        "结果目录": os.path.abspath(result_dir),
        "分析结果": {
            "血管分割": {
                "状态": results['blood_vessels'][1],
                "指标": vessel_metrics,
                "结果文件": os.path.abspath(saved_files['blood_vessels']) if 'blood_vessels' in saved_files else None
            },
            "出血分割": {
                "状态": results['haemorages'][1],
                "指标": haem_metrics,
                "结果文件": os.path.abspath(saved_files['haemorages']) if 'haemorages' in saved_files else None
            },
            "硬渗出物分割": {
                "状态": results['hard_exudates'][1],
                "指标": exudate_metrics,
                "结果文件": os.path.abspath(saved_files['hard_exudates']) if 'hard_exudates' in saved_files else None
            },
            "微动脉瘤分割": {
                "状态": results['microaneurysms'][1],
                "指标": micro_metrics,
                "结果文件": os.path.abspath(saved_files['microaneurysms']) if 'microaneurysms' in saved_files else None
            },
            "视盘分割": {
                "状态": results['optical_disc'][1],
                "指标": disc_metrics,
                "结果文件": os.path.abspath(saved_files['optical_disc']) if 'optical_disc' in saved_files else None
            },
            "软渗出物分割": {
                "状态": results['soft_exudates'][1],
                "指标": sex_metrics,
                "结果文件": os.path.abspath(saved_files['soft_exudates']) if 'soft_exudates' in saved_files else None
            },
            "DR分级": {
                "结果": results.get('classification', (None, "未进行分类"))[0],
                "严重程度": results.get('classification', (None, "未进行分类"))[1]
            }
        },
        "量化分析指标": {k: float(v) if isinstance(v, (np.float32, np.float64)) else
        int(v) if isinstance(v, (np.int32, np.int64)) else v
                         for k, v in all_clinical_metrics.items()}
    }
    end_time_total = time.time()
    print(f"\n总分析耗时: {end_time_total - start_time_total:.2f} 秒")
    
    # print(f"\n分析完成！结果已保存到: {result_dir}")
    # print(f"详细报告: {report_path}")

    # 在控制台打印临床指标
    print("\n量化分析指标:")
    print("=" * 30)
    for metric_name, value in all_clinical_metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.6f}")
        else:
            print(f"{metric_name}: {value}")

    # 输出JSON结果到控制台
    print("\nJSON格式的完整分析结果:")
    print("=" * 30)
    print(json.dumps(analysis_results, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()