#!/usr/bin/env python3
"""
DME (糖尿病黄斑水肿) 风险评估工具

该工具整合以下分析：
1. 使用 AutoMorphalyzer 计算视杯视盘直径
2. 使用 Fovea_OD_localization 定位黄斑中心凹
3. 使用 DR-Detection-and-Clasification-System 计算硬性渗出物
4. 基于 ETDRS 标准计算 DME 风险等级

原理：
- 使用视盘直径 (DD) 作为参照尺度
- 计算硬性渗出物到黄斑中心凹的最小距离
- 根据距离判断 DME 风险等级：
  * 距离 < 1/3 DD (约500μm): 高风险 DME/CSME
  * 1/3 DD <= 距离 < 1 DD: 中等风险
  * 距离 >= 1 DD: 低风险
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加项目根目录到 Python 路径
sys.path.insert(0, PROJECT_ROOT)


def get_conda_env_python_path(env_name):
    """
    安全地获取指定 conda 环境的 Python 路径，自动过滤非 JSON 内容。
    """
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("Conda 未安装或不在 PATH 中。请确保 'conda' 命令可用。")

    if result.returncode != 0:
        raise RuntimeError(f"conda info 命令失败: {result.stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("conda info 无输出。")

    # 提取 JSON
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")

    json_str = stdout[start:end]

    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")

    # 获取环境路径
    root_prefix = info.get("root_prefix")
    envs = info.get("envs", [])

    if not root_prefix:
        raise RuntimeError("conda info 输出中缺少 root_prefix。")

    # 构建环境名到路径的映射
    env_paths = {"base": root_prefix}
    for env_path in envs:
        name = os.path.basename(env_path)
        env_paths[name] = env_path

    # 确定目标路径
    if not env_name or env_name == "base":
        target_path = root_prefix
    else:
        target_path = env_paths.get(env_name)
        if not target_path:
            available = list(env_paths.keys())
            raise RuntimeError(f"Conda 环境 '{env_name}' 不存在。可用环境: {available}")

    # 拼接 Python 可执行文件路径
    if sys.platform.startswith("win"):
        python_exec = os.path.join(target_path, "python.exe")
    else:
        python_exec = os.path.join(target_path, "bin", "python")

    if not os.path.isfile(python_exec):
        raise RuntimeError(f"在路径 {target_path} 中未找到 Python 可执行文件。")

    return python_exec


def run_python_script(script_path: str, args: List[str], env_name: Optional[str] = None, cwd: Optional[str] = None) -> Dict:
    """
    运行 Python 脚本并解析 JSON 输出

    Args:
        script_path: 脚本路径
        args: 脚本参数列表
        env_name: conda 环境名称（可选）
        cwd: 工作目录（可选）

    Returns:
        解析后的 JSON 结果
    """
    if env_name:
        # 获取 conda 环境的 Python 路径
        python_path = get_conda_env_python_path(env_name)
        cmd = [python_path, script_path] + args
    else:
        cmd = [sys.executable, script_path] + args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd
    )

    if result.returncode != 0:
        raise RuntimeError(f"脚本执行失败: {result.stderr}")

    # 提取 JSON 输出
    stdout = result.stdout.strip()
    start = stdout.find('{')
    end = stdout.rfind('}') + 1

    if start == -1 or end == 0:
        raise RuntimeError(f"未找到 JSON 输出: {stdout}")

    json_str = stdout[start:end]
    return json.loads(json_str)


def calculate_optic_disc_diameter(optic_disc_mask_path: str) -> float:
    """
    从视盘分割 mask 计算视盘直径（像素）

    Args:
        optic_disc_mask_path: 视盘分割 mask 图像路径

    Returns:
        视盘直径（像素）
    """
    # 读取视盘 mask
    mask = cv2.imread(optic_disc_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取视盘 mask: {optic_disc_mask_path}")

    # 二值化
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("视盘 mask 中未找到有效轮廓")

    # 获取最大轮廓（视盘区域）
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter = radius * 2

    # 也可以使用椭圆拟合获得长轴和短轴
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        # 使用主轴作为直径
        diameter = major_axis

    return float(diameter)


def extract_hard_exudate_coordinates(hard_exudate_mask_path: str) -> List[Tuple[float, float]]:
    """
    从硬性渗出物 mask 提取所有渗出物的中心坐标

    Args:
        hard_exudate_mask_path: 硬性渗出物分割 mask 路径

    Returns:
        渗出物坐标列表 [(x1, y1), (x2, y2), ...]
    """
    # 读取硬性渗出物 mask
    mask = cv2.imread(hard_exudate_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取硬性渗出物 mask: {hard_exudate_mask_path}")

    # 二值化
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 查找所有渗出物轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []
    for contour in contours:
        # 过滤掉过小的区域（噪声）
        if cv2.contourArea(contour) < 5:
            continue

        # 计算轮廓的质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            coordinates.append((cx, cy))

    return coordinates


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """计算两点之间的欧氏距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def run_fovea_localization(abs_image_path: str, output_dir: str) -> Dict:
    """
    步骤 1: 定位黄斑中心凹和视盘（独立任务）

    Returns:
        包含黄斑和视盘坐标的结果字典
    """
    print("[任务 1/3] 开始: 定位黄斑中心凹和视盘...")

    fovea_script = os.path.join(PROJECT_ROOT, "anatomical_segmentation",
                                 "Fovea_OD_localization_by_fundus_image_toolbox.py")
    temp_fovea_output = os.path.join(output_dir, "fovea_od_localization.png")
    abs_fovea_output = os.path.abspath(temp_fovea_output)

    fovea_result = run_python_script(
        fovea_script,
        [abs_image_path, abs_fovea_output],
        env_name="fundus_image_toolbox"
    )

    if fovea_result.get("status") != "success":
        raise RuntimeError(f"黄斑和视盘定位失败: {fovea_result}")

    print("[任务 1/3] 完成: 黄斑中心凹和视盘定位")
    return fovea_result


def run_automorphalyzer(abs_image_path: str, output_dir: str) -> Dict:
    """
    步骤 2: 使用 AutoMorphalyzer 计算视盘直径（独立任务）

    Returns:
        包含视盘直径信息的结果字典
    """
    print("[任务 2/3] 开始: 使用 AutoMorphalyzer 计算视盘直径...")

    automorph_script = "analyze_single.py"
    automorph_cwd = os.path.join(PROJECT_ROOT, "anatomical_segmentation", "AutoMorphalyzer")
    automorph_output_dir = os.path.join(output_dir, "automorph_results")

    # 创建输出目录
    os.makedirs(automorph_output_dir, exist_ok=True)

    # 调用 AutoMorphalyzer
    automorph_result = run_python_script(
        automorph_script,
        ["--input", abs_image_path, "--output", automorph_output_dir, "--device", "cuda:0"],
        env_name="automorph-env",
        cwd=automorph_cwd
    )

    # 从结果中提取视盘直径
    if automorph_result.get("images") and len(automorph_result["images"]) > 0:
        optic_disc_data = automorph_result["images"][0]["optic_disc"]
        disc_height = optic_disc_data.get("disc_height_px", -1)
        disc_width = optic_disc_data.get("disc_width_px", -1)

        if disc_height > 0 and disc_width > 0:
            disc_diameter = (disc_height + disc_width) / 2
            print(f"[任务 2/3] 完成: 视盘直径 {disc_diameter:.1f} 像素（高度: {disc_height:.1f}, 宽度: {disc_width:.1f}）")
            return {
                "disc_diameter": disc_diameter,
                "disc_height": disc_height,
                "disc_width": disc_width
            }
        else:
            raise ValueError("AutoMorphalyzer 未能检测到有效的视盘尺寸")
    else:
        raise ValueError("AutoMorphalyzer 返回结果为空")


def run_dr_detection(abs_image_path: str, output_dir: str) -> Dict:
    """
    步骤 3: 硬性渗出物检测（独立任务）

    Returns:
        包含硬性渗出物 mask 路径的结果字典
    """
    print("[任务 3/3] 开始: 检测硬性渗出物...")

    dr_script_name = "comprehensive_analysis.py"
    dr_cwd = os.path.join(PROJECT_ROOT, "Lesion_detection", "DR-Detection-and-Clasification-System")
    dr_output_dir = os.path.join(output_dir, "dr_analysis")

    # 创建输出目录
    os.makedirs(dr_output_dir, exist_ok=True)

    # 使用绝对路径和正确的工作目录
    abs_dr_output = os.path.abspath(dr_output_dir)

    dr_result = run_python_script(
        dr_script_name,
        [abs_image_path, "-o", abs_dr_output],
        env_name="DR-Detection-and-Clasification-System",
        cwd=dr_cwd
    )

    # 从 DR 分析结果中提取硬性渗出物 mask
    analysis_dirs = [d for d in os.listdir(dr_output_dir) if d.startswith("analysis_")]
    if not analysis_dirs:
        raise RuntimeError("DR 分析未生成结果目录")

    hard_exudate_mask_path = os.path.join(
        dr_output_dir,
        analysis_dirs[0],
        "hard_exudates_analysis.png"
    )

    if not os.path.exists(hard_exudate_mask_path):
        raise RuntimeError("硬性渗出物分割结果未找到")

    print("[任务 3/3] 完成: 硬性渗出物检测")
    return {"hard_exudate_mask_path": hard_exudate_mask_path}


def calculate_min_distance_to_fovea(
    fovea_coord: Tuple[float, float],
    exudate_coords: List[Tuple[float, float]]
) -> Optional[float]:
    """
    计算所有硬性渗出物到黄斑中心凹的最小距离

    Args:
        fovea_coord: 黄斑中心凹坐标 (x, y)
        exudate_coords: 渗出物坐标列表

    Returns:
        最小距离（像素），如果没有渗出物则返回 None
    """
    if not exudate_coords:
        return None

    distances = [calculate_distance(fovea_coord, exudate) for exudate in exudate_coords]
    return min(distances)


def assess_dme_risk(
    min_distance: Optional[float],
    disc_diameter: float
) -> Dict:
    """
    根据距离和视盘直径评估 DME 风险

    Args:
        min_distance: 渗出物到黄斑的最小距离（像素），None 表示无渗出物
        disc_diameter: 视盘直径（像素）

    Returns:
        风险评估结果字典
    """
    if min_distance is None:
        return {
            "risk_level": 0,
            "risk_category": "无风险/无硬性渗出",
            "description": "未检测到硬性渗出物，基于硬性渗出的 DME 风险极低",
            "clinical_note": "注意：此评估仅基于硬性渗出物。囊样水肿可能无明显渗出，建议结合 OCT 检查"
        }

    # 计算阈值
    threshold_csme = disc_diameter / 3  # 500μm 约等于 1/3 DD
    threshold_intermediate = disc_diameter  # 1 DD

    # 计算相对距离（以 DD 为单位）
    distance_in_dd = min_distance / disc_diameter

    if min_distance <= threshold_csme:
        return {
            "risk_level": 2,
            "risk_category": "高风险 DME/CSME",
            "description": f"硬性渗出物距离黄斑中心凹 {min_distance:.1f} 像素 (约 {distance_in_dd:.2f} DD)，小于 1/3 DD",
            "clinical_note": "符合临床显著黄斑水肿 (CSME) 标准中的'硬性渗出距黄斑中心 500μm 内'指征，强烈建议进行 OCT 检查和眼科会诊",
            "threshold_used": f"1/3 DD = {threshold_csme:.1f} 像素"
        }
    elif min_distance <= threshold_intermediate:
        return {
            "risk_level": 1,
            "risk_category": "中等风险",
            "description": f"硬性渗出物距离黄斑中心凹 {min_distance:.1f} 像素 (约 {distance_in_dd:.2f} DD)，在 1/3 DD 到 1 DD 之间",
            "clinical_note": "硬性渗出物位于黄斑区域，存在 DME 风险，建议定期随访和眼科评估",
            "threshold_used": f"1/3 DD = {threshold_csme:.1f} 像素, 1 DD = {threshold_intermediate:.1f} 像素"
        }
    else:
        return {
            "risk_level": 0,
            "risk_category": "低风险",
            "description": f"硬性渗出物距离黄斑中心凹 {min_distance:.1f} 像素 (约 {distance_in_dd:.2f} DD)，大于 1 DD",
            "clinical_note": "硬性渗出物远离黄斑区域，基于硬性渗出的 DME 风险较低，但仍建议定期检查",
            "threshold_used": f"1 DD = {threshold_intermediate:.1f} 像素"
        }


def visualize_dme_assessment(
    image_path: str,
    fovea_coord: Tuple[float, float],
    optic_disc_coord: Tuple[float, float],
    exudate_coords: List[Tuple[float, float]],
    disc_diameter: float,
    min_distance: Optional[float],
    risk_result: Dict,
    output_path: str
):
    """
    可视化 DME 评估结果

    Args:
        image_path: 原始眼底图像路径
        fovea_coord: 黄斑中心凹坐标
        optic_disc_coord: 视盘中心坐标
        exudate_coords: 渗出物坐标列表
        disc_diameter: 视盘直径
        min_distance: 最小距离
        risk_result: 风险评估结果
        output_path: 输出图像路径
    """
    # 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 调整图像大小以便显示
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 缩放坐标
        fovea_coord = (fovea_coord[0] * scale, fovea_coord[1] * scale)
        optic_disc_coord = (optic_disc_coord[0] * scale, optic_disc_coord[1] * scale)
        exudate_coords = [(x * scale, y * scale) for x, y in exudate_coords]
        disc_diameter *= scale
        if min_distance:
            min_distance *= scale

    # BGR 转 RGB（用于显示）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 绘制视盘（蓝色圆圈）
    cv2.circle(img_rgb, (int(optic_disc_coord[0]), int(optic_disc_coord[1])),
               int(disc_diameter / 2), (0, 0, 255), 2)
    cv2.putText(img_rgb, "OD", (int(optic_disc_coord[0]) - 20, int(optic_disc_coord[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 绘制黄斑中心凹（红色十字）
    fovea_x, fovea_y = int(fovea_coord[0]), int(fovea_coord[1])
    cv2.drawMarker(img_rgb, (fovea_x, fovea_y), (255, 0, 0),
                   cv2.MARKER_CROSS, 30, 3)
    cv2.putText(img_rgb, "Fovea", (fovea_x + 15, fovea_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 绘制距离圈（1/3 DD 和 1 DD）
    threshold_csme = int(disc_diameter / 3)
    threshold_intermediate = int(disc_diameter)
    cv2.circle(img_rgb, (fovea_x, fovea_y), threshold_csme, (255, 255, 0), 2)  # 黄色
    cv2.circle(img_rgb, (fovea_x, fovea_y), threshold_intermediate, (0, 255, 0), 2)  # 绿色

    # 绘制硬性渗出物（根据风险等级着色）
    risk_level = risk_result['risk_level']
    if risk_level == 2:
        exudate_color = (255, 0, 255)  # 品红色（高风险）
    elif risk_level == 1:
        exudate_color = (255, 165, 0)  # 橙色（中等风险）
    else:
        exudate_color = (128, 128, 128)  # 灰色（低风险）

    closest_exudate = None
    if min_distance and exudate_coords:
        # 找到最近的渗出物
        distances = [calculate_distance(fovea_coord, exudate) for exudate in exudate_coords]
        closest_idx = distances.index(min(distances))
        closest_exudate = exudate_coords[closest_idx]

    for i, (ex, ey) in enumerate(exudate_coords):
        if closest_exudate and (ex, ey) == closest_exudate:
            # 最近的渗出物用特殊标记
            cv2.circle(img_rgb, (int(ex), int(ey)), 8, exudate_color, -1)
            cv2.circle(img_rgb, (int(ex), int(ey)), 10, (255, 255, 255), 2)
            # 绘制连线
            cv2.line(img_rgb, (fovea_x, fovea_y), (int(ex), int(ey)), exudate_color, 2)
        else:
            cv2.circle(img_rgb, (int(ex), int(ey)), 5, exudate_color, -1)

    # 添加标注文本
    y_offset = 30
    cv2.putText(img_rgb, f"Risk: {risk_result['risk_category']}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(img_rgb, f"Min Distance: {min_distance:.1f}px" if min_distance else "No exudates",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(img_rgb, f"OD Diameter: {disc_diameter:.1f}px", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 保存结果
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)


def main(image_path: str, output_dir: str = None) -> Dict:
    """
    主函数：执行完整的 DME 风险评估流程（并发执行三个独立任务）

    Args:
        image_path: 输入眼底图像路径
        output_dir: 输出目录（可选）

    Returns:
        评估结果字典
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"图像文件不存在: {image_path}"}

    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "dme_assessment_results")
    os.makedirs(output_dir, exist_ok=True)

    # 使用绝对路径
    abs_image_path = os.path.abspath(image_path)

    print("=" * 80)
    print("DME 风险评估开始（并发执行模式）")
    print("=" * 80)
    print(f"输入图像: {abs_image_path}")
    print(f"输出目录: {output_dir}\n")

    try:
        # 并发执行三个独立任务
        print("正在并发执行 3 个独立任务...")
        print("-" * 80)

        fovea_result = None
        automorph_result = None
        dr_result = None
        errors = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交三个任务
            future_fovea = executor.submit(run_fovea_localization, abs_image_path, output_dir)
            future_automorph = executor.submit(run_automorphalyzer, abs_image_path, output_dir)
            future_dr = executor.submit(run_dr_detection, abs_image_path, output_dir)

            # 等待任务完成并收集结果
            for future in as_completed([future_fovea, future_automorph, future_dr]):
                try:
                    result = future.result()
                    if future == future_fovea:
                        fovea_result = result
                    elif future == future_automorph:
                        automorph_result = result
                    elif future == future_dr:
                        dr_result = result
                except Exception as e:
                    if future == future_fovea:
                        errors['fovea'] = str(e)
                    elif future == future_automorph:
                        errors['automorph'] = str(e)
                    elif future == future_dr:
                        errors['dr'] = str(e)

        print("-" * 80)
        print("所有任务执行完成\n")

        # 检查关键任务是否成功
        if 'fovea' in errors:
            return {"status": "error", "message": f"黄斑和视盘定位失败: {errors['fovea']}"}

        if 'dr' in errors:
            return {"status": "error", "message": f"硬性渗出物检测失败: {errors['dr']}"}

        # 提取黄斑和视盘坐标
        fovea_x = fovea_result["result"]["coordinates"]["fovea"]["x"]
        fovea_y = fovea_result["result"]["coordinates"]["fovea"]["y"]
        od_x = fovea_result["result"]["coordinates"]["optic_disc"]["x"]
        od_y = fovea_result["result"]["coordinates"]["optic_disc"]["y"]

        fovea_coord = (fovea_x, fovea_y)
        optic_disc_coord = (od_x, od_y)

        print(f"✓ 黄斑中心凹: ({fovea_x:.1f}, {fovea_y:.1f})")
        print(f"✓ 视盘中心: ({od_x:.1f}, {od_y:.1f})")

        # 处理 AutoMorphalyzer 结果（可能失败，需要回退）
        if 'automorph' in errors:
            print(f"⚠ AutoMorphalyzer 失败: {errors['automorph']}")
            print(f"⚠ 使用 OD-Fovea 距离估算视盘直径...")
            od_fovea_distance = calculate_distance(optic_disc_coord, fovea_coord)
            disc_diameter = od_fovea_distance / 2.5
            print(f"✓ 估算视盘直径: {disc_diameter:.1f} 像素（基于 OD-Fovea 距离）")
        else:
            disc_diameter = automorph_result["disc_diameter"]
            print(f"✓ 视盘直径: {disc_diameter:.1f} 像素")

        # 提取硬性渗出物坐标
        hard_exudate_mask_path = dr_result["hard_exudate_mask_path"]
        exudate_coords = extract_hard_exudate_coordinates(hard_exudate_mask_path)
        print(f"✓ 检测到 {len(exudate_coords)} 个硬性渗出物\n")

        # 计算 DME 风险
        print("=" * 80)
        print("计算 DME 风险...")
        print("=" * 80)
        min_distance = calculate_min_distance_to_fovea(fovea_coord, exudate_coords)
        risk_result = assess_dme_risk(min_distance, disc_diameter)

        print(f"\n风险等级: {risk_result['risk_category']}")
        print(f"描述: {risk_result['description']}")
        print(f"临床建议: {risk_result['clinical_note']}\n")

        # 生成可视化结果
        print("生成可视化结果...")
        visualization_path = os.path.join(output_dir, "dme_risk_assessment.png")
        visualize_dme_assessment(
            abs_image_path,
            fovea_coord,
            optic_disc_coord,
            exudate_coords,
            disc_diameter,
            min_distance,
            risk_result,
            visualization_path
        )
        print(f"✓ 可视化结果已保存: {visualization_path}\n")

        # 构建完整结果
        result = {
            "status": "success",
            "result": {
                "anatomical_features": {
                    "fovea_coordinates": {"x": fovea_x, "y": fovea_y},
                    "optic_disc_coordinates": {"x": od_x, "y": od_y},
                    "optic_disc_diameter_pixels": disc_diameter
                },
                "hard_exudates": {
                    "count": len(exudate_coords),
                    "coordinates": [{"x": x, "y": y} for x, y in exudate_coords],
                    "min_distance_to_fovea_pixels": min_distance if min_distance else None
                },
                "dme_risk_assessment": risk_result,
                "visualization_path": os.path.abspath(visualization_path),
                "output_directory": os.path.abspath(output_dir)
            }
        }

        print("=" * 80)
        print("DME 风险评估完成！")
        print("=" * 80)

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"DME 风险评估过程中出错: {str(e)}",
            "traceback": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DME (糖尿病黄斑水肿) 风险评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dme_risk_assessment.py input_image.jpg
  python dme_risk_assessment.py input_image.jpg -o ./results
        """
    )
    parser.add_argument("image_path", type=str, help="输入眼底图像路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出目录（默认: dme_assessment_results）")

    args = parser.parse_args()

    result = main(args.image_path, args.output)

    print("\n" + "=" * 60)
    print("DME 风险评估结果（JSON 格式）:")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
