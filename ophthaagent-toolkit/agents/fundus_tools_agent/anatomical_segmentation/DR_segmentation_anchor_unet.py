#!/usr/bin/env pyth#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETDRS (Early Treatment Diabetic Retinopathy Study) 网格分析工具

功能:
1.  自动调用外部工具，获取黄斑/视盘坐标、视盘直径和病灶位置。
2.  根据获取的参数，计算并生成标准的ETDRS网格。
3.  在眼底图像上绘制ETDRS网格，并对9个区域进行标注。
4.  判断每个病灶具体属于ETDRS网格的哪个区域。
5.  输出一个包含网格参数、病灶分布和可视化图像路径的JSON结果。

ETDRS 网格定义:
- 中心区域 (C): 半径为 0.33 DD 的圆形区域，以黄斑中心凹为中心。
- 内环 (Inner Ring): 位于半径 1 DD 的环状区域。
- 外环 (Outer Ring): 位于半径 2 DD 的环状区域。
- 内环和外环均被划分为四个象限: 上(Superior)、下(Inferior)、鼻侧(Nasal)、颞侧(Temporal)。
- 颞侧-鼻侧轴线由黄斑中心凹指向视盘中心。

9个区域命名:
- C (Central): 中心凹区域 (半径 0.5mm)
- SI (Superior Inner): 内环上方区域
- II (Inferior Inner): 内环下方区域
- NI (Nasal Inner): 内环鼻侧区域
- TI (Temporal Inner): 内环颞侧区域
- SO (Superior Outer): 外环上方区域
- IO (Inferior Outer): 外环下方区域
- NO (Nasal Outer): 外环鼻侧区域
- TO (Temporal Outer): 外环颞侧区域
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import math
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# 配置日志
logger = logging.getLogger(__name__)


# --- 自动化依赖注入部分 (参考 dme_risk_assessment.py) ---


# --- fundus_prep.py 中的图像裁剪函数 ---

def imread(file_path, c=None):
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)

    if im is None:
        raise Exception(f'Can not read image: {file_path}')

    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imwrite(file_path, image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)


def fold_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_mask_BZ(img):
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threshold = np.mean(gray_img) / 3 - 5
    _, mask = cv2.threshold(gray_img, max(5, threshold), 1, cv2.THRESH_BINARY)

    nn_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    new_mask = (1 - mask).astype(np.uint8)
    cv2.floodFill(new_mask, nn_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1] - 1, new_mask.shape[0] - 1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask


def _get_center_by_edge(mask):
    center = [0, 0]
    x = mask.sum(axis=1)
    center[0] = np.where(x > x.max() * 0.95)[0].mean()
    x = mask.sum(axis=0)
    center[1] = np.where(x > x.max() * 0.95)[0].mean()
    return center


def _get_radius_by_mask_center(mask, center):
    mask = mask.astype(np.uint8)
    ksize = max(mask.shape[1] // 400 * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    index = np.where(mask > 0)
    d_int = np.sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
    b_count = np.bincount(np.ceil(d_int).astype(int))
    radius = np.where(b_count > b_count.max() * 0.995)[0].max()
    return radius


def _get_circle_by_center_bbox(shape, center, bbox, radius):
    center_mask = np.zeros(shape=shape).astype('uint8')
    center_tmp = (int(center[0]), int(center[1]))
    center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)
    return center_mask


def get_mask(img):
    if img.ndim == 3:
        g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img = img.copy()
    else:
        raise Exception('image dim is not 1 or 3')
    h, w = g_img.shape
    shape = g_img.shape[0:2]
    tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask = get_mask_BZ(tg_img)
    center = _get_center_by_edge(tmp_mask)
    radius = _get_radius_by_mask_center(tmp_mask, center)

    center = [center[0], center[1]]
    radius = int(radius)
    s_h = max(0, int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h - s_h, 2 * radius), min(w - s_w, 2 * radius))
    tmp_mask = _get_circle_by_center_bbox(shape, center, bbox, radius)
    return tmp_mask, bbox, center, radius


def mask_image(img, mask):
    img[mask <= 0, ...] = 0
    return img


def remove_back_area(img, bbox=None, border=None):
    image = img
    if border is None:
        border = np.array((bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3], img.shape[0], img.shape[1]),
                          dtype=int)
    image = image[border[0]:border[1], border[2]:border[3], ...]
    return image, border


def supplemental_black_area(img, border=None):
    image = img
    h, v = img.shape[0:2]
    max_l = max(h, v)
    if image.ndim > 2:
        new_image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
    else:
        new_image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)

    top = (max_l - h) // 2
    left = (max_l - v) // 2

    new_image[top:top + h, left:left + v, ...] = image

    if border is None:
        border = (top, top + h, left, left + v, max_l)

    return new_image, border


def process_without_gb(img, label):
    radius_list, centre_list_w, centre_list_h = [], [], []
    borders = []
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img, bbox=bbox)
    mask, _ = remove_back_area(mask, border=r_border)
    if label is not None:
        label, _ = remove_back_area(label, bbox=bbox)
    borders.append(r_border)
    r_img, sup_border = supplemental_black_area(r_img)
    if label is not None:
        label, _ = supplemental_black_area(label)
    mask, _ = supplemental_black_area(mask, border=sup_border)
    borders.append(sup_border)

    radius_list.append(radius)
    centre_list_w.append(int(center[0]))
    centre_list_h.append(int(center[1]))
    return r_img, borders, (mask * 255).astype(np.uint8), label, radius_list, centre_list_w, centre_list_h


# --- 结束 fundus_prep.py 函数 ---


# 使用绝对路径添加 device_config.py 所在的目录到 sys.path
DEVICE_CONFIG_DIR = '<ANON_ABS_PATH>'
sys.path.insert(0, DEVICE_CONFIG_DIR)
from device_config import get_device
sys.path.pop(0)


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

    # 输出 stderr 中的进度信息
    if result.stderr and result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"脚本执行失败: {result.stderr}")

    # 提取 JSON 输出
    stdout = result.stdout.strip()
    
    # 新的 JSON 解析策略：使用 json.JSONDecoder 来解析所有 JSON 对象
    from json.decoder import JSONDecoder
    
    json_objects = []
    decoder = JSONDecoder()
    idx = 0
    while idx < len(stdout):
        try:
            # 跳过空白字符
            while idx < len(stdout) and stdout[idx].isspace():
                idx += 1
            
            if idx >= len(stdout):
                break
            
            # 尝试解析 JSON 对象
            obj, end = decoder.raw_decode(stdout[idx:])
            if isinstance(obj, dict):
                json_objects.append(obj)
                logger.info(f"  [DEBUG] 成功解析 JSON 对象 #{len(json_objects)}: {list(obj.keys())}")
            idx += end
        except json.JSONDecodeError:
            # 如果解析失败，跳过当前字符并继续
            idx += 1
    
    if not json_objects:
        print(f"--- DEBUG: Failed to find valid JSON in stdout. ---", file=sys.stderr)
        print(f"--- DEBUG: stdout 总长度: {len(stdout)} ---", file=sys.stderr)
        left_brace_count = stdout.count("{")
        right_brace_count = stdout.count("}")
        print(f"--- DEBUG: stdout 中 左大括号 的数量: {left_brace_count} ---", file=sys.stderr)
        print(f"--- DEBUG: stdout 中 右大括号 的数量: {right_brace_count} ---", file=sys.stderr)
        print(f"--- DEBUG: 完整 stdout 内容如下 ---", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(f"--- DEBUG: stdout 内容结束 ---", file=sys.stderr)
        raise RuntimeError(f"未找到 JSON 输出")
    
    # 返回最后一个有效的 JSON 对象（应该是最终结果）
    return json_objects[-1]


def find_latest_lesion_masks(output_dir: str, lesion_dir: Optional[str] = None) -> Dict[str, str]:
    """
    在指定的输出目录及其父目录中查找最新的病灶分割结果。

    该函数会导航到 'dr_analysis_unet' 子目录，找到时间戳最新的 'analysis_*' 文件夹，
    并返回其中四种主要病灶掩码的文件路径。

    Args:
        output_dir (str): 基础输出目录 (通常是 'intermediate_files_dir')。
        lesion_dir (Optional[str]): 直接提供的病灶分析目录路径（包含 dr_analysis_unet），如果提供则直接使用。

    Returns:
        Dict[str, str]: 一个字典，将病灶类型映射到其对应的掩码文件绝对路径。
    """
    print("[TASK 3/3] Locating latest lesion masks from file system...")

    # 如果直接提供了 lesion_dir，优先使用
    if lesion_dir and os.path.isdir(lesion_dir):
        dr_analysis_dir = lesion_dir
        print(f"Using provided lesion directory: {dr_analysis_dir}")
    else:
        # 搜索策略：从近到远
        search_dirs = [
            # 1. 当前工具的输出目录（向上两级）
            os.path.dirname(os.path.dirname(output_dir)),
            # 2. 当前工具的输出目录（向上三级）
            os.path.dirname(os.path.dirname(os.path.dirname(output_dir))),
            # 3. tool_outputs 根目录
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))),
        ]

        dr_analysis_dir = None
        for search_dir in search_dirs:
            candidate_dir = os.path.join(search_dir, "dr_analysis_unet")
            if os.path.isdir(candidate_dir):
                dr_analysis_dir = candidate_dir
                print(f"Found dr_analysis_unet at: {dr_analysis_dir}")
                break

    if not dr_analysis_dir:
        # 如果没找到，尝试查找所有以 lesions_ 开头的目录中的 dr_analysis_unet
        # 此时 search_dirs[2] 应该对应 tool_outputs 根目录
        tool_outputs_root = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))
        for item in os.listdir(tool_outputs_root):
            if "_lesions_" in item and os.path.isdir(os.path.join(tool_outputs_root, item)):
                candidate_dir = os.path.join(tool_outputs_root, item, "dr_analysis_unet")
                if os.path.isdir(candidate_dir):
                    dr_analysis_dir = candidate_dir
                    print(f"Found dr_analysis_unet in lesions directory: {dr_analysis_dir}")
                    break

        if not dr_analysis_dir:
            raise FileNotFoundError(f"Lesion analysis directory 'dr_analysis_unet' not found in any expected location")

    # 查找所有 'analysis_*' 文件夹
    try:
        analysis_dirs = [
            d for d in os.listdir(dr_analysis_dir)
            if d.startswith("analysis_") and os.path.isdir(os.path.join(dr_analysis_dir, d))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Lesion analysis directory not found or is not a directory: {dr_analysis_dir}")

    if not analysis_dirs:
        raise FileNotFoundError(f"No 'analysis_*' directories found in {dr_analysis_dir}")

    # 按名称（时间戳）排序，找到最新的目录
    analysis_dirs.sort(reverse=True)
    latest_analysis_dir = os.path.join(dr_analysis_dir, analysis_dirs[0])
    print(f"Found latest analysis directory: {latest_analysis_dir}")

    # 定义需要查找的病灶掩码文件名
    lesion_mask_files = {
        "hemorrhages": "haemorages_analysis.png",
        "hard_exudates": "hard_exudates_analysis.png",
        "microaneurysms": "microaneurysms_analysis.png",
        "soft_exudates": "soft_exudates_analysis.png"
    }

    # 构建并验证每个掩码文件的路径
    lesion_mask_paths = {}
    for lesion_type, mask_filename in lesion_mask_files.items():
        mask_path = os.path.join(latest_analysis_dir, mask_filename)
        if os.path.exists(mask_path):
            lesion_mask_paths[lesion_type] = os.path.abspath(mask_path)
        else:
            print(f"Warning: Lesion mask not found for '{lesion_type}' at {mask_path}")

    print(f"[TASK 3/3] Completed. Found {len(lesion_mask_paths)} lesion mask files.")
    return lesion_mask_paths


def run_fovea_od_localization(abs_image_path: str, output_dir: str) -> Dict:
    """
    步骤 2: 使用 Fovea_OD_localization_by_fundus_image_toolbox 计算 视盘 和 中心凹坐标

    Returns:
        包含两个坐标的结果字典
    """
    print("[TASK 1/3] Locating Fovea and Optic Disc...", file=sys.stderr)

    fol_script = "Fovea_OD_localization_by_fundus_image_toolbox.py"
    # 修复路径：从当前文件位置计算，确保路径正确
    fol_cwd = os.path.dirname(os.path.abspath(__file__))

    # 调用 Fovea_OD_localization_by_fundus_image_toolbox
    fol_result = run_python_script(
        fol_script,
        [abs_image_path, "no_save", "--device", "cpu"],
        env_name="fundus_image_toolbox",
        cwd=fol_cwd
    )

    if fol_result.get("status") != "success":
        raise RuntimeError(f"Fovea/OD localization failed: {fol_result}")
    # 将调试输出重定向到 stderr，以避免污染 stdout
    print(f"Fovea and OD detection result: {fol_result['result']['coordinates']}", file=sys.stderr)
    return fol_result['result']['coordinates']


def run_disc_diameter_estimation(abs_image_path: str, output_dir: str) -> float:
    """
    步骤 3: 使用 AutoMorphalyzer 计算视盘直径（独立任务）

    Returns:
        包含视盘直径信息的结果字典
    """
    print("[任务 2/3] 开始: 使用 AutoMorphalyzer 计算视盘直径...")
    automorph_script = "analyze_single.py"
    # 修复路径：从当前文件位置计算
    current_dir = os.path.dirname(os.path.abspath(__file__))
    automorph_cwd = os.path.join(current_dir, "AutoMorphalyzer")
    automorph_output_dir = os.path.join(output_dir, "automorph_results")

    # 创建输出目录
    os.makedirs(automorph_output_dir, exist_ok=True)

    # 调用 AutoMorphalyzer
    automorph_result = run_python_script(
        automorph_script,
        ["--input", abs_image_path, "--output", automorph_output_dir, "--device", "cpu"],
        env_name="automorph-env",
        cwd=automorph_cwd
    )

    # 从结果中提取视盘直径
    if automorph_result.get("images") and len(automorph_result["images"]) > 0:
        image_info = automorph_result["images"][0]
        optic_disc_data = image_info["optic_disc"]

        vessel_mask_path = image_info.get("vessel_mask_path")
        disc_mask_path = image_info.get("disc_mask_path")

        disc_height = optic_disc_data.get("disc_height_px", -1)
        disc_width = optic_disc_data.get("disc_width_px", -1)

        if disc_height > 0 and disc_width > 0:
            disc_diameter = (disc_height + disc_width) / 2
            print(f"[任务 2/3] 完成: 视盘直径 {disc_diameter:.1f} 像素（高度: {disc_height:.1f}, 宽度: {disc_width:.1f}）")
            return {
                "disc_diameter": disc_diameter,
                "disc_height": disc_height,
                "disc_width": disc_width,
                "vessel_mask_path": vessel_mask_path,
                "disc_mask_path": disc_mask_path
            }
        else:
            raise ValueError("AutoMorphalyzer 未能检测到有效的视盘尺寸")
    else:
        raise ValueError("AutoMorphalyzer 返回结果为空")


def run_lesion_detection(abs_image_path: str, output_dir: str) -> Dict[str, str]:
    """
    步骤 3: 定位病灶掩码文件（独立任务）
    不再将掩码转换为点，而是直接返回各类病灶掩码的路径。

    Returns:
        一个字典，将病灶类型映射到其对应的掩码文件路径。
    """
    print("[任务 3/3] 开始: 定位病灶掩码文件...")

    dr_script_name = "comprehensive_analysis.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 anatomical_segmentation 返回到 fundus_tools_agent 目录
    fundus_tools_agent_dir = os.path.dirname(current_dir)
    dr_cwd = os.path.join(fundus_tools_agent_dir, "Lesion_detection", "DR-Detection-and-Clasification-System")
    # 输出在dr_analysis目录下
    dr_output_dir = os.path.join(output_dir, "dr_analysis")
    os.makedirs(dr_output_dir, exist_ok=True)
    abs_dr_output = os.path.abspath(dr_output_dir)

    # 运行ddcs病灶检测脚本
    run_python_script(
        dr_script_name,
        [abs_image_path, "-o", abs_dr_output],
        env_name="ddcs",
        cwd=dr_cwd
    )

    # 在输出目录中找到分析结果文件夹
    analysis_dirs = [
        d for d in os.listdir(dr_output_dir)
        if d.startswith("analysis_") and os.path.isdir(os.path.join(dr_output_dir, d))
    ]
    if not analysis_dirs:
        raise RuntimeError("DR 分析未生成结果目录")

    # 按修改时间排序，选择最新的目录
    analysis_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(dr_output_dir, d)), reverse=True)
    analysis_dir = os.path.join(dr_output_dir, analysis_dirs[0])

    # 定义需要查找的病灶掩码
    lesion_mask_files = {
        "hard_exudates": "hard_exudates_analysis.png",
        "hemorrhages": "haemorages_analysis.png",
        "microaneurysms": "microaneurysms_analysis.png",
        "soft_exudates": "soft_exudates_analysis.png"
    }

    # 构建并验证每个掩码文件的路径, 只返回4类病灶的路径
    lesion_mask_paths = {}
    for lesion_type, mask_filename in lesion_mask_files.items():
        mask_path = os.path.join(analysis_dir, mask_filename)
        if os.path.exists(mask_path):
            lesion_mask_paths[lesion_type] = mask_path
            # 读取掩码图像并打印尺寸
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is not None:
                print(f"病灶 '{lesion_type}' 分割掩码尺寸: {mask_image.shape}")
            else:
                print(f"无法读取病灶 '{lesion_type}' 的掩码图像: {mask_path}")

    # 打印输入图像尺寸
    input_image = cv2.imread(abs_image_path)
    if input_image is not None:
        print(f"输入图像尺寸: {input_image.shape}")
    else:
        print(f"无法读取输入图像: {abs_image_path}")

    print(f"[任务 3/3] 完成. 找到 {len(lesion_mask_paths)} 个病灶掩码文件。")
    return lesion_mask_paths


def calculate_etdrs_grid_parameters(fovea_coord: Tuple[float, float], od_coord: Tuple[float, float], disc_diameter: float) -> Dict:
    """
    根据黄斑中心、视盘中心坐标和视盘直径，计算ETDRS网格的几何参数。
    """
    fovea_x, fovea_y = fovea_coord
    od_x, od_y = od_coord

    r_central = 0.33 * disc_diameter
    r_inner = 1.0 * disc_diameter
    r_outer = 2.0 * disc_diameter

    nasal_temporal_angle_rad = math.atan2(od_y - fovea_y, od_x - fovea_x)

    return {
        "center": fovea_coord,
        "disc_diameter": disc_diameter,
        "radii": {
            "central": r_central,
            "inner": r_inner,
            "outer": r_outer,
        },
        "axes_angles_rad": {
            "nasal": nasal_temporal_angle_rad % (2 * math.pi),
            "temporal": (nasal_temporal_angle_rad + math.pi) % (2 * math.pi),
            "superior": (nasal_temporal_angle_rad - math.pi / 2) % (2 * math.pi),
            "inferior": (nasal_temporal_angle_rad + math.pi / 2) % (2 * math.pi),
        }
    }


def draw_etdrs_grid(image: np.ndarray, grid_params: Dict, segmentation_data: Dict, lesion_mask_paths: Dict[str, str]) -> np.ndarray:
    """
    在眼底图上绘制ETDRS网格、解剖结构和各类病灶掩码。
    """
    vis_img = image.copy()
    DARKEN_ALPHA = 0.55
    vis_img = cv2.convertScaleAbs(vis_img, alpha=DARKEN_ALPHA, beta=0)
    h, w = vis_img.shape[:2]
    print(f"[DEBUG] Drawing on base image with resolution: {h}x{w}")

    transparent_overlay = np.zeros_like(vis_img, dtype=np.uint8)
    solid_overlay = np.zeros_like(vis_img, dtype=np.uint8)

    vessel_mask_path = segmentation_data.get("vessel_mask_path")
    if vessel_mask_path and os.path.exists(vessel_mask_path):
        vessel_mask = cv2.imread(vessel_mask_path, cv2.IMREAD_GRAYSCALE)
        if vessel_mask is not None:
            print(f"[DEBUG] Vessel mask original resolution: {vessel_mask.shape}", file=sys.stderr)
            vessel_mask = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            print(f"[DEBUG] Vessel mask resized to: {vessel_mask.shape}", file=sys.stderr)

            m = vessel_mask > 128  # bool mask

            # 你想要的“最终显示绿色”的目标色（OpenCV 是 BGR）
            vessel_color = np.array([50, 252, 50], dtype=np.float32)

            # 染色强度：0.25~0.55 常用；越大越绿、纹理越弱
            vessel_alpha = 0.25

            base = vis_img.astype(np.float32)
            base[m] = (1.0 - vessel_alpha) * base[m] + vessel_alpha * vessel_color
            vis_img = np.clip(base, 0, 255).astype(np.uint8)

    disc_mask_path = segmentation_data.get("disc_mask_path")
    if disc_mask_path and os.path.exists(disc_mask_path):
        disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
        if disc_mask is not None:
            print(f"[DEBUG] Disc mask original resolution: {disc_mask.shape}")
            disc_mask = cv2.resize(disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            print(f"[DEBUG] Disc mask resized to: {disc_mask.shape}")
            transparent_overlay[disc_mask > 128] = [225, 105, 65]

    lesion_colors = {
        "microaneurysms": (255, 0, 255),  # 
        "hard_exudates": (255, 255, 0),  # 
        "soft_exudates": (0, 255, 255),  # 
        "hemorrhages": (0, 0, 255),      # 
    }

    for lesion_type, mask_path in lesion_mask_paths.items():
        if os.path.exists(mask_path):
            lesion_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if lesion_mask is None:
                continue

            print(f"[DEBUG] Lesion '{lesion_type}' original mask resolution: {lesion_mask.shape}")

            if lesion_mask.shape[0] != h or lesion_mask.shape[1] != w:
                lesion_mask = cv2.resize(lesion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                print(f"[DEBUG] Lesion '{lesion_type}' resized to: {lesion_mask.shape}")

            # Binarize mask
            binary_mask = (lesion_mask > 128).astype(np.uint8)
            if not np.any(binary_mask):
                continue

            # 1. Dilate the mask to make it more visible
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # 2. Apply semi-transparent fill to the transparent_overlay
            color = lesion_colors.get(lesion_type, (255, 255, 255))
            transparent_overlay[dilated_mask > 0] = color

            # 3. Apply a brighter outline of the same color to the solid_overlay
            # Create a lighter/brighter color for the outline by adding 80 to each channel
            outline_color = tuple(min(c + 80, 255) for c in color)
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(solid_overlay, contours, -1, outline_color, 1)

    vis_img = cv2.addWeighted(vis_img, 1, transparent_overlay, 0.6, 0)

    solid_pixels = np.any(solid_overlay > [0, 0, 0], axis=-1)
    vis_img[solid_pixels] = solid_overlay[solid_pixels]

    center = tuple(map(int, grid_params["center"]))
    radii = grid_params["radii"]
    grid_color = (255, 0, 0)
    grid_thickness = 2

    cv2.circle(vis_img, center, int(radii["central"]), grid_color, grid_thickness)
    cv2.circle(vis_img, center, int(radii["inner"]), grid_color, grid_thickness)
    cv2.circle(vis_img, center, int(radii["outer"]), grid_color, grid_thickness)

    for angle_rad in grid_params["axes_angles_rad"].values():
        p1 = (int(center[0] + radii["central"] * math.cos(angle_rad)),
              int(center[1] + radii["central"] * math.sin(angle_rad)))
        p2 = (int(center[0] + radii["outer"] * math.cos(angle_rad)),
              int(center[1] + radii["outer"] * math.sin(angle_rad)))
        cv2.line(vis_img, p1, p2, grid_color, 1)

    return vis_img


def DR_segmentation_anchor(image_path: str, output_path: str, lesion_dir: Optional[str] = None) -> Dict:
    """
    主工作流程函数，串联所有分析步骤。
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    print("进入函数")
    abs_image_path = os.path.abspath(image_path)
    output_dir = os.path.abspath(output_path)
    os.makedirs(output_dir, exist_ok=True)
    intermediate_files_dir = os.path.join(output_dir, "intermediate_files")
    os.makedirs(intermediate_files_dir, exist_ok=True)

    print("Starting ETDRS Grid Analysis Workflow...")

    print("[TASK 0/4] Preprocessing: Cropping fundus area...")
    try:
        original_image = imread(abs_image_path)
        print(f"[DEBUG] Original input image resolution: {original_image.shape[:2]}")
        cropped_image, _, _, _, _, _, _ = process_without_gb(original_image, None)

        cropped_image_filename = os.path.basename(abs_image_path)
        cropped_image_path = os.path.join(intermediate_files_dir, cropped_image_filename)
        imwrite(cropped_image_path, cropped_image)
        print(f"Cropped image saved to: {cropped_image_path}")
    except Exception as e:
        return {"status": "error", "message": f"Failed during image cropping: {e}"}

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_coords = executor.submit(run_fovea_od_localization, cropped_image_path, intermediate_files_dir)
            future_diameter = executor.submit(run_disc_diameter_estimation, cropped_image_path, intermediate_files_dir)
            future_lesions_unet = executor.submit(find_latest_lesion_masks, intermediate_files_dir, lesion_dir)

            coords_result = future_coords.result()
            diameter_data = future_diameter.result()
            lesion_mask_paths = future_lesions_unet.result()

        fovea_coord = (coords_result['fovea']['x'], coords_result['fovea']['y'])
        od_coord = (coords_result['optic_disc']['x'], coords_result['optic_disc']['y'])

        fovea_od_dist = math.sqrt((fovea_coord[0] - od_coord[0]) ** 2 + (fovea_coord[1] - od_coord[1]) ** 2)
        disc_diameter_inferred = fovea_od_dist / 2.5
        grid_params = calculate_etdrs_grid_parameters(fovea_coord, od_coord, disc_diameter_inferred)

        image = cv2.imread(cropped_image_path)
        vis_image = draw_etdrs_grid(
            image,
            grid_params,
            segmentation_data=diameter_data,
            lesion_mask_paths=lesion_mask_paths
        )

        base_name = os.path.basename(image_path)
        file_stem, _ = os.path.splitext(base_name)
        annotated_filename = f"{file_stem}_annotated.png"

        anchor_dir = os.path.join(intermediate_files_dir, "anchor")
        os.makedirs(anchor_dir, exist_ok=True)
        final_save_path = os.path.join(anchor_dir, annotated_filename)

        cv2.imwrite(final_save_path, vis_image)

        res = {
            "status": "success",
            "inputs": {"image_path": image_path},
            "results": {
                "acquired_parameters": {
                    "fovea_coord": fovea_coord,
                    "od_coord": od_coord,
                    "disc_diameter_data": diameter_data
                },
                "output_visualization_path": os.path.abspath(final_save_path),
            }
        }
        print(res, file=sys.stderr)
        return res

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated ETDRS Grid Analysis Tool")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input fundus image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for all generated files.")
    parser.add_argument("--lesion_dir", type=str, required=False,
                        help="Optional: Path to the lesion analysis directory (containing dr_analysis_unet) from detect_lesions.")

    args = parser.parse_args()

    result = DR_segmentation_anchor(image_path=args.image_path, output_path=args.output_path, lesion_dir=args.lesion_dir)
    print(json.dumps(result, indent=4))
