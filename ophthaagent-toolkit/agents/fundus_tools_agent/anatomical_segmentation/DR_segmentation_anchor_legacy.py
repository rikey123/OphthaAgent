#!/usr/bin/env python3
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# --- 自动化依赖注入部分 (参考 dme_risk_assessment.py) ---

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from script output: {json_str}") from e

def run_fovea_od_localization(abs_image_path: str, output_dir: str) -> Dict:
    """
    步骤 2: 使用 Fovea_OD_localization_by_fundus_image_toolbox 计算 视盘 和 中心凹坐标

    Returns:
        包含两个坐标的结果字典
    """

    print("[TASK 1/3] Locating Fovea and Optic Disc...")

    fol_script = "Fovea_OD_localization_by_fundus_image_toolbox.py"
    fol_cwd = os.path.join(PROJECT_ROOT, "anatomical_segmentation")

    # 调用 Fovea_OD_localization_by_fundus_image_toolbox
    fol_result = run_python_script(
        fol_script,
        [abs_image_path, "no_save"],
        env_name="fundus_image_toolbox",
        cwd=fol_cwd
    )

    if fol_result.get("status") != "success": 
        raise RuntimeError(f"Fovea/OD localization failed: {fol_result}")
    print(fol_result['result']['coordinates'])

    return fol_result['result']['coordinates']

def run_disc_diameter_estimation(abs_image_path: str, output_dir: str) -> float:
    """
    步骤 3: 使用 AutoMorphalyzer 计算视盘直径（独立任务）

    Returns:
        包含视盘直径信息的结果字典
    """

    print("[任务 2/3] 开始: 使用 AutoMorphalyzer 计算视盘直径...")
    automorph_script = "analyze_single.py"
    automorph_cwd = os.path.join(PROJECT_ROOT, "anatomical_segmentation", "AutoMorphalyzer")
    automorph_output_dir = os.path.join(output_dir, "automorph_results")

    # 创建输出目录
    os.makedirs(automorph_output_dir, exist_ok=True)
    # 在脚本开头添加
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from device_config import get_device

    # 替换硬编码设备设置
    device = get_device()  # 替换 torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 调用 AutoMorphalyzer
    automorph_result = run_python_script(
        automorph_script,
        ["--input", abs_image_path, "--output", automorph_output_dir, "--device", device],
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
    dr_cwd = os.path.join(PROJECT_ROOT, "Lesion_detection", "DR-Detection-and-Clasification-System")
    dr_output_dir = os.path.join(output_dir, "dr_analysis")

    os.makedirs(dr_output_dir, exist_ok=True)
    abs_dr_output = os.path.abspath(dr_output_dir)

    # 运行病灶检测脚本
    run_python_script(
        dr_script_name,
        [abs_image_path, "-o", abs_dr_output],
        env_name="DR-Detection-and-Clasification-System",
        cwd=dr_cwd
    )

    # 在输出目录中找到分析结果文件夹
    analysis_dirs = [d for d in os.listdir(dr_output_dir) if d.startswith("analysis_") and os.path.isdir(os.path.join(dr_output_dir, d))]
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

    # 构建并验证每个掩码文件的路径
    lesion_mask_paths = {}
    for lesion_type, mask_filename in lesion_mask_files.items():
        mask_path = os.path.join(analysis_dir, mask_filename)
        if os.path.exists(mask_path):
            lesion_mask_paths[lesion_type] = mask_path
    
    print(f"[任务 3/3] 完成. 找到 {len(lesion_mask_paths)} 个病灶掩码文件。")
    return lesion_mask_paths



def calculate_etdrs_grid_parameters(fovea_coord: Tuple[float, float], od_coord: Tuple[float, float], disc_diameter: float) -> Dict:
    """
    根据黄斑中心、视盘中心坐标和视盘直径，计算ETDRS网格的几何参数。

    Args:
        fovea_coord (Tuple[float, float]): 黄斑中心的(x, y)坐标。
        od_coord (Tuple[float, float]): 视盘中心的(x, y)坐标。
        disc_diameter (float): 视盘的平均直径（像素）。

    Returns:
        Dict: 包含网格中心、半径、各象限轴角度等参数的字典。
    """
    # 提取黄斑和视盘的坐标
    fovea_x, fovea_y = fovea_coord
    od_x, od_y = od_coord

    r_central = 0.33 * disc_diameter
    r_inner = 1.0 * disc_diameter
    r_outer = 2.0 * disc_diameter

    # 计算从黄斑指向视盘的角度，此方向定义为鼻侧（Nasal）
    nasal_temporal_angle_rad = math.atan2(od_y - fovea_y, od_x - fovea_x)

    # 返回一个包含所有网格参数的字典
    return {
        "center": fovea_coord,  # 网格中心（黄斑中心）
        "disc_diameter": disc_diameter,
        "radii": {
            "central": r_central,
            "inner": r_inner,
            "outer": r_outer,
        },
        "axes_angles_rad": {
            # 定义四个主轴线的角度（弧度制）
            "nasal": nasal_temporal_angle_rad % (2 * math.pi),  # 鼻侧
            "temporal": (nasal_temporal_angle_rad + math.pi) % (2 * math.pi),  # 颞侧
            "superior": (nasal_temporal_angle_rad - math.pi / 2) % (2 * math.pi),  # 上侧
            "inferior": (nasal_temporal_angle_rad + math.pi / 2) % (2 * math.pi),  # 下侧
        }
    }



def draw_etdrs_grid(image: np.ndarray, grid_params: Dict, segmentation_data: Dict, lesion_mask_paths: Dict[str, str]) -> np.ndarray:
    """
    在眼底图上绘制ETDRS网格、解剖结构和各类病灶掩码。

    Args:
        image (np.ndarray): 输入的眼底图像 (OpenCV格式)。
        grid_params (Dict): ETDRS网格的几何参数。
        segmentation_data (Dict): 包含血管和视盘mask路径的字典。
        lesion_mask_paths (Dict[str, str]): 包含各类病灶掩码路径的字典。

    Returns:
        np.ndarray: 绘制了所有元素的可视化图像。
    """
    vis_img = image.copy()
    overlay = vis_img.copy()
    h, w = vis_img.shape[:2]

    # --- 步骤 1: 绘制半透明的解剖结构 (血管和视盘) ---
    vessel_mask_path = segmentation_data.get("vessel_mask_path")
    if vessel_mask_path and os.path.exists(vessel_mask_path):
        vessel_mask = cv2.imread(vessel_mask_path, cv2.IMREAD_GRAYSCALE)
        if vessel_mask is not None:
            vessel_mask = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[vessel_mask > 128] = [0, 255, 0]  # 绿色血管

    disc_mask_path = segmentation_data.get("disc_mask_path")
    if disc_mask_path and os.path.exists(disc_mask_path):
        disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
        if disc_mask is not None:
            disc_mask = cv2.resize(disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[disc_mask > 128] = [225, 105, 65]  # 浅蓝色视盘

    vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)

    # --- 步骤 2: 绘制ETDRS网格 ---
    center = tuple(map(int, grid_params["center"]))
    radii = grid_params["radii"]
    cv2.circle(vis_img, center, int(radii["central"]), (255, 0, 0), 2)      # 中心圈 -> 蓝色
    cv2.circle(vis_img, center, int(radii["inner"]), (255, 0, 0), 2)        # 第二圈 -> 蓝色
    cv2.circle(vis_img, center, int(radii["outer"]), (255, 0, 0), 2)   # 第三圈 -> 蓝色

    for angle_rad in grid_params["axes_angles_rad"].values():
        p1 = (int(center[0] + radii["central"] * math.cos(angle_rad)), int(center[1] + radii["central"] * math.sin(angle_rad)))
        p2 = (int(center[0] + radii["outer"] * math.cos(angle_rad)), int(center[1] + radii["outer"] * math.sin(angle_rad)))
        cv2.line(vis_img, p1, p2, (255, 255, 255), 1)

    # --- 步骤 3: 在最顶层直接绘制不同颜色的病灶掩码 ---
    lesion_colors = {
        "hard_exudates": [0, 255, 255],  # 黄色 (不变)
        "hemorrhages": [0, 0, 255],      # 正红色
        "microaneurysms": [102, 0, 51],   # 紫色
        "soft_exudates": [122, 190, 255]   # 新颜色
    }

    for lesion_type, mask_path in lesion_mask_paths.items():
        if os.path.exists(mask_path):
            # 读取拼接图并裁剪右半部分的掩码
            stitched_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if stitched_mask is None: 
                print(f"  - Failed to read mask file.") # DEBUG
                continue
            
            mask_h, mask_w = stitched_mask.shape
            lesion_mask = stitched_mask[:, mask_w//2:]

            # 确保掩码尺寸与主图一致
            if lesion_mask.shape[0] != h or lesion_mask.shape[1] != w:
                lesion_mask = cv2.resize(lesion_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 将掩码区域用指定颜色填充
            color = lesion_colors.get(lesion_type, [255, 255, 255]) # 默认为白色
            if np.any(lesion_mask > 128):
                print(f"  - Lesion pixels found. Applying color: {color}") # DEBUG
                vis_img[lesion_mask > 128] = color
            else:
                print(f"  - No lesion pixels found in mask.") # DEBUG
        else:
            print(f"  - Mask file does not exist.") # DEBUG

    return vis_img

# --- Main Workflow --- 

def DR_segmentation_anchor(image_path: str, output_path: str) -> Dict:
    """
    主工作流程函数，串联所有分析步骤。
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    abs_image_path = os.path.abspath(image_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    intermediate_files_dir = os.path.join(output_dir, "intermediate_files")
    os.makedirs(intermediate_files_dir, exist_ok=True)

    print("Starting ETDRS Grid Analysis Workflow...")
    try:
        # 并行运行所有数据采集任务
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_coords = executor.submit(run_fovea_od_localization, abs_image_path, intermediate_files_dir)
            future_diameter = executor.submit(run_disc_diameter_estimation, abs_image_path, intermediate_files_dir)
            future_lesions = executor.submit(run_lesion_detection, abs_image_path, intermediate_files_dir)

            coords_result = future_coords.result()
            diameter_data = future_diameter.result()
            lesion_mask_paths = future_lesions.result()

        fovea_coord = (coords_result['fovea']['x'], coords_result['fovea']['y'])
        od_coord = (coords_result['optic_disc']['x'], coords_result['optic_disc']['y'])

        # --- ETDRS 分析：使用黄斑-视盘距离推算视盘直径 ---
        fovea_od_dist = math.sqrt((fovea_coord[0] - od_coord[0])**2 + (fovea_coord[1] - od_coord[1])**2)
        disc_diameter_inferred = fovea_od_dist / 2.5
        grid_params = calculate_etdrs_grid_parameters(fovea_coord, od_coord, disc_diameter_inferred)

        # --- 可视化 ---
        image = cv2.imread(image_path)
        vis_image = draw_etdrs_grid(
            image,
            grid_params,
            segmentation_data=diameter_data,
            lesion_mask_paths=lesion_mask_paths
        )
        cv2.imwrite(output_path, vis_image)

        # --- 生成最终结果 ---
        return {
            "status": "success",
            "inputs": {"image_path": image_path},
            "results": {
                "acquired_parameters": {
                    "fovea_coord": fovea_coord,
                    "od_coord": od_coord,
                    "disc_diameter_data": diameter_data
                },
                "grid_parameters": grid_params,
                "lesion_mask_paths": lesion_mask_paths,
                "output_visualization_path": os.path.abspath(output_path),
            }
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated ETDRS Grid Analysis Tool")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input fundus image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output visualization image.")
    
    # 添加命令行参数解析
    args = parser.parse_args()
    
    # 调用主函数并打印JSON结果
    result = DR_segmentation_anchor(image_path=args.image_path, output_path=args.output_path)
    print(json.dumps(result, indent=4))
