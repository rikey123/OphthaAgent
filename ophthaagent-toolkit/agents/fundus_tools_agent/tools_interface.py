import os
import subprocess

import subprocess
import json
import os
import sys

import subprocess
import json
import os
import sys
import re


# 硬编码的 conda 环境 Python 路径（避免在 subprocess 中调用 conda info）
_HARDCODED_CONDA_PATHS = {
    "base": "<ANON_ABS_PATH>",
    "ietk": "<ANON_ABS_PATH>",
    "fundus_lesions_toolkit": "<ANON_ABS_PATH>",
    "dr_grading_dino": "<ANON_ABS_PATH>",
    "fundus_image_toolbox": "<ANON_ABS_PATH>",
    "dr_seg_unet": "<ANON_ABS_PATH>",
    "deepseenetplus": "<ANON_ABS_PATH>",
    "automorph-env": "<ANON_ABS_PATH>",
}

_CONDA_BASE_PATH = "<ANON_ABS_PATH>"


def get_conda_env_python_path(env_name):
    """
    获取指定 conda 环境的 Python 路径。
    优先使用硬编码路径，失败时回退到 conda info 动态查找。
    """
    # 优先使用硬编码路径
    if env_name in _HARDCODED_CONDA_PATHS:
        python_exec = _HARDCODED_CONDA_PATHS[env_name]
        if os.path.isfile(python_exec):
            return python_exec

    # 回退到动态查找
    conda_exe = "conda"
    try:
        subprocess.run([conda_exe, "--version"], capture_output=True)
    except FileNotFoundError:
        potential_paths = [
            "<ANON_ABS_PATH>",
            "<ANON_ABS_PATH>",
            os.path.expanduser("~/miniconda3/bin/conda"),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                conda_exe = path
                break
        else:
            # conda 不可用，尝试根据环境名构建路径
            if env_name and env_name != "base":
                python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
                if os.path.isfile(python_exec):
                    return python_exec
            raise RuntimeError(f"Conda 未安装且环境 '{env_name}' 未在硬编码路径中找到。")

    try:
        result = subprocess.run(
            [conda_exe, "info", "--json"],
            capture_output=True,
            text=True,
        )
    except Exception as e:
        # 尝试根据环境名构建路径
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError(f"调用 conda info 失败: {str(e)}")

    if result.returncode != 0:
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError(f"conda info 命令失败: {result.stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("conda info 无输出。")

    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")

    json_str = stdout[start:end]

    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")

    root_prefix = info.get("root_prefix")
    envs = info.get("envs", [])

    if not root_prefix:
        raise RuntimeError("conda info 输出中缺少 root_prefix。")

    env_paths = {"base": root_prefix}
    for env_path in envs:
        name = os.path.basename(env_path)
        env_paths[name] = env_path

    if not env_name or env_name == "base":
        target_path = root_prefix
    else:
        target_path = env_paths.get(env_name)
        if not target_path:
            available = list(env_paths.keys())
            raise RuntimeError(f"Conda 环境 '{env_name}' 不存在。可用环境: {available}")

    if sys.platform.startswith("win"):
        python_exec = os.path.join(target_path, "python.exe")
    else:
        python_exec = os.path.join(target_path, "bin", "python")

    if not os.path.isfile(python_exec):
        raise RuntimeError(f"在路径 {target_path} 中未找到 Python 可执行文件。")

    return python_exec

def enhance_fundus_image(image_path, output_path=None, show_results=True):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    path=get_conda_env_python_path('ietk')

    # 使用绝对路径来定位脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'preprocessing', 'ietk-ret.py')

    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
            encoding='utf-8',
            errors='replace'  )
    print(result.stdout)
    stdout = result.stdout.strip()
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def fundus_lesion_segmentation(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_file_dir, 'Lesion_detection', 'fundus-lesions-toolkit.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"fundus-lesions-toolkit脚本不存在: {script_path}")

    path = get_conda_env_python_path('fundus_lesions_toolkit')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def crop_by_fit(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_file_dir, 'preprocessing', 'crop_by_fundus_image_toolbox.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"crop脚本不存在: {script_path}")

    path = get_conda_env_python_path('fundus_image_toolbox')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def enhance_by_fit(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_file_dir, 'preprocessing', 'enhance_by_fundus_image_toolbox.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"enhance脚本不存在: {script_path}")

    path = get_conda_env_python_path('fundus_image_toolbox')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def quality_assess_by_fit(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    path = get_conda_env_python_path('fundus_image_toolbox')

    # 使用绝对路径来定位脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'image_quality', 'quality_assess_by_fundus_image_toolbox.py')

    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    stdout = result.stdout.strip()
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def fov_od_localization_by_fit(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_file_dir, 'anatomical_segmentation', 'Fovea_OD_localization_by_fundus_image_toolbox.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"Fovea定位脚本不存在: {script_path}")

    path = get_conda_env_python_path('fundus_image_toolbox')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    stdout = result.stdout.strip()
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info

def vessel_segment_by_fit(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path is not None:
        output_path = os.path.abspath(output_path)

    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_file_dir, 'anatomical_segmentation', 'Vessel_Segmentation_by_fundus_image_toolbox.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"Vessel分割脚本不存在: {script_path}")

    path = get_conda_env_python_path('fundus_image_toolbox')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, image_path]
    if output_path is not None:
        cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    stdout = result.stdout.strip()
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info
    
def DRgrading_by_DINO(image_path):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    # 获取当前文件所在目录，计算脚本的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(current_file_dir,'Lesion_detection','DRgrading','best_model.pth')

    script_path = os.path.join(current_file_dir, 'Lesion_detection','DRgrading', 'predict.py')

    # 确保脚本存在
    if not os.path.isfile(script_path):
        raise RuntimeError(f"脚本不存在: {script_path}")

    path = get_conda_env_python_path('dr_grading_dino')
    # 构建命令参数 - 只在output_path非None时添加
    cmd = [path, script_path, '--image_path', image_path,'--json_output','--model_path',weight_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8',
                                check=True,
                                errors='replace')
    except subprocess.CalledProcessError as e:
        print(f"DRgrading_by_DINO 子进程执行失败。")
        print(f"返回码: {e.returncode}")
        print(f"标准输出 (stdout):\n{e.stdout}")
        print(f"标准错误 (stderr):\n{e.stderr}")
        raise RuntimeError(f"DRgrading_by_DINO 失败: {e.stderr}") from e

    stdout = result.stdout.strip()
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("conda info 输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info =  {"status": "success", "result": json.loads(json_str)}
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 conda info 的 JSON 输出。错误: {e}")
    return info
    
if __name__ == "__main__":
    input='<ANON_ABS_PATH>'
    output1='<ANON_ABS_PATH>'
    output2='<ANON_ABS_PATH>'
    #crop_by_fit(input, input)
    # enhance_fundus_image(input, output1)
    # fundus_lesion_segmentation(output1, output2)
    fov_od_localization_by_fit(input, output1)
    #quality_assess_by_fit(input, output1)
    #vessel_segment_by_fit(input, output1)
    #enhance_by_fit(input, output1)
    DRgrading_by_DINO(input)



