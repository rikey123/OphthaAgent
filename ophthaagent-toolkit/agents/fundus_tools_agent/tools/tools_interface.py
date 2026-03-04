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

    # 回退到动态查找 conda info
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # conda 命令不可用，尝试根据环境名构建路径
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError(f"Conda 未安装或不在 PATH 中，且环境 '{env_name}' 未在硬编码路径中找到。")

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
    path=get_conda_env_python_path('ietk')
    result = subprocess.run([path, './agents/fundus_tools_agent/preprocessing/ietk-ret.py',image_path, output_path], capture_output=True, text=True,
            encoding='utf-8',
            errors='replace'  )
    print(result.stdout)

def fundus_lesion_segmentation(image_path, output_path=None):
    path = get_conda_env_python_path('fundus_lesions_toolkit')
    result = subprocess.run([path, './agents/fundus_tools_agent/Lesion_detection/fundus-lesions-toolkit.py', image_path, output_path], capture_output=True, text=True,
                            encoding='utf-8',
                            check=True,
                            errors='replace')
    print(result.stdout)

if __name__ == "__main__":
    input='test.png'
    output1='enhanced_fundus3.jpg'
    output2='lesions3.jpg'
    enhance_fundus_image(input, output1)
    fundus_lesion_segmentation(output1, output2)