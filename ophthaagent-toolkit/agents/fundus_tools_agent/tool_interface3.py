import os
import subprocess
import json
import sys
import re

"""
segment_by_ddcs + segment_by_unet + DR_segmentation_anchor_ETDRS
"""
# 导入 get_device
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from device_config import get_device
device = str(get_device())
print("device:", device)

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
    "mkcnet": "<ANON_ABS_PATH>",
    "lg": "<ANON_ABS_PATH>",
}

_CONDA_BASE_PATH = "<ANON_ABS_PATH>"


def get_conda_env_python_path(env_name):
    """获取 conda 环境 Python 路径。优先使用硬编码路径。"""
    # 优先使用硬编码路径
    if env_name in _HARDCODED_CONDA_PATHS:
        python_exec = _HARDCODED_CONDA_PATHS[env_name]
        if os.path.isfile(python_exec):
            return python_exec

    # 回退到动态查找
    conda_exe_path = "conda"
    conda_root = (
        os.path.dirname(os.path.dirname(sys.prefix))
        if "envs" in sys.prefix
        else sys.prefix
    )
    potential_conda_exe = os.path.join(conda_root, "bin", "conda")
    if os.path.exists(potential_conda_exe):
        conda_exe_path = potential_conda_exe

    try:
        result = subprocess.run(
            [conda_exe_path, "info", "--json"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError(f"Conda 未安装或不在 PATH 中。")

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
def DR_Grading(image_path):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)

    path = get_conda_env_python_path('mkcnet')
    # 使用绝对路径避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mkcnet_dir = os.path.join(script_dir, 'Lesion_detection', 'MKCNet')
    result = subprocess.run(
        [path, 'predict_single_image.py',"--image_path", image_path,"--datasets", 'DRAC', 'DEEPDR', 'EYEQ' ,"--model_type","MKCNet"], capture_output=True, text=True,
        encoding='utf-8',
        cwd=mkcnet_dir,
        errors='replace')
    print("STDERR:", result.stderr)  # 错误输出
    stdout = result.stdout.strip()
    print(f"[DEBUG] Raw output from segment_by_ddcs:\n{stdout}\n---")
    print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("DR_Grading 脚本输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 DR_Grading 脚本的 JSON 输出。错误: {e}")
    result = {
        "status": "success",
        "result": info
    }
    return result
def segment_by_ddcs(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)

    path = get_conda_env_python_path('ddcs')
    # 使用绝对路径避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ddcs_dir = os.path.join(script_dir, 'Lesion_detection', 'DR-Detection-and-Clasification-System')

    # 输出路径，新建dr_analysis_unet
    dr_output_dir = os.path.join(output_path, "dr_analysis_ddcs")
    os.makedirs(dr_output_dir, exist_ok=True)
    abs_dr_output = os.path.abspath(dr_output_dir)

    # 构建命令 - 只在output_path非None时添加
    cmd = [path, 'comprehensive_analysis.py', image_path]
    if output_path is not None:
        cmd.extend(['--output', abs_dr_output])

    result = subprocess.run(cmd, capture_output=True, text=True,
        encoding='utf-8',
        cwd=ddcs_dir,
        errors='replace')
    print("STDERR:", result.stderr)  # 错误输出
    stdout = result.stdout.strip()
    #print(f"[DEBUG] Raw output from segment_by_ddcs:\n{stdout}\n---")
    #print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("segment_by_ddcs 脚本输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 segment_by_ddcs 脚本的 JSON 输出。错误: {e}")
    result = {
        "status": "success",
        "result": info
    }
    print(result)
    return result

def segment_by_unet(image_path, output_path=None):
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)

    path = get_conda_env_python_path('dr_seg_unet')
    # 使用绝对路径避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ddcs_dir = os.path.join(script_dir, 'Lesion_detection', 'dr_single_image_analyzer',)

    # 输出路径，新建dr_analysis_unet
    dr_output_dir = os.path.join(output_path, "dr_analysis_unet")
    os.makedirs(dr_output_dir, exist_ok=True)
    abs_dr_output = os.path.abspath(dr_output_dir)

    # 构建命令 - 只在output_path非None时添加
    cmd = [path, 'dr_analyzer_resize.py',
        '--image', image_path,
        '--output', abs_dr_output,
        '--device', device,  # 🔧 使用全局设备配置
        '--target-resolution', '4288x2848',  # 模型最佳分辨率
        '--resize-method', 'padding']  # 使用保持宽高比 + padding 方法
    if output_path is not None:
        cmd.extend(['--output', abs_dr_output])

    result = subprocess.run(cmd, capture_output=True, text=True,
        encoding='utf-8',
        cwd=ddcs_dir,
        errors='replace')
    print("STDERR:", result.stderr)  # 错误输出
    stdout = result.stdout.strip()
    #print(f"[DEBUG] Raw output from segment_by_unet:\n{stdout}\n---")
    #print(result.stdout)
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("脚本输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析脚本的 JSON 输出。错误: {e}")
    result = {
        "status": "success",
        "result": info
    }
    return result


def DR_segmentation_anchor_ETDRS(image_path, output_path=None, lesion_dir=None):
    """
    使用 DR_segmentation_anchor_legacy copy.py 对眼底图像进行全自动ETDRS网格分析。
    
    Args:
        image_path: 输入图像路径
        output_path: 输出目录路径
        lesion_dir: detect_lesions 的输出目录（包含 dr_analysis_unet），可选
    """
    # 1. 转换为绝对路径
    image_path = os.path.abspath(image_path)
    if output_path:
        output_path = os.path.abspath(output_path)
    if lesion_dir:
        lesion_dir = os.path.abspath(lesion_dir)
    
    path = get_conda_env_python_path('OphthaAgent')
    # 3. 定义脚本路径和工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script_path = os.path.join(script_dir, 'anatomical_segmentation', 'DR_segmentation_anchor_unet.py')
    script_cwd = os.path.dirname(target_script_path)

    # 输出路径，新建dr_analysis_unet
    dr_output_dir = os.path.join(output_path, "dr_anchor")
    os.makedirs(dr_output_dir, exist_ok=True)
    abs_dr_output = os.path.abspath(dr_output_dir)

    # 4. 构建命令
    cmd = [path, target_script_path, '--image_path', image_path]
    if output_path:
        cmd.extend(['--output_path', abs_dr_output])
    if lesion_dir:
        cmd.extend(['--lesion_dir', lesion_dir])

    # 5. 执行子进程
    result = subprocess.run(cmd, capture_output=True, text=True,
        encoding='utf-8',
        cwd=script_cwd,
        errors='replace')

    # 6. 检查和解析输出
    if result.returncode != 0:
        # 如果脚本执行失败，返回错误信息
        error_message = f"DR_segmentation_anchor_ETDRS script failed with return code {result.returncode}.\\nSTDERR: {result.stderr}\\nSTDOUT: {result.stdout}"
        print(error_message) # 打印详细错误供调试
        raise RuntimeError(error_message)

    stdout = result.stdout.strip()
    #print("STDOUT:", stdout) # 打印标准输出供调试
    if result.stderr:
        print("STDERR:", result.stderr) # 打印错误输出供调试

    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("DR_segmentation_anchor_ETDRS 脚本输出中未找到 JSON 结构。")

    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"无法解析 DR_segmentation_anchor_ETDRS 脚本的 JSON 输出。错误: {e}")

    # 7. 返回结果
    final_result = {
        "status": "success",
        "result": info
    }
    return final_result


if __name__ == "__main__":
    #image_path="<ANON_ABS_PATH>"
    #image_path = "<ANON_ABS_PATH>"
    image_path = "<ANON_ABS_PATH>"
    #segment_by_ddcs(image_path,"<ANON_ABS_PATH>")
    segment_by_unet(image_path,"<ANON_ABS_PATH>")
    # DR_segmentation_anchor_ETDRS(image_path,"<ANON_ABS_PATH>")
    #print("anchor complete")
