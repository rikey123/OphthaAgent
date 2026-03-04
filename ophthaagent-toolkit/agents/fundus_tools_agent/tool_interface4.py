import os
import subprocess
import json
import sys
import re
import warnings

# Suppress the specific UserWarning from pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

"""
AMD_predict_fundus_by_deepseenet + Lesion_predict_OCT_by_opticnet + segment_by_AutoMorphalyzer + dme_risk_assessment + 
"""
# 在文件开头添加项目根目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 导入 get_device
from device_config import get_device
device = get_device()
print("device:",device)
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
#
def AMD_predict_fundus_by_deepseenet(image_path):
    # 获取当前文件的绝对路径，然后定位到deepseenet-plus目录
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    deepseenet_dir = os.path.join(current_file_dir, 'Lesion_detection', 'deepseenet-plus')

    # 确保目录存在
    if not os.path.isdir(deepseenet_dir):
        raise RuntimeError(f"deepseenet-plus目录不存在: {deepseenet_dir}")

    # 将image_path转换为绝对路径
    image_path_abs = os.path.abspath(image_path)

    path = get_conda_env_python_path('deepseenetplus')
    result = subprocess.run(
        [path,'analyze_single_image.py',
         image_path_abs,'--json'], capture_output=True, text=True,
        encoding='utf-8',
        cwd=deepseenet_dir,
        errors='replace')
    print("--- DEBUG: Raw STDERR from subprocess ---")
    print(result.stderr)
    print("--- END DEBUG ---")
    stdout = result.stdout.strip()
    print("--- DEBUG: Raw STDOUT from subprocess ---")
    print(stdout)
    print("--- END DEBUG ---")
    # 尝试从开头找 '{' 到匹配的 '}'，提取完整 JSON
    # 简单但有效的方法：找第一个 '{' 和最后一个 '}'
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("输出中未找到 JSON 结构。")
    json_str = stdout[start:end]
    print("--- DEBUG: Extracted string to be parsed as JSON ---")
    print(json_str)
    print("--- END DEBUG ---")
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 调试用：可临时打印出问题内容（生产环境慎用）
        # print("原始输出片段（前1000字符）:", stdout[:1000])
        raise RuntimeError(f"无法解析 JSON 输出。错误: {e}")
    result = {
        "status": "success",
        "result": info
    }
    return result
def Lesion_predict_OCT_by_opticnet(image_path):
    path = get_conda_env_python_path('opticnet')
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_file:
        output_file_path = tmp_file.name

    command = [
        path, 
        'kermany_inference.py', 
        '--imgpath', image_path, 
        '--kermany_weights', 'Kermany2018.hdf5',
        '--output', output_file_path
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True, cwd='./agents/fundus_tools_agent/Lesion_detection/OpticNet-71')
        with open(output_file_path, 'r') as f:
            result = json.load(f)
        return {"status": "success", "result": result}
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
def segment_by_AutoMorphalyzer(image_path, output_path='./result'):
     # 转换为绝对路径
    image_path = os.path.abspath(image_path)

    path = get_conda_env_python_path('automorph-env')
    # 使用绝对路径避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    automorph_dir = os.path.join(script_dir, 'anatomical_segmentation', 'AutoMorphalyzer')

    # AutoMorphalyzer 要求必须提供 --input 和 --output 参数
    # 如果未提供 output_path，生成临时输出路径
    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix='.json', prefix='automorph_')

    # 构建命令 - AutoMorphalyzer 必须同时提供 input 和 output
    cmd = [path, 'analyze_single.py',
           '--input', image_path,
           '--output', output_path]

    # 禁用代理环境变量，避免socks代理错误
    env = os.environ.copy()
    env['NO_PROXY'] = '*'
    env['no_proxy'] = '*'
    # 移除可能导致问题的代理设置
    for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                      'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']:
        env.pop(proxy_var, None)

    # 设置CUDA设备（如果环境中有）
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']

    result = subprocess.run(cmd, capture_output=True, text=True,
        encoding='utf-8',
        cwd=automorph_dir,
        env=env,
        errors='replace',
        timeout=300)  # 5分钟超时

    print("STDERR:", result.stderr[:500] if result.stderr else "None")  # 错误输出
    print("STDOUT:", result.stdout[:500] if result.stdout else "None")

    # 检查返回码
    if result.returncode != 0:
        return {
            "status": "error",
            "message": f"AutoMorphalyzer执行失败，返回码: {result.returncode}",
            "stderr": result.stderr[:1000] if result.stderr else ""
        }

    # AutoMorphalyzer将结果保存到output_path指定的路径
    # 实际保存的文件路径是 output_path/measurements.json
    json_file = os.path.join(output_path, 'measurements.json')

    if not os.path.exists(json_file):
        # 尝试直接使用output_path作为文件路径
        if os.path.exists(output_path) and output_path.endswith('.json'):
            json_file = output_path
        else:
            return {
                "status": "error",
                "message": f"AutoMorphalyzer结果文件不存在: {json_file}",
                "output_path": output_path,
                "stdout": result.stdout[:500] if result.stdout else ""
            }

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"无法解析JSON文件: {e}",
            "json_file": json_file
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"读取结果文件失败: {e}",
            "json_file": json_file
        }

    result = {
        "status": "success",
        "result": info
    }
    return result

def dme_risk_assessment(image_path, output_dir=None):
    """
    DME (糖尿病黄斑水肿) 风险评估

    该函数整合以下分析：
    1. 使用 AutoMorphalyzer 计算视杯视盘直径
    2. 使用 Fovea_OD_localization 定位黄斑中心凹
    3. 使用 DR-Detection-and-Clasification-System 检测硬性渗出物
    4. 基于 ETDRS 标准计算 DME 风险等级

    Args:
        image_path: 输入眼底图像路径
        output_dir: 输出目录（可选，默认为 dme_assessment_results）

    Returns:
        dict: 包含以下内容的结果字典：
            - status: "success" 或 "error"
            - result: 包含解剖特征、硬性渗出物信息和 DME 风险评估的详细结果
                - anatomical_features: 黄斑、视盘坐标和视盘直径
                - hard_exudates: 硬性渗出物数量、坐标和到黄斑的最小距离
                - dme_risk_assessment: 风险等级、描述和临床建议
                    - risk_level: 0 (低风险), 1 (中等风险), 2 (高风险)
                    - risk_category: 风险分类描述
                    - description: 详细说明
                    - clinical_note: 临床建议
                - visualization_path: 可视化结果图像路径
                - output_directory: 输出目录路径

    示例:
        result = dme_risk_assessment('/path/to/fundus_image.jpg')
        if result['status'] == 'success':
            risk_level = result['result']['dme_risk_assessment']['risk_level']
            print(f"DME 风险等级: {risk_level}")
    """
    # 构建脚本路径
    script_path = os.path.join(os.path.dirname(__file__), 'anatomical_segmentation', 'dme_risk_assessment_unet.py')

    # 构建命令
    cmd = [sys.executable, script_path, image_path]
    if output_dir:
        cmd.extend(['-o', output_dir])

    # 执行脚本
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode != 0:
        raise RuntimeError(f"DME 风险评估脚本执行失败 (ReturnCode {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    # 提取 JSON 输出
    stdout = result.stdout
    
    # 使用鲁棒的从后往前提取 JSON 逻辑
    info = None
    start_indices = [i for i, char in enumerate(stdout) if char == '{']
    end_indices = [i for i, char in enumerate(stdout) if char == '}']
    
    for start in reversed(start_indices):
        for end in reversed(end_indices):
            if end > start:
                candidate = stdout[start:end+1]
                try:
                    temp_info = json.loads(candidate)
                    if isinstance(temp_info, dict) and "status" in temp_info:
                        info = temp_info
                        break
                except json.JSONDecodeError:
                    continue
        if info is not None:
            break

    if info is None:
        raise RuntimeError(f"DME 风险评估输出中未找到有效的 JSON 结构。参考 STDOUT:\n{stdout}\nSTDERR:\n{result.stderr}")

    return info
if __name__ == "__main__":
    image_path="<ANON_ABS_PATH>"
    # oct_image_path="oct.jpg"
    image_path=os.path.abspath(image_path)
    #segment_by_AutoMorphalyzer(image_path)
    # oct_image_path=os.path.abspath(oct_image_path)
    #AMD_predict_fundus_by_deepseenet(image_path)
    #Lesion_predict_OCT_by_opticnet(oct_image_path)
    #print(dme_risk_assessment(image_path))
    print(dme_risk_assessment(image_path, output_dir='./dme_assessment_results'))

