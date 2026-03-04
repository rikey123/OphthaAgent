import os
import subprocess
import json
import sys
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

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
        # conda 不可用，尝试构建路径
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError("Conda is not installed or not in the PATH.")

    if result.returncode != 0:
        if env_name and env_name != "base":
            python_exec = os.path.join(_CONDA_BASE_PATH, "envs", env_name, "bin", "python")
            if os.path.isfile(python_exec):
                return python_exec
        raise RuntimeError(f"The conda info command failed: {result.stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("conda info has no output.")

    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        raise RuntimeError("No JSON structure was found in the conda info output.")

    json_str = stdout[start:end]

    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Unable to parse the JSON output of conda info. Error: {e}")

    root_prefix = info.get("root_prefix")
    envs = info.get("envs", [])

    if not root_prefix:
        raise RuntimeError("root_prefix is missing from the conda info output.")

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
            raise RuntimeError(f"The Conda environment '{env_name}' does not exist. Available environments: {available}")

    python_exec = os.path.join(target_path, "bin", "python") if not sys.platform.startswith("win") else os.path.join(target_path, "python.exe")

    if not os.path.isfile(python_exec):
        raise RuntimeError(f"The Python executable was not found in the path {target_path}.")

    return python_exec

def _run_and_parse_json(command):
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding='utf-8',
                            errors='replace')
    if result.returncode != 0:
        error_message = (
            f"Script execution failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Stderr:\n{result.stderr}\n"
            f"Stdout:\n{result.stdout}"
        )
        raise RuntimeError(error_message)
        
    stdout = result.stdout.strip()
    start = stdout.find('{')
    end = stdout.rfind('}') + 1
    if start == -1 or end == 0:
        print(f"Error: No JSON object found in the script output:\n{stdout}\n---", file=sys.stderr)
        raise RuntimeError("The script output does not contain a valid JSON object.")
    json_str = stdout[start:end]
    try:
        info = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from script output:\n{json_str}\n---", file=sys.stderr)
        raise RuntimeError(f"Failed to parse JSON from script output. Error: {e}")
    return info

def enhance_fundus_image(image_path, output_path, show_results=True):
    path=get_conda_env_python_path('ietk')
    script_path = os.path.join(_AGENT_ROOT, 'preprocessing', 'ietk-ret.py')
    command = [path, script_path, image_path, output_path]
    return _run_and_parse_json(command)

def fundus_lesion_segmentation(image_path, output_path):
    path = get_conda_env_python_path('fundus_lesions_toolkit')
    script_path = os.path.join(_AGENT_ROOT, 'Lesion_detection', 'fundus-lesions-toolkit.py')
    command = [path, script_path, image_path, output_path]
    return _run_and_parse_json(command)

def vessel_segmentation(image_path, output_dir):
    path = get_conda_env_python_path('ietk')
    script_path = os.path.join(_SCRIPT_DIR, 'AutoMorph', 'M2_Vessel_seg', 'test_outside_integrated.py')
    command = [
        path, 
        script_path, 
        '--image_path', image_path, 
        '--output_dir', output_dir,
        '--dataset', 'DRIVE',
        '--dataset_test', 'DRIVE',
        '--jn', 'unet_vessel_seg'
    ]
    return _run_and_parse_json(command)

def run_automorph_m0_m1(image_path):
    path = get_conda_env_python_path('lg')
    script_path = os.path.join(_AGENT_ROOT, 'image_quality', 'automorph_M0_M1.py')
    command = [path, script_path, image_path]
    return _run_and_parse_json(command)

def run_automorph_m2(image_path):
    path = get_conda_env_python_path('lg')
    script_path = os.path.join(_AGENT_ROOT, 'anatomical_segmentation', 'automorph_M2.py')
    command = [path, script_path, image_path]
    return _run_and_parse_json(command)

def run_quantitative_measurement(image_path):
    path = get_conda_env_python_path('lg')
    script_path = os.path.join(_AGENT_ROOT, 'measurement', 'quantitative_measurement.py')
    command = [path, script_path, image_path]
    return _run_and_parse_json(command)

if __name__ == "__main__":
    input_image = os.path.join(_SCRIPT_DIR, 'test1.jpg')
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"Test image 'test1.jpg' not found in script directory: {_SCRIPT_DIR}")

    print(f"--- Using test image: {input_image} ---")

    # Define output paths using absolute paths
    enhanced_image = os.path.join(_SCRIPT_DIR, 'enhanced_fundus.jpg')
    lesions_image = os.path.join(_SCRIPT_DIR, 'lesions.jpg')
    vessel_output_dir = os.path.join(_SCRIPT_DIR, 'vessel_seg_output')

    # Create output directories
    os.makedirs(vessel_output_dir, exist_ok=True)

    print("\n--- 1. Enhancing Fundus Image (IETK) ---")
    enhance_result = enhance_fundus_image(input_image, enhanced_image)
    print(json.dumps(enhance_result, indent=4))

    print("\n--- 2. Running AutoMorph M0/M1 ---")
    m0_m1_result = run_automorph_m0_m1(enhanced_image)
    print(json.dumps(m0_m1_result, indent=4))

    print("\n--- 3. Running AutoMorph M2 ---")
    m2_result = run_automorph_m2(enhanced_image)
    print(json.dumps(m2_result, indent=4))

    print("\n--- 4. Running Quantitative Measurement ---")
    quant_result = run_quantitative_measurement(enhanced_image)
    print(json.dumps(quant_result, indent=4))

    print("\n--- 5. Segmenting Fundus Lesions ---")
    lesion_result = fundus_lesion_segmentation(enhanced_image, lesions_image)
    print(json.dumps(lesion_result, indent=4))

    print("\n--- 6. Segmenting Vessels (IETK) ---")
    vessel_result = vessel_segmentation(enhanced_image, vessel_output_dir)
    print(json.dumps(vessel_result, indent=4))
