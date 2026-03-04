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


def get_conda_env_python_path(env_name):
    """
    安全地获取指定 conda 环境的 Python 路径，自动过滤非 JSON 内容。
    """
    try:
        # 调用 conda info --json，并确保 stderr 不污染 stdout
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True,
            text=True,
            # 不检查返回码，因为即使有警告也可能成功
        )
    except FileNotFoundError:
        raise RuntimeError("Conda 未安装或不在 PATH 中。请确保 'conda' 命令可用。")

    if result.returncode != 0:
        raise RuntimeError(f"conda info 命令失败: {result.stderr}")

    # 修复：提取第一个有效的 JSON 对象（忽略前面或后面的垃圾内容）
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("conda info 无输出。")

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
def DR_Grading(image_path):
    path = get_conda_env_python_path('mkcnet')
    result = subprocess.run(
        [path, 'predict_single_image.py',"--image_path", image_path,"--datasets", 'DRAC', 'DEEPDR', 'EYEQ' ,"--model_type","MKCNet"], capture_output=True, text=True,
        encoding='utf-8',
        cwd='./agents/fundus_tools_agent/Lesion_detection/MKCNet',
        errors='replace')
    print("STDERR:", result.stderr)  # 错误输出
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
    result = {
        "status": "success",
        "result": info
    }
    return result
def segment_by_ddcs(image_path, output_path=None, device=None):
    path = get_conda_env_python_path('DR-Detection-and-Clasification-System')
    
    # Prepare environment variables for the subprocess
    subprocess_env = os.environ.copy()

    result = subprocess.run(
        [path, 'comprehensive_analysis.py',
         image_path,'--output', output_path],
        capture_output=True, text=True,
        encoding='utf-8',
        cwd='./agents/fundus_tools_agent/Lesion_detection/DR-Detection-and-Clasification-System',
        errors='replace',
        env=subprocess_env)  # Pass the modified environment
    print("STDERR:", result.stderr)  # 错误输出
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
    result = {
        "status": "success",
        "result": info
    }
    return result




if __name__ == "__main__":
    # 1. 设置目录
    image_dir = "<ANON_ABS_PATH> Segmentation/1. Original Images/a. Training Set"
    output_dir = "<ANON_ABS_PATH>"
    json_output_path = os.path.join(output_dir, "segmentation_results.json")

    # 2. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 3. 检查输入目录是否存在
    if not os.path.isdir(image_dir):
        print(f"错误: 输入目录 '{image_dir}' 不存在或不是一个目录。")
        sys.exit(1)

    # 4. 存储所有结果
    all_results = []
    
    # 5. 遍历目录中的所有图片
    try:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"错误: 输入目录 '{image_dir}' 不存在。")
        sys.exit(1)

    print(f"在 '{image_dir}' 中找到 {len(image_files)} 张图片。")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        
        # 为每张图片创建一个唯一的输出子目录
        image_basename = os.path.splitext(image_name)[0]
        segmentation_output_path = os.path.join(output_dir, image_basename)
        os.makedirs(segmentation_output_path, exist_ok=True)

        print(f"--- 开始处理: {image_name} ---")
        
        try:
            # 调用分割函数，不传递 device 参数
            result = segment_by_ddcs(
                image_path=image_path,
                output_path=segmentation_output_path
            )
            
            # 将图片标识和结果添加到列表中
            result_entry = {
                "image_name": image_name,
                "output_path": segmentation_output_path,
                "status": "success",
                "analysis_result": result
            }
            all_results.append(result_entry)
            
            print(f"成功处理: {image_name}")

        except Exception as e:
            print(f"处理 '{image_name}' 时发生错误: {e}")
            # 记录失败信息
            error_entry = {
                "image_name": image_name,
                "status": "error",
                "message": str(e)
            }
            all_results.append(error_entry)
        
        print(f"--- 处理完成: {image_name} ---\n")

    # 6. 将所有结果写入一个JSON文件
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"所有结果已成功保存到: {json_output_path}")
    except Exception as e:
        print(f"将结果写入JSON文件时出错: {e}")




