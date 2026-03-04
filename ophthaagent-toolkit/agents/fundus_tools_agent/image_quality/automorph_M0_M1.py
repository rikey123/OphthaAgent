import os
import subprocess
import argparse
import shutil
import json
import csv
import pandas as pd
def get_row_by_basename(csv_dir, target_basename):
    csv_path = os.path.join(csv_dir, 'results_ensemble.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')

    # 提取 Name 列中每个路径的 basename（即文件名）
    df_basenames = df['Name'].apply(os.path.basename)

    # 找到 basename 等于 target_basename 的行
    matched_rows = df[df_basenames == target_basename]

    if matched_rows.empty:
        raise ValueError(f"No row found with basename = '{target_basename}' in 'Name' column")

    # 返回第一匹配行（如有多个，可根据需求调整）
    return matched_rows.iloc[0].to_dict()

def run_command(command, working_dir, shell=False):
    """在指定的工作目录中运行命令。"""
    env = os.environ.copy()
    automorph_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'tools', 'AutoMorph'))
    env['PYTHONPATH'] = f"{automorph_dir}:{env.get('PYTHONPATH', '')}"

    automorph_data_dir = os.getenv("AUTOMORPH_DATA")
    if automorph_data_dir:
        env['AUTOMORPH_DATA'] = automorph_data_dir

    if shell:
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        print(f"在 {working_dir} 中运行 shell 命令: '{cmd_str}'")
        process = subprocess.Popen(cmd_str, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, shell=True)
    else:
        print(f"在 {working_dir} 中运行命令: '{' '.join(command)}'")
        process = subprocess.Popen(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # 实时输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        if shell:
            cmd_to_print = command
        else:
            cmd_to_print = ' '.join(command)
        print(f"运行命令时出错: {cmd_to_print}。返回码: {rc}")
    return rc

def main(image_path):
    """
    主函数，按照正确的流程运行 AutoMorph 的 M0 和 M1 阶段，并输出质量评估结果。
    """
    script_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'tools', 'AutoMorph'))
    
    # --- 步骤 1: 将图像放入 AutoMorph/images 目录 ---
    images_dir = os.path.join(script_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"源图像未找到: {image_path}"}
        
    destination_image_path = os.path.join(images_dir, os.path.basename(image_path))
    shutil.copy(image_path, destination_image_path)

    # --- 步骤 2: 运行 generate_resolution.py ---
    run_command(['python', 'generate_resolution.py'], script_dir)

    # --- 步骤 3: 准备 AUTOMORPH_DATA 目录并清理结果 ---
    run_command(['python', 'automorph_data.py'], script_dir)
    
    automorph_data_dir = os.getenv("AUTOMORPH_DATA")
    if not automorph_data_dir:
        print("警告: 未设置 AUTOMORPH_DATA。假设结果在本地 ./Results 目录中。")
        results_dir_path = os.path.join(script_dir, "Results")
    else:
        results_dir_path = os.path.join(automorph_data_dir, "Results")

    if os.path.exists(results_dir_path):
        m0_path = os.path.join(results_dir_path, "M0")
        if os.path.exists(m0_path): shutil.rmtree(m0_path)
        m1_path = os.path.join(results_dir_path, "M1")
        if os.path.exists(m1_path): shutil.rmtree(m1_path)
    else:
        os.makedirs(results_dir_path)

    # --- 步骤 4: 图像预处理 (M0) ---
    print("### M0: 预处理开始 ###")
    m0_dir = os.path.join(script_dir, 'M0_Preprocess')
    if run_command(['python', 'EyeQ_process_main.py'], m0_dir) != 0:
        return {"status": "error", "message": "M0 预处理失败"}
    print("### M0: 预处理完成 ###")

    # --- 步骤 5: 图像质量评估 (M1) ---
    print("### M1: 图像质量评估开始 ###")
    m1_dir = os.path.join(script_dir, 'M1_Retinal_Image_quality_EyePACS')
    
    shell_command = "sed 's/\r$//' test_outside.sh | bash"
    if run_command(shell_command, m1_dir, shell=True) != 0:
        return {"status": "error", "message": "M1 质量评估失败 (test_outside.sh)"}
    
    if run_command(['python', 'merge_quality_assessment.py'], m1_dir) != 0:
        return {"status": "error", "message": "M1 质量评估失败 (merge_quality_assessment.py)"}
    print("### M1: 图像质量评估完成 ###")

    # --- 步骤 6: 检查质量评估文件夹 ---
    
    image_basename = os.path.basename(image_path)
    image_name_no_ext, _ = os.path.splitext(image_basename)
    # M1 阶段会将图像转换为 png 格式
    processed_image_name = f"{image_name_no_ext}.png"

    good_quality_path = os.path.join(results_dir_path, "M1", "Good_quality", processed_image_name)
    bad_quality_path = os.path.join(results_dir_path, "M1", "Bad_quality", processed_image_name)

    quality_label = None
    if os.path.exists(good_quality_path):
        quality_label = "Good"
    elif os.path.exists(bad_quality_path):
        quality_label = "Bad"

    if quality_label is None:
        return {"status": "error", "message": f"在 Good_quality 或 Bad_quality 目录中未找到对应的图像 '{processed_image_name}'"}

    assessment = "Good Quality" if quality_label == "Good" else "Bad quality"
    label_int = 1 if quality_label == "Good" else 0
    indicators = get_row_by_basename(m1_path, processed_image_name)
    return {
        "status": "success",
        "result": {
            "label": label_int,
            "assessment": assessment,
            "indicators": indicators
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为单个图像运行 AutoMorph 的 M0 和 M1 阶段，并输出JSON格式的质量评估结果。")
    parser.add_argument("image_path", type=str, help="输入图像的绝对路径。")
    args = parser.parse_args()

    result = main(args.image_path)
    print(json.dumps(result, ensure_ascii=False))