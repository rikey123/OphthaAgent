import os
import subprocess
import argparse
import json

AUTOMORPH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'AutoMorph'))

def run_command(command, working_dir, shell=False):
    """在指定的工作目录中运行命令。"""
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{AUTOMORPH_DIR}:{env.get('PYTHONPATH', '')}"
    automorph_data_dir = os.getenv("AUTOMORPH_DATA")
    if automorph_data_dir:
        env['AUTOMORPH_DATA'] = automorph_data_dir

    if shell:
        cmd_str = command
        print(f"在 {working_dir} 中运行 shell 命令: '{cmd_str}'")
        process = subprocess.Popen(cmd_str, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, shell=True, executable='/bin/bash')
    else:
        print(f"在 {working_dir} 中运行命令: '{' '.join(command)}'")
        process = subprocess.Popen(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    if rc != 0:
        cmd_to_print = command if shell else ' '.join(command)
        print(f"运行命令时出错: {cmd_to_print}。返回码: {rc}")
    return rc

def main(image_path):
    """
    主函数，为 Good_quality 目录下的图像运行 AutoMorph 的 M2 阶段，
    并返回与输入图像匹配的分割结果路径。
    """
    automorph_dir = AUTOMORPH_DIR

    # --- 步骤 1: 检查 M0/M1 阶段的产物 ---
    print("### 正在检查 M0/M1 阶段的产物... ###")
    good_quality_dir = os.path.join(automorph_dir, 'Results', 'M1', 'Good_quality')
    if not os.path.isdir(good_quality_dir) or not any(fname.endswith('.png') for fname in os.listdir(good_quality_dir)):
        return {"status": "error", "message": f"在 {good_quality_dir} 中没有找到 M1 阶段生成的 .png 图像。请确保 M0 和 M1 阶段已成功运行。"}

    crop_info_file = os.path.join(automorph_dir, 'Results', 'M0', 'crop_info.csv')
    if not os.path.isfile(crop_info_file):
        return {"status": "error", "message": f"M0 阶段的产物 {crop_info_file} 未找到。请确保 M0 阶段已成功运行。"}
    print("### M0/M1 产物检查通过。 ###")

    # --- 步骤 2: 运行分割模块 (M2) ---
    print("### M2: 分割模块开始 ###")
    m2_vessel_dir = os.path.join(automorph_dir, 'M2_Vessel_seg')
    print("--- 开始血管分割 ---")
    if run_command("sed 's/\r$//' test_outside.sh | bash", m2_vessel_dir, shell=True) != 0:
        return {"status": "error", "message": "M2 血管分割失败"}
    print("--- 血管分割完成 ---")

    m2_av_dir = os.path.join(automorph_dir, 'M2_Artery_vein')
    print("--- 开始动静脉分类 ---")
    if run_command("sed 's/\r$//' test_outside.sh | bash", m2_av_dir, shell=True) != 0:
        return {"status": "error", "message": "M2 动静脉分类失败"}
    print("--- 动静脉分类完成 ---")

    m2_disc_cup_dir = os.path.join(automorph_dir, 'M2_lwnet_disc_cup')
    print("--- 开始视杯视盘分割 ---")
    if run_command("sed 's/\r$//' test_outside.sh | bash", m2_disc_cup_dir, shell=True) != 0:
        return {"status": "error", "message": "M2 视杯视盘分割失败"}
    print("--- 视杯视盘分割完成 ---")
    print("### M2: 分割模块完成 ###")

    # --- 步骤 3: 查找并返回结果路径 ---
    print("### 正在查找分割结果... ###")
    image_name_no_ext, _ = os.path.splitext(os.path.basename(image_path))
    result_image_name = f"{image_name_no_ext}.png"
    
    results_m2_dir = os.path.join(automorph_dir, 'Results', 'M2')
    
    av_path = os.path.join(results_m2_dir, 'artery_vein', 'raw', result_image_name)
    bv_path = os.path.join(results_m2_dir, 'binary_vessel', 'raw', result_image_name)
    od_path = os.path.join(results_m2_dir, 'optic_disc_cup', 'raw', result_image_name)

    found_paths = {}
    all_found = True
    if os.path.exists(av_path):
        found_paths['artery_vein'] = av_path
    else:
        all_found = False

    if os.path.exists(bv_path):
        found_paths['binary_vessel'] = bv_path
    else:
        all_found = False

    if os.path.exists(od_path):
        found_paths['optic_disc_cup'] = od_path
    else:
        all_found = False

    if all_found:
        return {
            "status": "success",
            "result": found_paths
        }
    else:
        missing = [p for p in [av_path, bv_path, od_path] if not os.path.exists(p)]
        return {
            "status": "error",
            "message": f"未能找到所有分割结果。缺失的文件: {', '.join(missing)}"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为单个图像运行 AutoMorph 的 M2 阶段，并返回分割结果的路径。")
    parser.add_argument("image_path", type=str, help="输入图像的绝对路径。")
    args = parser.parse_args()
    
    result = main(args.image_path)
    print(json.dumps(result, ensure_ascii=False, indent=4))