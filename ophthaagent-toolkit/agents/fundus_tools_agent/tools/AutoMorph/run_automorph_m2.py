import os
import subprocess
import shutil
import glob

def run_command(command, working_dir, shell=False):
    """在指定的工作目录中运行命令。"""
    # 为子进程设置环境
    env = os.environ.copy()
    automorph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    env['PYTHONPATH'] = f"{automorph_dir}:{env.get('PYTHONPATH', '')}"
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

    # 实时输出
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

def main():
    """
    主函数，为 Good_quality 目录下的图像运行 AutoMorph 的 M2 阶段。
    结果将存储在 AutoMorph/Results 目录中。
    """
    automorph_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"AutoMorph 根目录: {automorph_dir}")

    # --- 步骤 1: 检查 M0/M1 阶段的产物 ---
    print("### 正在检查 M0/M1 阶段的产物... ###")
    good_quality_dir = os.path.join(automorph_dir, 'Results', 'M1', 'Good_quality')
    if not os.path.isdir(good_quality_dir) or not any(fname.endswith('.png') for fname in os.listdir(good_quality_dir)):
        print(f"错误: 在 {good_quality_dir} 中没有找到 M1 阶段生成的 .png 图像。")
        print("请确保 M0 和 M1 阶段已成功运行。")
        return

    crop_info_file = os.path.join(automorph_dir, 'Results', 'M0', 'crop_info.csv')
    if not os.path.isfile(crop_info_file):
        print(f"错误: M0 阶段的产物 {crop_info_file} 未找到。")
        print("请确保 M0 阶段已成功运行。")
        return
    print("### M0/M1 产物检查通过。 ###")

    # --- 步骤 2: 运行分割模块 (M2) ---
    print("### M2: 分割模块开始 ###")

    # M2_Vessel_seg
    m2_vessel_dir = os.path.join(automorph_dir, 'M2_Vessel_seg')
    print("--- 开始血管分割 ---")
    run_command("sed 's/\r$//' test_outside.sh | bash", m2_vessel_dir, shell=True)
    print("--- 血管分割完成 ---")

    # M2_Artery_vein
    m2_av_dir = os.path.join(automorph_dir, 'M2_Artery_vein')
    print("--- 开始动静脉分类 ---")
    run_command("sed 's/\r$//' test_outside.sh | bash", m2_av_dir, shell=True)
    print("--- 动静脉分类完成 ---")

    # M2_lwnet_disc_cup
    m2_disc_cup_dir = os.path.join(automorph_dir, 'M2_lwnet_disc_cup')
    print("--- 开始视杯视盘分割 ---")
    run_command("sed 's/\r$//' test_outside.sh | bash", m2_disc_cup_dir, shell=True)
    print("--- 视杯视盘分割完成 ---")

    print("### M2: 分割模块完成 ###")
    print("### 全部完成 ###")

if __name__ == "__main__":
    main()