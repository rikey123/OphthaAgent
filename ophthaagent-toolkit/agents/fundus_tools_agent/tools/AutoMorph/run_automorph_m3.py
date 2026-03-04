import os
import subprocess

def run_command(command, working_dir):
    """在指定的工作目录中运行命令。"""
    env = os.environ.copy()
    automorph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    env['PYTHONPATH'] = f"{automorph_dir}:{env.get('PYTHONPATH', '')}"
    
    process = subprocess.Popen(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    if rc != 0:
        print(f"运行命令时出错: {' '.join(command)}。返回码: {rc}")
    return rc

def main():
    """
    主函数，运行 AutoMorph 的 M3 阶段（特征测量）。
    """
    automorph_dir = os.path.abspath(os.path.dirname(__file__))

    print("### M3: 特征测量开始 ###")

    # --- 步骤 1: 在 M3_feature_zone/retipy 中运行脚本 ---
    m3_feature_zone_dir = os.path.join(automorph_dir, 'M3_feature_zone', 'retipy')
    
    run_command(['python', 'create_datasets_disc_centred_B.py'], m3_feature_zone_dir)
    run_command(['python', 'create_datasets_disc_centred_C.py'], m3_feature_zone_dir)
    run_command(['python', 'create_datasets_macular_centred_B.py'], m3_feature_zone_dir)
    run_command(['python', 'create_datasets_macular_centred_C.py'], m3_feature_zone_dir)

    # --- 步骤 2: 在 M3_feature_whole_pic/retipy 中运行脚本 ---
    m3_feature_whole_pic_dir = os.path.join(automorph_dir, 'M3_feature_whole_pic', 'retipy')

    run_command(['python', 'create_datasets_macular_centred.py'], m3_feature_whole_pic_dir)
    run_command(['python', 'create_datasets_disc_centred.py'], m3_feature_whole_pic_dir)

    # --- 步骤 3: 运行 csv_merge.py ---
    run_command(['python', 'csv_merge.py'], automorph_dir)


if __name__ == "__main__":
    main()