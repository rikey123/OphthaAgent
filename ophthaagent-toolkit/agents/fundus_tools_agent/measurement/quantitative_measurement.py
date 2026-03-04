import os
import subprocess
import sys
import json
import pandas as pd

# 确定 AutoMorph 的根目录
AUTOMORPH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'AutoMorph'))

def run_command(command, working_dir):
    """在指定的工作目录中运行命令，并实时打印输出。"""
    env = os.environ.copy()
    # 设置 AUTOMORPH_DATA 环境变量
    env['AUTOMORPH_DATA'] = AUTOMORPH_DIR
    # 将 AutoMorph 根目录和父目录添加到 PYTHONPATH
    env['PYTHONPATH'] = f"{AUTOMORPH_DIR}:{os.path.dirname(AUTOMORPH_DIR)}:{env.get('PYTHONPATH', '')}"
    
    try:
        process = subprocess.Popen(
            command, 
            cwd=working_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            env=env,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line.strip())

        process.wait()
        rc = process.poll()
        if rc != 0:
            print(f"命令 {' '.join(command)} 执行失败，返回码: {rc}", file=sys.stderr)
        return rc
    except FileNotFoundError:
        print(f"命令未找到: {command[0]}", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"执行命令时发生未知错误: {e}", file=sys.stderr)
        return -1

def get_features(csv_path, image_name):
    """从 CSV 文件中提取特定图像的特征，忽略文件扩展名。"""
    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        # 确保 'Name' 列是字符串类型
        df['Name'] = df['Name'].astype(str)
        
        # 从输入图像名中获取不带扩展名的部分
        image_basename = os.path.splitext(image_name)[0]
        
        # 在DataFrame中查找不带扩展名匹配的行
        # 使用 .str.rsplit('.', n=1).str[0] 更安全，以防文件名中有点
        image_row = df[df['Name'].str.rsplit('.', n=1).str[0] == image_basename]
        
        if image_row.empty:
            return {"error": f"Image '{image_basename}' not found in {os.path.basename(csv_path)}"}
            
        # 将 NaN 替换为 None (在 JSON 中会是 null)
        return image_row.iloc[0].where(pd.notnull(image_row.iloc[0]), None).to_dict()
    except Exception as e:
        return {"error": f"Failed to read or process {csv_path}: {e}"}

import shutil

def backup_and_restore_m3_results(m3_results_dir, backup_dir, action='backup'):
    """备份或恢复 M3 结果目录的内容。"""
    if action == 'backup':
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        if os.path.exists(m3_results_dir):
            shutil.copytree(m3_results_dir, backup_dir)
            print(f"Backed up contents of {m3_results_dir} to {backup_dir}")
    elif action == 'restore':
        if os.path.exists(backup_dir):
            # 先删除当前M3目录，再从备份恢复
            if os.path.exists(m3_results_dir):
                shutil.rmtree(m3_results_dir)
            shutil.copytree(backup_dir, m3_results_dir)
            print(f"Restored contents to {m3_results_dir} from {backup_dir}")
            shutil.rmtree(backup_dir) # 清理备份

def main(image_path):
    """
    主函数，运行 AutoMorph 的 M3 阶段（特征测量），并提取指定图像的结果。
    """
    automorph_dir = AUTOMORPH_DIR
    image_name_with_ext = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name_with_ext)[0]
    
    m3_results_dir = os.path.join(automorph_dir, 'Results', 'M3')
    backup_dir = os.path.join(automorph_dir, 'Results', 'M3_backup')

    # --- 备份 M3 目录 ---
    backup_and_restore_m3_results(m3_results_dir, backup_dir, action='backup')

    print(f"AutoMorph 根目录: {automorph_dir}")
    print(f"目标图像: {image_name_with_ext} (匹配名: {image_name_without_ext})")

    try:
        print("### M3: 特征测量开始 ###")

        # --- 步骤 1: 在 M3_feature_whole_pic/retipy 中运行脚本 (生成基础 CSV) ---
        m3_feature_whole_pic_dir = os.path.join(automorph_dir, 'M3_feature_whole_pic', 'retipy')
        scripts_whole = [
            'create_datasets_macular_centred.py',
            'create_datasets_disc_centred.py'
        ]
        for script in scripts_whole:
            if run_command(['python', script], m3_feature_whole_pic_dir) != 0:
                raise RuntimeError(f"Failed to run {script}")

        # --- 步骤 2: 在 M3_feature_zone/retipy 中运行脚本 (添加区域特征) ---
        m3_feature_zone_dir = os.path.join(automorph_dir, 'M3_feature_zone', 'retipy')
        scripts_zone = [
            'create_datasets_disc_centred_B.py',
            'create_datasets_disc_centred_C.py',
            'create_datasets_macular_centred_B.py',
            'create_datasets_macular_centred_C.py'
        ]
        for script in scripts_zone:
            if run_command(['python', script], m3_feature_zone_dir) != 0:
                raise RuntimeError(f"Failed to run {script}")

        # --- 步骤 3: 运行 csv_merge.py ---
        if run_command(['python', 'csv_merge.py'], automorph_dir) != 0:
            raise RuntimeError("Failed to run csv_merge.py")
            
        print("### M3: 特征测量完成 ###")
        print("### 开始提取特征 ###")

        # --- 步骤 4: 提取并输出特征 ---
        macular_csv_path = os.path.join(m3_results_dir, 'Macular_Features.csv')
        disc_csv_path = os.path.join(m3_results_dir, 'Disc_Features.csv')

        macular_features = get_features(macular_csv_path, image_name_with_ext)
        disc_features = get_features(disc_csv_path, image_name_with_ext)

        result = {
            "macular_features": macular_features,
            "disc_features": disc_features
        }
        
        # 检查是否有错误
        if "error" in macular_features and "error" in disc_features:
            print(json.dumps({"status": "error", "result": result}, ensure_ascii=False, indent=4))
        else:
            print(json.dumps({"status": "success", "result": result}, ensure_ascii=False, indent=4))

    except RuntimeError as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
    # finally:
    #     # --- 恢复 M3 目录 ---
    #     backup_and_restore_m3_results(m3_results_dir, backup_dir, action='restore')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"status": "error", "message": "Usage: python quantitative_measurement.py <image_path>"}, ensure_ascii=False))
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    
    if not os.path.exists(input_image_path):
        print(json.dumps({"status": "error", "message": f"Input image not found: {input_image_path}"}, ensure_ascii=False))
        sys.exit(1)
        
    main(input_image_path)