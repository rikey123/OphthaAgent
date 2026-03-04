import os
import subprocess

def main():
    python_path = "<ANON_ABS_PATH>"
    script_path = os.path.abspath("agents/fundus_tools_agent/Lesion_detection/MKCNet/evaluate_messidor2_5class.py")
    
    csv_path = "<ANON_ABS_PATH>"
    images_dir = "<ANON_ABS_PATH>"
    
    # Model paths (Assuming training saves here)
    model_dir = os.path.abspath("agents/fundus_tools_agent/Lesion_detection/MKCNet/model/MKCNet_DeepDRiD_5Class")
    model_path = os.path.join(model_dir, "best_model.pth")
    meta_model_path = os.path.join(model_dir, "best_model_LG.pth")
    
    cmd = [
        python_path,
        script_path,
        "--csv_path", csv_path,
        "--images_dir", images_dir,
        "--model_path", model_path,
        "--meta_model_path", meta_model_path,
        "--device", "cuda:1"
    ]
    
    print(f"Running evaluation: {' '.join(cmd)}")
    
    # Run in MKCNet dir
    cwd = os.path.abspath("agents/fundus_tools_agent/Lesion_detection/MKCNet")
    subprocess.run(cmd, cwd=cwd)

if __name__ == "__main__":
    main()
