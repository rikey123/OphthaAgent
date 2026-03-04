#!/usr/bin/env python3
"""
DR分析器环境配置脚本
自动安装依赖并验证环境
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示状态"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误信息: {e.stderr}")
        return False

def main():
    print("🚀 DR单张图片分析器环境配置")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("需要 Python >= 3.8")
        return False

    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查是否在conda环境中
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda环境: {conda_env}")
    else:
        print("⚠️  建议在Conda环境中运行")

    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
    except ImportError:
        print("⚠️  PyTorch未安装")

    # 安装依赖
    if os.path.exists("requirements.txt"):
        if run_command("pip install -r requirements.txt", "安装Python依赖"):
            print("✅ 依赖安装完成")
        else:
            print("❌ 依赖安装失败")
            return False
    else:
        print("❌ requirements.txt 文件不存在")
        return False

    # 验证安装
    print("\n🔍 验证安装...")
    try:
        import torch
        import monai
        import cv2
        import albumentations
        print("✅ 核心依赖验证通过")
    except ImportError as e:
        print(f"❌ 依赖验证失败: {e}")
        return False

    # 检查模型文件
    model_files = ["models/best.pth", "models/config.json"]
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"⚠️  模型文件缺失: {missing_files}")
        print("请确保模型文件存在")
    else:
        print("✅ 模型文件存在")

    print("\n🎉 环境配置完成！")
    print("\n使用方法:")
    print("1. 命令行使用:")
    print("   python dr_analyzer.py --image /path/to/image.jpg --output ./output")
    print("2. Python API使用:")
    print("   from dr_analyzer import DRAnalyzer")
    print("   analyzer = DRAnalyzer()")
    print("   result = analyzer.analyze_image('image.jpg', './output')")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
