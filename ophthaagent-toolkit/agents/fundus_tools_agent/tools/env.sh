#!/bin/bash
set -e  # 出错立即退出

# 初始化 conda（关键！）
eval "$(conda shell.bash hook)"

# 创建并安装 ietk 环境
echo "📦 创建 ietk 环境..."
conda create -n ietk python=3.11 -y
conda activate ietk
pip install ietk-ret

# 创建并安装 fundus_lesions_toolkit 环境
echo "📦 创建 fundus_lesions_toolkit 环境..."
conda create -n fundus_lesions_toolkit python=3.11 -y
conda activate fundus_lesions_toolkit

# 检查 fundus-lesions-toolkit 目录是否存在
if [ ! -d "fundus-lesions-toolkit" ]; then
    echo "❌ 错误: fundus-lesions-toolkit 目录不存在，请先克隆或放置该目录。"
    exit 1
fi

cd fundus-lesions-toolkit
echo "📥 安装 PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "📥 安装本地包..."
pip install .

# 创建并安装 fundus_image_toolbox 环境
echo "📦 创建 fundus_image_toolbox 环境..."
conda create -n fundus_image_toolbox python=3.9.19 pip -y
conda activate fundus_image_toolbox
pip install fundus_image_toolbox

# 创建并安装 AutoMorph 环境
echo "📦 创建 fundus_image_toolbox 环境..."
conda create -n fundus_image_toolbox python=3.9.19 pip -y
conda activate fundus_image_toolbox
pip install fundus_image_toolbox

echo "🎉 所有 Python 环境配置完成！"