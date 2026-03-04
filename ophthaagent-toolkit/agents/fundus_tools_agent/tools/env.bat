#!/bin/bash

# 自动安装 Python 环境和依赖的脚本
conda create -n ietk python = 3.11
conda create -n fundus_lesions_toolkit python =3.11
conda activate ietk
pip install ietk-ret
conda activate fundus_lesions_toolkit
cd fundus-lesions-toolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install .



echo "🎉 Python 环境配置完成！"
echo "💡 使用以下命令激活环境：source .venv/bin/activate"