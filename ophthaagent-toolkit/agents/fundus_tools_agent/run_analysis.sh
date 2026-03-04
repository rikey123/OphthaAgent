#!/bin/bash

# 获取脚本所在的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Conda 环境名称
CONDA_ENV_NAME="deepseenetplus"

# 查找 Conda 的安装路径
CONDA_BASE=$(conda info --base)

# 加载 Conda 的 shell 函数
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活 Conda 环境
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

# 手动设置 LD_LIBRARY_PATH，确保能找到Conda环境中的库文件
CONDA_LIB_PATH="$CONDA_BASE/envs/$CONDA_ENV_NAME/lib"
echo "Manually setting LD_LIBRARY_PATH to include: $CONDA_LIB_PATH"
export LD_LIBRARY_PATH="$CONDA_LIB_PATH:$LD_LIBRARY_PATH"

# 定义并执行主分析脚本
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/tool_interface4.py"
echo "\nRunning the analysis script..."
python "$PYTHON_SCRIPT_PATH"
