#!/bin/bash

# ===================================================================
# Hybrid CNN-ViT Network 训练脚本 - RetinalOCT数据集
# ===================================================================
# 
# 这个脚本用于训练混合CNN+Vision Transformer模型，在RetinalOCT数据集上进行8分类
#
# 主要参数说明：
# --epochs: 训练轮数（默认30轮）
# --bs: 批次大小（每次训练使用多少张图片）
# --lr: 学习率（控制模型参数更新的步长）
# --cfg_path: 模型配置文件路径
# --train_data: 训练数据路径
# --test_data: 测试数据路径（这里使用验证集）
# --dsname: 保存的模型文件名
# --msp: 模型保存路径
# --log_path: 日志保存路径
# --print_confusion_matrix: 是否打印混淆矩阵
# ===================================================================

# 激活conda虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mgunet_new

# 设置使用GPU 0(空闲GPU)
export CUDA_VISIBLE_DEVICES=0

# 设置项目根目录
PROJECT_DIR="<ANON_ABS_PATH>"
DATA_DIR="<ANON_ABS_PATH>"

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_DIR}/outputs/retinal_oct_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "========================================"
echo "开始训练 Hybrid CNN-ViT 模型"
echo "========================================"
echo "数据集: RetinalOCT (8分类)"
echo "训练数据: ${DATA_DIR}/train"
echo "验证数据: ${DATA_DIR}/val"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用GPU: 0"
echo "========================================"

# 进入项目目录
cd ${PROJECT_DIR}

# 执行训练
python experiment.py \
    --epochs 30 \
    --bs 32 \
    --lr 0.0001 \
    --cfg_path model_cfg_oct8.json \
    --train_data ${DATA_DIR}/train \
    --test_data ${DATA_DIR}/val \
    --dsname retinal_oct_model.pth \
    --msp ${OUTPUT_DIR} \
    --psl ${OUTPUT_DIR} \
    --print_confusion_matrix False

echo "========================================"
echo "训练完成！"
echo "模型保存位置: ${OUTPUT_DIR}/retinal_oct_model.pth"
echo "训练日志: ${OUTPUT_DIR}/log.txt"
echo "训练曲线: ${OUTPUT_DIR}/training_curves.png"
echo "========================================"
