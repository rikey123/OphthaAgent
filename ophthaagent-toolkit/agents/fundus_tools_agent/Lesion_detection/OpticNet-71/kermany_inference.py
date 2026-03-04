import tensorflow as tf
import keras
import argparse
import keras.backend as K
from keras.models import load_model
import cv2
import numpy as np
import json
import sys
import os
import traceback

# 在文件开头添加设备配置
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from device_config import get_device

def setup_tensorflow_device():
    """配置TensorFlow设备"""
    device = get_device()
    
    if str(device) == 'cpu':
        # 强制使用CPU
        tf.config.set_visible_devices([], 'GPU')
        print("TensorFlow configured to use CPU")
    elif 'cuda' in str(device):
        # 强制使用特定GPU
        gpu_idx = str(device).split(':')[-1] if ':' in str(device) else '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        
        # 配置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"TensorFlow configured to use GPU {gpu_idx}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    else:
        print("Using default TensorFlow device configuration")

def image_preprocessing(img):
    """
    预处理输入图像
    """
    img = cv2.imread(img)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    img = 1.0*img/255

    return img


def predict_with_kermany_model(img_path, weights_path):
    """
    使用Kermany2018权重进行预测
    """
    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    
    # Kermany2018数据集的类别
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    # 预处理图像
    processed_img = image_preprocessing(img_path)
    
    # 加载模型并预测
    try:
        K.clear_session()
        model = load_model(weights_path)
        preds = model.predict(processed_img, batch_size=None, steps=1)
    except Exception as e:
        print(f"加载模型或预测时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    # 整理预测结果
    preds = preds.ravel()
    result = {}
    for i, class_name in enumerate(classes):
        result[class_name] = float(np.around(preds[i], decimals=4))
    
    return result


def kermany_inference(img_path, kermany_weights):
    """
    使用Kermany2018预训练权重进行推理
    """
    try:
        # 对Kermany2018数据集的模型进行预测
        kermany_result = predict_with_kermany_model(img_path, kermany_weights)
        
        # 组合结果
        final_result = {
            "image_path": img_path,
            "predictions": {
                "kermany2018": kermany_result
            }
        }
        
        return final_result
    except Exception as e:
        print(f"推理过程中发生错误: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='使用Kermany2018预训练权重对OCT图像进行推理')
    parser.add_argument('--imgpath', type=str, required=True, help='待推理图像的路径')
    parser.add_argument('--kermany_weights', type=str, default='Kermany2018.hdf5', help='Kermany2018模型权重路径')
    parser.add_argument('--output', type=str, help='输出JSON文件路径（可选，默认为stdout）')
    
    args = parser.parse_args()
    setup_tensorflow_device()
    # 检查文件是否存在
    if not os.path.exists(args.imgpath):
        print(f"错误: 图像文件 '{args.imgpath}' 不存在")
        sys.exit(1)
    
    # 执行推理
    result = kermany_inference(args.imgpath, args.kermany_weights)
    
    # 输出结果
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"结果已保存到 {args.output}")
    else:
        print(json_output)


if __name__ == '__main__':
    main()