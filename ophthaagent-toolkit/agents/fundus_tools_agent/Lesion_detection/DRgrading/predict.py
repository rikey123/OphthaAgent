import os
import cv2
import torch
import json
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# 导入模型定义
from models import DRDINOv2Model

class GrahamEnhancement:
    """Graham图像增强"""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        blur = cv2.GaussianBlur(img, (0, 0), self.sigma)
        img = cv2.addWeighted(img, 4, blur, -4, 128)
        return Image.fromarray(img)

def get_transform(img_size):
    """获取图像预处理变换"""
    sigma = img_size / 30
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        GrahamEnhancement(sigma=sigma),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model(model_path, model_name, device):
    """加载模型和权重"""
    print(f"正在加载模型: {model_name}")
    model = DRDINOv2Model(model_name=model_name).to(device)

    # 检查权重文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重文件: {model_path}")

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    raw_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    print("模型加载成功!")
    return model

def predict_image(image_path, model, transform, device):
    """对单张图片进行预测"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    # 读取图片
    image = Image.open(image_path).convert('RGB')
    print(f"\n图片信息:")
    print(f"  路径: {image_path}")
    print(f"  原始尺寸: {image.size}")

    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(input_tensor)

    # 计算概率
    probabilities = torch.softmax(outputs, dim=1)[0]  # [5]
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities.cpu().numpy()

def print_results(predicted_class, probabilities):
    """美化输出预测结果"""
    dr_labels = {
        0: "0 - 正常",
        1: "1 - 轻度",
        2: "2 - 中度",
        3: "3 - 重度",
        4: "4 - 增生性"
    }

    print("\n" + "="*50)
    print("预测结果:")
    print("="*50)

    # 显示预测等级
    print(f"\n预测分级: {dr_labels[predicted_class]}")
    print(f"类别编号: {predicted_class}")

    # 显示概率分布
    print("\n各级别概率分布:")
    print("-"*50)

    for i in range(5):
        bar_length = int(probabilities[i] * 40)
        bar = "█" * bar_length
        print(f"  {dr_labels[i]:12s} | {probabilities[i]*100:6.2f}% {bar}")

    print("-"*50)
    print(f"\n置信度: {probabilities[predicted_class]*100:.2f}%")
    print("="*50)

def format_json_result(image_path, predicted_class, probabilities, model_name, img_size):
    """格式化 JSON 输出"""
    dr_labels = {
        0: "正常",
        1: "轻度",
        2: "中度",
        3: "重度",
        4: "增生性"
    }

    result = {
        "image_path": image_path,
        "prediction": {
            "class_id": int(predicted_class),
            "class_name": dr_labels[predicted_class],
            "confidence": float(probabilities[predicted_class])
        },
        "probabilities": {
            "0_normal": float(probabilities[0]),
            "1_mild": float(probabilities[1]),
            "2_moderate": float(probabilities[2]),
            "3_severe": float(probabilities[3]),
            "4_proliferative": float(probabilities[4])
        },
        "model_info": {
            "model_name": model_name,
            "input_size": img_size
        }
    }
    return result

def main(args):
    # 设置设备
    #device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # 添加项目根目录到 sys.path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
    from device_config import get_device
    device = get_device()
    print(f"使用设备: {device}")

    # 构建模型权重路径
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    if args.model_path:
        model_path = args.model_path

    # 加载模型
    model = load_model(model_path, args.model_name, device)

    # 获取预处理
    transform = get_transform(args.img_size)

    # 预测
    try:
        predicted_class, probabilities = predict_image(
            args.image_path, model, transform, device
        )

        # 格式化 JSON 结果
        json_result = format_json_result(
            args.image_path, predicted_class, probabilities,
            args.model_name, args.img_size
        )

        # 如果不启用 JSON 输出，则显示美化格式
        if not args.json_output:
            print_results(predicted_class, probabilities)

        # JSON 输出模式
        if args.json_output:
            # 打印 JSON 到控制台
            print("\n" + "="*50)
            print("JSON 格式输出:")
            print("="*50)
            print(json.dumps(json_result, indent=2, ensure_ascii=False))
            print("="*50)

        # 保存结果到文件
        if args.save_result:
            result_dir = args.output_dir if args.output_dir else "."
            os.makedirs(result_dir, exist_ok=True)

            # 根据 json_output 参数决定保存格式
            if args.json_output:
                result_file = os.path.join(result_dir, "prediction_result.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
            else:
                result_file = os.path.join(result_dir, "prediction_result.txt")
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Image Path: {args.image_path}\n")
                    f.write(f"Predicted Class: {predicted_class}\n")
                    f.write("Probabilities:\n")
                    for i in range(5):
                        f.write(f"  Class {i}: {probabilities[i]*100:.2f}%\n")

            print(f"\n结果已保存至: {result_file}")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR分级预测 - 单张图片")

    # 必需参数
    parser.add_argument('--image_path', type=str, required=True,
                        help='待预测的图片路径')

    # 可选参数
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重文件路径 (默认: output_dir/best_model.pth)')
    parser.add_argument('--model_name', type=str,
                        default='vit_base_patch14_dinov2.lvd142m',
                        help='模型名称 (默认: vit_base_patch14_dinov2.lvd142m)')
    parser.add_argument('--img_size', type=int, default=518,
                        help='输入图像尺寸 (默认: 518)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='输出目录 (默认: 当前目录)')
    parser.add_argument('--save_result', action='store_true',
                        help='是否保存预测结果到文件')
    parser.add_argument('--json_output', action='store_true',
                        help='使用 JSON 格式输出（同时影响控制台显示和文件保存格式）')

    args = parser.parse_args()
    main(args)
