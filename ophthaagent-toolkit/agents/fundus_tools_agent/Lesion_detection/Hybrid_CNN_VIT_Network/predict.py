
import torch
from torchvision import transforms
from PIL import Image
import argparse
import json
import os
import sys
# 从项目文件中导入模型定义
from models.model import VitConvNet

def predict(image_path, model_path, config_path, output_json=False):
    """
    使用指定的模型和配置对单个图像进行推理。

    参数:
    - image_path (str): 要进行推理的输入图像的路径。
    - model_path (str): 预训练模型的 .pth 文件的路径。
    - config_path (str): 定义模型架构的 JSON 配置文件的路径。
    - output_json (bool): 是否以JSON格式输出结果。
    """
    # 1. 加载模型配置
    try:
        with open(config_path, 'r') as f:
            model_cfg = json.load(f)
        # 修正：使用正确的键 'backbone_cfg' 和 'vit_cfg'
        backbone_cfg = model_cfg['backbone_cfg']
        vit_cfg = model_cfg['vit_cfg']
    except FileNotFoundError:
        print(f"错误: 配置文件未找到于 '{config_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析配置文件 '{config_path}'")
        return
    except KeyError:
        print("错误: 配置文件中缺少 'backbone_cfg' 或 'vit_cfg' 键")
        return

    # 2. 设置设备
    # 在文件开头添加
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from device_config import get_device

    # 替换原来的设备设置
    device = get_device()
    print(f"正在使用设备: {device}")

    # 3. 加载完整的预训练模型
    # .pth 文件包含的是完整的模型对象，而不是 state_dict，所以直接加载
    try:
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含自定义类的模型
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("已成功加载完整的预训练模型。")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到于 '{model_path}'")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
        
    # 4. 将模型移至设备并设置为评估模式
    model = model.to(device)
    model.eval() # 切换到评估模式

    # 5. 定义图像预处理
    # 重要：必须与训练时的预处理完全一致！
    # 训练时使用的是 utils.get_augmentation(train=False)，只做了ToTensor()
    # 但是需要确保图像尺寸正确，这里根据实际需要添加Resize
    # 注意：训练时ImageFolder会自动加载各种尺寸的图像
    # 推理时需要保证输入图像能被模型处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 只做ToTensor，与训练时一致
        # 不添加Normalize！训练时没有使用
    ])

    # 6. 加载和预处理图像
    if not os.path.exists(image_path):
        print(f"错误: 图像文件未找到于 '{image_path}'")
        return
        
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"原始图像尺寸: {image.size}")
        
        # 应用预处理变换（仅ToTensor，与训练一致）
        input_tensor = transform(image).unsqueeze(0)  # 添加批次维度 [1, C, H, W]
        print(f"输入张量形状: {input_tensor.shape}")
        print(f"张量值范围: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        
        input_tensor = input_tensor.to(device)
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return

    # 7. 执行推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 8. 映射结果到类别名称
    # 这些类别名称基于 RetinalOCT_Dataset 目录结构
    class_names = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
    predicted_class = class_names[predicted_idx.item()]

    if output_json:
        # JSON格式输出
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence.item(),
            "probabilities": {cls: prob.item() for cls, prob in zip(class_names, probabilities[0])}
        }
        print(json.dumps(result, ensure_ascii=False))
    else:
        # 原来的文本格式输出
        print("\n--- 推理完成 ---")
        print(f"图片路径: {os.path.basename(image_path)}")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence.item():.4f}")
        print("\n所有类别的概率分布:")
        for i, (cls, prob) in enumerate(zip(class_names, probabilities[0])):
            print(f"  {cls}: {prob.item():.4f}")
        print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 VitConvNet 模型对视网膜OCT图像进行分类。")
    parser.add_argument('--image_path', type=str, required=True, help='输入图像的路径。')
    parser.add_argument('--model_path', type=str, default='<ANON_ABS_PATH>', help='预训练模型的 .pth 文件的路径。')
    parser.add_argument('--config_path', type=str, default='<ANON_ABS_PATH>', help='模型配置 .json 文件的路径。')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出结果')
    
    args = parser.parse_args()

    predict(args.image_path, args.model_path, args.config_path, output_json=args.json)

    # 示例用法:
    # python predict.py --image_path /path/to/your/image.jpeg
    # python predict.py --image_path /path/to/your/image.jpeg --model_path /path/to/your/model.pth
