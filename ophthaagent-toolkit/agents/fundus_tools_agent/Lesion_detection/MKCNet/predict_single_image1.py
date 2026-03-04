import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import json
from model.model_manager import get_model
from configs.defaults import _C as cfg_default
import numpy as np


def load_model(cfg, model_path, meta_model_path=None):
    """
    加载预训练模型
    """
    # 获取模型
    if cfg.MODEL.NAME in cfg.ALL_MODELS:
        if cfg.MODEL.NAME in ['MKCNet', 'FirstOrder_MKCNet']:
            psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
            model, metalearner = get_model(cfg, psi)

            # 加载模型权重
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"成功加载模型权重: {model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            if meta_model_path and os.path.exists(meta_model_path):
                metalearner.load_state_dict(torch.load(meta_model_path))
                print(f"成功加载Meta模型权重: {meta_model_path}")
            elif cfg.MODEL.NAME in ['MKCNet', 'FirstOrder_MKCNet']:
                raise FileNotFoundError(f"Meta模型文件不存在: {meta_model_path}")

            return model, metalearner
        else:
            model = get_model(cfg)

            # 加载模型权重
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"成功加载模型权重: {model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            return model, None
    else:
        raise ValueError(f"不支持的模型: {cfg.MODEL.NAME}")


def preprocess_image(image_path, cfg):
    """
    预处理图像
    """
    # 打开图像
    if cfg.DATASET.NAME in ['DEEPDR', 'EYEQ']:
        image = Image.open(image_path).convert('RGB')
    else:
        image = Image.open(image_path).convert('L')

    # 定义预处理步骤
    means = cfg.DATASET.NORMALIZATION_MEAN
    stds = cfg.DATASET.NORMALIZATION_STD

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    # 应用预处理
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加batch维度


def predict_single_image(image_path, cfg, model, metalearner=None):
    """
    对单张图像进行预测
    """
    # 预处理图像
    image_tensor = preprocess_image(image_path, cfg)
    image_tensor = image_tensor.cuda()

    # 设置模型为评估模式
    model.eval()
    if metalearner:
        metalearner.eval()

    # 进行预测
    with torch.no_grad():
        if cfg.MODEL.NAME in ['MKCNet', 'FirstOrder_MKCNet']:
            output_T, output_M, output_IQ = model(image_tensor)
            # 应用softmax获取概率
            softmax = torch.nn.Softmax(dim=1)
            prob_T = softmax(output_T).cpu().numpy()[0]
            prob_IQ = softmax(output_IQ).cpu().numpy()[0]

            # 获取预测结果
            pred_T = torch.argmax(output_T, dim=1).cpu().numpy()[0]
            pred_IQ = torch.argmax(output_IQ, dim=1).cpu().numpy()[0]

            return {
                'diagnosis_pred': int(pred_T),
                'diagnosis_probs': [float(p) for p in prob_T],
                'image_quality_pred': int(pred_IQ),
                'image_quality_probs': [float(p) for p in prob_IQ]
            }
        else:
            output = model(image_tensor)
            # 应用softmax获取概率
            softmax = torch.nn.Softmax(dim=1)
            prob_T = softmax(output).cpu().numpy()[0]

            # 获取预测结果
            pred_T = torch.argmax(output, dim=1).cpu().numpy()[0]

            return {
                'diagnosis_pred': int(pred_T),
                'diagnosis_probs': [float(p) for p in prob_T]
            }


def get_dr_labels(dataset_name, num_classes):
    """
    获取DR标签的含义
    """
    if dataset_name in ['DRAC', 'DEEPDR', 'EYEQ'] and num_classes == 3:
        return {
            0: "无明显视网膜病变或轻微病变",
            1: "中度病变",
            2: "严重病变"
        }
    else:
        return {i: f"类别 {i}" for i in range(num_classes)}


def get_model_paths(model_type, dataset):
    """
    根据模型类型和数据集获取模型路径列表（尝试多种可能的路径）
    """
    paths_to_try = []

    # 添加特定的路径组合
    if model_type == 'MKCNet' and dataset == 'EYEQ':
        # EYEQ数据集使用FirstOrder_MKCNet模型
        paths_to_try.append("./model/FirstOrder_MKCNet_EYEQ")
    elif model_type == 'FirstOrder_MKCNet' and dataset == 'EYEQ':
        paths_to_try.append("./model/FirstOrder_MKCNet_EYEQ")

    # 添加默认路径
    paths_to_try.append(f"./model/{model_type}_{dataset}")

    # 如果是MKCNet模型，也尝试FirstOrder_MKCNet路径
    if model_type == 'MKCNet':
        paths_to_try.append(f"./model/FirstOrder_MKCNet_{dataset}")

    # 如果是FirstOrder_MKCNet模型，也尝试MKCNet路径
    if model_type == 'FirstOrder_MKCNet':
        paths_to_try.append(f"./model/MKCNet_{dataset}")

    return paths_to_try


def find_model_files(model_type, dataset):
    """
    查找模型文件的实际路径
    """
    paths_to_try = get_model_paths(model_type, dataset)

    for model_dir in paths_to_try:
        if os.path.exists(model_dir):
            model_path = os.path.join(model_dir, "best_model.pth")
            meta_model_path = None

            if model_type in ['MKCNet', 'FirstOrder_MKCNet']:
                meta_model_path = os.path.join(model_dir, "best_model_LG.pth")

            if os.path.exists(model_path):
                return model_path, meta_model_path, model_dir

    # 如果没有找到，返回默认路径用于错误提示
    default_dir = f"./model/{model_type}_{dataset}"
    model_path = os.path.join(default_dir, "best_model.pth")
    meta_model_path = None
    if model_type in ['MKCNet', 'FirstOrder_MKCNet']:
        meta_model_path = os.path.join(default_dir, "best_model_LG.pth")

    return model_path, meta_model_path, default_dir


def main():
    parser = argparse.ArgumentParser(description='对单张图片进行DR分析')
    parser.add_argument('--image_path', type=str, required=True, help='待分析图像的路径')
    parser.add_argument('--datasets', type=str, nargs='+',
                        choices=['DRAC', 'DEEPDR', 'EYEQ'],
                        default=['DRAC'], help='使用的数据集配置列表')
    parser.add_argument('--model_type', type=str,
                        choices=['MKCNet', 'FirstOrder_MKCNet', 'VanillaNet', 'CANet', 'DETACH'],
                        default='MKCNet', help='使用的模型类型')
    parser.add_argument('--output_json', type=str, help='输出JSON文件路径')

    args = parser.parse_args()

    # 检查图像文件是否存在
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像文件不存在: {args.image_path}")

    # 存储所有结果
    results = {}

    # 对每个数据集配置进行分析
    for dataset in args.datasets:
        print(f"\n正在处理 {dataset} 数据集配置...")

        # 设置配置
        cfg = cfg_default.clone()
        cfg.MODEL.NAME = args.model_type
        cfg.DATASET.NAME = dataset
        cfg.merge_from_file(f"./configs/datasets/{dataset}.yaml")

        # 查找模型文件路径
        model_path, meta_model_path, model_dir = find_model_files(args.model_type, dataset)

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}，跳过此数据集")
            continue

        # 加载模型
        try:
            model, metalearner = load_model(cfg, model_path, meta_model_path)
            model.cuda()

            # 进行预测
            result = predict_single_image(args.image_path, cfg, model, metalearner)
            results[dataset] = result
            print(f"成功处理 {dataset} 数据集配置 (使用模型路径: {model_dir})")

        except Exception as e:
            print(f"处理 {dataset} 数据集时出错: {str(e)}")
            continue

    # 准备JSON输出数据
    output_data = {
        "image_path": args.image_path,
        "model_type": args.model_type,
        "results": {}
    }

    for dataset, result in results.items():
        cfg = cfg_default.clone()
        cfg.MODEL.NAME = args.model_type
        cfg.DATASET.NAME = dataset
        cfg.merge_from_file(f"./configs/datasets/{dataset}.yaml")

        # 获取标签含义
        dr_labels = get_dr_labels(dataset, cfg.DATASET.NUM_T)

        # 构建详细结果
        detailed_result = {
            "diagnosis": {
                "predicted_class": result['diagnosis_pred'],
                "class_label": dr_labels.get(result['diagnosis_pred'], '未知'),
                "probabilities": [
                    {
                        "class": i,
                        "label": dr_labels.get(i, f'类别 {i}'),
                        "probability": prob
                    }
                    for i, prob in enumerate(result['diagnosis_probs'])
                ]
            }
        }

        if 'image_quality_pred' in result:
            iq_labels = {0: "高质量", 1: "低质量"} if cfg.DATASET.NUM_IQ == 2 else {i: f"质量等级 {i}" for i in
                                                                                    range(cfg.DATASET.NUM_IQ)}
            detailed_result["image_quality"] = {
                "predicted_class": result['image_quality_pred'],
                "class_label": iq_labels.get(result['image_quality_pred'], '未知'),
                "probabilities": [
                    {
                        "class": i,
                        "label": iq_labels.get(i, f'等级 {i}'),
                        "probability": prob
                    }
                    for i, prob in enumerate(result['image_quality_probs'])
                ]
            }

        output_data["results"][dataset] = detailed_result

    # 打印结果
    if not results:
        print("没有成功处理任何数据集配置")
        output_data["error"] = "没有成功处理任何数据集配置"
    else:
        print("\n" + "=" * 60)
        print("DR多数据集配置分析结果")
        print("=" * 60)
        print(f"图像路径: {args.image_path}")
        print(f"使用模型: {args.model_type}")

        for dataset, result in results.items():
            cfg = cfg_default.clone()
            cfg.MODEL.NAME = args.model_type
            cfg.DATASET.NAME = dataset
            cfg.merge_from_file(f"./configs/datasets/{dataset}.yaml")

            print(f"\n{dataset} 数据集配置结果:")
            print("-" * 40)

            # 获取标签含义
            dr_labels = get_dr_labels(dataset, cfg.DATASET.NUM_T)

            print(f"疾病诊断结果:")
            print(f"  预测类别: {result['diagnosis_pred']} ({dr_labels.get(result['diagnosis_pred'], '未知')})")
            print(f"  预测概率:")
            for i, prob in enumerate(result['diagnosis_probs']):
                print(f"    {dr_labels.get(i, f'类别 {i}')}: {prob:.6f}")

            if 'image_quality_pred' in result:
                print(f"图像质量评估结果:")
                iq_labels = {0: "高质量", 1: "低质量"} if cfg.DATASET.NUM_IQ == 2 else {i: f"质量等级 {i}" for i in
                                                                                        range(cfg.DATASET.NUM_IQ)}
                print(
                    f"  预测类别: {result['image_quality_pred']} ({iq_labels.get(result['image_quality_pred'], '未知')})")
                print(f"  预测概率:")
                for i, prob in enumerate(result['image_quality_probs']):
                    print(f"    {iq_labels.get(i, f'等级 {i}')}: {prob:.6f}")

    # 保存JSON文件
    if args.output_json:
        try:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到JSON文件: {args.output_json}")
        except Exception as e:
            print(f"保存JSON文件时出错: {str(e)}")
    else:
        # 如果没有指定输出文件，则打印JSON到控制台
        print("\n" + "=" * 60)
        print("JSON格式结果:")
        print("=" * 60)
        print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()