#!/usr/bin/env python3
"""
DR分析器使用示例
演示如何使用DR分析器对单张图片进行分割分析
"""

import os
from dr_analyzer import DRAnalyzer

def main():
    # 示例参数
    image_path = "/path/to/your/fundus/image.jpg"  # 请替换为实际的图像路径
    output_dir = "./example_output"

    # 检查输入图像是否存在
    if not os.path.exists(image_path):
        print("示例图像不存在，请设置正确的图像路径")
        print(f"请将 image_path 变量设置为实际的眼底图像路径")
        print(f"当前设置: {image_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("初始化DR分析器...")
    # 创建分析器
    analyzer = DRAnalyzer(
        model_path="models/best.pth",
        config_path="models/config.json",
        device="auto"  # 自动选择设备
    )

    print(f"开始分析图像: {image_path}")
    # 分析图像
    result = analyzer.analyze_image(image_path, output_dir)

    print("分析完成！")
    print(f"结果已保存至: {output_dir}")
    print(f"JSON结果: {os.path.join(output_dir, 'dr_analysis_result.json')}")

    # 打印关键指标
    print("\n=== 分析结果摘要 ===")
    for lesion_name, lesion_data in result["分析结果"].items():
        print(f"\n{lesion_name}:")
        for metric_name, metric_value in lesion_data["指标"].items():
            print(f"  {metric_name}: {metric_value}")

    print(f"\n可视化结果:")
    for lesion_name, lesion_data in result["分析结果"].items():
        print(f"  {lesion_name}: {lesion_data['结果文件']}")

if __name__ == "__main__":
    main()
