import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ietk.methods as methods
import ietk.util as util


def enhance_fundus_image(image_path, output_path=None, show_results=True):
    """
    对单张眼底图像进行增强的完整流程

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        show_results: 是否显示结果
    """
    def preprocess_fundus_image(img):
        """预处理眼底图像，获取前景区域"""
        # 获取前景掩码（眼底区域）
        I, fg = util.center_crop_and_get_foreground_mask(img, crop=True)
        # 获取背景掩码
        bg = util.get_background(I)
        # 将背景设为黑色
        I[bg] = 0
        return I, fg, bg
    # 1. 加载图像
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0

    # 2. 预处理
    I, fg, bg = preprocess_fundus_image(img_array)
    # 方法1: 亮度调整
    brightened = methods.brighten_darken(I, 'A+B+X', fg)
    # 方法2: 锐化
    #sharpened = methods.sharpen(I, bg, t=0.15)
    # 方法3: 组合增强
    #combined = methods.sharpen(brightened, bg, t=0.15)
    # 4. 显示结果
    if show_results:
        #fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(I)
        axes[0].set_title('原始图像', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(brightened)
        axes[1].set_title('亮度调整 (A+B+X)', fontsize=14)
        axes[1].axis('off')

        #axes[1, 0].imshow(sharpened)
        #axes[1, 0].set_title('锐化增强', fontsize=14)
        #axes[1, 0].axis('off')

        #axes[1, 1].imshow(combined)
        #axes[1, 1].set_title('组合增强', fontsize=14)
        #axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
    # 5. 保存结果
    if output_path:
        clipped_img = np.clip(brightened, 0, 1)
        enhanced_img = (clipped_img * 255).astype(np.uint8)
        Image.fromarray(enhanced_img).save(output_path)
        #plt.imsave(output_path, combined)

    return output_path



# 使用示例
if __name__ == "__main__":
    # 替换为您的图像路径
    if len(sys.argv) <= 1:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]
    # image_path = "test1.jpg"
    # output_path = "test.png"
    results = enhance_fundus_image(
        image_path=image_path,
        output_path=output_path,
        show_results=True
    )
    result={
        "status": "success",
        "result": {
            "save_path": output_path,
        }
    }
    print(json.dumps(result))