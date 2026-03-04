#质量评估
import json
from typing import List, Tuple

import fundus_image_toolbox as fit
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

def load_image(input_path):
    """
    加载图像

    Args:
        input_path (str): 输入图像路径

    Returns:
        numpy.ndarray: 图像数组
    """
    try:
        img = plt.imread(input_path)
        # 如果图像是RGBA格式，转换为RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        # 如果图像是浮点型且范围在0-1之间，转换为0-255范围的uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        return img
    except Exception as e:
        raise Exception(f"加载图像时出错: {str(e)}")
def draw_points_on_image(image: np.ndarray, points: List[Tuple[float, float]], colors: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for (x, y), c in zip(points, colors):
        ax.scatter([x], [y], c=c, s=120, marker="x", linewidths=3)
    ax.axis("off")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return plot_img
def save_image(path , image):
    # Convert to PIL Image and save
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image)
    pil_img.save(path)
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) <= 1:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]
    # image_path = 'normal.jpg'
    # output_path = 'test.png'
    fundus1 = load_image(image_path)
        #  关键修复：转换为 uint8
    if fundus1.dtype == np.float32 or fundus1.dtype == np.float64:
        # 假设值在 [0, 1] 范围内
        fundus1 = (fundus1 * 255).astype(np.uint8)
    elif fundus1.dtype != np.uint8:
        # 兜底：强制转为 uint8
        fundus1 = fundus1.astype(np.uint8)
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    from device_config import get_device
    device = get_device()
    # 检查命令行参数，如果提供了 --device，则覆盖默认值
    if "--device" in sys.argv:
        try:
            device_index = sys.argv.index("--device")
            device = sys.argv[device_index + 1]
        except (ValueError, IndexError):
            pass # 如果 --device 格式不正确，则忽略并使用默认值
    fovea_model, _ = fit.load_fovea_od_model(device=device)
    coordinates = fovea_model.predict([fundus1])
    print(coordinates)
    coord_result = {
        "status": "success",
        "result":{
            "save_path": output_path,
            "coordinates":{
                "fovea": {
                    "x": float(coordinates[0]),
                    "y": float(coordinates[1])
                },
                "optic_disc": {
                    "x": float(coordinates[2]),
                    "y": float(coordinates[3])
                }
                }
            }


    }
    print(json.dumps(coord_result))
    # with open('test.json', 'w', encoding='utf-8') as f:
    #     json.dump(coord_result, f, ensure_ascii=False, indent=2)
    # 如果指定了输出图像路径，则绘制坐标点并保存图像
    # if output_path:
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     # ax.imshow(img)
    #     # 绘制黄斑中心凹（红色）
    #     ax.scatter(coordinates[0], coordinates[1], c='r', s=50, label='黄斑中心凹 (Fovea)')
    #     # 绘制视盘（蓝色）
    #     ax.scatter(coordinates[2], coordinates[3], c='b', s=50, label='视盘 (Optic Disc)')
    #     ax.legend()
    #     ax.set_title('黄斑中心凹和视盘定位结果')
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     plt.close()



