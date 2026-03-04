#质量评估
import json
from typing import List

import fundus_image_toolbox as fit
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
def save_image(path , image):
    # Convert to PIL Image and save
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image)
    pil_img.save(path)


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

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) <= 1:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]
    # image_path = 'test.jpg'
    # output_path = 'test.png'
    """
        分割血管

        Args:
            input_path (str): 输入图像路径
            output_path (str): 输出图像路径

        Returns:
            dict: 处理结果信息
        """
    img = load_image(image_path)
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    from device_config import get_device
    device = get_device()
    ensemble = fit.load_segmentation_ensemble(device)
    vessel_masks = fit.ensemble_predict_segmentation(ensemble, [img], threshold=0.5, size=(512, 512))
    print(vessel_masks.shape)
    plt.imsave(output_path, vessel_masks, cmap='gray')
    result = {
        "status": "success",
        "result": {
            "save_path": output_path,
        }
    }
    print(json.dumps(result))


