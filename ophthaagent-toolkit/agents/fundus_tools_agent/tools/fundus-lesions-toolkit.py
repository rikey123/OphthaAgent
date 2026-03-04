import json
import sys
import numpy as np
from PIL import Image
import os
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
from fundus_lesions_toolkit.models import segment
from fundus_lesions_toolkit.constants import DEFAULT_COLORS, LESIONS
from fundus_lesions_toolkit.utils.images import open_image
from fundus_lesions_toolkit.utils.visualization import plot_image_and_mask
from fundus_lesions_toolkit.models.detection import count_lesions
if __name__ == "__main__":
    # 替换为您的图像路径
    # if len(sys.argv) <= 1:
    #     print(json.dumps({"error": "Missing arguments"}))
    #     sys.exit(1)

    # image_path = sys.argv[1]
    # output_path = sys.argv[2]
    image_path = 'test.png'
    output_path = 'lesions.png'
    img = open_image(image_path)
    pred = segment(img,device='cuda')
    plot_image_and_mask(img, pred, alpha=0.8, title='My segmentation', colors=DEFAULT_COLORS, labels=LESIONS)
    if img.dtype != np.uint8:
        # If it's float in [0,1], scale to [0,255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)
    # 计数病变
    counts = count_lesions(img, size_threshold=15,
                           device='cuda', train_datasets='IDRID')

    # 输出结果
    for lesion, count in counts.items():
        print(f"Number of {lesion}: {count}")