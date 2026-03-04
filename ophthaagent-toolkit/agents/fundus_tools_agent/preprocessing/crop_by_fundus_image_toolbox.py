#圆形裁切
import json

import fundus_image_toolbox as fit
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) <= 1:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]
    # image_path = 'test.jpg'
    # output_path = 'test.png'
    fundus1 = plt.imread(image_path)
        #  关键修复：转换为 uint8
    if fundus1.dtype == np.float32 or fundus1.dtype == np.float64:
        # 假设值在 [0, 1] 范围内
        fundus1 = (fundus1 * 255).astype(np.uint8)
    elif fundus1.dtype != np.uint8:
        # 兜底：强制转为 uint8
        fundus1 = fundus1.astype(np.uint8)
    fundus1_cropped = fit.crop(fundus1, size=(512,512))
    plt.imsave(output_path, fundus1_cropped)
    result={
        "status": "success",
        "result": {
            "save_path": output_path,
        }
    }
    print(json.dumps(result))
