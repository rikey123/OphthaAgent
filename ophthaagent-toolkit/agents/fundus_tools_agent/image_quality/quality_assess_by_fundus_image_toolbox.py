#质量评估
import json
import os

import fundus_image_toolbox as fit
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [convert(x) for x in o]
    return o
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    # output_path = sys.argv[2]
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
        import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    from device_config import get_device
    device = get_device()
    ensemble = fit.load_quality_ensemble(device=device)
    confs, labels = fit.ensemble_predict_quality(ensemble, [fundus1])
    labels = int(labels)
    confs = convert(confs)
    result={
        "status": "success",
        "result": {
            "lable": labels,
            "confs": confs,
            "assessment": "合格" if labels else "不合格",
        }
    }
    print(json.dumps(result,ensure_ascii=False))



