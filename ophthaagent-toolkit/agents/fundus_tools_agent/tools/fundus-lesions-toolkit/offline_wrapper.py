
import os
# 在导入任何HuggingFace相关模块之前设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 现在可以安全导入fundus_lesions_toolkit
from fundus_lesions_toolkit.models import segment, count_lesions
from fundus_lesions_toolkit.constants import Dataset

def safe_segment(image, **kwargs):
    """安全的分割函数，使用离线模式"""
    try:
        return segment(image, **kwargs)
    except Exception as e:
        print(f"分割失败: {e}")
        return None

def safe_count_lesions(image, **kwargs):
    """安全的病变计数函数，使用离线模式"""
    try:
        return count_lesions(image, **kwargs)
    except Exception as e:
        print(f"病变计数失败: {e}")
        return {}
