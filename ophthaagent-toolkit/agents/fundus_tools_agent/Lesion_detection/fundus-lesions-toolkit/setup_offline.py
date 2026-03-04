"""
设置离线环境的脚本
"""
import os
import sys

def setup_offline_environment():
    """设置HuggingFace离线环境"""
    
    # 设置环境变量
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # 设置缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = cache_dir
    
    print("✅ 已设置离线环境变量:")
    print(f"   HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE')}")
    print(f"   TRANSFORMERS_OFFLINE = {os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"   HF_DATASETS_OFFLINE = {os.environ.get('HF_DATASETS_OFFLINE')}")
    print(f"   缓存目录: {cache_dir}")
    
    return True

def check_cache_status():
    """检查本地缓存状态"""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if os.path.exists(cache_dir):
        print(f"✅ 缓存目录存在: {cache_dir}")
        
        # 检查模型缓存
        hub_dir = os.path.join(cache_dir, "hub")
        if os.path.exists(hub_dir):
            models = [d for d in os.listdir(hub_dir) if d.startswith("models--")]
            print(f"   已缓存的模型数量: {len(models)}")
            for model in models:
                print(f"   - {model}")
        else:
            print("   ⚠️  hub目录不存在")
    else:
        print(f"❌ 缓存目录不存在: {cache_dir}")

def create_offline_wrapper():
    """创建离线包装器"""
    wrapper_code = '''
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
'''
    
    with open("offline_wrapper.py", "w", encoding="utf-8") as f:
        f.write(wrapper_code)
    
    print("✅ 已创建 offline_wrapper.py")
    print("使用方法:")
    print("   from offline_wrapper import safe_segment, safe_count_lesions")

if __name__ == "__main__":
    print("🔧 设置HuggingFace离线环境")
    print("=" * 50)
    
    # 设置离线环境
    setup_offline_environment()
    print()
    
    # 检查缓存状态
    check_cache_status()
    print()
    
    # 创建离线包装器
    create_offline_wrapper()
    print()
    
    print("🎉 设置完成！现在可以使用离线模式了")
    print("\n推荐使用方法:")
    print("1. 重启Python解释器")
    print("2. 导入: from offline_wrapper import safe_segment")
    print("3. 使用: result = safe_segment(your_image, device='cpu')")
