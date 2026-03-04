import os
import sys

# mimicking Fovea_OD_localization_by_fundus_image_toolbox.py
sys.path.append('<ANON_ABS_PATH>')
try:
    from device_config import get_device
    print(f"get_device() returns: {get_device()}")
except ImportError:
    print("Could not import device_config")
except Exception as e:
    print(f"Error: {e}")
