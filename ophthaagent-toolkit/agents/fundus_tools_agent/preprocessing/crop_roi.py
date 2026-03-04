"""
Crop ROI (Region of Interest) 工具
支持bbox裁剪和可选的超分辨率增强
"""
import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def crop_fundus_roi(
    image_path: str,
    bbox: dict,
    output_path: str = None,
    super_resolution: bool = False,
    sr_scale: int = 2
):
    """
    裁剪眼底图像ROI区域，可选超分辨率增强

    Args:
        image_path: 输入图像路径
        bbox: 边界框 {"x1": int, "y1": int, "x2": int, "y2": int}
        output_path: 输出路径，如果为None则自动生成
        super_resolution: 是否启用超分辨率
        sr_scale: 超分倍数 (2 or 4)

    Returns:
        dict: {
            "status": "success",
            "result": {
                "cropped_image_path": str,
                "original_bbox": dict,
                "cropped_size": {"width": int, "height": int},
                "super_resolution_applied": bool
            }
        }
    """
    try:
        # 验证输入
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "error": f"图像文件不存在: {image_path}"
            }

        # 加载图像
        img = Image.open(image_path)
        img_width, img_height = img.size

        # 验证和调整bbox
        x1 = max(0, int(bbox.get("x1", 0)))
        y1 = max(0, int(bbox.get("y1", 0)))
        x2 = min(img_width, int(bbox.get("x2", img_width)))
        y2 = min(img_height, int(bbox.get("y2", img_height)))

        if x2 <= x1 or y2 <= y1:
            return {
                "status": "error",
                "error": f"无效的bbox: ({x1}, {y1}, {x2}, {y2})"
            }

        # 裁剪
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_width, cropped_height = cropped_img.size

        # 超分辨率增强（如果启用）
        sr_applied = False
        if super_resolution:
            try:
                cropped_img = apply_super_resolution(cropped_img, sr_scale)
                sr_applied = True
            except Exception as e:
                print(f"警告: 超分辨率处理失败，使用原始裁剪结果: {e}", file=sys.stderr)

        # 生成输出路径
        if output_path is None:
            # ✅ 修复：默认使用 tool_outputs 目录而不是数据集目录
            base_name = Path(image_path).stem
            ext = Path(image_path).suffix
            output_dir = "./tool_outputs"  # ✅ 默认使用工具输出目录
            os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
            sr_suffix = f"_sr{sr_scale}x" if sr_applied else ""
            output_path = str(Path(output_dir) / f"{base_name}_crop_{x1}_{y1}_{x2}_{y2}{sr_suffix}{ext}")
        elif os.path.isdir(output_path):
            # ✅ 如果 output_path 是目录，在目录中生成文件名
            base_name = Path(image_path).stem
            ext = Path(image_path).suffix
            sr_suffix = f"_sr{sr_scale}x" if sr_applied else ""
            output_path = str(Path(output_path) / f"{base_name}_crop_{x1}_{y1}_{x2}_{y2}{sr_suffix}{ext}")
        # else: output_path 是完整的文件路径，直接使用

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # 保存裁剪后的图像
        cropped_img.save(output_path, quality=95)

        # ✅ 转换为绝对路径
        output_path_abs = os.path.abspath(output_path)

        # 返回结果
        result = {
            "status": "success",
            "result": {
                "cropped_image_path": output_path_abs,  # ✅ 返回绝对路径
                "original_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "cropped_size": {"width": cropped_width, "height": cropped_height},
                "super_resolution_applied": sr_applied,
                "sr_scale": sr_scale if sr_applied else 1
            }
        }

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": f"裁剪处理失败: {str(e)}"
        }


def apply_super_resolution(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    应用超分辨率增强

    使用简单的双三次插值作为基础实现
    生产环境可以替换为Real-ESRGAN或SwinIR
    """
    # 方案1: 简单的双三次插值（快速但效果一般）
    new_size = (img.width * scale, img.height * scale)
    img_sr = img.resize(new_size, Image.BICUBIC)

    # TODO: 可选替换为深度学习超分模型
    # 如果有GPU和Real-ESRGAN模型:
    # from basicsr.archs.rrdbnet_arch import RRDBNet
    # from realesrgan import RealESRGANer
    # model = RRDBNet(...)
    # upsampler = RealESRGANer(...)
    # img_sr = upsampler.enhance(img_array)

    return img_sr


def smart_crop_macula(image_path: str, fovea_center: dict = None, size: int = 500):
    """
    智能裁剪黄斑区域

    Args:
        image_path: 图像路径
        fovea_center: 黄斑中心坐标 {"x": int, "y": int}，如果为None则使用图像中心
        size: 裁剪区域大小（正方形边长）

    Returns:
        dict: 裁剪结果
    """
    if fovea_center is None:
        # 如果没有提供黄斑坐标，使用图像中心
        img = Image.open(image_path)
        fovea_center = {"x": img.width // 2, "y": img.height // 2}

    # 构建bbox
    half_size = size // 2
    bbox = {
        "x1": fovea_center["x"] - half_size,
        "y1": fovea_center["y"] - half_size,
        "x2": fovea_center["x"] + half_size,
        "y2": fovea_center["y"] + half_size
    }

    return crop_fundus_roi(image_path, bbox, super_resolution=True, sr_scale=2)


def smart_crop_optic_disc(image_path: str, od_center: dict = None, size: int = 400):
    """
    智能裁剪视盘区域

    Args:
        image_path: 图像路径
        od_center: 视盘中心坐标 {"x": int, "y": int}
        size: 裁剪区域大小（正方形边长）

    Returns:
        dict: 裁剪结果
    """
    if od_center is None:
        # 默认假设视盘在图像右侧1/3处
        img = Image.open(image_path)
        od_center = {"x": int(img.width * 0.65), "y": img.height // 2}

    half_size = size // 2
    bbox = {
        "x1": od_center["x"] - half_size,
        "y1": od_center["y"] - half_size,
        "x2": od_center["x"] + half_size,
        "y2": od_center["y"] + half_size
    }

    return crop_fundus_roi(image_path, bbox, super_resolution=True, sr_scale=2)


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="眼底图像ROI裁剪工具")
    parser.add_argument("image_path", help="输入图像路径")
    parser.add_argument("bbox", help='边界框JSON字符串，例如: \'{"x1":100,"y1":200,"x2":300,"y2":400}\'')
    parser.add_argument("output_path", nargs="?", default=None, help="输出路径（可选）")
    parser.add_argument("--sr", action="store_true", help="启用超分辨率")
    parser.add_argument("--sr-scale", type=int, default=2, choices=[2, 4], help="超分倍数")
    parser.add_argument("--mode", choices=["custom", "macula", "optic_disc"], default="custom",
                       help="裁剪模式")

    args = parser.parse_args()

    # 解析bbox
    if args.mode == "custom":
        try:
            bbox = json.loads(args.bbox)
        except json.JSONDecodeError:
            print(json.dumps({
                "status": "error",
                "error": "无效的bbox JSON格式"
            }))
            sys.exit(1)

        result = crop_fundus_roi(
            args.image_path,
            bbox,
            args.output_path,
            args.sr,
            args.sr_scale
        )
    elif args.mode == "macula":
        # bbox参数作为fovea center
        try:
            fovea_center = json.loads(args.bbox)
        except:
            fovea_center = None
        result = smart_crop_macula(args.image_path, fovea_center)
    else:  # optic_disc
        try:
            od_center = json.loads(args.bbox)
        except:
            od_center = None
        result = smart_crop_optic_disc(args.image_path, od_center)

    # 输出JSON结果
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 返回状态码
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
