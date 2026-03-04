"""
图像处理器
负责图像编码、裁剪、缩略图生成等
"""
import base64
import io
import os
from typing import Optional, Tuple, Dict
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器：base64编码、裁剪、超分等"""

    @staticmethod
    def encode_image_to_base64(
        image_path: str,
        resize: Optional[Tuple[int, int]] = None,
        quality: int = 95
    ) -> Optional[str]:
        """
        编码图片为base64，可选resize以减小token消耗

        Args:
            image_path: 图像路径
            resize: 可选的resize尺寸 (width, height)
            quality: JPEG质量 (1-100)

        Returns:
            base64编码的字符串，失败返回None
        """
        if not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            return None

        try:
            img = Image.open(image_path)

            # 转换为RGB（如果是RGBA或其他格式）
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize（如果指定）
            if resize:
                img = img.resize(resize, Image.LANCZOS)

            # 编码为JPEG格式
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img_bytes = buffer.getvalue()

            # Base64编码
            base64_str = base64.b64encode(img_bytes).decode("utf-8")

            logger.debug(f"图像编码成功: {image_path}, 大小: {len(base64_str)} 字符")
            return base64_str

        except Exception as e:
            logger.error(f"图像编码失败: {image_path}, 错误: {e}")
            return None

    @staticmethod
    def create_thumbnail(
        image_path: str,
        max_size: int = 1024,
        quality: int = 85
    ) -> Optional[str]:
        """
        创建缩略图用于初步观察（节省token）

        Args:
            image_path: 图像路径
            max_size: 缩略图最大边长
            quality: JPEG质量

        Returns:
            base64编码的缩略图字符串
        """
        if not os.path.exists(image_path):
            return None

        try:
            img = Image.open(image_path)

            # 保持纵横比缩放
            img.thumbnail((max_size, max_size), Image.LANCZOS)

            # 编码
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img_bytes = buffer.getvalue()

            base64_str = base64.b64encode(img_bytes).decode("utf-8")

            logger.debug(f"缩略图生成成功: {image_path}, 尺寸: {img.size}")
            return base64_str

        except Exception as e:
            logger.error(f"缩略图生成失败: {image_path}, 错误: {e}")
            return None

    @staticmethod
    def prepare_images_for_lvlm(
        image_path: str,
        include_thumbnail: bool = True,
        include_full: bool = True,
        thumbnail_size: int = 512,
        full_size: int = 2048
    ) -> Dict[str, str]:
        """
        准备多个尺度的图像供LVLM使用

        Args:
            image_path: 图像路径
            include_thumbnail: 是否包含缩略图
            include_full: 是否包含全尺寸图
            thumbnail_size: 缩略图最大边长
            full_size: 全尺寸图最大边长

        Returns:
            dict: {
                "thumbnail": base64_str or None,
                "full": base64_str or None,
                "original_size": {"width": int, "height": int}
            }
        """
        result = {
            "thumbnail": None,
            "full": None,
            "original_size": None
        }

        if not os.path.exists(image_path):
            logger.warning(f"图像不存在: {image_path}")
            return result

        try:
            # 获取原始尺寸
            with Image.open(image_path) as img:
                result["original_size"] = {"width": img.width, "height": img.height}

            # 生成缩略图
            if include_thumbnail:
                result["thumbnail"] = ImageProcessor.create_thumbnail(
                    image_path,
                    max_size=thumbnail_size
                )

            # 生成全尺寸图（可能缩放）
            if include_full:
                img = Image.open(image_path)
                if max(img.size) > full_size:
                    # 需要缩放
                    ratio = full_size / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    result["full"] = ImageProcessor.encode_image_to_base64(
                        image_path,
                        resize=new_size
                    )
                else:
                    # 直接使用原图
                    result["full"] = ImageProcessor.encode_image_to_base64(image_path)

            return result

        except Exception as e:
            logger.error(f"图像准备失败: {image_path}, 错误: {e}")
            return result

    @staticmethod
    def crop_and_encode(
        image_path: str,
        bbox: Dict[str, int],
        output_path: Optional[str] = None,
        super_resolution: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        裁剪ROI并编码

        Args:
            image_path: 图像路径
            bbox: 边界框 {"x1": int, "y1": int, "x2": int, "y2": int}
            output_path: 可选的输出路径
            super_resolution: 是否启用超分

        Returns:
            (cropped_path, base64_str): 裁剪后的图片路径和base64编码
        """
        try:
            # 调用crop_roi工具
            from preprocessing.crop_roi import crop_fundus_roi

            result = crop_fundus_roi(
                image_path,
                bbox,
                output_path,
                super_resolution
            )

            if result["status"] == "success":
                cropped_path = result["result"]["cropped_image_path"]
                base64_str = ImageProcessor.encode_image_to_base64(cropped_path)
                return cropped_path, base64_str
            else:
                logger.error(f"裁剪失败: {result.get('error')}")
                return None, None

        except Exception as e:
            logger.error(f"裁剪编码失败: {e}")
            return None, None

    @staticmethod
    def get_image_info(image_path: str) -> Optional[Dict]:
        """
        获取图像基本信息

        Returns:
            dict: {
                "width": int,
                "height": int,
                "format": str,
                "mode": str,
                "size_kb": float
            }
        """
        if not os.path.exists(image_path):
            return None

        try:
            img = Image.open(image_path)
            file_size = os.path.getsize(image_path) / 1024  # KB

            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_kb": round(file_size, 2)
            }
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            return None

    @staticmethod
    def estimate_base64_token_count(base64_str: str, chars_per_token: int = 4) -> int:
        """
        估算base64字符串会消耗的token数量

        Args:
            base64_str: base64编码字符串
            chars_per_token: 平均每个token包含的字符数

        Returns:
            估算的token数量
        """
        if not base64_str:
            return 0
        return len(base64_str) // chars_per_token


# 测试代码
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    test_image = "./test_images/sample.jpg"

    print("=== 测试图像信息获取 ===")
    info = ImageProcessor.get_image_info(test_image)
    print(json.dumps(info, indent=2))

    print("\n=== 测试缩略图生成 ===")
    thumbnail_b64 = ImageProcessor.create_thumbnail(test_image, max_size=512)
    if thumbnail_b64:
        print(f"缩略图大小: {len(thumbnail_b64)} 字符")
        print(f"预估token: {ImageProcessor.estimate_base64_token_count(thumbnail_b64)}")

    print("\n=== 测试多尺度准备 ===")
    images = ImageProcessor.prepare_images_for_lvlm(
        test_image,
        include_thumbnail=True,
        include_full=True
    )
    print(f"原始尺寸: {images['original_size']}")
    if images['thumbnail']:
        print(f"缩略图: {len(images['thumbnail'])} 字符")
    if images['full']:
        print(f"全图: {len(images['full'])} 字符")
