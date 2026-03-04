 #!/usr/bin/env python3
"""
DR单张图片分割分析器
基于训练好的Attention-UNet模型，对单张眼底图像进行糖尿病视网膜病变分割
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# [临时修改] 硬编码设备配置为 cuda:0，后续将使用 device_config
def get_device():
    """临时硬编码返回 cuda:0"""
    return 'cuda:7' if torch.cuda.is_available() else 'cpu'

# MONAI for AttentionUnet
try:
    from monai.networks.nets import AttentionUnet
except ImportError:
    raise ImportError("Please install MONAI: pip install monai")

# 导入数据集处理函数
from utils.dataset import get_val_transforms

# 配置常量
IDRID_MEAN = (0.485, 0.456, 0.406)
IDRID_STD = (0.229, 0.224, 0.225)

# 病灶类型映射
LESION_NAMES = ["MA", "HE", "EX", "SE"]
LESION_CHINESE_NAMES = {
    "MA": "microaneurysms",
    "HE": "haemorages",
    "EX": "hard_exudates",
    "SE": "soft_exudates"
}

# 颜色映射 (BGR格式，用于OpenCV)
COLOR_MAP = {
    "MA": (255, 0, 0),      # 红色 - 微动脉瘤
    "HE": (0, 0, 255),      # 蓝色 - 出血
    "EX": (0, 255, 0),      # 绿色 - 硬渗出物
    "SE": (255, 255, 0)     # 青色 - 软渗出物
}


class DRAnalyzer:
    """DR分割分析器"""

    def __init__(self, model_path: str, config_path: str, device: str = "auto",
                 target_resolution: Tuple[int, int] = None,
                 resize_method: str = "padding"):
        """
        初始化分析器

        Args:
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda', 'cuda:0'等)
            target_resolution: 目标分辨率 (width, height)，用于模型推理
                              None 表示使用原始分辨率
                              默认 (4288, 2848) 为模型最佳分辨率
            resize_method: resize 方法
                          - "direct": 直接 resize（可能变形）
                          - "padding": 保持宽高比 + 填充黑边（推荐）
        """
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # 分辨率适配配置
        self.target_resolution = target_resolution  # (width, height)
        self.resize_method = resize_method
        self.use_resize = target_resolution is not None

        if self.use_resize:
            self.logger.info(f"启用分辨率适配: {target_resolution[0]}x{target_resolution[1]}")
            self.logger.info(f"Resize 方法: {resize_method}")

        # [最终诊断] 打印进程启动时的CUDA_VISIBLE_DEVICES值
        self.logger.info(f"诊断信息：进程启动时 CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

        # 设置设备
        if device == "auto":
            self.device = torch.device(get_device())
        else:
            self.device = torch.device(device)

        self.logger.info(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            # self.logger.info(f"CUDA 设备 ID: {torch.cuda.current_device()}")
            pass

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # [新增] 删除可能引起冲突的设备配置，确保全局配置生效
        if 'cuda_visible_devices' in self.config:
            del self.config['cuda_visible_devices']

        # 构建模型
        self.model = self.build_model()
        self.load_weights(model_path)

        # 设置模型参数
        self.patch_size = self.config.get('patch_size', 512)
        self.stride = self.config.get('stride', 256)
        self.num_classes = self.config.get('num_classes', 4)

        self.logger.info(f"模型配置: patch_size={self.patch_size}, stride={self.stride}, num_classes={self.num_classes}")

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def build_model(self) -> nn.Module:
        """构建Attention-UNet模型"""
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=self.config.get('num_classes', 4),
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            dropout=0.0,
        ).to(self.device)

        return model

    def load_weights(self, model_path: str):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)

        # 处理可能的state_dict嵌套
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.logger.info(f"成功加载模型权重: {model_path}")

    def _resize_image_direct(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        方法1: 直接 resize 图像到目标尺寸（可能变形）

        Args:
            image: 输入图像 (H, W, 3)
            target_size: 目标尺寸 (width, height)

        Returns:
            resized_image: resize 后的图像
        """
        target_w, target_h = target_size
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _resize_image_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        方法2: 保持宽高比 resize，不足部分填充黑边

        Args:
            image: 输入图像 (H, W, 3)
            target_size: 目标尺寸 (width, height)

        Returns:
            resized_image: resize + padding 后的图像
            offset: (x_offset, y_offset) 图像在画布中的偏移
            scale: 缩放比例
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例（保持宽高比）
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize 图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建目标尺寸的黑色画布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 居中放置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas, (x_offset, y_offset), scale

    def _resize_mask_back(self, mask: torch.Tensor, original_size: Tuple[int, int],
                         resize_info: dict = None) -> torch.Tensor:
        """
        将预测 mask resize 回原始尺寸

        Args:
            mask: 预测 mask (num_classes, H, W)
            original_size: 原始图像尺寸 (height, width)
            resize_info: resize 信息（用于 padding 方法）
                        包含 offset 和 scale

        Returns:
            resized_mask: resize 回原始尺寸的 mask (num_classes, original_h, original_w)
        """
        original_h, original_w = original_size
        _, curr_h, curr_w = mask.shape

        # 如果当前尺寸已经和原始尺寸一致，直接返回
        if curr_h == original_h and curr_w == original_w:
            return mask

        # 如果使用了 padding 方法，需要先裁剪再 resize
        if self.resize_method == "padding" and resize_info is not None:
            x_offset, y_offset = resize_info['offset']
            scale = resize_info['scale']

            # 计算原始图像在 resize 后画布中的区域
            # new_w = original_w * scale
            # new_h = original_h * scale
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            # 裁剪掉黑边
            mask_cropped = mask[:, y_offset:y_offset+new_h, x_offset:x_offset+new_w]

            # Resize 到原始尺寸
            mask_resized = torch.nn.functional.interpolate(
                mask_cropped.unsqueeze(0),  # (1, num_classes, new_h, new_w)
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (num_classes, original_h, original_w)
        else:
            # 直接 resize（direct 方法或没有 padding 信息）
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0),  # (1, num_classes, H, W)
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (num_classes, original_h, original_w)

        return mask_resized

    def sliding_window_predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        对单张图做滑动窗口推理

        Args:
            image: 输入图像张量 (1, 3, H, W)

        Returns:
            probs: 预测概率 (num_classes, H, W)
        """
        _, _, H, W = image.shape
        full = torch.zeros((self.num_classes, H, W), dtype=torch.float32)
        count = torch.zeros((self.num_classes, H, W), dtype=torch.float32)

        def _positions_1d(length: int, patch_size: int, stride: int):
            """Generate sliding window start indices that guarantee edge coverage."""
            if length <= patch_size:
                return [0]
            last = length - patch_size
            pos = list(range(0, last + 1, stride))
            if pos[-1] != last:
                pos.append(last)
            return pos

        with torch.no_grad():
            ys = _positions_1d(H, self.patch_size, self.stride)
            xs = _positions_1d(W, self.patch_size, self.stride)

            for y in ys:
                for x in xs:
                    patch = image[:, :, y : y + self.patch_size, x : x + self.patch_size].to(self.device)
                    logits = self.model(patch)  # (1, num_classes, patch_size, patch_size)

                    if logits.dim() == 4:
                        logits = logits.squeeze(0)  # (num_classes, patch_size, patch_size)

                    probs = torch.sigmoid(logits)  # (num_classes, patch_size, patch_size)

                    full[:, y : y + self.patch_size, x : x + self.patch_size] += probs.cpu()
                    count[:, y : y + self.patch_size, x : x + self.patch_size] += 1

        # 平均概率
        full = full / (count + 1e-8)
        return full

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int], dict]:
        """
        预处理输入图像

        Args:
            image_path: 图像文件路径

        Returns:
            image_tensor: 预处理后的张量 (1, 3, H, W)
            original_image: 原始图像数组 (H, W, 3)
            original_size: 原始图像尺寸 (height, width)
            resize_info: resize 信息（用于 padding 方法）
                        {'offset': (x, y), 'scale': float} 或 None
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = original_image.shape[:2]

        # 初始化 resize_info
        resize_info = None
        processed_image = original_image

        # 如果需要 resize 到目标分辨率
        if self.use_resize:
            if self.resize_method == "direct":
                # 方法1: 直接 resize
                self.logger.info(f"使用 direct resize: {original_w}x{original_h} -> {self.target_resolution[0]}x{self.target_resolution[1]}")
                processed_image = self._resize_image_direct(original_image, self.target_resolution)

            elif self.resize_method == "padding":
                # 方法2: 保持宽高比 + padding
                self.logger.info(f"使用 padding resize: {original_w}x{original_h} -> {self.target_resolution[0]}x{self.target_resolution[1]}")
                processed_image, offset, scale = self._resize_image_with_padding(original_image, self.target_resolution)
                resize_info = {'offset': offset, 'scale': scale}
                self.logger.info(f"  缩放比例: {scale:.4f}, 偏移: ({offset[0]}, {offset[1]})")
        else:
            self.logger.info(f"使用原始分辨率: {original_w}x{original_h}")

        # 转换为PIL Image用于albumentations
        pil_image = Image.fromarray(processed_image)

        # 使用验证时的数据增强
        transform = get_val_transforms()
        transformed = transform(image=processed_image)
        image_tensor = transformed['image'].unsqueeze(0)  # (1, 3, H, W)

        return image_tensor, original_image, (original_h, original_w), resize_info

    def analyze_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        分析单张图像

        Args:
            image_path: 输入图像路径
            output_dir: 基础输出目录

        Returns:
            分析结果字典
        """
        # --- 确保路径是绝对的 ---
        output_dir = os.path.abspath(output_dir)
        image_path = os.path.abspath(image_path)
        os.makedirs(output_dir, exist_ok=True)

        # --- Create a timestamped output directory ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"analysis_{timestamp_str}"
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.logger.info(f"结果将保存到: {session_dir}")

        # 预处理图像（返回原始尺寸和 resize 信息）
        image_tensor, original_image, original_size, resize_info = self.preprocess_image(image_path)
        original_h, original_w = original_size
        processed_h, processed_w = image_tensor.shape[2], image_tensor.shape[3]

        self.logger.info(f"原始图像尺寸: {original_w}x{original_h}")
        self.logger.info(f"处理后图像尺寸: {processed_w}x{processed_h}")

        # 推理
        self.logger.info("开始分割推理...")
        preds = self.sliding_window_predict(image_tensor)  # (num_classes, processed_h, processed_w)

        # 关键步骤：如果进行了 resize，需要将预测结果 resize 回原始尺寸
        if self.use_resize and (processed_w != original_w or processed_h != original_h):
            self.logger.info(f"将预测结果 resize 回原始尺寸: {original_w}x{original_h}")
            preds = self._resize_mask_back(preds, (original_h, original_w), resize_info)
        else:
            self.logger.info("无需 resize，使用原始分辨率")

        # 计算指标（使用原始尺寸）
        analysis_results = self._calculate_metrics(preds, original_h, original_w)

        # 生成可视化结果
        vis_file_paths = self._generate_visualizations(preds, original_image, session_dir)

        # 构建结果 - 按照用户提供的原始模板格式
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "分析时间": timestamp,
            "输入图像": os.path.abspath(image_path),
            "结果目录": os.path.abspath(session_dir),
            "分析结果": {},
            "量化分析指标": {}
        }

        # 添加各病灶的分析结果 - 按照原始模板格式
        lesion_mapping = {
            "MA": "微动脉瘤",
            "HE": "出血",
            "EX": "硬渗出物",
            "SE": "软渗出物"
        }

        for i, lesion in enumerate(LESION_NAMES):
            lesion_result = analysis_results[lesion]
            chinese_name = lesion_mapping[lesion]

            result["分析结果"][f"{chinese_name}分割"] = {
                "状态": f"{chinese_name}分割完成",
                "指标": lesion_result["指标"],
                "结果文件": vis_file_paths[lesion]
            }

        # 添加量化分析指标 - 按照原始模板格式
        for i, lesion in enumerate(LESION_NAMES):
            lesion_result = analysis_results[lesion]
            chinese_name = lesion_mapping[lesion]

            if lesion == "MA":
                result["量化分析指标"]["微动脉瘤数量"] = lesion_result["指标"]["微动脉瘤数量"]
                result["量化分析指标"]["微动脉瘤区域占比"] = lesion_result["指标"]["微动脉瘤区域占比"]
            elif lesion == "HE":
                result["量化分析指标"]["出血区域占比"] = lesion_result["指标"]["出血区域占比"]
                result["量化分析指标"]["出血区域总大小"] = lesion_result["指标"]["出血区域总大小"]
            elif lesion == "EX":
                result["量化分析指标"]["硬渗出物区域占比"] = lesion_result["指标"]["硬渗出物区域占比"]
                result["量化分析指标"]["硬渗出物分布指数"] = lesion_result["指标"]["硬渗出物分布指数"]
                result["量化分析指标"]["硬渗出物平均面积"] = lesion_result["指标"]["硬渗出物平均面积"]
            elif lesion == "SE":
                result["量化分析指标"]["软渗出物区域占比"] = lesion_result["指标"]["软渗出物区域占比"]
                result["量化分析指标"]["软渗出物分布指数"] = lesion_result["指标"]["软渗出物分布指数"]
                result["量化分析指标"]["软渗出物平均面积"] = lesion_result["指标"]["软渗出物平均面积"]

        return result

    def _calculate_metrics(self, preds: torch.Tensor, H: int, W: int) -> Dict[str, Dict]:
        """
        计算各病灶的分割指标

        Args:
            preds: 预测概率 (num_classes, H, W)
            H, W: 图像尺寸

        Returns:
            各病灶的指标字典
        """
        total_pixels = H * W
        results = {}

        for i, lesion in enumerate(LESION_NAMES):
            pred_mask = (preds[i] > 0.5).float().numpy()
            pred_pixels = pred_mask.sum()

            # 计算指标
            area_ratio = float(pred_pixels / total_pixels)

            if lesion == "MA":
                # 微动脉瘤：计算数量（使用连通组件）
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    pred_mask.astype(np.uint8), connectivity=8
                )
                # 排除背景 (label 0)
                num_components = max(0, num_labels - 1)

                results[lesion] = {
                    "指标": {
                        "微动脉瘤数量": int(num_components),
                        "微动脉瘤区域占比": area_ratio
                    }
                }

            elif lesion == "HE":
                # 出血：计算区域占比和总大小
                results[lesion] = {
                    "指标": {
                        "出血区域占比": area_ratio,
                        "出血区域总大小": float(pred_pixels)
                    }
                }

            elif lesion == "EX":
                # 硬渗出物：计算区域占比、分布指数、平均面积
                if pred_pixels > 0:
                    # 计算连通组件
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        pred_mask.astype(np.uint8), connectivity=8
                    )
                    if num_labels > 1:  # 排除背景
                        areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景的面积
                        avg_area = float(np.mean(areas))
                        # 分布指数：面积标准差/平均面积
                        if len(areas) > 1:
                            distribution_index = float(np.std(areas) / np.mean(areas))
                        else:
                            distribution_index = 0.0
                    else:
                        avg_area = 0.0
                        distribution_index = 0.0
                else:
                    avg_area = 0.0
                    distribution_index = 0.0

                results[lesion] = {
                    "指标": {
                        "硬渗出物区域占比": area_ratio,
                        "硬渗出物分布指数": distribution_index,
                        "硬渗出物平均面积": avg_area
                    }
                }

            elif lesion == "SE":
                # 软渗出物：计算区域占比、分布指数、平均面积
                if pred_pixels > 0:
                    # 计算连通组件
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        pred_mask.astype(np.uint8), connectivity=8
                    )
                    if num_labels > 1:  # 排除背景
                        areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景的面积
                        avg_area = float(np.mean(areas))
                        # 分布指数：面积标准差/平均面积
                        if len(areas) > 1:
                            distribution_index = float(np.std(areas) / np.mean(areas))
                        else:
                            distribution_index = 0.0
                    else:
                        avg_area = 0.0
                        distribution_index = 0.0
                else:
                    avg_area = 0.0
                    distribution_index = 0.0

                results[lesion] = {
                    "指标": {
                        "软渗出物区域占比": area_ratio,
                        "软渗出物分布指数": distribution_index,
                        "软渗出物平均面积": avg_area
                    }
                }

        return results

    def _generate_visualizations(self, preds: torch.Tensor, original_image: np.ndarray, output_dir: str) -> Dict[str, str]:
        """
        生成可视化结果:
        1. 为每个病灶生成黑白二值图，但使用原始文件名保存。
        2. 生成一个包含所有病灶的彩色叠加综合图。

        Args:
            preds: 预测概率 (num_classes, H, W)
            original_image: 原始图像 (H, W, 3)
            output_dir: 输出目录

        Returns:
            文件路径映射字典，与原始结构保持一致。
        """
        # 打印输入图像尺寸
        self.logger.info(f"输入图像尺寸: {original_image.shape[1]}x{original_image.shape[0]}")

        H, W = original_image.shape[:2]
        file_paths = {}

        # 1. 为每个病灶生成独立的黑白二值图，但使用原始文件名
        for i, lesion in enumerate(LESION_NAMES):
            pred_mask = (preds[i] > 0.5).float().numpy()

            # 打印病灶掩码尺寸
            self.logger.info(f"病灶 '{LESION_CHINESE_NAMES[lesion]}' 分割掩码尺寸: {pred_mask.shape[1]}x{pred_mask.shape[0]}")

            # 创建黑白二值图像
            binary_image = np.zeros((H, W), dtype=np.uint8)
            binary_image[pred_mask > 0.5] = 255

            # 使用原始文件名保存
            output_path = os.path.join(output_dir, f"{LESION_CHINESE_NAMES[lesion].lower()}_analysis.png")
            cv2.imwrite(output_path, binary_image)

            file_paths[lesion] = output_path
            #self.logger.info(f"已保存 {lesion} 的二值掩码图到: {output_path}")

        # 2. 生成并保存综合彩色叠加图
        combined_overlay = original_image.copy().astype(np.float32)
        for i, lesion in enumerate(LESION_NAMES):
            pred_mask = (preds[i] > 0.5).float().numpy()
            # 创建一个临时的彩色掩码用于叠加
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            color_mask[pred_mask > 0.5] = COLOR_MAP[lesion]
            # 叠加到综合图上
            combined_overlay = cv2.addWeighted(combined_overlay, 0.8, color_mask.astype(np.float32), 0.2, 0)

        # 为综合图添加图例
        combined_overlay_with_legend = self._add_legend_to_image(combined_overlay.astype(np.uint8), LESION_NAMES)

        # 保存综合图
        combined_path = os.path.join(output_dir, "dr_combined_analysis.png")
        cv2.imwrite(combined_path, cv2.cvtColor(combined_overlay_with_legend, cv2.COLOR_RGB2BGR))
        self.logger.info(f"已保存综合彩色叠加图到: {combined_path}")

        return file_paths

    def _add_legend_to_image(self, image: np.ndarray, lesions_to_show: List[str]) -> np.ndarray:
        """
        向图像添加图例

        Args:
            image: 输入图像 (H, W, 3)
            lesions_to_show: 要显示的病灶类型列表

        Returns:
            添加了图例的图像
        """
        H, W = image.shape[:2]
        legend_image = image.copy()

        # 图例参数
        legend_height = 30
        legend_margin = 10
        text_margin = 5
        color_box_size = 20

        # 为每个病灶添加图例条目
        for i, lesion in enumerate(lesions_to_show):
            color = COLOR_MAP[lesion]
            # 使用英文名称避免乱码
            english_name = {
                "MA": "MA",
                "HE": "HE",
                "EX": "EX",
                "SE": "SE"
            }[lesion]

            # 计算位置
            y_start = H - legend_height * (len(lesions_to_show) - i) + legend_margin
            y_end = y_start + color_box_size

            # 绘制颜色方块
            cv2.rectangle(legend_image,
                         (legend_margin, y_start),
                         (legend_margin + color_box_size, y_end),
                         color, -1)

            # 添加边框
            cv2.rectangle(legend_image,
                         (legend_margin, y_start),
                         (legend_margin + color_box_size, y_end),
                         (255, 255, 255), 1)

            # 添加文字
            text_x = legend_margin + color_box_size + text_margin
            text_y = y_start + color_box_size - 5

            cv2.putText(legend_image, english_name,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(legend_image, english_name,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        return legend_image


def main():
    parser = argparse.ArgumentParser(description="DR单张图片分割分析器")
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument( "--model", "-m", default="models/best.pth", help="模型权重文件路径")
    parser.add_argument("--config", "-c", default="models/config.json", help="模型配置文件路径")
    parser.add_argument("--device", "-d", default="auto", help="计算设备 (auto/cpu/cuda/cuda:0等)")

    # 新增：分辨率适配参数
    parser.add_argument("--target-resolution", "-tr", type=str, default=None,
                        help="目标分辨率，格式: WxH (例如: 4288x2848)。如果为 None 则使用原始分辨率")
    parser.add_argument("--resize-method", "-rm", type=str, default="padding",
                        choices=["direct", "padding"],
                        help="Resize 方法: 'direct' (直接resize，可能变形) 或 'padding' (保持宽高比+填充黑边，推荐)")

    args = parser.parse_args()

    # 确保输出路径是绝对路径
    output_path = os.path.abspath(args.output)

    # 解析目标分辨率
    target_resolution = None
    if args.target_resolution:
        try:
            width, height = map(int, args.target_resolution.lower().split('x'))
            target_resolution = (width, height)
        except Exception as e:
            raise ValueError(f"无效的目标分辨率格式: {args.target_resolution}。请使用格式: WxH (例如: 4288x2848)")

    try:
        # 初始化分析器
        analyzer = DRAnalyzer(
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            target_resolution=target_resolution,
            resize_method=args.resize_method
        )
        # 运行分析
        result = analyzer.analyze_image(args.image, output_path)

        # 打印最终的JSON结果到stdout
        # 确保即时在管道中也能被捕获
        print(json.dumps(result, ensure_ascii=False, indent=4))

    except Exception as e:
        # 捕获初始化或执行过程中的任何错误
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"执行 dr_analyzer.py 失败: {e}", exc_info=True)
        # 打印错误信息的JSON到stdout，以便上层脚本捕获
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, ensure_ascii=False, indent=4))
        sys.exit(1) # 以非零状态码退出，明确表示失败


if __name__ == "__main__":
    main()

