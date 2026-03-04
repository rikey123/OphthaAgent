
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch


IDRID_MEAN = (0.485, 0.456, 0.406)
IDRID_STD = (0.229, 0.224, 0.225)


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                ],
                p=0.5,
            ),
            A.Normalize(mean=IDRID_MEAN, std=IDRID_STD),
            ToTensorV2(),
        ]
    )


def _additional_targets_for_masks(num_masks: int) -> Dict[str, str]:
    # Albumentations: apply same geometric transform to all masks.
    # We keep the primary key 'mask' unused in multi-mask mode and use mask0..mask{C-1}.
    return {f'mask{i}': 'mask' for i in range(num_masks)}


def get_train_transforms_multi(num_masks: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                ],
                p=0.5,
            ),
            A.Normalize(mean=IDRID_MEAN, std=IDRID_STD),
            ToTensorV2(),
        ],
        additional_targets=_additional_targets_for_masks(num_masks),
    )


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IDRID_MEAN, std=IDRID_STD),
        ToTensorV2(),
    ])


def get_val_transforms_multi(num_masks: int) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IDRID_MEAN, std=IDRID_STD),
            ToTensorV2(),
        ],
        additional_targets=_additional_targets_for_masks(num_masks),
    )


def _mask_suffix_from_dir(mask_dir: str) -> str:
    if 'Microaneurysms' in mask_dir:
        return 'MA'
    if 'Haemorrhages' in mask_dir:
        return 'HE'
    if 'Hard Exudates' in mask_dir:
        return 'EX'
    if 'Soft Exudates' in mask_dir:
        return 'SE'
    if 'Optic Disc' in mask_dir:
        return 'OD'
    raise ValueError(f'Unrecognized mask dir: {mask_dir}')


def load_idrid_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_idrid_merged_mask(image_name: str, mask_dirs: List[str]) -> np.ndarray:
    """返回单通道二值 mask (H,W), 值为 0/1 float32。"""
    base_name = os.path.splitext(image_name)[0]
    merged: Optional[np.ndarray] = None

    for mdir in mask_dirs:
        suffix = _mask_suffix_from_dir(mdir)
        mpath = os.path.join(mdir, f'{base_name}_{suffix}.tif')
        if not os.path.exists(mpath):
            continue
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if merged is None:
            merged = m.astype(np.uint8)
        else:
            merged = np.maximum(merged, m.astype(np.uint8))

    if merged is None:
        # 没有 mask 文件时返回全 0
        # 注意：调用方应确保这个情况不会影响训练（比如漏配目录）
        raise FileNotFoundError(f'No mask found for {image_name} in provided mask_dirs')

    return (merged > 0).astype(np.float32)


def load_idrid_multi_masks(image_name: str, mask_dirs: List[str]) -> np.ndarray:
    """返回多通道二值 mask (C,H,W), C=len(mask_dirs), 值为 0/1 float32。

    注意：IDRiD 的某些类别在“该图没有该病灶”时可能不会提供对应的 mask 文件。
    这种情况下将该通道视为全 0，而不是报错。
    """
    base_name = os.path.splitext(image_name)[0]
    masks: List[Optional[np.ndarray]] = []
    shape_hw: Optional[Tuple[int, int]] = None

    for mdir in mask_dirs:
        suffix = _mask_suffix_from_dir(mdir)
        mpath = os.path.join(mdir, f'{base_name}_{suffix}.tif')
        if not os.path.exists(mpath):
            masks.append(None)
            continue

        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            masks.append(None)
            continue

        if shape_hw is None:
            shape_hw = (int(m.shape[0]), int(m.shape[1]))
        masks.append((m > 0).astype(np.float32))

    # Determine shape; if all masks are missing, we cannot infer H,W.
    if shape_hw is None:
        raise FileNotFoundError(f'All class masks are missing for {image_name} in provided mask_dirs')

    filled: List[np.ndarray] = []
    for m in masks:
        if m is None:
            filled.append(np.zeros(shape_hw, dtype=np.float32))
        else:
            if m.shape[:2] != shape_hw:
                raise ValueError(f'Mask shape mismatch for {image_name}: got {m.shape[:2]} vs {shape_hw}')
            filled.append(m)

    return np.stack(filled, axis=0)


def _positions_1d(length: int, patch_size: int, stride: int) -> List[int]:
    """Generate sliding window start indices that guarantee edge coverage."""
    if length <= patch_size:
        return [0]
    last = length - patch_size
    pos = list(range(0, last + 1, stride))
    if pos[-1] != last:
        pos.append(last)
    return pos


class IDRiDPatchDataset(Dataset):
    """将 IDRiD 大图切为 patch，用于训练/验证。

    通过缓存机制避免每个 patch 重复读取整张大图。
    返回:
      image: Tensor [3,H,W]
      mask:  Tensor [1,H,W]
    """

    def __init__(
        self,
        image_dir: str,
        mask_dirs: List[str],
        patch_size: int = 512,
        stride: int = 256,
        transform: Optional[A.Compose] = None,
        oversample_factor: int = 5,
        return_multi_masks: bool = False,
        hard_negative: bool = False,
        hard_negative_topk: float = 0.3,
        hard_negative_oversample: int = 2,
    ):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.oversample_factor = max(1, oversample_factor)
        self.return_multi_masks = return_multi_masks
        self.hard_negative = bool(hard_negative)
        self.hard_negative_topk = float(hard_negative_topk)
        self.hard_negative_oversample = max(1, int(hard_negative_oversample))
        self.num_masks = len(mask_dirs)
        self._default_transform = get_val_transforms_multi(self.num_masks) if self.return_multi_masks else get_val_transforms()

        self.image_names = sorted([n for n in os.listdir(image_dir) if n.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
        if len(self.image_names) == 0:
            raise FileNotFoundError(f'No images found in {image_dir}')

        # patch 索引
        positive: List[Tuple[str, int, int]] = []
        negative: List[Tuple[str, int, int]] = []
        negative_scored: List[Tuple[float, str, int, int]] = []
        for name in self.image_names:
            img_path = os.path.join(self.image_dir, name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = load_idrid_merged_mask(name, self.mask_dirs)

            # Hard negative scoring: use edge/gradient density as a cheap proxy for
            # vessel/noisy structures that look like lesions.
            grad_integral = None
            if self.hard_negative:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(gx, gy)
                grad_integral = cv2.integral(grad)

            ys = _positions_1d(h, patch_size, stride)
            xs = _positions_1d(w, patch_size, stride)
            for y in ys:
                for x in xs:
                    sub_mask = mask[y : y + patch_size, x : x + patch_size]
                    if sub_mask.sum() > 0:
                        positive.append((name, y, x))
                    else:
                        negative.append((name, y, x))
                        if grad_integral is not None:
                            # integral image is (H+1, W+1)
                            y2 = y + patch_size
                            x2 = x + patch_size
                            s = (
                                grad_integral[y2, x2]
                                - grad_integral[y, x2]
                                - grad_integral[y2, x]
                                + grad_integral[y, x]
                            )
                            score = float(s) / float(patch_size * patch_size)
                            negative_scored.append((score, name, y, x))

        # Hard negative mining: oversample top-k negatives with high edge density.
        hard_negative_patches: List[Tuple[str, int, int]] = []
        if self.hard_negative and negative_scored:
            topk_ratio = min(max(self.hard_negative_topk, 0.0), 1.0)
            k = max(1, int(round(len(negative_scored) * topk_ratio)))
            negative_scored.sort(key=lambda t: t[0], reverse=True)
            hard_negative_patches = [(n, y, x) for _s, n, y, x in negative_scored[:k]]

        if hard_negative_patches:
            negative = hard_negative_patches * self.hard_negative_oversample + negative

        self.patches = positive * self.oversample_factor + negative
        if not self.patches:
            raise ValueError('No patches collected after oversampling; check masks')
        random.shuffle(self.patches)

        # 缓存（按 DataLoader worker 各自维护）
        self._cache_name: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_mask: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.patches)

    def _get_cached(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_name != name:
            img_path = os.path.join(self.image_dir, name)
            img = load_idrid_image_rgb(img_path)
            mask = load_idrid_merged_mask(name, self.mask_dirs)
            if mask.shape[:2] != img.shape[:2]:
                raise ValueError(f'Mask/Image shape mismatch for {name}: {mask.shape} vs {img.shape}')
            self._cache_name = name
            self._cache_img = img
            self._cache_mask = mask
        assert self._cache_img is not None and self._cache_mask is not None
        return self._cache_img, self._cache_mask

    def __getitem__(self, idx: int):
        name, y, x = self.patches[idx]
        img, mask = self._get_cached(name)

        img_patch = img[y : y + self.patch_size, x : x + self.patch_size]
        aug_transform = self.transform or self._default_transform

        if not self.return_multi_masks:
            mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]
            aug = aug_transform(image=img_patch, mask=mask_patch)
            img_patch_t = aug['image']
            mask_patch_t = aug['mask'].unsqueeze(0)
            return img_patch_t, mask_patch_t

        masks_multi = load_idrid_multi_masks(name, self.mask_dirs)  # (C,H,W)
        masks_multi_patch = masks_multi[:, y : y + self.patch_size, x : x + self.patch_size]

        data = {'image': img_patch}
        for i in range(self.num_masks):
            data[f'mask{i}'] = masks_multi_patch[i]
        aug = aug_transform(**data)
        img_patch_t = aug['image']
        masks_t = [aug[f'mask{i}'] for i in range(self.num_masks)]
        masks_t = torch.stack(masks_t, dim=0).float()
        return img_patch_t, masks_t


class IDRiDFullImageDataset(Dataset):
    """用于评估：返回整张图（Tensor）和整张 mask。"""

    def __init__(
        self,
        image_dir: str,
        mask_dirs: List[str],
        transform: Optional[A.Compose] = None,
        return_multi_masks: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform or get_val_transforms()
        self.return_multi_masks = return_multi_masks

        self.image_names = sorted([n for n in os.listdir(image_dir) if n.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
        if len(self.image_names) == 0:
            raise FileNotFoundError(f'No images found in {image_dir}')

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img = load_idrid_image_rgb(os.path.join(self.image_dir, name))
        mask_merged = load_idrid_merged_mask(name, self.mask_dirs)
        aug = self.transform(image=img, mask=mask_merged)
        img_t = aug['image']
        mask_merged_t = aug['mask'].unsqueeze(0)

        if not self.return_multi_masks:
            return img_t, mask_merged_t, name

        # 注意：multi masks 不做几何增强（当前 val transform 只有 normalize，所以不会破坏对齐）
        masks_multi = load_idrid_multi_masks(name, self.mask_dirs)
        masks_multi_t = torch.from_numpy(masks_multi).float()
        return img_t, mask_merged_t, masks_multi_t, name
