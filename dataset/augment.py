# augment.py
# -*- coding: utf-8 -*-
"""
augment.py(YOLOv2 数据增强:Darknet 风格近似)
================================================

核心目标:
1) 训练增强:jitter(scale+pad+crop+resize) + flip + HSV(hue/sat/exposure)
2) 验证增强:仅 Resize + 归一化 + ToTensor

说明:
- YOLOv2 / Darknet 通常使用 0~1 归一化(/255)，不做 ImageNet mean/std
- Albumentations 的 HueSaturationValue / RandomBrightnessContrast 只是近似 hue/sat/exposure
"""

from __future__ import annotations

from typing import Tuple, Optional
import os

# 避免 albumentations 更新提示
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_yolov2_transforms(
    img_size: int = 416,
    *,
    # Darknet 常用 padding 灰色背景
    pad_value: int = 114,
    # bbox 过滤(你可以先保持与 YOLOv1 类似)
    bbox_min_area: float = 1.0,
    bbox_min_visibility: float = 0.10,
    # jitter 强度(近似 darknet jitter，建议 0.2~0.3)
    jitter_scale_limit: float = 0.20,
    # 是否引入小角度旋转(论文提到 rotations，但 darknet 检测训练并不常开；默认关闭更稳)
    rotate_limit_deg: float = 0.0,
    rotate_p: float = 0.0,
    # HSV 扰动(近似 hue/sat/exposure)
    hsv_p: float = 0.80,
    hue_shift_limit: int = 10,
    sat_shift_limit: int = 30,
    val_shift_limit: int = 30,
    # 亮度对比度(可选)
    bc_p: float = 0.20,
    brightness_limit: float = 0.20,
    contrast_limit: float = 0.20,
    # 归一化:YOLOv2 更贴近 darknet -> 0~1(不减均值、不除 std)
    normalize_to_01: bool = True,
) -> Tuple[A.Compose, A.Compose]:
    """
    功能:
        构建 YOLOv2 的 train/val transforms(Albumentations)，并支持按 img_size 生成不同版本。

    输入:
        img_size:
            当前训练尺度(multi-scale 时会变化)，例如 320~608 step 32
        normalize_to_01:
            True:输出范围约为 0~1(用 A.Normalize(mean=0,std=1,max=255))
            False:保持 0~255 的 float32(仅 ToTensorV2)

    输出:
        train_transform, val_transform
    """
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=bbox_min_area,
        min_visibility=bbox_min_visibility,
    )

    train_ops = []

    # 1) jitter:随机缩放 + pad + 随机裁剪回固定尺寸 + resize(保证最终尺寸一致)
    # 近似 darknet:随机改变视野，再缩放回网络输入尺寸(非 letterbox)
    train_ops += [
        A.RandomScale(scale_limit=jitter_scale_limit, p=0.50),

        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
            p=1.0,
        ),
        # RandomCrop 是“平移 + 裁剪”的近似(对 bbox 会做裁切与过滤)
        A.RandomCrop(height=img_size, width=img_size, p=0.50),

        # 强制回到固定尺寸(避免上面随机操作导致尺寸不一致)
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
    ]

    # 2) flip
    train_ops += [
        A.HorizontalFlip(p=0.50),
    ]

    # 3) rotation(默认关闭，更贴近常见 darknet 检测训练)
    if rotate_limit_deg > 0.0 and rotate_p > 0.0:
        train_ops += [
            A.Rotate(
                limit=float(rotate_limit_deg),
                border_mode=cv2.BORDER_CONSTANT,
                value=(pad_value, pad_value, pad_value),
                p=float(rotate_p),
            )
        ]

    # 4) HSV / exposure
    train_ops += [
        A.HueSaturationValue(
            hue_shift_limit=int(hue_shift_limit),
            sat_shift_limit=int(sat_shift_limit),
            val_shift_limit=int(val_shift_limit),
            p=float(hsv_p),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=float(brightness_limit),
            contrast_limit=float(contrast_limit),
            p=float(bc_p),
        ),
    ]

    # 5) normalize + tensor
    if normalize_to_01:
        train_ops += [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    else:
        train_ops += [
            ToTensorV2(),
        ]

    train_transform = A.Compose(train_ops, bbox_params=bbox_params)

    # -------------------------
    # val:仅 resize + normalize
    # -------------------------
    val_ops = [
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
    ]
    if normalize_to_01:
        val_ops += [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    else:
        val_ops += [
            ToTensorV2(),
        ]

    val_transform = A.Compose(val_ops, bbox_params=bbox_params)

    return train_transform, val_transform
