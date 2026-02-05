# dataset/augment.py
# -*- coding: utf-8 -*-
"""
入口:
    build_yolov2_transforms(img_size, ...)

出口:
    train_transform, val_transform (Albumentations Compose)

YOLOv2 对齐要点（工程近似）:
- 不使用 letterbox（这里采用“stretch 到正方形” + 随机 crop/translate 的近似 jitter）
- Color distortion（hue/sat/exposure 的近似）
- 最终强制 Resize 到 img_size（用于 multi-scale：img_size 会随 batch 变化）

注意:
- 这里使用 Albumentations 的 bbox_params 同步变换 bbox（format=pascal_voc）
- normalize_to_01=True 时，输出是 normalize 后的 tensor（和你 YOLOv1 augment 一致）
"""

from __future__ import annotations

from typing import Tuple
import os
import cv2

# 避免 albumentations 更新提示
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_yolov2_transforms(
    img_size: int = 416,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    bbox_min_area: float = 1.0,
    bbox_min_visibility: float = 0.10,
    pad_value: int = 114,
    jitter_scale_limit: float = 0.20,
    rotate_limit_deg: float = 0.0,
    rotate_p: float = 0.0,
    normalize_to_01: bool = True,
) -> Tuple[A.Compose, A.Compose]:
    """
    功能:
        构建 YOLOv2 风格增强（Albumentations 近似）

    输入:
        img_size:
            最终输出尺寸（multi-scale 时会动态传入不同 size）
        jitter_scale_limit:
            尺度抖动强度（近似 darknet jitter 的一部分效果）
        pad_value:
            padding 填充值（你之前用 114，保持一致）
        normalize_to_01:
            True -> Normalize(mean,std,max_pixel_value=255.0) + ToTensorV2
            False -> 仅 ToTensorV2（输出仍是 0~255 float tensor）

    输出:
        train_transform, val_transform
    """
    img_size = int(img_size)

    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=float(bbox_min_area),
        min_visibility=float(bbox_min_visibility),
    )

    # -------------------------
    # 【新增】YOLOv2 训练增强（近似）
    # 说明：
    # - YOLOv2/darknet 常见是 jitter + random crop/translate + flip + hsv
    # - Albumentations 无法 1:1 复刻 darknet 的 “new_ar + scale + dx/dy” 那套，
    #   这里用 RandomScale + PadIfNeeded + RandomCrop 组合去近似
    # -------------------------
    train_ops = [
        # 1) 随机尺度抖动
        A.RandomScale(scale_limit=float(jitter_scale_limit), p=0.50),

        # 2) Pad 到至少 img_size，再随机裁剪回 img_size（近似平移+裁剪）
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(int(pad_value), int(pad_value), int(pad_value)),
            p=1.0,
        ),
        A.RandomCrop(height=img_size, width=img_size, p=0.50),

        # 3) 最终强制到固定尺寸（multi-scale 的关键：保证 batch 内一致）
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),

        # 4) 水平翻转
        A.HorizontalFlip(p=0.50),

        # 5) 颜色扰动（近似 hue/sat/exposure）
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.80,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.20,
        ),
    ]

    # 可选：旋转（检测里不一定需要，默认关）
    if float(rotate_limit_deg) > 0.0 and float(rotate_p) > 0.0:
        train_ops.insert(
            3,
            A.Rotate(limit=float(rotate_limit_deg), border_mode=cv2.BORDER_CONSTANT,
                     value=(int(pad_value), int(pad_value), int(pad_value)), p=float(rotate_p))
        )

    if bool(normalize_to_01):
        train_ops += [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
        val_ops = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    else:
        train_ops += [ToTensorV2()]
        val_ops = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            ToTensorV2(),
        ]

    train_transform = A.Compose(train_ops, bbox_params=bbox_params)
    val_transform = A.Compose(val_ops, bbox_params=bbox_params)

    return train_transform, val_transform
