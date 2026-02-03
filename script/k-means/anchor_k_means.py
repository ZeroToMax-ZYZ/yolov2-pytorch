# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import os
import csv

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.augment import build_yolov2_transforms


VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]

@dataclass
class Sample:
    """
    功能:
        保存一条样本路径信息
    """
    img_path: str
    csv_path: str


def read_voc_csv(csv_path: str, class_to_id: Dict[str, int]) -> Tuple[List[List[float]], List[int]]:
    """
    功能:
        读取 VOC 风格 csv(name,xmin,ymin,xmax,ymax),输出 bboxes 与 class_ids

    输入:
        csv_path:
            csv 文件路径
        class_to_id:
            类别名到类别 id 的映射

    输出:
        bboxes:
            List[[x_min, y_min, x_max, y_max]](float)
        class_ids:
            List[int]
    """
    bboxes: List[List[float]] = []
    class_ids: List[int] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 兼容:空文件/只有表头
    if len(rows) <= 1:
        return bboxes, class_ids

    for row in rows[1:]:
        if len(row) < 5:
            continue
        name = str(row[0]).strip()
        if name not in class_to_id:
            # 遇到未知类别:跳过(也可以选择 raise)
            continue
        try:
            x_min = float(row[1])
            y_min = float(row[2])
            x_max = float(row[3])
            y_max = float(row[4])
        except ValueError:
            continue

        # 过滤异常框
        if x_max <= x_min or y_max <= y_min:
            continue

        bboxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(class_to_id[name])

    return bboxes, class_ids


def clip_bbox_xyxy(b: List[float], w: int, h: int) -> List[float]:
    """
    功能:
        将 bbox 裁剪到图像范围内,避免越界

    输入:
        b: [x_min, y_min, x_max, y_max]
        w,h: 图像宽高

    输出:
        裁剪后的 bbox(float)
    """
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return [x1, y1, x2, y2]


class VOCDataset(Dataset):
    """
    功能:
        读取 VOC 检测数据(images + targets/csv),并输出 yolov2 训练所需的:
        - 图像张量 img_tensor
        - yolov2 label 张量 yolo_tensor

    输入(构造参数):
        base_path:
            数据集根目录,内部包含 images/ 与 targets/
        transform:
            Albumentations Compose(需要支持 bbox)
        img_dir_name / target_dir_name:
            子目录名,默认与示例一致
        img_size:
            最终输出尺寸(默认 448)
        S:
            yolov2 grid 数(默认 7)
        classes:
            类别列表(默认 VOC_CLASSES)

    输出(__getitem__):
        img_tensor: torch.float32, (3, img_size, img_size)
        yolo_tensor: torch.float32, (S, S, 5 + C)
    """
    def __init__(
        self,
        base_path: str,
        transform: Optional[Any] = None,
        img_dir_name: str = "images",
        target_dir_name: str = "targets",
        img_size: int = 448,
        S: int = 7,
        classes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.img_dir = os.path.join(base_path, img_dir_name)
        self.target_dir = os.path.join(base_path, target_dir_name)

        self.img_size = int(img_size)
        self.S = int(S)
        self.transform = transform

        self.classes = classes if classes is not None else VOC_CLASSES
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        self.C = len(self.classes)

        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Sample]:
        """
        功能:
            扫描 images/ 下所有图片,匹配 targets/ 下同名 csv

        规则:
            - 图片后缀支持 .jpg/.jpeg/.png(只要 stem 同名)
            - 如果 csv 不存在则跳过
            - 最终按文件名排序,保证可复现
        """
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Target dir not found: {self.target_dir}")

        exts = {".jpg", ".jpeg", ".png"}
        names = sorted(os.listdir(self.img_dir))

        samples: List[Sample] = []
        for fn in names:
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in exts:
                continue
            img_path = os.path.join(self.img_dir, fn)
            csv_path = os.path.join(self.target_dir, stem + ".csv")
            if not os.path.exists(csv_path):
                continue
            samples.append(Sample(img_path=img_path, csv_path=csv_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 1) 读图(cv2 默认 BGR)
        img_bgr = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {sample.img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2) 读标注(原图坐标系)
        bboxes, class_ids = read_voc_csv(sample.csv_path, self.class_to_id)

        # 3) Albumentations 增强(会同步变换 bbox)
        if self.transform is not None:
            out = self.transform(image=img, bboxes=bboxes, class_labels=class_ids)
            img_t = out["image"]               # torch.Tensor(CHW) if ToTensorV2 used
            bboxes_t = list(out["bboxes"])     # List[Tuple[float,float,float,float]]
            class_ids_t = list(out["class_labels"])
        
        else:
            # 不传 transform 时,做一个最小可用版本:Resize 到 img_size
            img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            # 手动缩放 bbox
            h0, w0 = img.shape[:2]
            sx = float(self.img_size) / float(w0)
            sy = float(self.img_size) / float(h0)
            bboxes_t = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bboxes]
            class_ids_t = class_ids

            # 转 torch.Tensor(CHW)
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().to(torch.float32)

        # 4) 编码 yolov2 label(基于最终图像尺寸)
        #    注意:如果 img_t 是 CHW,则 W=img_t.shape[2], H=img_t.shape[1]
        
        out_h = int(img_t.shape[1])
        out_w = int(img_t.shape[2])
        
        yolo_t = encode_yolov2_targets(
            bboxes_xyxy=[list(map(float, b)) for b in bboxes_t],
            class_ids=[int(c) for c in class_ids_t],
            img_w=out_w,
            img_h=out_h,
            S=self.S,
            C=self.C,
        )

        return img_t, yolo_t


