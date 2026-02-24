# -*- coding: utf-8 -*-
"""
入口:
    VOCDataset(base_path, transform, anchors_rel, ...)
出口:
    __getitem__ -> (img_tensor, yolo_tensor)
        img_tensor: (3, H, W) torch.float32
        yolo_tensor: (S, S, A, 5 + C) torch.float32   或 (S, S, A*(5+C))
            对每个 anchor 的 5+C:
                [0] = x（cell 内 offset，范围 (0,1)，推荐与 sigmoid(tx) 对齐）
                [1] = y（cell 内 offset，范围 (0,1)，推荐与 sigmoid(ty) 对齐）
                [2] = tw* = ln(w_rel / p_w_rel)   （对齐 bw = pw * exp(tw)）
                [3] = th* = ln(h_rel / p_h_rel)
                [4] = obj (0/1)
                [5:] = one-hot(C)

YOLOv2 细节对齐（核心）:
1) x,y: cell 内 offset，训练时预测端用 sigmoid 约束到 (0,1)
2) w,h: 相对 anchor prior 的指数缩放 -> 监督量用 ln(w_rel/p_w_rel), ln(h_rel/p_h_rel)
3) 每个 cell 有 A 个 anchors：每个 (cell,anchor) 最多负责 1 个 GT
4) anchor priors 由训练集 bbox 尺寸做 k-means 聚类得到：distance = 1 - IoU(wh, centroid)

标注格式:
    targets/*.csv, 第一行 header, 后续每行:
    name,x_min,y_min,x_max,y_max
    坐标为像素(原图坐标系)

重要工程建议（与你之前讨论一致）:
- anchors 建议离线计算一次，并保存 JSON；Dataset 读取即可
- 不建议在 __getitem__ 内触发聚类（多进程 DataLoader 会重复算）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import os
import csv
import json
import math

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from icecream import ic
# 你自己的增强构建函数（若没有可先不传 transform）
# from dataset.augment import build_yolov2_transforms


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


# ============================================================
# 1) 读取 CSV 标注（与 YOLOv1 版一致）
# ============================================================
def read_voc_csv(csv_path: str, class_to_id: Dict[str, int]) -> Tuple[List[List[float]], List[int]]:
    """
    功能:
        读取 VOC 风格 csv(name,xmin,ymin,xmax,ymax), 输出 bboxes 与 class_ids
    """
    bboxes: List[List[float]] = []
    class_ids: List[int] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 兼容: 空文件/只有表头
    if len(rows) <= 1:
        return bboxes, class_ids

    for row in rows[1:]:
        if len(row) < 5:
            continue
        name = str(row[0]).strip()
        if name not in class_to_id:
            continue
        try:
            x_min = float(row[1])
            y_min = float(row[2])
            x_max = float(row[3])
            y_max = float(row[4])
        except ValueError:
            continue

        if x_max <= x_min or y_max <= y_min:
            continue

        bboxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(int(class_to_id[name]))

    return bboxes, class_ids


def clip_bbox_xyxy(b: List[float], w: int, h: int) -> List[float]:
    """
    功能:
        将 bbox 裁剪到图像范围内，避免越界
    """
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return [x1, y1, x2, y2]


# ============================================================
# 2) 【新增】YOLOv2 anchor 聚类：全 torch Tensor 实现
# ============================================================
def iou_wh_torch(wh: torch.Tensor, anchors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    功能:
        在 (w,h) 空间计算 IoU（假设中心对齐），用于：
        - k-means distance = 1 - IoU
        - anchor 匹配 best anchor

    输入:
        wh:
            (N,2) 或 (...,2)
        anchors:
            (K,2)

    输出:
        ious:
            (N,K) 或 (...,K) 取决于 wh 的维度
    """
    # wh: (..., 2), anchors: (K, 2)
    w = wh[..., 0:1]  # (...,1)
    h = wh[..., 1:2]  # (...,1)

    aw = anchors[:, 0].view(1, -1)  # (1,K)
    ah = anchors[:, 1].view(1, -1)  # (1,K)
    # 当两个矩形中心重合时，它们的交集宽度等于 min(w1, w2)，交集高度等于 min(h1, h2)
    inter = torch.minimum(w, aw) * torch.minimum(h, ah)  # (...,K)
    union = w * h + aw * ah - inter + eps
    return inter / union


def load_anchors_json(json_path: str) -> List[Tuple[float, float]]:
    """
    功能:
        从 json 读取 anchors_rel
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data.get("anchors_rel", [])
    return [(float(a[0]), float(a[1])) for a in anchors]


def encode_yolov2_targets(
    bboxes_xyxy: List[List[float]],
    class_ids: List[int],
    img_w: int,
    img_h: int,
    S: int,
    C: int,
    anchors_rel: List[Tuple[float, float]],
    *,
    flatten_targets: bool = False,
    min_box_size_px: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    功能:
        将 (xyxy + class_id) 编码成 YOLOv2 的 (S,S,A,5+C) label 张量

    输入:
        bboxes_xyxy:
            N 个 bbox,每个 [x_min, y_min, x_max, y_max]，像素坐标，基于最终输入图像尺寸
        class_ids:
            每个 bbox 对应的类别 id(0..C-1)
        img_w, img_h:
            最终输入图像尺寸（例如 416x416；multi-scale 时可能变化）
        S:
            grid 数（通常 S = img_size/32，例如 416/32=13）
        C:
            类别数（VOC=20）
        anchors_rel:
            anchors prior，使用相对输入尺寸归一化 (w_rel,h_rel)，范围 (0,1]
        flatten_targets:
            True -> (S,S,A*(5+C))，False -> (S,S,A,5+C)
        min_box_size_px:
            可选：过滤极小框（默认 0 关闭）
        eps:
            数值稳定用

    输出:
        yolo_tensor:
            (S,S,A,5+C) 或 (S,S,A*(5+C))

    编码细节（按 YOLOv2 参数化监督）:
        - x,y：cell 内 offset ∈ (0,1)
        - tw,th：ln(w_rel/p_w_rel), ln(h_rel/p_h_rel)
        - obj：1
        - cls：one-hot
    """
    assert len(anchors_rel) > 0, "YOLOv2 需要 anchors_rel（请先做 k-means 聚类）"
    A = int(len(anchors_rel))

    # 【修改】YOLOv2: 每个 cell 有 A 个 anchor
    yolo = torch.zeros((S, S, A, 5 + C), dtype=torch.float32)

    if len(bboxes_xyxy) == 0:
        return yolo.view(S, S, A * (5 + C)) if flatten_targets else yolo

    cell_w = float(img_w) / float(S)
    cell_h = float(img_h) / float(S)

    # 【新增】同一 (cell,anchor) 最多写一个 GT
    used = torch.zeros((S, S, A), dtype=torch.bool)

    # 提前转 anchors
    anc = torch.tensor(anchors_rel, dtype=torch.float32)  # (A,2)

    for b, cid in zip(bboxes_xyxy, class_ids):
        x1, y1, x2, y2 = clip_bbox_xyxy(b, img_w, img_h)

        bw = x2 - x1
        bh = y2 - y1

        # 合法性兜底
        if bw <= 0.0 or bh <= 0.0:
            continue

        # 可选极小框过滤（默认关闭）
        if min_box_size_px > 0.0 and (bw < min_box_size_px or bh < min_box_size_px):
            continue

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        grid_x = int(cx // cell_w)
        grid_y = int(cy // cell_h)

        # 防止越界
        grid_x = max(0, min(S - 1, grid_x))
        grid_y = max(0, min(S - 1, grid_y))

        # cell 内 offset（归一化到 (0,1)，避免 0/1 导致后续 logit/数值边界问题）
        offset_x = (cx - grid_x * cell_w) / cell_w
        offset_y = (cy - grid_y * cell_h) / cell_h
        offset_x = float(max(eps, min(1.0 - eps, offset_x)))
        offset_y = float(max(eps, min(1.0 - eps, offset_y)))

        # GT 宽高归一化到相对输入尺度
        w_rel = float(bw) / float(img_w)
        h_rel = float(bh) / float(img_h)

        if w_rel <= 0.0 or h_rel <= 0.0:
            continue

        # 计算 best anchor（基于 wh IoU）
        gt_wh = torch.tensor([[w_rel, h_rel]], dtype=torch.float32)  # (1,2)
        ious = iou_wh_torch(gt_wh, anc)                              # (1,A)
        order = torch.argsort(ious[0], descending=True)              # (A,)

        # 【新增】如果 best anchor 已占用，尝试次优 anchor
        chosen_a: Optional[int] = None
        for ai in order.tolist():
            if not bool(used[grid_y, grid_x, ai].item()):
                chosen_a = int(ai)
                break
        if chosen_a is None:
            # 该 cell 的所有 anchor 都占了（极少发生），跳过
            continue

        used[grid_y, grid_x, chosen_a] = True

        aw_rel = float(anchors_rel[chosen_a][0])
        ah_rel = float(anchors_rel[chosen_a][1])
        aw_rel = max(aw_rel, eps)
        ah_rel = max(ah_rel, eps)

        # YOLOv2 参数化监督：tw*=ln(w_rel/p_w_rel), th*=ln(h_rel/p_h_rel)
        tw = float(math.log(max(w_rel, eps) / aw_rel))
        th = float(math.log(max(h_rel, eps) / ah_rel))

        onehot = F.one_hot(torch.tensor(int(cid), dtype=torch.int64), num_classes=C).to(torch.float32)

        label = torch.zeros((5 + C,), dtype=torch.float32)
        label[0] = float(offset_x)
        label[1] = float(offset_y)
        label[2] = float(tw)
        label[3] = float(th)
        label[4] = 1.0
        label[5:] = onehot

        yolo[grid_y, grid_x, chosen_a, :] = label

    return yolo.view(S, S, A * (5 + C)) if flatten_targets else yolo


# ============================================================
# 4) 【修改】YOLOv2 Dataset（与 YOLOv1 Dataset 风格一致）
# ============================================================
class VOCDataset(Dataset):
    """
    功能:
        读取 VOC 检测数据(images + targets/csv),并输出 YOLOv2 训练所需的:
        - 图像张量 img_tensor
        - YOLOv2 label 张量 yolo_tensor（anchor-based）

    输入(构造参数):
        base_path:
            数据集根目录,内部包含 images/ 与 targets/
        transform:
            Albumentations Compose(需要支持 bbox), 且输出 out["image"] 为 torch.Tensor(CHW)
        anchors_rel: 【新增】
            YOLOv2 必需：k-means 得到的 priors (w_rel,h_rel)
        img_size: 【修改】
            YOLOv2 默认检测分辨率常用 416
        stride: 【新增】
            默认为 32，用于推导 S = img_size/stride
        S:
            若不传则自动由 img_size/stride 推导
        flatten_targets: 【新增】
            True -> 输出 (S,S,A*(5+C))，False -> 输出 (S,S,A,5+C)

    输出(__getitem__):
        img_tensor: torch.float32, (3, img_size, img_size)
        yolo_tensor: torch.float32, (S, S, A, 5 + C) 或 (S, S, A*(5+C))
    """

    def __init__(
        self,
        base_path: str,
        transform: Optional[Any] = None,
        *,
        anchors_rel: List[Tuple[float, float]],     # 【新增】
        img_dir_name: str = "images",
        target_dir_name: str = "targets",
        img_size: int = 416,                        # 【修改】默认 416
        stride: int = 32,                           # 【新增】默认 32
        S: Optional[int] = None,                    # 【新增】可自动推导
        classes: Optional[List[str]] = None,
        flatten_targets: bool = False,              # 【新增】
        min_box_size_px: float = 0.0,               # 【新增】默认不额外过滤
    ) -> None:
        super().__init__()

        self.base_path = base_path
        self.img_dir = os.path.join(base_path, img_dir_name)
        self.target_dir = os.path.join(base_path, target_dir_name)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Target dir not found: {self.target_dir}")

        self.transform = transform
        self.img_size = int(img_size)
        self.stride = int(stride)

        if S is None:
            if self.img_size % self.stride != 0:
                raise ValueError(f"img_size({self.img_size}) must be divisible by stride({self.stride})")
            self.S = self.img_size // self.stride
        else:
            self.S = int(S)

        self.classes = classes if classes is not None else VOC_CLASSES
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        self.C = int(len(self.classes))

        if len(anchors_rel) == 0:
            raise ValueError("anchors_rel is empty. Please run k-means clustering to get priors first.")
        self.anchors_rel = [(float(w), float(h)) for (w, h) in anchors_rel]  # (A,2)

        self.flatten_targets = bool(flatten_targets)
        self.min_box_size_px = float(min_box_size_px)

        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Sample]:
        """
        功能:
            扫描 images/ 下所有图片,匹配 targets/ 下同名 csv
        """
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

        # 1) 读图
        img_bgr = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {sample.img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2) 读标注（原图坐标系）
        bboxes, class_ids = read_voc_csv(sample.csv_path, self.class_to_id)

        # 3) 增强（会同步变换 bbox）
        if self.transform is not None:
            out = self.transform(image=img, bboxes=bboxes, class_labels=class_ids)
            img_t = out["image"]               # torch.Tensor(CHW) if ToTensorV2 used
            bboxes_t = list(out["bboxes"])     # List[Tuple[float,float,float,float]]
            class_ids_t = list(out["class_labels"])
        else:
            # 最小可用：Resize 到 img_size（stretch）
            img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            # 手动缩放 bbox
            h0, w0 = img.shape[:2]
            sx = float(self.img_size) / float(w0)
            sy = float(self.img_size) / float(h0)
            bboxes_t = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bboxes]
            class_ids_t = class_ids

            # 转 torch.Tensor(CHW)，保持 0~255 的 float32（与你 YOLOv1 dataset 一致）
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().to(torch.float32)

        # 4) 编码 YOLOv2 label（基于最终图像尺寸）
        out_h = int(img_t.shape[1])
        out_w = int(img_t.shape[2])

        yolo_t = encode_yolov2_targets(
            bboxes_xyxy=[list(map(float, b)) for b in bboxes_t],
            class_ids=[int(c) for c in class_ids_t],
            img_w=out_w,
            img_h=out_h,
            S=self.S,
            C=self.C,
            anchors_rel=self.anchors_rel,
            flatten_targets=self.flatten_targets,
            min_box_size_px=self.min_box_size_px,
        )

        return img_t, yolo_t


if __name__ == "__main__":
    cfg = {
        # "train_path": r"/root/autodl-tmp/YOLOv1_dataset/train",
        "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train",
        "anchors_json": r"dataset/anchors_k5.json",
    }
    anchors_rel = load_anchors_json(cfg["anchors_json"])
    ds = VOCDataset(
                base_path=cfg["train_path"],
                anchors_rel=anchors_rel,
            )
    
    print(len(ds))
    