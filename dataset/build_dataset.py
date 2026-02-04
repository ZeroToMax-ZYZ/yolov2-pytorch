# build_dataset.py
# -*- coding: utf-8 -*-
"""
build_dataset.py（YOLOv2：multi-scale DataLoader + 数据集构建）
==============================================================

你当前目标：
- 复现 YOLOv2 的 multi-scale training：每 10 个 batch 随机切输入尺寸
  sizes = {320, 352, ..., 608}，步长 32
- YOLOv2 训练增强使用 augment.py 的 build_yolov2_transforms(img_size)
- ignore 机制放到 loss 内实现（dataset 不输出 ignore mask）

关键实现策略（避免 PyTorch DataLoader 预取/多进程导致尺寸混乱）：
1) 使用 MultiScaleBatchSampler：每个 batch 的 index 都带上 img_size -> (idx, img_size)
2) 使用 MultiScaleDatasetRouter：内部维护 size->dataset 的映射，
   __getitem__((idx,size)) 直接路由到对应固定 size 的 dataset 实例，保证一个 batch 内张量尺寸一致

这样做的好处：
- 不需要在训练 loop 里手动 set_img_size
- 不受 prefetch_factor / persistent_workers 影响
- 多进程 worker 稳定可复现

你需要做的唯一前提：
- 你已经有 YOLOv2 的 VOCDataset（anchor-based label 编码）
  其构造接口应类似：
      VOCDataset(base_path=..., transform=..., anchors_rel=..., img_size=..., stride=32, classes=..., flatten_targets=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Iterable, Iterator

import os
import math

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from dataset.augment import build_yolov2_transforms

# =========================
# 你需要根据你的文件名调整这里的 import
# =========================
# 例如你把我之前给你的 YOLOv2 dataset 存成了 dataset/VOC_dataset_yolov2.py
from dataset.VOC_dataset import VOCDataset, compute_anchors_from_voc_csvs, load_anchors_json


VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]


def build_multiscale_sizes(
    ms_min: int = 320,
    ms_max: int = 608,
    ms_step: int = 32,
) -> List[int]:
    """
    功能：
        生成 YOLOv2 multi-scale 的尺寸列表：
            [320, 352, ..., 608]
    """
    ms_min = int(ms_min)
    ms_max = int(ms_max)
    ms_step = int(ms_step)
    if ms_min > ms_max:
        raise ValueError(f"ms_min({ms_min}) > ms_max({ms_max})")
    if ms_step <= 0:
        raise ValueError(f"ms_step must be > 0, got {ms_step}")
    sizes = list(range(ms_min, ms_max + 1, ms_step))
    return sizes


class MultiScaleBatchSampler:
    """
    功能：
        生成 batch 索引，并为每个 batch 附带一个 img_size：
            batch = [(idx1, size), (idx2, size), ..., (idxB, size)]
        从而保证：
        - 一个 batch 内的所有样本都用同一个输入尺寸
        - 每隔 interval_batches 个 batch 切一次 size（随机从 sizes 中采样）

    注意：
        这是一个“batch sampler”（直接产出 batch），因此 DataLoader 里不要再传 batch_size/shuffle。
    """

    def __init__(
        self,
        indices: List[int],
        batch_size: int,
        *,
        sizes: List[int],
        interval_batches: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.indices = list(map(int, indices))
        self.batch_size = int(batch_size)
        self.sizes = list(map(int, sizes))
        self.interval_batches = int(interval_batches)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self.sizes) == 0:
            raise ValueError("sizes is empty")
        if self.interval_batches <= 0:
            raise ValueError("interval_batches must be > 0")

        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        功能：
            每个 epoch 开始前调用一次，保证 shuffle 与 size 采样可复现且 epoch 间变化
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        # 1) 生成当 epoch 的索引序列
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000003)

        if self.shuffle:
            perm = torch.randperm(len(self.indices), generator=g).tolist()
            idxs = [self.indices[i] for i in perm]
        else:
            idxs = list(self.indices)

        # 2) 按 batch_size 切分
        batches: List[List[int]] = []
        for i in range(0, len(idxs), self.batch_size):
            b = idxs[i:i + self.batch_size]
            if len(b) < self.batch_size and self.drop_last:
                continue
            batches.append(b)

        # 3) 每 interval_batches 个 batch 采样一次新的 size
        cur_size = int(self.sizes[int(torch.randint(0, len(self.sizes), (1,), generator=g).item())])
        for bi, b in enumerate(batches):
            if bi % self.interval_batches == 0:
                cur_size = int(self.sizes[int(torch.randint(0, len(self.sizes), (1,), generator=g).item())])
            yield [(int(x), int(cur_size)) for x in b]


class MultiScaleDatasetRouter(Dataset):
    """
    功能：
        内部维护 size -> dataset 的映射：
            ds_map[size] = VOCDataset(..., img_size=size, transform=build_yolov2_transforms(size)[0], ...)

        __getitem__ 接收：
            1) (idx, size) ：multi-scale batch sampler 的输出
            2) idx(int)    ：兼容普通用法（此时 fallback 到 default_size）

    优点：
        - 不需要修改你的 VOCDataset 代码
        - 不需要共享变量，不会被 DataLoader 的预取打乱
        - 多进程稳定
    """

    def __init__(
        self,
        *,
        base_path: str,
        anchors_rel: List[Tuple[float, float]],
        sizes: List[int],
        stride: int = 32,
        classes: Optional[List[str]] = None,
        flatten_targets: bool = False,
        min_box_size_px: float = 0.0,
        normalize_to_01: bool = True,
        pad_value: int = 114,
        bbox_min_area: float = 1.0,
        bbox_min_visibility: float = 0.10,
        jitter_scale_limit: float = 0.20,
    ) -> None:
        super().__init__()

        self.sizes = list(map(int, sizes))
        if len(self.sizes) == 0:
            raise ValueError("sizes is empty")

        self.default_size = int(self.sizes[len(self.sizes) // 2])  # 随便取一个居中值兜底
        self.ds_map: Dict[int, Dataset] = {}

        classes = classes if classes is not None else VOC_CLASSES

        for sz in self.sizes:
            train_tf, _ = build_yolov2_transforms(
                img_size=sz,
                pad_value=pad_value,
                bbox_min_area=bbox_min_area,
                bbox_min_visibility=bbox_min_visibility,
                jitter_scale_limit=jitter_scale_limit,
                rotate_limit_deg=0.0,
                rotate_p=0.0,
                normalize_to_01=normalize_to_01,
            )

            ds = VOCDataset(
                base_path=base_path,
                transform=train_tf,
                anchors_rel=anchors_rel,
                img_size=int(sz),
                stride=int(stride),
                S=None,
                classes=classes,
                flatten_targets=flatten_targets,
                min_box_size_px=float(min_box_size_px),
            )
            self.ds_map[int(sz)] = ds

        # 所有 ds 的样本列表应一致（同一 base_path，按文件名排序），长度应该相同
        any_ds = next(iter(self.ds_map.values()))
        self._length = len(any_ds)

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, item: Any):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            idx = int(item[0])
            sz = int(item[1])
        else:
            idx = int(item)
            sz = int(self.default_size)

        if sz not in self.ds_map:
            # 理论不会发生：batch sampler 只会采样 sizes 内的 size
            sz = int(self.default_size)

        return self.ds_map[sz][idx]


def _resolve_anchors_rel(cfg: Dict[str, Any], class_to_id: Dict[str, int]) -> List[Tuple[float, float]]:
    """
    功能：
        1) 若 cfg["anchors_rel"] 直接提供，则直接使用
        2) 否则读取 cfg["anchors_json"]
        3) 若 json 不存在且 cfg["auto_compute_anchors"]=True，则从 train_path 扫描计算并保存

    备注：
        这里假设你的 YOLOv2 dataset 文件里实现了：
            compute_anchors_from_voc_csvs(...)
            load_anchors_json(...)
    """
    if cfg.get("anchors_rel", None) is not None:
        anchors_rel = [(float(a[0]), float(a[1])) for a in cfg["anchors_rel"]]
        if len(anchors_rel) == 0:
            raise ValueError("cfg['anchors_rel'] is empty")
        return anchors_rel

    anchors_json = cfg.get("anchors_json", None)
    if anchors_json is None:
        # 默认保存到 train_path 下
        anchors_json = os.path.join(cfg["train_path"], "anchors_k5.json")

    if os.path.exists(anchors_json):
        return load_anchors_json(anchors_json)

    if not bool(cfg.get("auto_compute_anchors", True)):
        raise FileNotFoundError(f"anchors_json not found: {anchors_json}")

    k = int(cfg.get("anchors_k", 5))
    seed = int(cfg.get("anchors_seed", 0))
    min_box_size_px = float(cfg.get("anchors_min_box_size_px", 0.0))

    anchors_rel = compute_anchors_from_voc_csvs(
        base_path=cfg["train_path"],
        class_to_id=class_to_id,
        k=k,
        seed=seed,
        out_json=anchors_json,
        min_box_size_px=min_box_size_px,
    )
    return anchors_rel


def build_dataset(cfg: Dict[str, Any]):
    """
    功能：
        构建 YOLOv2 的 train_loader / val_loader，并支持 multi-scale。

    输入 cfg 关键字段（建议）：
        train_path: str
        test_path: str
        batch_size: int
        num_workers: int
        persistent_workers: bool
        input_size: int            # 用作 val_size 默认值（val 固定尺寸）
        stride: int = 32
        debug_mode: Optional[float] = None

        multi_scale: bool = True
        ms_min: int = 320
        ms_max: int = 608
        ms_step: int = 32
        ms_interval: int = 10      # 每多少个 batch 切一次尺寸

        anchors_json: Optional[str]
        auto_compute_anchors: bool = True
        anchors_k: int = 5
        anchors_seed: int = 0

        normalize_to_01: bool = True
        pad_value: int = 114
        bbox_min_area: float = 1.0
        bbox_min_visibility: float = 0.10
        jitter_scale_limit: float = 0.20

    输出：
        train_loader, val_loader, extra
            extra["anchors_rel"]
            extra["ms_sizes"]
            extra["batch_sampler"]（方便你在训练 loop 每个 epoch 调一次 set_epoch）
    """
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 8))
    persistent_workers = bool(cfg.get("persistent_workers", True))
    stride = int(cfg.get("stride", 32))

    classes = cfg.get("classes", VOC_CLASSES)
    seen_classes = [str(x) for x in classes]
    class_to_id = {n: i for i, n in enumerate(seen_classes)}

    anchors_rel = _resolve_anchors_rel(cfg, class_to_id)

    # -------------------------
    # multi-scale sizes（train 用）
    # -------------------------
    multi_scale = bool(cfg.get("multi_scale", True))
    if multi_scale:
        ms_sizes = build_multiscale_sizes(
            ms_min=int(cfg.get("ms_min", 320)),
            ms_max=int(cfg.get("ms_max", 608)),
            ms_step=int(cfg.get("ms_step", 32)),
        )
        ms_interval = int(cfg.get("ms_interval", 10))
    else:
        ms_sizes = [int(cfg.get("input_size", 416))]
        ms_interval = 10

    # -------------------------
    # 构建 train dataset（router）
    # -------------------------
    train_router = MultiScaleDatasetRouter(
        base_path=cfg["train_path"],
        anchors_rel=anchors_rel,
        sizes=ms_sizes,
        stride=stride,
        classes=seen_classes,
        flatten_targets=bool(cfg.get("flatten_targets", False)),
        min_box_size_px=float(cfg.get("min_box_size_px", 0.0)),
        normalize_to_01=bool(cfg.get("normalize_to_01", True)),
        pad_value=int(cfg.get("pad_value", 114)),
        bbox_min_area=float(cfg.get("bbox_min_area", 1.0)),
        bbox_min_visibility=float(cfg.get("bbox_min_visibility", 0.10)),
        jitter_scale_limit=float(cfg.get("jitter_scale_limit", 0.20)),
    )

    # -------------------------
    # debug mode：只采样 train 的一部分索引
    # 注意：val 仍使用全量（你也可以按需缩小）
    # -------------------------
    base_indices = list(range(len(train_router)))
    debug_mode = cfg.get("debug_mode", None)
    if debug_mode is not None:
        ratio = float(debug_mode)
        take = max(1, int(len(base_indices) * ratio))
        g = torch.Generator()
        g.manual_seed(int(cfg.get("debug_seed", 0)))
        perm = torch.randperm(len(base_indices), generator=g).tolist()
        base_indices = [base_indices[i] for i in perm[:take]]
        print("⚠️ debug mode : training subset len:", len(base_indices))

    # -------------------------
    # train batch sampler（multi-scale 由这里控制）
    # -------------------------
    train_batch_sampler = MultiScaleBatchSampler(
        indices=base_indices,
        batch_size=batch_size,
        sizes=ms_sizes,
        interval_batches=int(ms_interval),
        shuffle=True,
        drop_last=bool(cfg.get("drop_last", False)),
        seed=int(cfg.get("sampler_seed", 0)),
    )

    train_loader = DataLoader(
        train_router,
        batch_sampler=train_batch_sampler,   # 关键：不要再传 batch_size/shuffle
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
    )

    # -------------------------
    # val：固定尺寸（通常用 416）
    # -------------------------
    val_size = int(cfg.get("val_size", cfg.get("input_size", 416)))
    _, val_tf = build_yolov2_transforms(
        img_size=val_size,
        pad_value=int(cfg.get("pad_value", 114)),
        bbox_min_area=float(cfg.get("bbox_min_area", 1.0)),
        bbox_min_visibility=float(cfg.get("bbox_min_visibility", 0.10)),
        jitter_scale_limit=float(cfg.get("jitter_scale_limit", 0.20)),
        rotate_limit_deg=0.0,
        rotate_p=0.0,
        normalize_to_01=bool(cfg.get("normalize_to_01", True)),
    )

    val_dataset = VOCDataset(
        base_path=cfg["test_path"],
        transform=val_tf,
        anchors_rel=anchors_rel,
        img_size=val_size,
        stride=stride,
        S=None,
        classes=seen_classes,
        flatten_targets=bool(cfg.get("flatten_targets", False)),
        min_box_size_px=float(cfg.get("min_box_size_px", 0.0)),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
    )

    extra = {
        "anchors_rel": anchors_rel,
        "ms_sizes": ms_sizes,
        "batch_sampler": train_batch_sampler,  # 训练 loop 每个 epoch 调 set_epoch(epoch)
        "val_size": val_size,
    }

    print("train len:", len(train_router), "| val len:", len(val_dataset))
    print("multi_scale:", multi_scale, "| sizes:", ms_sizes, "| interval:", ms_interval)
    print("anchors_rel:", anchors_rel)

    return train_loader, val_loader, extra
