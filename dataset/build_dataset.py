# dataset/build_dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import os
import math
import json

import torch
from torch.utils.data import Dataset, DataLoader

from dataset.augment import build_yolov2_transforms
from dataset.VOC_dataset import VOCDataset  # 你按自己的实际路径修改


# ============================================================
# 1) 生成 multi-scale 候选尺寸
# ============================================================
def build_multiscale_sizes(ms_min: int = 320, ms_max: int = 608, ms_step: int = 32) -> List[int]:
    """
    作用：
        生成 YOLOv2 multi-scale 的候选输入边长列表：
            [320, 352, 384, ..., 608]
    """
    return list(range(int(ms_min), int(ms_max) + 1, int(ms_step)))


# ============================================================
# 2) 生成 “batch -> size” 的计划表（schedule）
# ============================================================
def generate_size_schedule(
    num_batches: int,
    sizes: List[int],
    interval_batches: int = 10,
    *,
    seed: int = 0,
    epoch: int = 0,
) -> torch.Tensor:
    """
    作用：
        生成 schedule：长度 = num_batches
        schedule[b] = 第 b 个 batch 使用的输入尺寸（img_size）

    关键：
        - 每 interval_batches（默认 10）个 batch 换一次随机尺寸
        - 这就是 YOLOv2 论文里的 multi-scale training 机制

    可复现性：
        - 使用 seed + epoch 生成固定随机序列
        - 同一个 epoch 的 schedule 固定，方便你复现实验
    """
    num_batches = int(num_batches)
    interval_batches = int(interval_batches)

    # 1) 计算要采样多少次 size（每 interval_batches 个 batch 采样一次）
    # math.ceil 对浮点数向上取整
    num_groups = int(math.ceil(num_batches / float(interval_batches)))

    # 2) 用固定的随机源，保证复现
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(epoch) * 1000003)

    sizes_t = torch.tensor(sizes, dtype=torch.int64)

    # 3) 先为每个 group 抽一个 size
    group_idx = torch.randint(low=0, high=sizes_t.numel(), size=(num_groups,), generator=g)
    group_sizes = sizes_t[group_idx]  # (num_groups,)

    # 4) 把 group_sizes 展开到 batch 级别
    schedule = group_sizes.repeat_interleave(interval_batches)[:num_batches].contiguous()
    return schedule  # int64, (num_batches,)


# ============================================================
# 3) Router Dataset：支持 __getitem__((idx, size))
# ============================================================
class MultiScaleDatasetRouter(Dataset):
    """
    作用：
        让你的 DataLoader 可以喂给 Dataset 一个 (idx, size)：
            router[(idx, size)] -> ds_map[size][idx]

    为什么要 Router？
        - 你的 VOCDataset 通常是“固定 img_size”的（初始化时传 img_size）
        - multi-scale 训练要求 “一个 batch 里所有样本用同一个 img_size”
        - 所以我们为每个 size 创建一个 dataset 实例，然后按 size 路由即可

    你要记住的一个关键点：
        - ds_map 中不同 size 的 dataset 扫描样本顺序必须一致
        - 你只要在 VOCDataset._collect_samples() 里按文件名排序，通常就天然一致
    """

    def __init__(
        self,
        base_path: str,
        anchors_rel: List[Tuple[float, float]],
        sizes: List[int],
        *,
        stride: int = 32,
        classes: Optional[List[str]] = None,
        normalize_to_01: bool = True,
        pad_value: int = 114,
        bbox_min_area: float = 1.0,
        bbox_min_visibility: float = 0.10,
        jitter_scale_limit: float = 0.20,
        flatten_targets: bool = False,
        min_box_size_px: float = 0.0,
    ) -> None:
        super().__init__()

        # 候选 size 列表
        self.sizes = list(map(int, sizes))
        self.default_size = int(self.sizes[len(self.sizes) // 2])

        # 为每个 size 创建一个“固定 img_size”的 dataset
        ''' 
        ds_map = {
            sz1: VOCDataset(),
            sz2: VOCDataset(),
        }
        '''
        self.ds_map: Dict[int, Dataset] = {}
        for sz in self.sizes:
            train_tf, _ = build_yolov2_transforms(
                img_size=int(sz),
                pad_value=int(pad_value),
                bbox_min_area=float(bbox_min_area),
                bbox_min_visibility=float(bbox_min_visibility),
                jitter_scale_limit=float(jitter_scale_limit),
                normalize_to_01=bool(normalize_to_01),
            )

            ds = VOCDataset(
                base_path=base_path,
                transform=train_tf,
                anchors_rel=anchors_rel,
                img_size=int(sz),
                stride=int(stride),
                classes=classes,
                flatten_targets=bool(flatten_targets),
                min_box_size_px=float(min_box_size_px),
            )

            self.ds_map[int(sz)] = ds

        # Router 的长度 = 任意一个 ds 的长度（它们应该一样）
        self._len = len(next(iter(self.ds_map.values())))

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, item: Any):
        # DataLoader 的 batch_sampler 会喂给我们 (idx, size)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            idx = int(item[0])
            sz = int(item[1])
        else:
            # 如果你手动 router[idx]，则走 default_size
            idx = int(item)
            sz = int(self.default_size)

        return self.ds_map[sz][idx]


# ============================================================
# 4) BatchSampler：把 schedule[b] 的 size 绑定到第 b 个 batch
# ============================================================
@dataclass
class ScheduleConfig:
    """
    用 dataclass 管理 schedule 参数，便于你在 cfg 里配置
    """
    sizes: List[int]
    interval_batches: int = 10
    seed: int = 0


class ScheduledMultiScaleBatchSampler:
    """
    作用：
        给 DataLoader 提供 batch 级别的采样逻辑：
        - 每个 epoch 预生成 schedule（batch->size）
        - 每次迭代返回一个 batch：
              [(idx, size), (idx, size), ...]
          这样 Dataset 就能按 size 做正确的 resize/encode

    学习重点：
        - 你需要理解：DataLoader 允许你传 batch_sampler
        - 传了 batch_sampler 后，DataLoader 就不再关心 batch_size/shuffle
        - batch_sampler 控制 “怎么把 index 组成 batch”
    """

    def __init__(
        self,
        indices: List[int],
        batch_size: int,
        schedule_cfg: ScheduleConfig,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.indices = list(map(int, indices))
        self.batch_size = int(batch_size)
        self.cfg = schedule_cfg
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.epoch = 0
        self.schedule: Optional[torch.Tensor] = None

    def set_epoch(self, epoch: int) -> None:
        """
        你每个 epoch 开始前调用一次：
            sampler.set_epoch(epoch)

        它会生成本 epoch 的 schedule（固定下来，方便复现）
        """
        self.epoch = int(epoch)

        # 计算本 epoch 有多少个 batch
        n = len(self.indices)
        if self.drop_last:
            num_batches = n // self.batch_size
        else:
            num_batches = int(math.ceil(n / float(self.batch_size)))

        # 生成 batch->size 计划表
        self.schedule = generate_size_schedule(
            num_batches=num_batches,
            sizes=self.cfg.sizes,
            interval_batches=self.cfg.interval_batches,
            seed=self.cfg.seed,
            epoch=self.epoch,
        )

    def __len__(self) -> int:
        n = len(self.indices)
        if self.drop_last:
            return n // self.batch_size
        return int(math.ceil(n / float(self.batch_size)))

    def __iter__(self):
        """
        注意：
            这里按你的要求，不使用 yield。
            我们直接构造 all_batches，然后 return iter(all_batches)

        输出：
            Iterator[ List[(idx,size), ...] ]
        """
        # 如果用户忘记 set_epoch，就默认用当前 epoch=0
        if self.schedule is None:
            self.set_epoch(self.epoch)

        schedule = self.schedule  # (num_batches,)

        # 1) epoch 内 shuffle（用 seed+epoch 可复现）
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.cfg.seed) + int(self.epoch) * 1000003)

        if self.shuffle:
            perm = torch.randperm(len(self.indices), generator=g).tolist()
            idxs = [self.indices[i] for i in perm]
        else:
            idxs = list(self.indices)

        # 2) 切 batch（batch 内只是 idx）
        batches_idx: List[List[int]] = []
        for i in range(0, len(idxs), self.batch_size):
            b = idxs[i:i + self.batch_size]
            if self.drop_last and len(b) < self.batch_size:
                continue
            batches_idx.append(b)

        # 3) 把 size 绑定到 batch（一个 batch 用一个 size）
        num_batches = min(len(batches_idx), int(schedule.numel()))
        all_batches: List[List[Tuple[int, int]]] = []

        for bi in range(num_batches):
            size = int(schedule[bi].item())
            all_batches.append([(int(ii), size) for ii in batches_idx[bi]])
        # 此时组合出来的all_batches是一个list，里面的每个元素都是一个子list，内容为(idx,size)
        return iter(all_batches)


# ============================================================
# 5) anchors 读取（学习版：只从 json 读）
# ============================================================
def load_anchors_json(json_path: str) -> List[Tuple[float, float]]:
    """
    作用：
        从 json 读取 anchors_rel
    json 结构示例：
        {"anchors_rel": [[w_rel,h_rel], ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data["anchors_rel"]
    return [(float(a[0]), float(a[1])) for a in anchors]


# ============================================================
# 6) build_dataset(cfg)：组装 DataLoader
# ============================================================
def build_dataset(cfg: Dict[str, Any]):
    """
    你会在训练脚本里这样用：
        train_loader, val_loader, extra = build_dataset(cfg)

    训练时每个 epoch 开始前必须做：
        extra["train_batch_sampler"].set_epoch(epoch)

    这样 multi-scale 才会生效。
    """
    VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
    ]
    # -------------------------
    # (1) 基本配置
    # -------------------------
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 8))
    persistent_workers = bool(cfg.get("persistent_workers", True))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))

    classes = VOC_CLASSES               # 例如 VOC_CLASSES
    stride = int(cfg.get("stride", 32))      # YOLOv2 backbone 下采样 32 倍

    # -------------------------
    # (2) anchors
    # -------------------------
    anchors_rel = load_anchors_json(cfg["anchors_json"])

    # -------------------------
    # (3) multi-scale sizes + schedule cfg
    # -------------------------
    sizes = build_multiscale_sizes(
        ms_min=int(cfg.get("ms_min", 320)),
        ms_max=int(cfg.get("ms_max", 608)),
        ms_step=int(cfg.get("ms_step", 32)),
    )

    schedule_cfg = ScheduleConfig(
        sizes=sizes,
        interval_batches=int(cfg.get("ms_interval", 10)),  # YOLOv2：每 10 个 batch 换一次
        seed=int(cfg.get("schedule_seed", 0)),
    )

    # -------------------------
    # (4) train dataset：Router（按 size 路由）
    # -------------------------
    # 调用的时候，给size和idx，返回对应的样本
    train_ds = MultiScaleDatasetRouter(
        base_path=cfg["train_path"],
        anchors_rel=anchors_rel,
        sizes=sizes,
        stride=stride,
        classes=classes,
        normalize_to_01=bool(cfg.get("normalize_to_01", True)),
        pad_value=int(cfg.get("pad_value", 114)),
        bbox_min_area=float(cfg.get("bbox_min_area", 1.0)),
        bbox_min_visibility=float(cfg.get("bbox_min_visibility", 0.10)),
        jitter_scale_limit=float(cfg.get("jitter_scale_limit", 0.20)),
        flatten_targets=bool(cfg.get("flatten_targets", False)),
        min_box_size_px=float(cfg.get("min_box_size_px", 0.0)),
    )

    # -------------------------
    # (5) train indices（学习版：直接全量）
    # -------------------------
    train_indices = list(range(len(train_ds)))

    # -------------------------
    # (6) train batch sampler：schedule 绑定 size
    # -------------------------
    train_batch_sampler = ScheduledMultiScaleBatchSampler(
        indices=train_indices,
        batch_size=batch_size,
        schedule_cfg=schedule_cfg,
        shuffle=True,
        drop_last=bool(cfg.get("drop_last", False)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,  # 注意：用了 batch_sampler 就不要传 batch_size/shuffle
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # -------------------------
    # (7) val dataset：固定尺寸（通常 416）
    # -------------------------
    val_size = int(cfg.get("val_size", 416))
    _, val_tf = build_yolov2_transforms(
        img_size=val_size,
        pad_value=int(cfg.get("pad_value", 114)),
        jitter_scale_limit=0.0,  # val 不做 jitter
        normalize_to_01=bool(cfg.get("normalize_to_01", True)),
    )

    val_ds = VOCDataset(
        base_path=cfg["test_path"],
        transform=val_tf,
        anchors_rel=anchors_rel,
        img_size=val_size,
        stride=stride,
        classes=classes,
        flatten_targets=bool(cfg.get("flatten_targets", False)),
        min_box_size_px=float(cfg.get("min_box_size_px", 0.0)),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    extra = {
        "anchors_rel": anchors_rel,
        "sizes": sizes,
        "train_batch_sampler": train_batch_sampler,  # 训练每 epoch 调 set_epoch
        "val_size": val_size,
    }
    
    return train_loader, val_loader, extra
