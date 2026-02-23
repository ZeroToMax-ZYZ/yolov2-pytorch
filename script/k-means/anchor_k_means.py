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

def kmeans_anchors_iou_torch(
    wh: torch.Tensor,
    k: int,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    功能:
        用 YOLOv2 论文思路做 dimension clusters（全 torch 实现）:
            distance = 1 - IoU(wh, centroid)

    输入:
        wh:
            (N,2)，归一化之后的相对宽高 (w_rel, h_rel)，范围 (0,1]
        k:
            anchor 数（VOC 常用 5）
        seed:
            固定随机种子保证可复现
        max_iter:
            最大迭代次数
        tol:
            收敛阈值：loss 或 centroid 变化足够小则停止

    输出:
        centroids:
            (k,2) torch.float32，按面积从小到大排序
    """
    assert wh.ndim == 2 and wh.shape[1] == 2
    assert k > 0 and wh.shape[0] >= k
    # (N,2)
    wh = wh.to(dtype=torch.float32, device="cpu").contiguous()

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    # 随机初始化：从样本中抽 k 个点
    perm = torch.randperm(wh.shape[0], generator=g)
    centroids = wh[perm[:k]].clone()  # (k,2)

    last_loss: Optional[float] = None

    for _ in range(int(max_iter)):
        # wh        :全部的label的wh信息   -->(N,2) ([40058, 2])
        # centroids :提取出来的初始聚类中心 -->(K,2) ([5, 2])
        # iou       :计算出来N个wh与K个anchor的iou，输出为[N,K] ([40058, 5])
        ious = iou_wh_torch(wh, centroids, eps=eps)
        # 论文公式
        dist = 1.0 - ious
        # # 找到每个样本距离最近的簇中心 (即 IoU 最大的 anchor)
        # assign为1-ious最小的wh对应的位置索引(也就是wh与anchor的iou最大的位置索引排序)
        assign = torch.argmin(dist, dim=1)  # (N,) ，值为 0 ~ k-1
        # ic(assign.shape)
        # 全部的行，1-iou最小的那一列
        # 也就是提取每个样本对应的最小距离，并求平均值作为当前的 Loss
        loss = float(dist[torch.arange(wh.shape[0]), assign].mean().item())
        # 退出条件：上一次的loss和这一次的loss的差值小于阈值
        if last_loss is not None and abs(last_loss - loss) < tol:
            break
        # 刷新loss
        last_loss = loss
        # new_centroids 是第 t+1 次迭代的结果
        new_centroids = centroids.clone()
        for j in range(k):
            # 找到属于第 j 个簇的所有样本的掩码
            mask = (assign == j)
            if not bool(mask.any().item()):
                # 如果某个初始中心运气太差，没有一个样本离它最近
                # 策略: 重新在数据集中随机选一个点代替它，强行把这一簇救活
                ridx = int(torch.randint(low=0, high=wh.shape[0], size=(1,), generator=g).item())
                new_centroids[j] = wh[ridx]   
            else:
                # 【关键差异】这里使用了 Median (中位数) 而不是 Mean (均值)
                # 标准 K-Means 使用 Mean。但 bbox 尺寸往往有长尾分布（极大的框），
                # Mean 容易受离群点影响。Median 更稳健，得到的 Anchor 更具代表性。
                new_centroids[j] = wh[mask].median(dim=0).values

        # centroid 收敛判定
        # # 如果新旧中心点的坐标变化极其微小，也视为收敛
        if float(torch.max(torch.abs(new_centroids - centroids)).item()) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # 按面积从小到大排序，便于稳定输出
    areas = centroids[:, 0] * centroids[:, 1]
    order = torch.argsort(areas)
    centroids = centroids[order]
    return centroids


def save_anchors_json(
    json_path: str,
    anchors_rel: List[Tuple[float, float]],
    *,
    k: int,
    seed: int,
    num_boxes: int,
    mean_iou: float,
) -> None:
    """
    功能:
        保存 anchors 到 json，便于复现实验与复用
    """
    payload = {
        "k": int(k),
        "seed": int(seed),
        "num_boxes": int(num_boxes),
        "mean_iou": float(mean_iou),
        "anchors_rel": [(float(w), float(h)) for (w, h) in anchors_rel],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def load_anchors_json(json_path: str) -> List[Tuple[float, float]]:
    """
    功能:
        从 json 读取 anchors_rel
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data.get("anchors_rel", [])
    return [(float(a[0]), float(a[1])) for a in anchors]


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


def mean_iou_best_anchor(wh: torch.Tensor, anchors: torch.Tensor) -> float:
    """
    功能:
        计算每个 bbox 与 best anchor 的 IoU 的平均值（常用于衡量 anchors 质量）
    """
    wh = wh.to(dtype=torch.float32, device="cpu").contiguous()
    anchors = anchors.to(dtype=torch.float32, device="cpu").contiguous()
    ious = iou_wh_torch(wh, anchors)  # (N,K)
    best = torch.max(ious, dim=1).values
    return float(best.mean().item())


def compute_anchors_from_voc_csvs(
    base_path: str,                 # 数据集根目录
    class_to_id: Dict[str, int],    # 类别映射（主要用于 read_voc_csv 过滤非法类别）
    img_dir_name: str = "images",   # 图片文件夹名
    target_dir_name: str = "targets", # 标注文件夹名
    k: int = 5,                     # 聚类簇数（VOC标准是5）
    seed: int = 0,                  # 随机种子
    out_json: Optional[str] = None, # 结果保存路径
    min_box_size_px: float = 0.0,   # 过滤极小框的阈值
) -> List[Tuple[float, float]]:
    """
    功能:
        遍历 VOC 风格 targets/*.csv，收集所有 bbox 的 (w_rel,h_rel)，做 k-means(1-IoU)，得到 anchors_rel。
        注意：这里以“原图尺度”统计 w_rel/h_rel，不使用随机增强后的框（避免 anchors 漂移）。

    输入:
        base_path:
            数据集根目录，内部包含 images/ 与 targets/
        class_to_id:
            类别名->id
        k:
            anchor 个数（VOC 论文常用 5）
        seed:
            随机种子（用于初始化聚类）
        out_json:
            若提供则保存 JSON
        min_box_size_px:
            可选：过滤极小框（论文未给阈值；默认 0 不额外过滤，仅几何合法性过滤）

    输出:
        anchors_rel:
            List[(w_rel, h_rel)]，相对原图归一化
    """
    # 拼接完整的文件夹路径
    img_dir = os.path.join(base_path, img_dir_name)
    target_dir = os.path.join(base_path, target_dir_name)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Target dir not found: {target_dir}")

    exts = {".jpg", ".jpeg", ".png"}
    names = sorted(os.listdir(img_dir))
    # 初始化一个空列表，用来装所有的 (w_rel, h_rel) 数据
    wh_list: List[Tuple[float, float]] = []

    for fn in names:
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue

        img_path = os.path.join(img_dir, fn)
        csv_path = os.path.join(target_dir, stem + ".csv")
        if not os.path.exists(csv_path):
            continue

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        if w0 <= 0 or h0 <= 0:
            continue

        bboxes, _ = read_voc_csv(csv_path, class_to_id)
        for b in bboxes:
            # 处理所有的bbox
            x1, y1, x2, y2 = b
            bw = x2 - x1
            bh = y2 - y1

            # 合法性兜底
            if bw <= 0.0 or bh <= 0.0:
                continue
            # 可选：极小框过滤（默认关闭）
            if min_box_size_px > 0.0 and (bw < min_box_size_px or bh < min_box_size_px):
                continue
            # wh是 归一化的
            w_rel = float(bw) / float(w0)
            h_rel = float(bh) / float(h0)

            if w_rel <= 0.0 or h_rel <= 0.0:
                continue
            wh_list.append((w_rel, h_rel))

    if len(wh_list) < k:
        raise RuntimeError(f"not enough boxes for k={k}, got {len(wh_list)}")

    wh = torch.tensor(wh_list, dtype=torch.float32)  # (N,2)
    # 此时anchors是经过聚类之后得到的， k,2, 归一化的
    anchors = kmeans_anchors_iou_torch(wh, k=k, seed=seed)  # (k,2)

    # 将结果转回 Python 列表，方便阅读和 JSON 序列化
    anchors_rel = [(float(a[0].item()), float(a[1].item())) for a in anchors]
    miou = mean_iou_best_anchor(wh, anchors)

    if out_json is not None:
        save_anchors_json(
            json_path=out_json,
            anchors_rel=anchors_rel,
            k=k,
            seed=seed,
            num_boxes=int(wh.shape[0]),
            mean_iou=miou,
        )

    return anchors_rel


# ============================================================
# 5) 【新增】可选：脚本模式下计算 anchors 并保存（不使用 argparse）
# ============================================================
if __name__ == "__main__":
    # 你可以把这里当作“离线聚类脚本”的最小入口
    # 按需修改 base_path，并确保 images/ 与 targets/ 存在
    VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
    ]

    base_path = r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train"  # 修改为你的训练集路径
    classes = VOC_CLASSES
    class_to_id = {n: i for i, n in enumerate(classes)}
    out_json = os.path.join(r"dataset", "anchors_k5.json")
    # out_json = os.path.join("anchors_k5.json")
    # anchors_rel = compute_anchors_from_voc_csvs(
    #         base_path=base_path,
    #         class_to_id=class_to_id,
    #         k=5,
    #         seed=0,
    #         out_json=out_json,
    #         min_box_size_px=0.0,
    #     )
    if not os.path.exists(out_json):
        anchors_rel = compute_anchors_from_voc_csvs(
            base_path=base_path,
            class_to_id=class_to_id,
            k=5,
            seed=0,
            out_json=out_json,
            min_box_size_px=0.0,
        )
        print("anchors_rel saved:", anchors_rel)
    else:
        anchors_rel = load_anchors_json(out_json)
        print("anchors_rel loaded:", anchors_rel)

    # 打印在 416 尺度下的像素 anchors
    # img_size = 416
    # anchors_px = [(a[0] * img_size, a[1] * img_size) for a in anchors_rel]
    # print("anchors_px@416:", anchors_px)








