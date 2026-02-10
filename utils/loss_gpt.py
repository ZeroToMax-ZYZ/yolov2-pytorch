# -*- coding: utf-8 -*-
"""
Yolov2Loss（对齐你图片风格的 YOLOv2 损失函数：noobj + prior + truth）
=================================================================

入口:
    loss_fn = Yolov2Loss(cfg, ic_debug=False)
    total_loss, items = loss_fn(pred_tensor, gt_tensor, epoch)

输入:
    pred_tensor: (bs, S, S, A, 5 + C)
        5+C = [t_x, t_y, t_w, t_h, t_o, cls_logits...]
        注意：t_o 是 logit，内部会做 sigmoid 得到 conf
    gt_tensor: (bs, S, S, A, 5 + C)
        5+C = [x, y, tw*, th*, obj, cls_onehot...]
        - x,y: cell 内 offset ∈ (0,1)
        - tw*,th*: ln(w_rel/p_w_rel), ln(h_rel/p_h_rel)
        - obj: 1 表示该 (i,j,k) 是负责该 GT 的 anchor，否则 0
        - cls: one-hot

输出:
    total_loss: 标量 tensor
    items: dict[str, tensor]，包含各分量（已做归一化时也是标量）
"""

import json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic


def load_anchors_json(json_path: str) -> List[Tuple[float, float]]:
    """
    作用：
        从 json 读取 anchors_rel

    json 结构示例：
        {"anchors_rel": [[w_rel,h_rel], ...]}

    返回：
        List[(w_rel, h_rel)]，长度 = A
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data["anchors_rel"]
    return [(float(a[0]), float(a[1])) for a in anchors]


def _build_grid_xy(
    S: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建网格坐标（grid_x, grid_y），用于把 cell 内 offset 转成全图归一化坐标

    返回：
        grid_x: (1, S, S, 1, 1)  列坐标 j
        grid_y: (1, S, S, 1, 1)  行坐标 i
    """
    ys = torch.arange(S, device=device, dtype=dtype)
    xs = torch.arange(S, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (S,S)
    grid_x = grid_x.view(1, S, S, 1, 1)
    grid_y = grid_y.view(1, S, S, 1, 1)
    return grid_x, grid_y


def _decode_xywh_to_xyxy(
    tx: torch.Tensor,
    ty: torch.Tensor,
    tw: torch.Tensor,
    th: torch.Tensor,
    anchors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按 YOLOv2 参数化解码 pred/gt 的 (x,y,w,h)，并输出 xyxy（归一化到 [0,1] 尺度）

    输入:
        tx,ty,tw,th: (bs, S, S, A, 1)
            - 如果是 pred：tx,ty 是网络输出（需要 sigmoid），tw,th 走 exp
            - 如果是 gt：
                x,y 已经是 cell 内 offset（不需要 sigmoid），但为了复用，本函数对 tx,ty 默认当作“需要 sigmoid 的 t 空间”
                因此 gt 的解码建议走专用函数 _decode_gt_to_xyxy

        anchors: (A, 2)  anchors_rel，w/h 是相对整图归一化

    返回:
        xyxy: (bs, S, S, A, 4)
        wh  : (bs, S, S, A, 2)  (w,h) 归一化尺度
    """
    bs, S, _, A, _ = tx.shape
    device, dtype = tx.device, tx.dtype
    grid_x, grid_y = _build_grid_xy(S, device=device, dtype=dtype)

    # anchors: (A,2) -> (1,1,1,A,1)
    anchor_w = anchors[:, 0].view(1, 1, 1, A, 1).to(device=device, dtype=dtype)
    anchor_h = anchors[:, 1].view(1, 1, 1, A, 1).to(device=device, dtype=dtype)

    bx = (torch.sigmoid(tx) + grid_x) / float(S)
    by = (torch.sigmoid(ty) + grid_y) / float(S)
    bw = torch.exp(tw) * anchor_w
    bh = torch.exp(th) * anchor_h

    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    x2 = bx + bw * 0.5
    y2 = by + bh * 0.5

    xyxy = torch.cat([x1, y1, x2, y2], dim=-1)
    wh = torch.cat([bw, bh], dim=-1)
    return xyxy, wh


def _decode_gt_to_xyxy(
    gt_x: torch.Tensor,
    gt_y: torch.Tensor,
    gt_tw: torch.Tensor,
    gt_th: torch.Tensor,
    anchors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    专门解码 gt（因为 gt_x/gt_y 已经是 cell 内 offset，不需要 sigmoid）

    输入:
        gt_x,gt_y,gt_tw,gt_th: (bs,S,S,A,1)
        anchors: (A,2)

    返回:
        gt_xyxy: (bs,S,S,A,4) 归一化到 [0,1]
        gt_wh  : (bs,S,S,A,2)
    """
    bs, S, _, A, _ = gt_x.shape
    device, dtype = gt_x.device, gt_x.dtype
    grid_x, grid_y = _build_grid_xy(S, device=device, dtype=dtype)

    anchor_w = anchors[:, 0].view(1, 1, 1, A, 1).to(device=device, dtype=dtype)
    anchor_h = anchors[:, 1].view(1, 1, 1, A, 1).to(device=device, dtype=dtype)

    bx = (gt_x + grid_x) / float(S)
    by = (gt_y + grid_y) / float(S)
    bw = torch.exp(gt_tw) * anchor_w
    bh = torch.exp(gt_th) * anchor_h

    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    x2 = bx + bw * 0.5
    y2 = by + bh * 0.5

    gt_xyxy = torch.cat([x1, y1, x2, y2], dim=-1)
    gt_wh = torch.cat([bw, bh], dim=-1)
    return gt_xyxy, gt_wh


def _bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    计算 IoU（xyxy 格式）

    输入:
        a: (..., 4)
        b: (..., 4)  与 a 需要可 broadcast

    输出:
        iou: broadcast 后的 (...,)
    """
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h

    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = area_a + area_b - inter + eps
    return inter / union


class Yolov2Loss(nn.Module):
    def __init__(self, cfg: Dict, ic_debug: bool = False):
        """
        cfg 建议包含：
            - num_classes: int
            - anchors_path: str
            - device: torch.device
            - loss_dtype: torch.dtype

            - lambda_noobj: float
            - lambda_prior: float
            - lambda_coord: float
            - lambda_obj: float
            - lambda_class: float

            - ignore_thresh: float   # 例如 0.6
            - warmup_epochs: int     # 例如 12（你现在的 prior_loss）
            - prior_on_bg_only: bool # 默认 True

            - normalize: bool        # 默认 True：按正样本数/负样本数归一化
            - cls_loss: str          # "mse" or "ce"，默认 "mse"（对齐你图）
        """
        super().__init__()
        self.cfg = cfg
        self.ic_debug = ic_debug

        self.num_classes = int(cfg["num_classes"])
        anchors = load_anchors_json(cfg["anchors_path"])
        self.register_buffer(
            "anchors",
            torch.tensor(anchors, dtype=cfg["loss_dtype"], device=cfg["device"]),
            persistent=False,
        )

        self.ignore_thresh = float(cfg.get("ignore_thresh", 0.6))
        self.warmup_epochs = int(cfg.get("warmup_epochs", 0))
        self.prior_on_bg_only = bool(cfg.get("prior_on_bg_only", True))

        self.normalize = bool(cfg.get("normalize", True))
        self.cls_loss = str(cfg.get("cls_loss", "mse")).lower()

        self.lambda_noobj = float(cfg.get("lambda_noobj", 0.5))
        self.lambda_prior = float(cfg.get("lambda_prior", 0.01))
        self.lambda_coord = float(cfg.get("lambda_coord", 5.0))
        self.lambda_obj = float(cfg.get("lambda_obj", 1.0))
        self.lambda_class = float(cfg.get("lambda_class", 1.0))

        self.eps = 1e-9

    @torch.no_grad()
    def _max_iou_to_any_gt_per_image(
        self,
        pred_xyxy: torch.Tensor,
        gt_xyxy: torch.Tensor,
        gt_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个预测框与该图片所有 GT 的最大 IoU（用于 noobj ignore）

        输入:
            pred_xyxy: (bs, S, S, A, 4)
            gt_xyxy  : (bs, S, S, A, 4)
            gt_obj   : (bs, S, S, A, 1)

        输出:
            max_iou: (bs, S, S, A, 1)
        """
        bs = pred_xyxy.shape[0]
        S = pred_xyxy.shape[1]
        A = pred_xyxy.shape[3]
        Npred = S * S * A

        pred_flat = pred_xyxy.view(bs, Npred, 4)
        gt_flat = gt_xyxy.view(bs, Npred, 4)
        gt_obj_flat = gt_obj.view(bs, Npred)

        max_iou_all = pred_flat.new_zeros((bs, Npred))

        for n in range(bs):
            pos_idx = torch.nonzero(gt_obj_flat[n] > 0.5, as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                # 当前图片没有 GT，则 max_iou 全 0
                continue

            g = gt_flat[n, pos_idx]  # (Ng,4)
            p = pred_flat[n]         # (Npred,4)

            # iou_matrix: (Npred, Ng)
            iou_matrix = _bbox_iou_xyxy(p[:, None, :], g[None, :, :])
            max_iou_all[n] = iou_matrix.max(dim=1).values

        return max_iou_all.view(bs, S, S, A, 1)

    def forward(
        self,
        pred_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        输入:
            pred_tensor: (bs,S,S,A,5+C)
            gt_tensor  : (bs,S,S,A,5+C)

        输出:
            total_loss: 标量
            items: 各分量（标量）
        """
        cfg = self.cfg
        device = pred_tensor.device
        dtype = pred_tensor.dtype

        # -------- 1) 拆分 pred / gt --------
        tx = pred_tensor[..., 0:1]
        ty = pred_tensor[..., 1:2]
        tw = pred_tensor[..., 2:3]
        th = pred_tensor[..., 3:4]
        to = pred_tensor[..., 4:5]        # logit
        cls_logits = pred_tensor[..., 5:] # (bs,S,S,A,C)

        gt_x = gt_tensor[..., 0:1]
        gt_y = gt_tensor[..., 1:2]
        gt_tw = gt_tensor[..., 2:3]
        gt_th = gt_tensor[..., 3:4]
        gt_obj = gt_tensor[..., 4:5]      # 0/1
        gt_cls = gt_tensor[..., 5:]       # one-hot

        # anchor buffer 对齐 device/dtype（防止你之后把模型搬到 cuda）
        anchors = self.anchors.to(device=device, dtype=dtype)

        # 正样本 mask（对齐图里的 1^truth）
        pos_mask = (gt_obj > 0.5)  # (bs,S,S,A,1) bool
        bg_mask = ~pos_mask        # 背景

        # -------- 2) 解码得到 pred/gt box（用于 IoU 和 noobj ignore）--------
        pred_xyxy, _ = _decode_xywh_to_xyxy(tx, ty, tw, th, anchors)
        gt_xyxy, _ = _decode_gt_to_xyxy(gt_x, gt_y, gt_tw, gt_th, anchors)

        # pred conf
        pred_conf = torch.sigmoid(to)  # (bs,S,S,A,1)

        # -------- 3) part1: noobj loss（对齐图片：1_{maxIoU<thresh} * lambda_noobj * (0-conf)^2）--------
        # max_iou: (bs,S,S,A,1) 仅用于决定 noobj 是否忽略
        max_iou = self._max_iou_to_any_gt_per_image(pred_xyxy, gt_xyxy, gt_obj)
        ignore_mask = (max_iou >= self.ignore_thresh)  # 与任意 GT 很像 -> 忽略 noobj
        noobj_mask = bg_mask & (~ignore_mask)          # 真正要当背景惩罚的框

        loss_noobj_map = self.lambda_noobj * noobj_mask.to(dtype) * (pred_conf ** 2)

        # -------- 4) part2: prior(warmup) loss（对齐图片：1_{t<warmup} * lambda_prior * sum(prior-b)^2）--------
        # 这里用 t 空间 prior=0 的写法（更贴近你当前代码注释）
        # tx=0,ty=0 -> sigmoid=0.5；tw=0,th=0 -> wh=anchor
        if epoch < self.warmup_epochs:
            if self.prior_on_bg_only:
                prior_mask = bg_mask.to(dtype)
            else:
                prior_mask = torch.ones_like(gt_obj, dtype=dtype, device=device)

            prior_term = (tx ** 2) + (ty ** 2) + (tw ** 2) + (th ** 2)
            loss_prior_map = self.lambda_prior * prior_mask * prior_term
        else:
            loss_prior_map = torch.zeros_like(gt_obj, dtype=dtype, device=device)

        # -------- 5) part3: truth（红框）：coord + obj + class --------
        # 5.1 coord（xy 用 sigmoid(tx/ty) 对齐 gt_x/gt_y；wh 用 t 空间对齐 gt_tw/gt_th）
        pred_x = torch.sigmoid(tx)
        pred_y = torch.sigmoid(ty)

        coord_term = (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2 + (tw - gt_tw) ** 2 + (th - gt_th) ** 2
        loss_coord_map = self.lambda_coord * pos_mask.to(dtype) * coord_term

        # 5.2 obj（目标为 IoU(pred_box, gt_box)，通常 detach）
        # 对齐图里：(IOU_truth^k - b^o_ijk)^2
        iou_pos = _bbox_iou_xyxy(pred_xyxy, gt_xyxy).unsqueeze(-1)  # (bs,S,S,A,1)
        iou_target = iou_pos.detach()
        loss_obj_map = self.lambda_obj * pos_mask.to(dtype) * ((pred_conf - iou_target) ** 2)

        # 5.3 class
        if self.cls_loss == "mse":
            # 对齐你图：MSE 风格
            pred_prob = F.softmax(cls_logits, dim=-1)
            cls_term = ((pred_prob - gt_cls) ** 2).sum(dim=-1, keepdim=True)  # (bs,S,S,A,1)
            loss_cls_map = self.lambda_class * pos_mask.to(dtype) * cls_term
        elif self.cls_loss == "ce":
            # 更标准的实现（可选）：cross entropy
            # 只在正样本上计算
            pos_mask_flat = pos_mask.squeeze(-1).reshape(-1)  # (bs*S*S*A,)
            if pos_mask_flat.any():
                logits_flat = cls_logits.reshape(-1, self.num_classes)
                target_flat = gt_cls.argmax(dim=-1).reshape(-1)
                loss_ce = F.cross_entropy(logits_flat[pos_mask_flat], target_flat[pos_mask_flat], reduction="mean")
                # 把它当成标量项加入（为了统一 items）
                loss_cls_map = torch.zeros_like(gt_obj, dtype=dtype, device=device)
                loss_cls_scalar = self.lambda_class * loss_ce
            else:
                loss_cls_map = torch.zeros_like(gt_obj, dtype=dtype, device=device)
                loss_cls_scalar = torch.zeros((), dtype=dtype, device=device)
        else:
            raise ValueError(f"cls_loss 只支持 'mse' 或 'ce'，你给的是: {self.cls_loss}")

        # -------- 6) 归一化 & 汇总 --------
        # 你如果想完全对齐 darknet “sum” 风格，把 normalize=False 即可
        num_pos = pos_mask.to(dtype).sum()
        num_noobj = noobj_mask.to(dtype).sum()
        num_prior = (bg_mask.to(dtype).sum() if self.prior_on_bg_only else torch.tensor(float(gt_obj.numel()), device=device, dtype=dtype))

        if self.normalize:
            loss_noobj = loss_noobj_map.sum() / (num_noobj + self.eps)
            loss_prior = loss_prior_map.sum() / (num_prior + self.eps)
            loss_coord = loss_coord_map.sum() / (num_pos + self.eps)
            loss_obj = loss_obj_map.sum() / (num_pos + self.eps)

            if self.cls_loss == "mse":
                loss_cls = loss_cls_map.sum() / (num_pos + self.eps)
            else:
                loss_cls = loss_cls_scalar
        else:
            loss_noobj = loss_noobj_map.sum()
            loss_prior = loss_prior_map.sum()
            loss_coord = loss_coord_map.sum()
            loss_obj = loss_obj_map.sum()

            if self.cls_loss == "mse":  
                loss_cls = loss_cls_map.sum()
            else:
                loss_cls = loss_cls_scalar

        total = loss_noobj + loss_prior + loss_coord + loss_obj + loss_cls

        items = {
            "total": total.detach(),
            "noobj": loss_noobj.detach(),
            "prior": loss_prior.detach(),
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "class": loss_cls.detach(),
            "num_pos": num_pos.detach(),
            "num_noobj": num_noobj.detach(),
        }

        if self.ic_debug:
            ic(total)
            ic(items)

        return total, items


if __name__ == "__main__":
    # 简单自测：形状对齐 + 能反传
    bs, S, A, C = 2, 13, 5, 20
    pred = torch.randn(bs, S, S, A, 5 + C, dtype=torch.float32)
    gt = torch.zeros(bs, S, S, A, 5 + C, dtype=torch.float32)

    # 构造一个正样本：随机挑一个 cell+anchor
    i, j, k = 3, 7, 2
    gt[:, i, j, k, 0] = 0.4  # x offset
    gt[:, i, j, k, 1] = 0.6  # y offset
    gt[:, i, j, k, 2] = 0.0  # tw*
    gt[:, i, j, k, 3] = 0.0  # th*
    gt[:, i, j, k, 4] = 1.0  # obj
    gt[:, i, j, k, 5 + 5] = 1.0  # class id=5 one-hot

    cfg = {
        "num_classes": C,
        "anchors_path": r"D:\1AAAAAstudy\python_base\pytorch\my_github_workspace\yolov2-pytorch\dataset\anchors_k5.json",
        "loss_dtype": torch.float32,
        "device": torch.device("cpu"),

        "lambda_noobj": 0.5,
        "lambda_prior": 0.01,
        "lambda_coord": 5.0,
        "lambda_obj": 1.0,
        "lambda_class": 1.0,

        "ignore_thresh": 0.6,
        "warmup_epochs": 12,
        "prior_on_bg_only": True,

        "normalize": True,
        "cls_loss": "mse",  # 或 "ce"
    }

    loss_fn = Yolov2Loss(cfg, ic_debug=True)
    total, items = loss_fn(pred, gt, epoch=0)

    total.backward()
    print("OK, loss backward success.")
