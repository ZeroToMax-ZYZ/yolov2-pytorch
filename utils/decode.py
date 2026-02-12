'''

把模型的输出和标注文件
从xywh-conf-cls(相对于grid cell左上角的偏移【0-1】)
转化为xyxyconf cls(相对于全图，在grid cell坐标系下的偏移【0-7】)


dataset的数据
yolo_tensor:
    (bs,S,S,A,5+C)
编码细节（按 YOLOv2 参数化监督）:
- x,y:cell 内 offset ∈ (0,1)
- tw,th:ln(w_rel/p_w_rel), ln(h_rel/p_h_rel)
- obj:1
- cls:one-hot

模型的预测输出:
(bs,S,S,A,5+C)
5+C: t_x, t_y, t_w, t_h, t_o, cls logits

转化之后的格式：
坐标系：全图偏移，grdi cell坐标系下【0-7】
decode之后，类别信息依旧为onehot
输出的格式：
gt:
list: [[xyxy-cls],[]]

pred: 
tensor: 2-13-13-5-20 注意，此时的cls为onehot


'''
import torch

import json
from typing import List, Tuple

from icecream import ic
from dataclasses import dataclass

@dataclass
class LabelDecode:
    bboxes: torch.Tensor
    labels: torch.Tensor

@dataclass
class PredDecode:
    bboxes: torch.Tensor
    labels: torch.Tensor
    confs: torch.Tensor

def load_anchors_json(json_path: str) -> List[Tuple[float, float]]:
    """
    作用:
        从 json 读取 anchors_rel
    json 结构示例:
        {"anchors_rel": [[w_rel,h_rel], ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data["anchors_rel"]
    return [(float(a[0]), float(a[1])) for a in anchors]


def _meshgrid(x):
    '''
    将xy从相对grid cell内部的偏移量转换为相对于整个图片的偏移量(grid cell坐标系下)
    x : [2, 13, 13, 5, 1]
    return: [2, 13, 13, 5, 1]
    '''
    grid_size = x.shape[1]
    loss_dtype = x.dtype
    loss_device = x.device
    # (2, 13, 13, 5, 1)
    i = torch.arange(0, grid_size, dtype=loss_dtype, device=loss_device)
    j = torch.arange(0, grid_size, dtype=loss_dtype, device=loss_device)

    ii, jj = torch.meshgrid(i, j, indexing='ij')

    ii = ii.reshape(1, grid_size, grid_size, 1, 1).expand(x.shape)
    jj = jj.reshape(1, grid_size, grid_size, 1, 1).expand(x.shape)

    return ii, jj


def _xywh2xyxy(x,y,w,h):
    '''
    输入
    x: grid cell内部的偏移量
    y: grid cell内部的偏移量
    w: b_w
    h: b_h
    '''
    grid_size = x.shape[1]
    ii, jj = _meshgrid(x)
    
    center_x = x + jj
    center_y = y + ii
    center_w = w * grid_size
    center_h = h * grid_size

    x1 = center_x - center_w/2
    y1 = center_y - center_h/2
    x2 = center_x + center_w/2
    y2 = center_y + center_h/2

    return x1, y1, x2, y2


def _twh2bwh(tw, th, anchors):
    '''
    按照yolov2的参数化方式，转化wh
    tw: (2, 13, 13, 5, 1)
    anchors: (k, 2)
    '''
    bs = tw.shape[0]
    grid_size = tw.shape[1]
    num_anchors = tw.shape[3]
    # (k, 1) --> (1, 1, 1, k, 1) --> (2, 13, 13, k, 1)
    anchor_w = anchors[:, 0].reshape(1, 1, 1, num_anchors, 1).expand(bs, grid_size, grid_size, num_anchors, 1)
    anchor_h = anchors[:, 1].reshape(1, 1, 1, num_anchors, 1).expand(bs, grid_size, grid_size, num_anchors, 1)

    b_w = torch.exp(tw) * anchor_w
    b_h = torch.exp(th) * anchor_h
    return b_w, b_h



def _txy2bxy(tx, ty):
    '''
    此时是针对pred的，pred预测出来的tx-ty按照yolov2参数化公式，先sigmod
    '''
    return torch.sigmoid(tx), torch.sigmoid(ty)


def decode_labels_list(gt, cfg):
    '''
    输入：bs*7*7*A*(xywh-conf-cls) cls为onehot
    以及anchors信息

    从偏移量转化为grid坐标系
    返回值：
    一个list，元素的数量为bs，每个tensor为该batch内的[nums, 5] xywh-cls 此时的cls为具体的类别
    '''
    anchors = torch.tensor(load_anchors_json(cfg["anchors_path"]), dtype=cfg["loss_dtype"], device=cfg["device"])
    bs = gt.shape[0]
    S = gt.shape[1]
    gt_x = gt[:, :, :, :, 0:1] # ([2, 13, 13, 5, 1])
    gt_y = gt[:, :, :, :, 1:2] # ([2, 13, 13, 5, 1])
    gt_w = gt[:, :, :, :, 2:3] # ([2, 13, 13, 5, 1])
    gt_h = gt[:, :, :, :, 3:4] # ([2, 13, 13, 5, 1])
    gt_conf = gt[:, :, :, :, 4:5] # ([2, 13, 13, 5, 1])
    gt_cls = gt[:, :, :, :, 5:] # ([2, 13, 13, 5, 20])

    gt_b_w, gt_b_h = _twh2bwh(gt_w, gt_h, anchors)

    # ([2, 13, 13, 5, 1])
    gt_x1, gt_y1, gt_x2, gt_y2 = _xywh2xyxy(gt_x, gt_y, gt_b_w, gt_b_h)

    # ([2, 13, 13, 5, 20]) --> ([2, 13, 13, 5, 1])
    gt_cls_argmax = gt_cls.argmax(dim=-1,keepdim=True)
    # ([2, 13, 13, 5, 5])
    comb_gt = torch.cat((gt_x1, gt_y1, gt_x2, gt_y2, gt_cls_argmax),dim=-1)
    # 提取出来含有gt的mask信息
    obj_mask = (gt_conf[:, :, :, :, 0] > 0.5) # 2-13-13-5

    out_list = []
    # ic(obj_mask.shape)
    for batch in range(bs):
        batch_list = []
        # 取出来对应batch的xywhcls信息和对应的mask
        batch_comb = comb_gt[batch] # batch_comb.shape: torch.Size([13, 13, 5, 5])
        batch_mask = obj_mask[batch] # batch_mask.shape: torch.Size([13, 13, 5])
        ic(batch_comb.shape, batch_mask.shape)
        # 提取
        picked = batch_comb[batch_mask] # ([271, 5])
        ic(picked.shape)
        if picked.numel() == 0:
            out_list.append(torch.zeros((0, 5), device=gt.device, dtype=gt.dtype))
        else:
            out_list.append(picked)
    # ic(out_list[0].shape)
    return out_list



def decode_preds(preds, cfg, B=2, conf_thresh=0.01):
    '''
    bs*13*13*5*25
    x-y-w-h-conf-cls
    '''
    anchors = torch.tensor(load_anchors_json(cfg["anchors_path"]), dtype=cfg["loss_dtype"], device=cfg["device"])
    bs = preds.shape[0]
    S = preds.shape[1]
    A = preds.shape[3]
    pred_x = preds[:, :, :, :, 0:1] # ([2, 13, 13, 5, 1])
    pred_y = preds[:, :, :, :, 1:2] # ([2, 13, 13, 5, 1])
    pred_w = preds[:, :, :, :, 2:3] # ([2, 13, 13, 5, 1])
    pred_h = preds[:, :, :, :, 3:4] # ([2, 13, 13, 5, 1])

    pred_b_x, pred_b_y = _txy2bxy(pred_x, pred_y)
    pred_b_w, pred_b_h = _twh2bwh(pred_w, pred_h, anchors)

    pred_conf = preds[:, :, :, :, 4:5] # ([2, 13, 13, 5, 1])
    pred_cls = preds[:, :, :, :, 5:] # ([2, 13, 13, 5, 20])
    # ([2, 13, 13, 5, 1])
    pred_x1, pred_y1, pred_x2, pred_y2 = _xywh2xyxy(pred_b_x, pred_b_y, pred_b_w, pred_b_h)
    # 扩展类别
    num_classes = pred_cls.shape[-1]

    # ([2, 13, 13, 5, 25])
    out_pred = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2, pred_conf, pred_cls),dim=-1)

    # conf filter 
    # ic(out_pred.shape)
    # mask = pred_conf.reshape(bs, S, S, B) > conf_thresh
    # out_bbox = out_pred[mask]
    # ic(out_bbox.shape)
    return out_pred



    # ic(out_pred.shape)

if __name__ == "__main__":
    cfg = {
        "num_classes": 20,
        "anchors_path": r'D:\1AAAAAstudy\python_base\pytorch\my_github_workspace\yolov2-pytorch\dataset\anchors_k5.json',
        "loss_dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    # test_gt = torch.randn(2, 13, 13, 5, 25)
    # decode_labels_list(test_gt, cfg) # [num, 6] num为标签的数量, 6为   x1-y1-x2-y2-conf-cls
    test_pred = torch.randn(2, 13, 13, 5, 25)
    out_pred = decode_preds(test_pred, cfg) # [num, 6] num为预测框中经过conf过滤后的数量, 6为   x1-y1-x2-y2-conf-cls
    ic(out_pred.shape) # ([2, 7, 7, 2, 25])
