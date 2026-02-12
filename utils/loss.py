import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import json

from typing import List, Tuple
from icecream import ic

'''
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
'''

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


def _anchor_iou(w,h,anchors):
    '''
    先验anchor框和标注框进行匹配，iou大的标注框的对应预测框来负责预测

    gt的w/h : (2, 13, 13, 5, 1)
    anchors: (k, 2)

    return: (2, 13, 13, 5)
    '''
    w = w.squeeze(-1)
    h = h.squeeze(-1)
    bs = w.shape[0]
    grid_size = w.shape[1]
    num_anchors = w.shape[3]
    # 整理anchor的顺序，由wh : (5, 2) -->w:[2, 13, 13, 5] 和 h:[2, 13, 13, 5]
    anchors_w = anchors[:, 0].reshape(1, 1, 1, num_anchors).expand(bs, grid_size, grid_size, num_anchors)
    anchors_h = anchors[:, 1].reshape(1, 1, 1, num_anchors).expand(bs, grid_size, grid_size, num_anchors)
    # 计算iou
    
    inter = torch.min(w, anchors_w) * torch.min(h, anchors_h)
    area_anchors = anchors_w * anchors_h
    area_wh = w * h
    
    union = area_wh + area_anchors - inter + 1e-8
    iou = inter / union
    # ic(iou.shape)
    return iou


def _bbox_iou_xyxy(bbox1, bbox2, eps):
    """
    计算 IoU（xyxy 格式）

    输入:
        a: (..., 4)
        b: (..., 4)  与 a 需要可 broadcast

    输出:
        iou: broadcast 后的 (...,)
    """
    ax1, ay1, ax2, ay2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
    bx1, by1, bx2, by2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]

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


def _pred_gt_iou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2, gt_o):
    '''  
    计算各个预测框和所有gt的iou
    已有的pred_x1, pred_y1, pred_x2, pred_y2 : [2, 13, 13, 5, 1]
    已有的gt_x1, gt_y1, gt_x2, gt_y2 : [2, 13, 13, 5, 1]

    先组合为 [2, 13, 13, 5, 4]
    预测框: [2, 13, 13, 5, 4]
    gt: [2, 13, 13, 5, 4]

    计算iou
    预测框的数量Npred = 13*13*5 = 845
    gt的数量Ngt = 真实gt的数量
    iou: [2, 13, 13, 5, Ngt]

    然后进一步计算在Ngt中的最大值
    return [2, 13, 13, 5, 1]
    '''
    pred_dtype = pred_x1.dtype
    pred_device = pred_x1.device

    bs = pred_x1.shape[0]
    grid_size = pred_x1.shape[1]
    num_anchors = pred_x1.shape[3]
    Npred = grid_size * grid_size * num_anchors
    # ([2, 13, 13, 5, 4])
    pred_bbox = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2), dim=4)
    # ([2, 13, 13, 5, 4])
    gt_bbox = torch.cat((gt_x1, gt_y1, gt_x2, gt_y2), dim=4)
    # gt_mask: 标注的mask ([2, 13, 13, 5, 1])
    gt_mask = gt_o > 0.5

    # gt_mask_flat.shape: torch.Size([2, 845])
    gt_mask_flat = gt_mask.view(bs, Npred)
    # max_iou_all.shape: torch.Size([2, 845])
    # 意义为: 每个img中，最大的iou对应的值，方便后续阈值判断
    max_iou_all = torch.zeros((bs, Npred), dtype=pred_dtype, device=pred_device)
    
    gt_flat = gt_bbox.view(bs, Npred, 4)
    pred_flat = pred_bbox.view(bs, Npred, 4)

    for n in range(bs):
        ''' 
        gt_mask_flat[n].shape: torch.Size([845])
        torch.nonzero(gt_mask_flat[n], as_tuple=False).shape: torch.Size([238, 1])
        pos_idx.shape: torch.Size([238])
        '''
        # 找出该图片中所有正样本（真实 GT）所在的位置索引
        pos_idx = torch.nonzero(gt_mask_flat[n], as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0:
            # 若该图片没有任何 GT（Ng==0），则 best_iou 保持为 0
            continue

        # 取出该图片中所有真实 GT 框坐标 ([238, 4])
        g = gt_flat[n, pos_idx]
        # 取出该图片的所有预测框坐标 ([845, 4])
        p = pred_flat[n]

        # 调整形状，我们期望的是每个预测框都与所有的gt框计算出来iou
        # 所以最后应当是 [845, 1, 4] 与 [1, 238, 4] 计算iou
        # 结果为 [845, 238] ,含义为:845个预测框，每个预测框对应238个gt框的iou值
        g_expand = g.reshape(1, -1, 4).expand(p.shape[0], -1, -1) # ([845, 238, 4])
        p_expand = p.reshape(-1, 1, 4).expand(-1, g.shape[0], -1) # ([845, 238, 4])

        # 计算iou
        iou = _bbox_iou_xyxy(p_expand, g_expand, eps=1e-8) # ([845, 239])
        # 取出最大iou的值，赋值给对应bs中
        max_iou, _ = iou.max(dim=1) # torch.Size([845])
        max_iou_all[n] = max_iou
    
    # 处理完成全部的bs之后，整理形状
    max_iou_all = max_iou_all.reshape(bs, grid_size, grid_size, num_anchors, 1)
    return max_iou_all


def iter2epoch(cfg):
    '''
    计算前12800次迭代转化为epoch是多少
    '''
    batch_size = cfg["batch_size"]
    train_size = cfg["train_size"]
    num_iter = 12800
    num_epoch = num_iter * batch_size // train_size
    return num_epoch


class Yolov2Loss(nn.Module):
    def __init__(self, cfg, ic_debug=False):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]
        self.anchors = torch.tensor(load_anchors_json(cfg["anchors_json"]), dtype=cfg["loss_dtype"], device=cfg["device"])
        self.ic_debug = ic_debug
        self.prior_loss = iter2epoch(cfg) # 前epoch个要计算先验框loss

    def forward(self, pred_tensor, gt_tensor, epoch):
        bs = pred_tensor.shape[0]
        grid_size = pred_tensor.shape[1]
        num_anchors = pred_tensor.shape[3]
        # pred_tensor: (bs,S,S,A,5+C)
        # gt_tensor: (bs,S,S,A,5+C)
        # 提取预测出来的xywh和obj
        pred_x = pred_tensor[:, :, :, :, 0:1] # tx ([2, 13, 13, 5, 1])
        pred_y = pred_tensor[:, :, :, :, 1:2] # ty
        pred_w = pred_tensor[:, :, :, :, 2:3] # tw
        pred_h = pred_tensor[:, :, :, :, 3:4] # th
        pred_xywh = pred_tensor[:, :, :, :, 0:4]
        pred_o = pred_tensor[:, :, :, :, 4:5] # ([2, 13, 13, 5, 1])
        pred_cls = pred_tensor[:, :, :, :, 5:] # ([2, 13, 13, 5, 20])
        if self.ic_debug:
            ic(pred_x.shape)
            ic(pred_cls.shape)
        # 提取gt的信息
        gt_x = gt_tensor[:, :, :, :, 0:1] # ([2, 13, 13, 5, 1])
        gt_y = gt_tensor[:, :, :, :, 1:2] # ([2, 13, 13, 5, 1])
        gt_w = gt_tensor[:, :, :, :, 2:3] # ([2, 13, 13, 5, 1])
        gt_h = gt_tensor[:, :, :, :, 3:4] # ([2, 13, 13, 5, 1])
        gt_xywh = gt_tensor[:, :, :, :, 0:4]
        gt_o = gt_tensor[:, :, :, :, 4:5] # ([2, 13, 13, 5, 1])
        gt_cls = gt_tensor[:, :, :, :, 5:] # ([2, 13, 13, 5, 20])
        if self.ic_debug:
            ic(gt_x.shape)
            ic(gt_cls.shape)
        # 由t_w,t_h 转为b_w, b_h
        # ([2, 13, 13, 5, 1])
        pred_b_x, pred_b_y = _txy2bxy(pred_x, pred_y)
        pred_b_w, pred_b_h = _twh2bwh(pred_w, pred_h, self.anchors)
        gt_b_w, gt_b_h = _twh2bwh(gt_w, gt_h, self.anchors)

        pred_x1, pred_y1, pred_x2, pred_y2 = _xywh2xyxy(pred_b_x, pred_b_y, pred_b_w, pred_b_h)
        gt_x1, gt_y1, gt_x2, gt_y2 = _xywh2xyxy(gt_x, gt_y, gt_b_w, gt_b_h)
        ''' 
        part 1
        计算各个预测框和所有gt的iou
        已有的pred_x1, pred_y1, pred_x2, pred_y2 : [2, 13, 13, 5, 1]
        已有的gt_x1, gt_y1, gt_x2, gt_y2 : [2, 13, 13, 5, 1]

        先组合为 [2, 13, 13, 5, 4]
        预测框: [2, 13, 13, 5, 4]
        gt: [2, 13, 13, 5, 4]

        计算iou
        预测框的数量Npred = 13*13*5 = 845
        gt的数量Ngt = 真实gt的数量
        iou: [2, 13, 13, 5, Ngt]

        选择最大的
        max_iou: [2, 13, 13, 5]
        然后阈值过滤，选出来小于0.6的
        '''
        # 得到pred的bbox与每个gt的iou值
        # max_iou_all: [2, 13, 13, 5, 1]
        max_iou_all = _pred_gt_iou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2, gt_o)
        # 阈值过滤
        ignore_mask = (max_iou_all < self.cfg["ignore_threshold"]) # ([2, 13, 13, 5, 1])
        # 同时要排除掉所有含有gt的grid cell
        bg_mask = (gt_o <= 0.5) # ([2, 13, 13, 5, 1])
        noobj_mask = bg_mask * ignore_mask # ([2, 13, 13, 5, 1])
        # 对齐监督空间
        pred_conf = torch.sigmoid(pred_o)
        loss_noobj = self.cfg["lambda_noobj"] * noobj_mask * (0-pred_conf)**2

        '''
        part 2 先验框引导损失

        让模型去尝试拟合我们之前通过kmeans计算出来的先验anchor
        这样的话sigmod(tx)和sigmoid(ty)应该尝试靠近0.5,也就是中心点 --> tx,ty尝试靠近0
        t_w和t_h 应该尝试靠近0 
        '''
        if epoch < self.prior_loss:
            # 前12800次迭代(转化为epoch单位)
            # pred_xywh: (2, 13, 13, 5, 4)
            loss_prior = self.cfg["lambda_prior"] * (0-pred_xywh)**2
        else:
            loss_prior = torch.tensor(0, dtype=self.cfg["loss_dtype"], device=self.cfg["device"])

        '''
        part 3 正样本误差
        首先提取到mask
        mask分为两部分:1:负责的grid cell mask 2:负责的anchor mask
        1:负责的grid cell mask
        gt的中心点落在哪个 Grid Cell，就由哪个 Grid Cell 负责。
        2:负责的anchor mask
        在该 Grid Cell 的 5 个 Anchor 中，谁的形状（长宽比）和gt最像（IoU 最大），谁就负责。
        '''
        '''
        2026-02-10 
        note: 错误的尝试,对于positive_mask的处理过于复杂，其实在dataset里面就已经处理好了，直接提取gt_o即可.
        
        # 负责的grid cell mask 
        gt_grid = (gt_o > 0.5) # ([2, 13, 13, 5, 1])
        # 最终变为 2, 13, 13
        gt_grid_mask = torch.any(gt_grid, dim=3) # ([2, 13, 13, 1])
        gt_grid_mask = gt_grid_mask.expand(bs, grid_size, grid_size, num_anchors) # ([2, 13, 13, 5])

        # 负责的anchor mask
        anchor_iou = _anchor_iou(gt_b_w, gt_b_h, self.anchors) # ([2, 13, 13, 5])
        _, anchor_idx = torch.max(anchor_iou, dim=3) # ([2, 13, 13])
        # ic(anchor_idx.shape)
        anchor_mask = F.one_hot(anchor_idx, num_anchors) # ([2, 13, 13, 5])
        
        # 正样本mask
        positive_mask = (gt_grid * anchor_mask).unsqueeze(-1) # ([2, 13, 13, 5])
        '''
        positive_mask = (gt_o > 0.5) # ([2, 13, 13, 5, 1])
        # ic(positive_mask.shape)
        # part 3.1 正样本坐标损失
        # 转化为loss函数的拟合目标xy [由监督空间决定]
        fitt_x, fitty = torch.sigmoid(pred_x), torch.sigmoid(pred_y)
        fitt_w, fitt_h = pred_w, pred_h
        fitt_xywh = torch.cat((fitt_x, fitty, fitt_w, fitt_h), dim=4) # ([2, 13, 13, 5, 4])
        positive_mask_expand = positive_mask.expand(bs, grid_size, grid_size, num_anchors, 4) # ([2, 13, 13, 5, 4])
    
        loss_positive_xywh = self.cfg["lambda_coord"] * positive_mask_expand * (gt_xywh - fitt_xywh)**2
        # part 3.2 正样本置信度损失
        # 预测出来的置信度逼近真实的pred与gt的iou
        # pred_o: [2, 13, 13, 5, 1]    max_iou_all: [2, 13, 13, 5, 1]
        # 转换到监督空间
        pred_conf = torch.sigmoid(pred_o)
        # 求对应的iou
        pred_bbox_xyxy = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2), dim=4)  # (bs,S,S,A,4)
        gt_bbox_xyxy   = torch.cat((gt_x1,   gt_y1,   gt_x2,   gt_y2),   dim=4)  # (bs,S,S,A,4)

        conf_iou = _bbox_iou_xyxy(pred_bbox_xyxy, gt_bbox_xyxy, eps=1e-8)
        loss_positive_conf = self.cfg["lambda_obj"] * positive_mask * (conf_iou - pred_conf.detach())**2

        # part 3.3 正样本类别损失
        # 预测出来的类别应该与gt的类别一致
        # 类别先做softmax
        pred_prob = F.softmax(pred_cls, dim=-1)
        loss_positive_cls = self.cfg["lambda_cls"] * positive_mask * (gt_cls - pred_prob)**2


        loss_final = torch.sum(loss_noobj) + torch.sum(loss_prior) + torch.sum(loss_positive_xywh) + torch.sum(loss_positive_conf) + torch.sum(loss_positive_cls)

        # ic(loss_final)
        return loss_final / bs



if __name__ == "__main__":
    test_pred = torch.randn(2, 13, 13, 5, 5+20)
    test_gt = torch.randn(2, 13, 13, 5, 5+20)
    cfg = {
        "num_classes": 20,
        "anchors_json": r'D:\1AAAAAstudy\python_base\pytorch\my_github_workspace\yolov2-pytorch\dataset\anchors_k5.json',
        "loss_dtype": torch.float32,
        "device": torch.device("cpu"),
        "lambda_noobj": 0.5,
        "lambda_obj": 5,
        "lambda_prior": 0.01,
        "lambda_coord": 1,
        "lambda_cls": 1,

        "ignore_threshold": 0.6, # loss part1 参数
    }
    
    loss_func = Yolov2Loss(cfg, ic_debug=True)

    loss = loss_func(test_pred, test_gt, 0)