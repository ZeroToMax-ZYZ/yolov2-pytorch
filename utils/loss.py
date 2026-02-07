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
- x,y：cell 内 offset ∈ (0,1)
- tw,th：ln(w_rel/p_w_rel), ln(h_rel/p_h_rel)
- obj：1
- cls：one-hot


模型的预测输出：
(bs,S,S,A,5+C)
5+C: t_x, t_y, t_w, t_h, t_o, cls logits
'''

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
    w/h : (2, 13, 13, 5, 1)
    anchors: (k, 2)

    return: (2, 13, 13, 5, k)
    '''
    bs = w.shape[0]
    grid_size = w.shape[1]
    num_anchors = w.shape[3]
    # [k,2] --> [1,1,1,k,1] --> [2,13,13,k,2]
    anchors_w = anchors[:, 0].reshape(1, 1, 1, 1, num_anchors).expand(bs, grid_size, grid_size, 1, num_anchors)
    anchors_h = anchors[:, 1].reshape(1, 1, 1, 1, num_anchors).expand(bs, grid_size, grid_size, 1, num_anchors)
    # 计算iou
    inter = torch.min(w, anchors_w) * torch.min(h, anchors_h)
    area_anchors = anchors_w * anchors_h
    area_wh = w * h
    union = area_wh + area_anchors - inter + 1e-8
    iou = inter / union
    # ic(iou.shape)
    return iou


def _bbox_iou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2):

    pred_bbox = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2), dim=4)
    gt_bbox = torch.cat((gt_x1, gt_y1, gt_x2, gt_y2), dim=4)
    ic(pred_bbox.shape)



class Yolov2Loss(nn.Module):
    def __init__(self, cfg, prior_loss, ic_debug=False):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]
        self.anchors = torch.tensor(load_anchors_json(cfg["anchors_path"]), dtype=cfg["loss_dtype"], device=cfg["device"])
        self.ic_debug = ic_debug
        self.prior_loss = prior_loss

    def forward(self, pred_tensor, gt_tensor, epoch):
        bs = pred_tensor.shape[0]
        grid_size = pred_tensor.shape[1]
        num_anchors = pred_tensor.shape[3]
        # pred_tensor: (bs,S,S,A,5+C)
        # gt_tensor: (bs,S,S,A,5+C)
        # 提取预测出来的xywh和obj
        pred_x = pred_tensor[:, :, :, :, 0:1] # ([2, 13, 13, 5, 1])
        pred_y = pred_tensor[:, :, :, :, 1:2] 
        pred_w = pred_tensor[:, :, :, :, 2:3]
        pred_h = pred_tensor[:, :, :, :, 3:4]
        pred_xywh = pred_tensor[:, :, :, :, 0:4]
        pred_o = pred_tensor[:, :, :, :, 4:5] 
        pred_cls = pred_tensor[:, :, :, :, 5:] # ([2, 13, 13, 5, 20])
        if self.ic_debug:
            ic(pred_x.shape)
            ic(pred_cls.shape)
        # 提取gt的信息
        gt_x = gt_tensor[:, :, :, :, 0:1]
        gt_y = gt_tensor[:, :, :, :, 1:2]
        gt_w = gt_tensor[:, :, :, :, 2:3]
        gt_h = gt_tensor[:, :, :, :, 3:4]
        gt_o = gt_tensor[:, :, :, :, 4:5]
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
        _bbox_iou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2)


        

        # noobj_mask
        # loss_noobj = cfg["lambda_noobj"] * noobj_mask * (-pred_o)**2

        # part 2
        # 先验框引导损失
        ''' 
        让模型去尝试拟合我们之前通过kmeans计算出来的先验anchor
        这样的话sigmod(tx)和sigmoid(ty)应该尝试靠近0.5,也就是中心点 --> tx,ty尝试靠近0
        t_w和t_h 应该尝试靠近0 
        '''
        # if epoch < self.prior_loss:
        #     # 前12800次迭代(转化为epoch单位)
        #     # pred_xywh: (2, 13, 13, 5, 4)
        #     loss_prior = cfg["lambda_prior"] * (0-pred_xywh)**2
        # else:
        #     loss_prior = torch.tensor(0, dtype=cfg["loss_dtype"], device=cfg["device"])


        # part 3
        # 正样本 定位误差
        ''' 
        落点原则：狗的中心点落在哪个 Grid Cell，就由哪个 Grid Cell 负责。
        形状原则：在该 Grid Cell 的 5 个 Anchor 中，谁的形状（长宽比）和狗最像（IoU 最大），谁就负责。
        '''
       
        # # ([2, 13, 13, 5, 1])
        # anchor_iou = _anchor_iou(pred_b_w, pred_b_h, self.anchors)
        # # anchor_idx: ([2, 13, 13])
        # _, anchor_idx = torch.max(anchor_iou.squeeze_(4), dim=3)
        # # anchor_mask: ([2, 13, 13, 5])
        # anchor_mask = F.one_hot(anchor_idx, num_anchors)
        # ic(anchor_mask.shape)




if __name__ == "__main__":
    test_pred = torch.randn(2, 13, 13, 5, 5+20)
    test_gt = torch.randn(2, 13, 13, 5, 5+20)
    cfg = {
        "num_classes": 20,
        "anchors_path": r'D:\1AAAAAstudy\python_base\pytorch\my_github_workspace\yolov2-pytorch\dataset\anchors_k5.json',
        "loss_dtype": torch.float32,
        "device": torch.device("cpu"),
        "lambda_noobj": 0.5,
        "lambda_prior": 0.01
    }
    
    loss_func = Yolov2Loss(cfg, prior_loss=12, ic_debug=True)

    loss = loss_func(test_pred, test_gt, 0)