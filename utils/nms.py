'''
输入经过decode的模型预测值（全图偏移，grdi cell坐标系下【0-7】）
经过nms过滤，输出过滤后的值，含义保持一致。
此时的nms函数的返回值是一个list，长度对应bs大小

关于类别：
decode之后，类别信息依旧为onehot
nms之后，类别信息为具体的类别
'''
import torch

from icecream import ic

def box_iou(box1, box2):
    '''
    box1 [4]: xyxy 最大的box
    box2 [nums, 4]除了最大的以外所有的box
    '''
    inter_x1 = torch.max(box1[0], box2[:, 0])
    inter_x2 = torch.min(box1[2], box2[:, 2])
    inter_y1 = torch.max(box1[1], box2[:, 1])
    inter_y2 = torch.min(box1[3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = box1_area + box2_area - inter + 1e-6 # 1e-6 防止除0
    iou = inter / union
    
    return iou
    
    
def nms(out_pred, conf_thresh=0.1, iou_thresh=0.5, topk_per_class=10):
    '''
    out_pred [2,7,7,2,xyxy-conf-cls]
    '''
    # base info
    bs, S, S, B, dim = out_pred.shape
    num_classes = dim - 5
    out_boxes = []
    
    for b in range(bs):
        b_out_pred = out_pred[b, :, :, :, :] # [7,7,2,6]
        b_out_pred = b_out_pred.reshape(S*S*B, -1) # [98, 6]
        batch_boxes = []
        # 逐个类别进行nms
        for c in range(num_classes):
            boxes = b_out_pred[:, :4]
            # 逐个类别的全概率
            # cls_scores.shape --> [98]
            cls_scores = b_out_pred[:, 4] * b_out_pred[:, 5+c]
            # 筛选大于阈值的box
            valid_boxes = boxes[cls_scores > conf_thresh] # ([48, 4])
            valid_scores = cls_scores[cls_scores > conf_thresh] # ([48])
            
            if topk_per_class > 0 and valid_scores.numel() > topk_per_class:
                sorted_scores, topk_idx = torch.topk(
                    valid_scores,
                    k=int(topk_per_class),
                    largest=True,
                    sorted=True,
                )
                sorted_boxes = valid_boxes[topk_idx]
            else:
                # 全概率排序，从大到小
                sorted_scores, sorted_idx = valid_scores.sort(descending=True) # ([48])
                sorted_boxes = valid_boxes[sorted_idx] # ([48, 4])
                
            keep_boxes = []
            
            while sorted_scores.shape[0] > 0:
                # 第一名
                best_box = sorted_boxes[0] # torch.Size([4])
                best_score = sorted_scores[0] # torch.Size([])  0 维张量（标量张量）
                # ic(best_box.shape)
                # ic(best_score.unsqueeze(0))
                c_tensor = torch.tensor([c], device=best_box.device, dtype=best_box.dtype)
                best_tensor = torch.cat([best_box, best_score.unsqueeze(0), c_tensor]) # torch.Size([6])
                keep_boxes.append(best_tensor)
                
                if sorted_scores.shape[0] == 1:
                    break
                
                # 比较iou
                iou = box_iou(best_box, sorted_boxes[1:])
                # 舍弃大于阈值的box，继续判定小于阈值的box
                # 一定要小心，提取的时候要排除第一个
                sorted_scores = sorted_scores[1:][iou < iou_thresh]
                sorted_boxes = sorted_boxes[1:][iou < iou_thresh]
                
            
            # 一个类别处理结束：
            if len(keep_boxes) > 0:
                batch_boxes.extend(keep_boxes)
                
                

        # 整个batch结束：
        if len(batch_boxes) > 0:
            out_boxes.append(torch.stack(batch_boxes))
        else:
            # 如果是空的
            out_boxes.append(torch.zeros(0, 6).to(out_pred.device))
            
    # ic(out_boxes)   
    return out_boxes


if __name__ == "__main__":
    test_tensor = torch.randn(2, 7, 7, 2, 6)
    out_boxes = nms(test_tensor, conf_thresh=0.01, iou_thresh=0.5)
    ic(out_boxes[0].shape)
    ic(len(out_boxes))