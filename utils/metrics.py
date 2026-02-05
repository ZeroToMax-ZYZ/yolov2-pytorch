'''
计算目标检测相关指标
包括:
map50
map50:95

输入约定:
1) preds_nms_all:
    List[torch.Tensor],长度=图片数
    每张图一个 tensor,形状:
        (Ni, 6) = [x1, y1, x2, y2, score, cls_id]
    - 坐标:grid 坐标系(0~S)
    - score:来自 nms 里 conf * cls_score
    - cls_id:在 nms 里被拼成 float(需要在 metrics 中转 long)

2) gts_all( decode_labels_list 的输出):
    List[torch.Tensor],长度=图片数
    每张图一个 tensor,形状:
        (Mi, 5) = [x1, y1, x2, y2, cls_id]
    - 坐标:grid 坐标系(0~S)
    - cls_id:是 float,需要转 long

'''

import torch
import numpy as np

from icecream import ic

def box_iou(box1, box2):
    '''
    box1 [4]: xyxy 最大的box
    box2 [nums, 4]除了最大的以外所有的box

    返回值：
    [nums]
    '''
    inter_x1 = torch.max(box1[0], box2[:, 0])
    inter_x2 = torch.min(box1[2], box2[:, 2])
    inter_y1 = torch.max(box1[1], box2[:, 1])
    inter_y2 = torch.min(box1[3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    box1_area = (box1[2] - box1[0]).clamp(min=0) * (box1[3] - box1[1]).clamp(min=0)
    box2_area = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    
    union = box1_area + box2_area - inter + 1e-6 # 1e-6 防止除0
    iou = inter / union
    
    return iou

def detach_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x

def compute_ap(Recall, Precision):
    '''
    先补齐边界点
    从后往前取最大值
    用小矩形面积累加作为整个曲线面积
    '''
    if Recall.numel() == 0 or Precision.numel() == 0:
        return 0.0
    
    s_device = Recall.device
    s_dtype = Recall.dtype

    # 补点
    completed_Recall = torch.cat([torch.tensor([0.0], device=s_device, dtype=s_dtype), 
                                  Recall, 
                                  torch.tensor([1.0], device=s_device, dtype=s_dtype)])
    completed_Precision = torch.cat([torch.tensor([0.0], device=s_device, dtype=s_dtype), 
                                     Precision, 
                                     torch.tensor([0.0], device=s_device, dtype=s_dtype)])

    # ic(completed_Precision.shape)
    ap_area = 0.0
    
    # ic(completed_Precision)
    # 精度包络
    for i in range(completed_Precision.shape[0] - 1, 0, -1):
        completed_Precision[i-1] = torch.maximum(completed_Precision[i], completed_Precision[i-1])
    
    # Recall的拐点
    # 如果当前点和前一个点发生变化，就求面积
    for j in range(1, completed_Recall.shape[0] - 1):
        if completed_Recall[j] != completed_Recall[j-1]:
            # 取该点左下角区域的面积
            w = completed_Recall[j] - completed_Recall[j-1]
            h = completed_Precision[j]
            ap_area += w.item() * h.item()
        
    return ap_area


    


        
        


def compute_map(preds_nms_all, gts_all, num_classes, metrics_dtype=torch.float32, eps=1e-6):
    '''
    1) preds_nms_all:
    List[torch.Tensor],长度=图片数
    每张图一个 tensor,形状:
        (Ni, 6) = [x1, y1, x2, y2, score, cls_id]
    - 坐标:grid 坐标系(0~S)
    - score:来自 nms 里 conf * cls_score
    - cls_id:在 nms 里被拼成 float(需要在 metrics 中转 long)

    2) gts_all( decode_labels_list 的输出):
        List[torch.Tensor],长度=图片数
        每张图一个 tensor,形状:
            (Mi, 5) = [x1, y1, x2, y2, cls_id]
        - 坐标:grid 坐标系(0~S)
        - cls_id:是 float,需要转 long
    '''
    # 构建[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    iou_threshs = [round(i, 2) for i in np.arange(0.50, 0.96, 0.05)]

    # 结果容器 torch.Size([10, 20]) 10对应的是10个不同的iou阈值，20对应的是20个类别
    ap_thresh_cls = torch.full((len(iou_threshs), num_classes),fill_value=float("nan"), dtype=metrics_dtype)

    # detach and move to cpu
    preds_nms_all = [detach_cpu(preds_nms) for preds_nms in preds_nms_all]
    gts_all = [detach_cpu(gts) for gts in gts_all]

    # 每个类别
    for c in range(num_classes):
        gt_bbox_pre_img = []
        gt_bbox_pre_img_count = 0
        # 收集gt的bbox和对应的数量，注意，此时gt_bbox_pre_img中的每个tensor代表每张图片对应的gt bbox
        for gt in gts_all:
            # 如果某张图片中，没有gt，则添加一个空tensor
            if gt.numel() == 0:
                gt_bbox_pre_img.append(torch.zeros((0, 4), dtype=metrics_dtype))
                continue
            # bbox --> [nums, 4] : {[nums, xyxy]}
            bbox = gt[:, :4].to(metrics_dtype)
            labels = gt[:, 4].to(torch.long)
            bbox_cls = bbox[labels == c]
            gt_bbox_pre_img.append(bbox_cls)
            gt_bbox_pre_img_count += int(bbox_cls.shape[0])

        # 统计某个类别下全部图片的gt数量，如果没有就跳过
        if gt_bbox_pre_img_count == 0:
            continue

        # 收集某个类别下全部图片的预测结果
        # 注意，此时的预测信息统计应当携带img id，便于后续的ap计算
        # 此时的pred_bbox_pre_img中的每一个元素代表每一个预测框。与前面的gt_list区分
        pred_bbox_pre_img = [] # [[img_id, score, bbox]]
        # pred_bbox_pre_img_count = 0

        for img_id, pred in enumerate(preds_nms_all):
            if pred.numel() == 0:
                continue
            bbox = pred[:, :4].to(metrics_dtype)
            scores = pred[:, 4].to(metrics_dtype)
            labels = pred[:, 5].to(torch.long)

            # 按照类别提取出来对应的预测结果
            bbox_cls = bbox[labels == c]
            scores_cls = scores[labels == c]

            # 此时bbox_cls, scores_cls都是某张图片的预测结果中某个类别的结果
            # 添加到pred_bbox_pre_img要以每个结果为单位，而不是以每个图片为单位
            for k in range(bbox_cls.shape[0]):
                pred_bbox_pre_img.append([img_id, float(scores_cls[k].item()), bbox_cls[k]])
                # pred_bbox_pre_img_count += 1

        # 此时针对的是 有GT但是没有pred
        if len(pred_bbox_pre_img) == 0:
            # 意味着所有的TP都是0，ap自然是0
            ap_thresh_cls[:, c] = 0
            continue

        # 按照score,从大到小排序
        # def return_score(x):
        #     return x[1]
        # pred_bbox_pre_img.sort(key=return_score, reverse=True)
        pred_bbox_pre_img.sort(key=lambda x: x[1], reverse=True)

        # 按照iou_thresh中逐个计算
        for iou_idx, iou_thresh in enumerate(iou_threshs):
            # gt的标记，确保一个类别中的gt只能被使用一次
            matched_flags = []
            for gt_bbox in gt_bbox_pre_img:
                # gt_bbox: [nums, 4] 含义为某张图片的gt信息
                matched_flags.append(torch.zeros(gt_bbox.shape[0], dtype=torch.bool))

            TP = torch.zeros(len(pred_bbox_pre_img), dtype=metrics_dtype)
            FP = torch.zeros(len(pred_bbox_pre_img), dtype=metrics_dtype)

            for i, (img_id, score, pred_bbox) in enumerate(pred_bbox_pre_img):
                gt_bbox = gt_bbox_pre_img[img_id]

                if gt_bbox.numel() == 0:
                    # 没有gt，则FP+1
                    FP[i] = 1.0
                    continue
                # 计算某个pred出来的bbox和全部的gt_bbox的iou结果
                ious = box_iou(pred_bbox, gt_bbox)
                # ic(ious.shape)
                max_iou, max_idx = ious.max(dim=0)

                # 判定TP iou > iou_thresh 并且 该gt没有被使用过
                iou_bool = float(max_iou.item()) >= iou_thresh
                matched_bool = (matched_flags[img_id][max_idx] == 0)
                if iou_bool and matched_bool:
                    # 匹配成功，TP+1，gt标记为已使用
                    TP[i] = 1.0
                    matched_flags[img_id][max_idx] = 1
                else:
                    # 没有匹配成功，FP+1
                    FP[i] = 1.0

            # TP_cumulative
            TP_cum = torch.cumsum(TP, dim=0)
            # FP_cumulative
            FP_cum = torch.cumsum(FP, dim=0)
            # 某个类别的某个iou阈值下的 Recall
            # 因为前面已经按照全概率从大到小排序过了，
            # 所以现在的Recall和Precesion里面的内容就是排序之后的。
            Recall = TP_cum / (gt_bbox_pre_img_count + eps)
            Precesion = TP_cum / (TP_cum + FP_cum + eps)

            AP = compute_ap(Recall, Precesion)
            # 储存每个iou_idx下每个类别的AP
            ap_thresh_cls[iou_idx, c] = AP

    # 所有类别计算完成之后,求均值
    map_pre_iou = {}
    for iou_idx, iou_thresh in enumerate(iou_threshs):
        AP_cls = ap_thresh_cls[iou_idx]
        # 排除掉异常值
        val_mask = torch.isfinite(AP_cls)
        AP_cls_vaild = AP_cls[val_mask]
        MAP_cls = AP_cls_vaild.mean().item()
        map_pre_iou[iou_thresh] = MAP_cls
        
    if len(map_pre_iou) > 0:
        map_50_95 = torch.tensor(list(map_pre_iou.values()), dtype=metrics_dtype).mean().item()
    else:
        map_50_95 = 0.0

    metrice_dict = {
        "map50": map_pre_iou.get(0.50, 0.0),
        "map50-95": map_50_95,
        "map_pre_iou": map_pre_iou,
    }
    return metrice_dict
    

if __name__ == "__main__":
    test_mode = "full"
    if test_mode == "calculate_ap":
        # test for ap
        # 用例 1：常见情况（缺 0/1 + 有重复 recall）
        # 0.49
        # rec = [0.10, 0.40, 0.40, 0.70]
        # prec = [1.00, 0.80, 0.60, 0.50]
        # AP = 0.670000
        rec = [0.20, 0.35, 0.90]
        prec = [0.90, 0.50, 0.70]
        # 0.48
        rec = [0.60]
        prec = [0.80]
        ap = compute_ap(torch.tensor(rec), torch.tensor(prec))
        ic(ap)

    if test_mode == "full":
        test_preds_nms_all = [torch.randn(5, 6),
                            torch.randn(3, 6),]
        test_gts_all = [torch.randn(4, 5),
                        torch.randn(2, 5),]
        metrice_dict = compute_map(test_preds_nms_all, test_gts_all, num_classes=20)
        ic(metrice_dict)
