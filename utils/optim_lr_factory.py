import torch
import torch.optim as optim

import torch.nn as nn
from icecream import ic
'''
优化器和学习率的工厂函数
目前支持的优化器
SGD
Adam

目前支持的学习率调度器
StepLR
CosineAnnealingLR

'''
def build_optimizer(model, cfg):
    optimizer = cfg["optimizer"]["type"]

    if optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    elif optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    else:
        raise ValueError(f"❗ Unsupported optimizer type: {optimizer}")

    return optimizer

def build_lr_scheduler(optimizer, cfg):
    lr_scheduler = cfg["optimizer"]["lr_scheduler"]["type"]

    if lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["optimizer"]["lr_scheduler"]["step_size"],
            gamma=cfg["optimizer"]["lr_scheduler"]["gamma"],
        )

    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["optimizer"]["lr_scheduler"]["T_max"],
            eta_min=cfg["optimizer"]["lr_scheduler"]["eta_min"],
        )

    elif lr_scheduler == "YOLOv1DetLR":
        # -----------------------------
        # YOLOv1 检测阶段：warmup + 分段常数
        # -----------------------------
        sch_cfg = cfg["optimizer"]["lr_scheduler"]

        # 你调试时可以开 ic，正式训练建议关掉
        # ic(sch_cfg)

        lr_warmup_start = float(sch_cfg.get("lr_warmup_start", 1e-3))
        lr_base = float(sch_cfg.get("lr_base", 1e-2))

        warmup_epochs = int(sch_cfg.get("warmup_epochs", 5))
        phase1_epochs = int(sch_cfg.get("phase1_epochs", 75))
        phase2_epochs = int(sch_cfg.get("phase2_epochs", 30))
        phase3_epochs = int(sch_cfg.get("phase3_epochs", 30))

        if warmup_epochs < 0:
            raise ValueError("warmup_epochs 必须 >= 0")
        if phase1_epochs <= 0:
            raise ValueError("phase1_epochs 必须 > 0")
        if warmup_epochs > 0 and phase1_epochs < warmup_epochs:
            raise ValueError("phase1_epochs 必须 >= warmup_epochs（phase1 含 warmup）")

        # 关键校验：LambdaLR 返回的是“倍率”，会乘以 optimizer 的 lr
        # 所以 optimizer 的初始 lr 必须等于 lr_base，否则比例会错
        opt_lr0 = float(optimizer.param_groups[0]["lr"])
        if abs(opt_lr0 - lr_base) / max(lr_base, 1e-12) > 1e-6:
            raise ValueError(
                f"optimizer 初始 lr={opt_lr0} 与 lr_base={lr_base} 不一致。"
                "请把 cfg['optimizer']['lr'] 设为 lr_base。"
            )

        def lr_lambda(epoch: int) -> float:
            """
            入口：
                epoch: int（0-based）
            出口：
                scale: float（乘在 optimizer.base_lr 上的倍率）
            """
            # -------- warmup：epoch=0 就是 warmup_start（不再用 epoch+1）--------
            if warmup_epochs > 0 and epoch < warmup_epochs:
                # t in [0, 1)
                t = float(epoch) / float(warmup_epochs)
                lr = lr_warmup_start + (lr_base - lr_warmup_start) * t
                return lr / lr_base

            # -------- phase1：到 phase1_epochs-1 都是 lr_base --------
            if epoch < phase1_epochs:
                return 1.0

            # -------- phase2：lr_base * 0.1 --------
            if epoch < phase1_epochs + phase2_epochs:
                return 0.1

            # -------- phase3：lr_base * 0.01 --------
            if epoch < phase1_epochs + phase2_epochs + phase3_epochs:
                return 0.01

            # 超出计划：保持最后一段
            return 0.01

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



    else:
        raise ValueError(f"❗ Unsupported lr_scheduler type: {lr_scheduler}")
    
    return scheduler

def build_loss_fn(cfg):
    loss_fu = cfg["loss_fn"]

    if loss_fu == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()

    else:
        raise ValueError(f"❗ Unsupported loss function type: {loss_fu}")
    
    return loss_function

