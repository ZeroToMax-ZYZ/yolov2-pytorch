import torch

from dataset.build_dataset import build_dataset

from nets.build_model import build_model
from dataset.build_dataset import build_dataset
from pre_weights.load_preweights import load_yolov2_backbone_pretrained_to_detector

from utils.optim_lr_factory import build_optimizer, build_lr_scheduler
from utils.loss import Yolov2Loss
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_logger, save_config




import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 获取当前device0的显卡型号
    GPU_model = torch.cuda.get_device_name(0)
    config = {
        "exp_time": exp_time,
        "GPU_model": GPU_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "exp1_full",
        "model_name": "YOLOv2",
        "save_interval": 10,
        # dataset config
        # D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset
        # /root/autodl-tmp/dataset_full/YOLOv1_dataset/train
        "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train",
        "test_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\test",
        "anchors_json": r"dataset/anchors_k5.json",
        "stride": 32,
        
        "pre_weights": r"pre_weights/best_model.pth",
        # test model 
        "debug_mode": 0.2, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "num_classes": 20,
        "input_size": 448,
        "batch_size": 32,
        "epochs": 135,
        "metric_interval": 5, # 每间隔几轮评估一次
        "num_workers": 8,
        "persistent_workers": True,
        "S": 7,
        "B": 2,
        # loss config
        "loss_dtype": torch.float32,
        "lambda_noobj": 0.5,
        "lambda_obj": 5,
        "lambda_prior": 0.01,
        "lambda_coord": 1,
        "lambda_cls": 1,
        "ignore_threshold": 0.6, # loss part1 参数

        "profile_time" : False,
        "profile_cuda_sync" : False,
        "optimizer": {
            "type": "SGD",
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            # "lr_scheduler":{
            #     "type": "StepLR",
            #     "step_size": 30,
            #     "gamma": 0.1,
            # }
            
            "lr_scheduler": {
                "type": "YOLOv1DetLR",
                "lr_warmup_start": 0.0001,
                "lr_base": 1e-3,
                "warmup_epochs": 10,
                "phase1_epochs": 75,
                "phase2_epochs": 30,
                "phase3_epochs": 30,
            },
        },
        # "optimizer": {
        #     "type": "Adam",
        #     "lr": 0.0001,
        #     "lr_scheduler": {
        #         "type": "CosineAnnealingLR",
        #         "T_max": 100,
        #         "eta_min": 1e-6,
        #     },
        #     "weight_decay": 1e-4,
        # }

    }
    config["exp_name"] += str("_" + exp_time)
    return config

def train():
    state = None
    cfg = base_config()
    model = build_model(cfg).to(cfg["device"])
    # 加载预训练权重
    if cfg["pre_weights"] is not None:
        load_yolov2_backbone_pretrained_to_detector(
        detector=model,
        ckpt_path=cfg["pre_weights"],
        backbone_module_names=("stage1", "stage2", "pool5", "stage3"),
        map_location="cpu",
        strict=False,
        verbose=True,
        auto_strip_common_root=True,
    )
    
    train_loader, test_loader, extra = build_dataset(cfg)
    # add train size
    cfg["train_size"] = int(len(train_loader))
    # save config
    save_config(cfg)

    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg)
    loss_fn = Yolov2Loss(cfg, ic_debug=True)

    for epoch in range(cfg["epochs"]):
        extra["train_batch_sampler"].set_epoch(epoch)
        metrics, state= fit_one_epoch(
            epoch, cfg, model, train_loader, test_loader, loss_fn, optimizer, lr_scheduler, state
        )
        # save logs and model
        save_logger(model, metrics, cfg, state)


if __name__ == '__main__':
    train()