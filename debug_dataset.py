from dataset.build_dataset import build_dataset
import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "exp_name": "exp2_5090_448_full",
    "model_name": "YOLOv2",
    "save_interval": 10,
    # dataset config
    # D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset
    # /root/autodl-tmp/dataset_full/YOLOv1_dataset/train
    # /root/autodl-tmp/YOLOv1_dataset
    "train_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\train",
    "test_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\YOLOv1_dataset\test",
    "anchors_json": r"dataset/anchors_k5.json",
    "stride": 32,
    "pre_weights": r"pre_weights/best_model_448.pth",
    # test model 
    "debug_mode": None, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
    "num_classes": 20,
    "input_size": 448,
    "batch_size": 64,
    "epochs": 160,
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

    "nms_device": "cpu",
    "profile_time" : False,
    "profile_cuda_sync" : False,
    "optimizer": {
        "type": "SGD",
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_scheduler": {
            "type": "YOLOv2DetLR",
            "lr_warmup_start": 0.0001,
            "lr_base": 1e-3,
            "warmup_epochs": 2,
            "phase1_epochs": 60,
            "phase2_epochs": 90,
        },
    },
}

train_loader, val_loader, extra_info = build_dataset(config)
print(extra_info)