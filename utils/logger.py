import os
import torch
from matplotlib import pyplot as plt
import json

def save_csv(metrics, csv_path):
    # 字段列表
    fields = [
        "epoch",       # 整数
        "train_loss",  # 浮点数
        "train_map50",  # 浮点数
        "train_map50_95",  # 浮点数
        "val_loss",    # 浮点数
        "val_map50",    # 浮点数
        "val_map50_95",    # 浮点数
        "lr",          # 浮点数（学习率）
        "epoch_time"   # 浮点数（耗时）
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            header = ",".join(fields) + "\n"
            f.write(header)
    
    with open(csv_path, "a") as f:
        values = []
        for field in fields:
            data = metrics[field]
            # format fit
            if isinstance(data, int):
                values.append(str(data))

            elif isinstance(data, float):
                if field != "lr":
                    # 除学习率之外的数据保留5位小数
                    values.append("{:.5f}".format(data))
                else:
                    values.append(str(data))
            else:
                values.append(str(data))

        row = ",".join(values) + "\n"
        f.write(row)


def plot_metrics(cfg, csv_path, plt_path):
    # 1. 设置全局字体为 Times New Roman
    plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    
    epochs = []
    train_losses, val_losses = [], []
    train_top1s, val_top1s = [], []
    train_top5s, val_top5s = [], []

    # 数据读取
    with open(csv_path, "r") as f:
        next(f)  # 跳过表头
        for line in f:
            items = line.strip().split(",")
            if len(items) < 7: continue
            epoch, t_loss, t_top1, t_top5, v_loss, v_top1, v_top5, _, _ = items
            
            epochs.append(int(epoch))
            train_losses.append(float(t_loss))
            val_losses.append(float(v_loss))
            train_top1s.append(float(t_top1))
            val_top1s.append(float(v_top1))
            train_top5s.append(float(t_top5))
            val_top5s.append(float(v_top5))

    # 获取最高准确率用于标题展示
    max_val_top1 = max(val_top1s)
    max_val_top5 = max(val_top5s)

    plt.figure(figsize=(14, 6))

    # --- 左图：Loss ---
    plt.subplot(1, 2, 1)
    # marker='x' 标注数据点，markersize 控制大小
    plt.plot(epochs, train_losses, label='Train Loss', marker='x', markersize=4, linewidth=1)
    plt.plot(epochs, val_losses, label='test Loss', marker='x', markersize=4, linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.7) # 开启网格
    plt.legend()

    # --- 右图：Accuracy (map50 & map50-95) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_top1s, label='Train map50', marker='x', markersize=4, linewidth=1)
    plt.plot(epochs, val_top1s, label='test map50', marker='x', markersize=4, linewidth=1)
    plt.plot(epochs, train_top5s, label='Train map50-95', marker='x', markersize=4, linestyle='--')
    plt.plot(epochs, val_top5s, label='test map50-95', marker='x', markersize=4, linestyle='--')
    
    plt.xlabel('Epochs')
    plt.ylabel('map')
    
    # 动态标题：包含最高 map50 Acc
    plt.title(f'Accuracy (Max test map50: {max_val_top1:.2f}%)')
    
    plt.grid(True, linestyle='--', alpha=0.7) # 开启网格
    plt.legend()

    plt.tight_layout()
    plt.savefig(plt_path, dpi=300) # 建议增加 dpi 提高清晰度
    # plt.show()


def save_logger(model, metrics, cfg, state):
    base_logs_path = os.path.join("logs", "logs_upload", cfg["exp_name"])
    base_weights_path = os.path.join("logs", "logs_weights", cfg["exp_name"])

    csv_path = os.path.join(base_logs_path, "metrics.csv")
    plt_path = os.path.join(base_logs_path, "metrics.png")
    model_path = os.path.join(base_weights_path, "weights")

    save_csv(metrics, csv_path)
    plot_metrics(cfg, csv_path, plt_path)
    save_model(model, cfg, csv_path, model_path, metrics, state)



def save_model(model, cfg, csv_path, model_path, metrics, state):
    '''
    save best and last model
    and every cfg["save_interval"] epoch
    '''
    # newest metrics
    val_map50 = metrics["val_map50"]
    
    best_val_map50 = state.val_map50
    
    # best
    if val_map50 >= best_val_map50:
        torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
        print(f"✅ Best model saved with Val map50 Accuracy: {val_map50:.2f}%")
    
    # every save_interval
    if (metrics["epoch"] % cfg["save_interval"]) == 0:
        torch.save(model.state_dict(), os.path.join(model_path, f"model_epoch_{metrics['epoch']}_valtop1_{val_map50:.2f}.pth"))
        # print(f"Model saved at epoch {metrics['epoch']}")

    # last
    torch.save(model.state_dict(), os.path.join(model_path, "last_model.pth"))




def save_config(cfg):
    base_logs_path = os.path.join("logs", "logs_upload", cfg["exp_name"])
    base_weights_path = os.path.join("logs", "logs_weights", cfg["exp_name"], "weights")

    if not os.path.exists(base_logs_path):
        os.makedirs(base_logs_path)
    if not os.path.exists(base_weights_path):
        os.makedirs(base_weights_path)

    config_path = os.path.join(base_logs_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)