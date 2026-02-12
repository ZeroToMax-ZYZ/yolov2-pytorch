import torch
# from utils.logger import logger

from utils.metrics import compute_map
from utils.decode import decode_preds, decode_labels_list
from utils.nms import nms

from dataclasses import dataclass

import time
from tqdm import tqdm
from icecream import ic


@dataclass
class Checkpoint:
    train_map50: float = 0.0
    train_map50_95: float = 0.0
    val_map50: float = 0.0
    val_map50_95: float = 0.0
    

def fit_train_epoch(epoch, cfg, model, train_loader, loss_fn, optimizer):
    '''
    '''
    model.train()

    train_loss = 0.0
    samples = 0
    epoch_preds = []
    epoch_gts = []

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")

    for images, labels in train_bar:
        bs = images.shape[0]
        samples += bs

        img = images.to(cfg["device"])
        label = labels.to(cfg["device"])

        outputs = model(img)
        loss = loss_fn(outputs, label)


        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ''' 
        outputs: (bs, S, S, (B*5+num_classes)) ([64, 7, 7, 30])
        label: (bs, S, S, (5+num_classes)) ([64, 7, 7, 25])
        out_decode:  ([64, 7, 7, 2, 25])
        '''

        if (epoch + 1) % cfg["metric_interval"] == 0:
            with torch.no_grad():

                out_decode = decode_preds(outputs.detach(), B=2, conf_thresh=0.01)
                out_boxes = nms(out_decode)
                epoch_preds.extend([b.detach().cpu() for b in out_boxes])

                label_decode = decode_labels_list(label.detach())
                epoch_gts.extend([b.detach().cpu() for b in label_decode])
                
        # 恢复到整个bs的损失
        train_loss += loss.item() * bs

        # updata bar
        train_bar.set_postfix(loss=f"{train_loss/(samples):.4f}")
    
    if (epoch + 1) % cfg["metric_interval"] == 0:
        metrics_dict = compute_map(epoch_preds, epoch_gts, cfg["num_classes"], metrics_dtype=torch.float32, eps=1e-6)
    else:
        metrics_dict = {}

    epoch_loss = train_loss / samples

    return epoch_loss, metrics_dict

def fit_val_epoch(epoch, cfg, model, val_loader, loss_fn):
    '''
    return: epoch_loss, epoch_top1(0-1), epoch_top5(0-1)
    '''
    model.eval()

    val_loss = 0.0
    samples = 0
    epoch_preds = []
    epoch_gts = []

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]")

    with torch.no_grad():
        for images, labels in val_bar:
            bs = images.shape[0]
            samples += bs

            img = images.to(cfg["device"])
            label = labels.to(cfg["device"])

            outputs = model(img)
            loss = loss_fn(outputs, label)
            if (epoch + 1) % cfg["metric_interval"] == 0:
                
                out_decode = decode_preds(outputs.detach(), B=2, conf_thresh=0.01)
                out_boxes = nms(out_decode)
                epoch_preds.extend([b.detach().cpu() for b in out_boxes])

                label_decode = decode_labels_list(label.detach())
                epoch_gts.extend([b.detach().cpu() for b in label_decode])

            val_loss += loss.item() * bs

            # update bar
            val_bar.set_postfix(loss=f"{val_loss/(samples):.4f}")

        if (epoch + 1) % cfg["metric_interval"] == 0:
            metrics_dict = compute_map(epoch_preds, epoch_gts, cfg["num_classes"], metrics_dtype=torch.float32, eps=1e-6)
        else:
            metrics_dict = {}

        epoch_loss = val_loss / samples


        return epoch_loss, metrics_dict


def fit_one_epoch(epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, state=None):
    '''
    return: train_loss, train_top1(0-1), train_top5(0-1),
            val_loss, val_top1(0-1), val_top5(0-1)
    '''
    # yolov1的学习率策略应当把scheduler放到开头，第一个epoch的lr就应当是warmup的
    lr_scheduler.step(epoch)

    if state is None:
        state = Checkpoint()

    start_time = time.time()
    train_loss, train_metrics = fit_train_epoch(
        epoch, cfg, model, train_loader, loss_fn, optimizer,
    )
    val_loss, val_metrics = fit_val_epoch(
        epoch, cfg, model, val_loader, loss_fn,
    )


    end_time = time.time()
    epoch_time = end_time - start_time # (s)


    if (epoch + 1) % cfg["metric_interval"] == 0:
        state.train_map50 = train_metrics.get("map50", 0.0)
        state.train_map50_95 = train_metrics.get("map50-95", 0.0)
        state.val_map50 = val_metrics.get("map50", 0.0)
        state.val_map50_95 = val_metrics.get("map50-95", 0.0)


    metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_map50": state.train_map50,
        "train_map50_95": state.train_map50_95,
        "val_loss": val_loss,
        "val_map50": state.val_map50,
        "val_map50_95": state.val_map50_95,
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": epoch_time,
    }
    return metrics, state