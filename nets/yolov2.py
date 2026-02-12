import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
from icecream import ic


class CBR(nn.Module):
    """
    Conv + BN + LeakyReLU
    """
    def __init__(self, in_c: int, out_c: int, ksize: int, stride: int, pad: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, ksize, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Reorg(nn.Module):
    """
    YOLOv2 passthrough 的 reorg 层(stride=2)
    输入:  (N, C, H, W)  要求 H,W 能被 stride 整除
    输出:  (N, C*stride*stride, H/stride, W/stride)
    在yolov2中,经过passthrough处理后,通道数变为原来的4倍,尺寸变为原来的一半
    作用：
    - 将高分辨率特征(比如 26x26)“重排”到低分辨率(13x13)并增加通道数,
      从而与深层特征做 concat(passthrough / route)。
    """
    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8, 64, 26, 26
        n, c, h, w = x.shape
        s = self.stride
        assert h % s == 0 and w % s == 0, f"[Reorg] H,W 必须可被 stride={s} 整除,got {(h,w)}"
        # before: (bs, c, h, w) (bs, 64, 26, 26)
        # (N, C, H/s, s, W/s, s) (bs, 64, 13, 2, 13, 2)
        x = x.view(n, c, h // s, s, w // s, s)
        # (N, C, s, s, H/s, W/s) (bs, 64, 2, 2, 13, 13)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (N, C*s*s, H/s, W/s) (bs, 256, 13, 13)
        x = x.view(n, c * s * s, h // s, w // s)
        return x


class YOLOv2(nn.Module):
    """
    YOLOv2 检测网络(Darknet-19 backbone + passthrough head)

    入口:
        x: (N, 3, H, W)  检测训练通常用 H=W=416(或其它 32 的倍数)
    出口:
        若 reshape_output=True:
            out: (N, S, S, A, 5 + C)
                 其中 5+C = [t_x, t_y, t_w, t_h, t_o, cls_logits...]
                 注意：这里保持 raw 预测(loss 里再做 sigmoid/softmax 等监督空间对齐)
        若 reshape_output=False:
            out: (N, A*(5+C), S, S)

    说明:
        - backbone 对齐 YOLOv2 论文中的 Darknet-19(5 次下采样,stride=32)
        - head 使用 passthrough：从 stride=16 的特征(26x26, 512c) -> 1x1 降维到 64c -> reorg -> 13x13, 256c
          与 stride=32 的特征(13x13, 1024c) concat -> 1280c -> conv -> 输出
    """
    def __init__(
        self,
        num_classes: int = 20,
        num_anchors: int = 5,
        ic_debug: bool = False,
        reshape_output: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.ic_debug = ic_debug
        self.reshape_output = reshape_output

        # -------------------------
        # Darknet-19 Backbone
        # -------------------------
        # 输入 416x416 时：
        # after pool1 -> 208
        # after pool2 -> 104
        # after pool3 -> 52
        # after pool4 -> 26   (passthrough 源)
        # after pool5 -> 13   (主干输出)
        self.stage1 = nn.Sequential(
            CBR(3, 32, 3, 1, 1),          # 32*416*416
            nn.MaxPool2d(2, 2),           # 32*208*208
            CBR(32, 64, 3, 1, 1),         # 64*208*208
            nn.MaxPool2d(2, 2),           # 64*104*104
            CBR(64, 128, 3, 1, 1),        # 128*104*104
            CBR(128, 64, 1, 1, 0),        # 64*104*104
            CBR(64, 128, 3, 1, 1),        # 128*104*104
            nn.MaxPool2d(2, 2),           # 128*52*52
            CBR(128, 256, 3, 1, 1),       # 256*52*52
            CBR(256, 128, 1, 1, 0),       # 128*52*52
            CBR(128, 256, 3, 1, 1),       # 256*52*52
            nn.MaxPool2d(2, 2),           # 256*26*26
        )

        # 这里输出是 26x26 的 512c(passthrough 源)
        self.stage2 = nn.Sequential(
            CBR(256, 512, 3, 1, 1),       # 512*26*26
            CBR(512, 256, 1, 1, 0),       # 256*26*26
            CBR(256, 512, 3, 1, 1),       # 512*26*26
            CBR(512, 256, 1, 1, 0),       # 256*26*26
            CBR(256, 512, 3, 1, 1),       # 512*26*26
        )

        self.pool5 = nn.MaxPool2d(2, 2)   # 512*13*13

        # 这里输出是 13x13 的 1024c(主干输出)
        self.stage3 = nn.Sequential(
            CBR(512, 1024, 3, 1, 1),      # 1024*13*13
            CBR(1024, 512, 1, 1, 0),      # 512*13*13
            CBR(512, 1024, 3, 1, 1),      # 1024*13*13
            CBR(1024, 512, 1, 1, 0),      # 512*13*13
            CBR(512, 1024, 3, 1, 1),      # 1024*13*13
        )

        # -------------------------
        # YOLOv2 Head (passthrough)
        # -------------------------
        # passthrough: 26x26,512 -> 26x26,64 -> reorg -> 13x13,256
        self.passthrough_conv = CBR(512, 64, 1, 1, 0)
        self.reorg = Reorg(stride=2)

        # 主分支：13x13,1024 -> 再堆叠两层 3x3
        self.head_conv = nn.Sequential(
            CBR(1024, 1024, 3, 1, 1),
            CBR(1024, 1024, 3, 1, 1),
        )

        # concat 后：1024 + 256 = 1280
        self.fuse_conv = CBR(1024 + 256, 1024, 3, 1, 1)

        out_ch = self.num_anchors * (5 + self.num_classes)
        # 最后一层保持 raw logits(不加 BN/激活),对齐 YOLOv2 的实现习惯
        self.pred = nn.Conv2d(1024, out_ch, kernel_size=1, stride=1, padding=0)

    def _reshape_pred(self, p: torch.Tensor) -> torch.Tensor:
        """
        将 (N, A*(5+C), S, S) -> (N, S, S, A, 5+C)
        """
        n, ch, s, _ = p.shape
        a = self.num_anchors
        k = 5 + self.num_classes
        assert ch == a * k, f"[YOLOv2] pred channels mismatch: {ch} vs {a}*{k}"
        p = p.view(n, a, k, s, s).permute(0, 3, 4, 1, 2).contiguous()
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone
        x = self.stage1(x)          # -> 26x26,256
        x_26 = self.stage2(x)       # -> 26x26,512  (passthrough 源)
        x_13 = self.pool5(x_26)     # -> 13x13,512
        x_13 = self.stage3(x_13)    # -> 13x13,1024

        # head 主分支
        x_head = self.head_conv(x_13)  # -> 13x13,1024

        # passthrough 分支
        x_pt = self.passthrough_conv(x_26)  # 26x26,64
        x_pt = self.reorg(x_pt)             # 13x13,256

        # concat + fuse
        x_fuse = torch.cat([x_pt, x_head], dim=1)  # 13x13,1280
        x_fuse = self.fuse_conv(x_fuse)            # 13x13,1024

        # pred
        p = self.pred(x_fuse)  # (N, A*(5+C), S, S)

        if self.ic_debug:
            print("=== YOLOv2 Debug Shapes ===")
            ic(x_26.shape)     # passthrough 源
            ic(x_13.shape)     # backbone 输出
            ic(x_pt.shape)     # reorg 后
            ic(x_head.shape)   # head 输出
            ic(p.shape)        # raw pred

        if self.reshape_output:
            return self._reshape_pred(p)  # (N,S,S,A,5+C)
        return p


class YOLOv2_Classifier(nn.Module):
    """
    YOLOv2 用于 ImageNet 预训练的分类头(复用 Darknet-19 backbone)

    入口:
        x: (N,3,H,W)  常用 224x224
    出口:
        logits: (N, num_classes)

    说明:
        - 直接拿 YOLOv2 的 backbone(到 stage3 结束)
        - GAP + FC 做分类
    """
    def __init__(self, num_classes: int = 1000, ic_debug: bool = False, dropout_p: float = 0.1):
        super().__init__()
        self.ic_debug = ic_debug

        # 复用 YOLOv2 的 backbone 部分(不含检测头)
        backbone = YOLOv2(num_classes=20, num_anchors=5, ic_debug=False, reshape_output=False)
        self.stage1 = backbone.stage1
        self.stage2 = backbone.stage2
        self.pool5 = backbone.pool5
        self.stage3 = backbone.stage3

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool5(x)
        x = self.stage3(x)          # (N,1024,13,13)

        if self.ic_debug:
            print("=== YOLOv2_Classifier Debug Shapes ===")
            ic(x.shape)

        x = self.gap(x)             # (N,1024,1,1)
        x = x.view(x.shape[0], -1)  # (N,1024)
        x = self.drop(x)
        x = self.fc(x)              # (N,num_classes)

        if self.ic_debug:
            ic(x.shape)
        return x


if __name__ == "__main__":

    mode = "det"  # "det" or "cls"

    if mode == "det":
        model = YOLOv2(num_classes=20, num_anchors=5, ic_debug=False, reshape_output=True)
        x = torch.randn(2, 3, 416, 416)
        y = model(x)
        # y: (N,13,13,5,25)  25 = 5 + 20
        ic(y.shape)

    if mode == "cls":
        model = YOLOv2_Classifier(num_classes=1000, ic_debug=True)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        ic(y.shape)
