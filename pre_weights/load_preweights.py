# pre_weights/load_yolov2_preweights.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
from collections import Counter

import torch
import torch.nn as nn


def load_yolov2_backbone_pretrained_to_detector(
    detector: nn.Module,
    ckpt_path: str,
    *,
    backbone_module_names: Tuple[str, ...] = ("stage1", "stage2", "pool5", "stage3"),
    map_location: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
    auto_strip_common_root: bool = True,
) -> Dict[str, Any]:
    """
    入口：
        detector:
            - 你的 YOLOv2 检测模型实例（包含 stage1/stage2/pool5/stage3 这些骨干模块）
        ckpt_path:
            - 你训练好的 YOLOv2_Classifier 权重路径（.pt/.pth）
            - 支持以下保存格式：
                A) torch.save(model.state_dict())
                B) torch.save({"state_dict": model.state_dict()})
                C) torch.save({"model": model.state_dict()})
        backbone_module_names:
            - 认为“哪些模块属于骨干”，默认与你当前网络一致：
              ("stage1", "stage2", "pool5", "stage3")
        map_location:
            - torch.load 用的 map_location（建议默认 cpu 更稳）
        strict:
            - 是否严格匹配 detector 的全部参数（一般 strict=False，只加载骨干）
        verbose:
            - 是否打印加载统计信息
        auto_strip_common_root:
            - 是否自动剥掉公共根前缀，例如 "model." / "net." / "backbone." 这类
              （当你保存 ckpt 时外面包了一层命名空间，会很有用）

    出口：
        report: dict
            - total_src_keys: ckpt 原始 key 总数
            - used_root_prefix: 自动剥掉的公共根前缀（如果没剥则为空串）
            - total_candidate_backbone_keys: ckpt 中识别到的“骨干候选”key 数
            - total_loaded_keys: 实际加载到 detector 的 key 数
            - missing_keys / unexpected_keys: 来自 detector.load_state_dict 的返回
            - loaded_keys: 成功加载的 key 列表（可用于你人工 spot-check）
    """

    # -------------------------
    # 1) 读取 checkpoint / state_dict
    # -------------------------
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            # 可能本身就是 state_dict
            state = ckpt
    else:
        raise TypeError(f"[load_yolov2] ckpt 不是 dict，实际类型={type(ckpt)}")

    if not isinstance(state, dict):
        raise TypeError(f"[load_yolov2] state_dict 不是 dict，实际类型={type(state)}")

    # -------------------------
    # 2) 去掉 DataParallel 前缀：module.
    # -------------------------
    def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(k.startswith("module.") for k in sd.keys()):
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    state = _strip_module_prefix(state)

    # -------------------------
    # 3) 自动识别并剥掉“公共根前缀”（可选）
    #    例如：k="model.stage1.0.conv.weight" -> root="model."
    #         k="backbone.stage2.3.bn.weight" -> root="backbone."
    # -------------------------
    def _find_common_root_prefix(keys: List[str], module_names: Tuple[str, ...]) -> str:
        """
        找到形如 "<root><module>." 里最常见的 <root>。
        如果没有明显公共 root，则返回 ""。
        """
        candidates: List[str] = []
        for k in keys:
            for m in module_names:
                token = m + "."
                idx = k.find(token)
                if idx > 0:
                    candidates.append(k[:idx])  # 注意：这里包含末尾的 '.' 之前所有字符
        if not candidates:
            return ""

        cnt = Counter(candidates)
        root, freq = cnt.most_common(1)[0]

        # 一个简单阈值：至少命中一定数量，才认为它是“公共根”
        # 你也可以把这个阈值调严/调松
        if freq >= max(10, len(keys) // 20):
            return root
        return ""

    used_root_prefix = ""
    if auto_strip_common_root:
        keys_list = list(state.keys())
        root = _find_common_root_prefix(keys_list, backbone_module_names)
        if root:
            used_root_prefix = root
            new_state: Dict[str, torch.Tensor] = {}
            for k, v in state.items():
                if k.startswith(root):
                    new_state[k[len(root):]] = v
                else:
                    new_state[k] = v
            state = new_state

    # -------------------------
    # 4) 从 ckpt 里抽取“骨干候选参数”
    # -------------------------
    def _is_backbone_key(k: str, module_names: Tuple[str, ...]) -> bool:
        return any(k.startswith(m + ".") for m in module_names)

    candidate_backbone: Dict[str, torch.Tensor] = {
        k: v for k, v in state.items() if _is_backbone_key(k, backbone_module_names)
    }

    # 如果一个都没找到，做一次兜底：直接拿 key 交集（对齐 detector 的 state_dict）
    det_sd = detector.state_dict()
    if len(candidate_backbone) == 0:
        # 兜底策略：如果你的 ckpt 就是 detector 风格（或已经是 stage1.*），直接匹配即可
        candidate_backbone = {k: v for k, v in state.items() if k in det_sd}

    # -------------------------
    # 5) 按 key + shape 过滤（防止 silent mismatch）
    # -------------------------
    filtered_state: Dict[str, torch.Tensor] = {}
    skipped_shape_mismatch: List[str] = []
    for k, v in candidate_backbone.items():
        if k in det_sd:
            if det_sd[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped_shape_mismatch.append(k)

    # -------------------------
    # 6) 加载到 detector
    # -------------------------
    load_ret = detector.load_state_dict(filtered_state, strict=strict)

    missing_keys = list(load_ret.missing_keys) if hasattr(load_ret, "missing_keys") else []
    unexpected_keys = list(load_ret.unexpected_keys) if hasattr(load_ret, "unexpected_keys") else []

    loaded_keys = sorted(list(filtered_state.keys()))

    report: Dict[str, Any] = {
        "ckpt_path": ckpt_path,
        "total_src_keys": len(state),
        "used_root_prefix": used_root_prefix,
        "backbone_module_names": backbone_module_names,
        "total_candidate_backbone_keys": len(candidate_backbone),
        "total_loaded_keys": len(loaded_keys),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "skipped_shape_mismatch_keys": skipped_shape_mismatch[:50],  # 太长就截断
        "skipped_shape_mismatch_count": len(skipped_shape_mismatch),
    }

    if verbose:
        print("[load_yolov2_backbone_pretrained_to_detector] done")
        print(f"  ckpt_path                 : {ckpt_path}")
        print(f"  used_root_prefix          : {used_root_prefix if used_root_prefix else '(none)'}")
        print(f"  src_total_keys            : {report['total_src_keys']}")
        print(f"  candidate_backbone_keys   : {report['total_candidate_backbone_keys']}")
        print(f"  loaded_keys               : {report['total_loaded_keys']}")
        if report["skipped_shape_mismatch_count"] > 0:
            print(f"  shape_mismatch_skipped    : {report['skipped_shape_mismatch_count']} (top10 below)")
            print(f"    {report['skipped_shape_mismatch_keys'][:10]}")
        # missing_keys 里通常会包含 detector 的 head 参数（passthrough_conv/head_conv/fuse_conv/pred），这是正常的
        if len(missing_keys) > 0:
            print(f"  missing_keys (top10)      : {missing_keys[:10]}")
        if len(unexpected_keys) > 0:
            print(f"  unexpected_keys (top10)   : {unexpected_keys[:10]}")

    return report


# -------------------------
# 使用示例
# -------------------------
if __name__ == "__main__":
    from nets.yolov2 import YOLOv2  # 你按自己的工程路径修改

    detector = YOLOv2(num_classes=20, num_anchors=5, reshape_output=True)
    ckpt_path = "pre_weights/best_model.pth"  # 你训练分类骨干得到的 ckpt

    rep = load_yolov2_backbone_pretrained_to_detector(
        detector=detector,
        ckpt_path=ckpt_path,
        backbone_module_names=("stage1", "stage2", "pool5", "stage3"),
        map_location="cpu",
        strict=False,
        verbose=True,
        auto_strip_common_root=True,
    )

    # 你可以重点检查 rep["loaded_keys"] 是否覆盖 stage1/stage2/stage3 的 conv/bn 参数
