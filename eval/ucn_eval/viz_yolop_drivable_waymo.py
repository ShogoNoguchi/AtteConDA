#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Waymo + YOLOP ドライバブル可視化専用スクリプト (viz-only)
- Uni-ControlNet の EX1 / EX2 とは独立に、Drivable マスクの可視化だけを再計算する
- --viz-only フラグが指定されているときのみ実行

使い方 (EX1 の例):
python viz_yolop_drivable_waymo.py \
  --viz-only \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root /home/shogo/coding/datasets/WaymoV2/UniControlNet_offline \
  --splits training validation testing \
  --camera front \
  --out-root /data/ucn_eval_cache_ex1/viz_ex1_drivable_fixed \
  --samples-per-split 4
"""

import argparse
import logging
import os
from logging import handlers
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _setup_logger(out_root: str) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "viz_yolop_drivable.log")
    logger = logging.getLogger("viz_yolop_drivable")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        )
    )

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


# ============================================================
# Waymo パス列挙
# ============================================================

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _list_waymo_pairs(
    orig_root: str, gen_root: str, split: str, camera: str
) -> List[Tuple[str, str]]:
    """
    元 RGB と生成画像 F(X) のパスを対応付けて列挙
    - orig_root/{split}/{camera}/{segment}/{stem}.jpg
    - gen_root/{split}/{camera}/{segment}/{stem}_ucn.png
    """
    base_orig = os.path.join(orig_root, split, camera)
    base_gen = os.path.join(gen_root, split, camera)
    pairs: List[Tuple[str, str]] = []

    if not os.path.isdir(base_orig):
        return pairs

    for root, _, files in os.walk(base_orig):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in ALLOWED_IMG_EXT:
                continue
            orig_path = os.path.join(root, fname)

            rel_dir = os.path.relpath(root, base_orig)
            gen_dir = os.path.join(base_gen, rel_dir)
            gen_path = os.path.join(gen_dir, stem + "_ucn.png")
            if os.path.exists(gen_path):
                pairs.append((orig_path, gen_path))

    return sorted(pairs)


# ============================================================
# YOLOP 推論 (Drivable)
# ============================================================

def _load_yolop(logger: logging.Logger) -> torch.nn.Module:
    logger.info("[YOLOP] load via torch.hub hustvl/yolop (pretrained=True, trust_repo=True)")
    model = torch.hub.load(
        "hustvl/yolop",
        "yolop",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _prep_img_for_yolop(bgr: np.ndarray) -> torch.Tensor:
    """
    YOLOP への簡易前処理:
    - 640x384 にリサイズ
    - RGB に変換
    - ToTensor + ImageNet 正規化
    """
    h, w = bgr.shape[:2]
    img_resized = cv2.resize(bgr, (640, 384), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    tensor = transform(rgb)  # (3, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def _infer_drivable_mask(
    model: torch.nn.Module, bgr: np.ndarray, prob_thr: float = 0.5
) -> np.ndarray:
    """
    YOLOP から drivable マスク (bool HxW) を取り出す。
    - da_seg_out: (B, 2, H, W) のログイットを想定 (channel 1 = drivable)
    """
    with torch.no_grad():
        inp = _prep_img_for_yolop(bgr)
        det_out, da_seg_out, ll_seg_out = model(inp)

        # da_seg_out: (1, 2, H, W)
        da_logits = da_seg_out[0]  # (2, H, W)
        da_prob = torch.softmax(da_logits, dim=0)  # (2, H, W)
        drivable_prob = da_prob[1]  # channel 1 を drivable と仮定
        mask = (drivable_prob > prob_thr).cpu().numpy().astype(np.uint8)  # {0,1}

    return mask  # 640x384


# ============================================================
# 可視化
# ============================================================

def _overlay_drivable(
    bgr: np.ndarray, mask_640x384: np.ndarray, color_bgr: Tuple[int, int, int]
) -> np.ndarray:
    """
    - 入力 bgr: 元画像 (任意解像度)
    - mask_640x384: YOLOP 出力の 640x384 マスク (0 or 1)
    - 出力: bgr と同じ解像度の overlay 画像
    """
    h, w = bgr.shape[:2]
    mask_resized = cv2.resize(
        mask_640x384.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_bool = mask_resized.astype(bool)

    overlay = bgr.copy()
    color = np.array(color_bgr, dtype=np.uint8)

    alpha = 0.6  # 色の強さ
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * color.astype(np.float32)
    ).astype(np.uint8)

    return overlay


def _viz_pair(
    model: torch.nn.Module,
    orig_path: str,
    gen_path: str,
    out_path: str,
    logger: logging.Logger,
) -> None:
    orig_bgr = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    gen_bgr = cv2.imread(gen_path, cv2.IMREAD_COLOR)
    if orig_bgr is None or gen_bgr is None:
        logger.warning("画像が読めません: %s / %s", orig_path, gen_path)
        return

    # YOLOP は 640x384 入力で処理
    mask_orig = _infer_drivable_mask(model, orig_bgr)
    mask_gen = _infer_drivable_mask(model, gen_bgr)

    overlay_orig = _overlay_drivable(orig_bgr, mask_orig, (255, 255, 0))  # シアン系
    overlay_gen = _overlay_drivable(gen_bgr, mask_gen, (0, 255, 255))    # イエロー系

    # 高さを合わせて横並び
    h = min(overlay_orig.shape[0], overlay_gen.shape[0])
    overlay_orig_r = cv2.resize(
        overlay_orig, (int(overlay_orig.shape[1] * h / overlay_orig.shape[0]), h)
    )
    overlay_gen_r = cv2.resize(
        overlay_gen, (int(overlay_gen.shape[1] * h / overlay_gen.shape[0]), h)
    )
    concat = np.concatenate([overlay_orig_r, overlay_gen_r], axis=1)

    _ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, concat)
    logger.info("viz saved: %s", out_path)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="YOLOP-based drivable visualization for Waymo (viz-only)"
    )
    ap.add_argument(
        "--viz-only",
        action="store_true",
        help="このフラグが指定されているときのみ可視化を実行する安全スイッチ。",
    )
    ap.add_argument(
        "--orig-root",
        type=str,
        required=True,
        help="元 RGB 画像のルート (Waymo extracted)",
    )
    ap.add_argument(
        "--gen-root",
        type=str,
        required=True,
        help="生成画像 F(X) のルート (Uni-ControlNet 推論結果)",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training", "validation", "testing"],
        help="対象とする split のリスト",
    )
    ap.add_argument(
        "--camera",
        type=str,
        default="front",
        help="カメラ名 (front 等)",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="可視化結果を書き出すルート",
    )
    ap.add_argument(
        "--samples-per-split",
        type=int,
        default=4,
        help="各 split ごとに可視化するペア数 (0 なら全件)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.viz_only:
        print("安全のため、--viz-only が指定されたときのみ実行します。")
        return

    logger = _setup_logger(args.out_root)
    logger.info("==== Drivable viz-only モード開始 ====")
    logger.info("orig-root: %s", args.orig_root)
    logger.info("gen-root : %s", args.gen_root)
    logger.info("splits   : %s", " ".join(args.splits))
    logger.info("camera   : %s", args.camera)
    logger.info("out-root : %s", args.out_root)

    yolop = _load_yolop(logger)

    total_viz = 0
    for split in args.splits:
        pairs = _list_waymo_pairs(args.orig_root, args.gen_root, split, args.camera)
        logger.info("[%s] pairs=%d", split, len(pairs))
        if not pairs:
            continue

        if args.samples_per_split > 0:
            pairs = pairs[: args.samples_per_split]

        for orig_path, gen_path in pairs:
            rel = os.path.relpath(orig_path, os.path.join(args.orig_root, split, args.camera))
            rel_dir = os.path.dirname(rel)
            stem = os.path.splitext(os.path.basename(orig_path))[0]
            out_path = os.path.join(
                args.out_root,
                split,
                args.camera,
                rel_dir,
                f"{stem}_drivable_viz.jpg",
            )
            _viz_pair(yolop, orig_path, gen_path, out_path, logger)
            total_viz += 1

    logger.info("総可視化ペア数: %d", total_viz)
    logger.info("==== Drivable viz-only 完了 ====")


if __name__ == "__main__":
    main()
