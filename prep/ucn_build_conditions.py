#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/home/shogo/coding/prep/ucn_build_conditions.py

学習用データセット（BDD10K / Cityscapes / GTA5 / nuImages(front) / BDD100K）のRGBから、
Uni-ControlNet 入力要件に合致する条件マップ（Depth / Edge / Semseg）を一括生成するユーティリティ。

- 実行環境: 既存 Docker 画像（ucn-eval）を流用（PyTorch 2.7.0+cu128, CUDA 12.8 固定）
- Depth: Metric3Dv2(ONNX, CUDA)の出力を「ロバストmin-max正規化→反転（奥=黒/手前=白）→ 3chグレースケールJPG」
- Edge: Canny(100,200,blur=3) → 3ch JPG（白エッジ/黒背景）
- Semseg: OneFormer(Cityscapes)の trainId を Cityscapes カラーマップで可視化 → JPG

出力（本番・チェック共通仕様）:
  /data/ucn_condmaps/{DATASET_KEY}/{depth|edge|semseg}/{rel_dir}/.../{stem}_{depth|edge|semseg}.jpg
  ※ Cityscapes / BDD10K(10K) / BDD100K は入力側に train/val/test があるため rel_dir に split が含まれます。
  ※ GTA5 / nuImages(front) は入力側に split が無い構造のため、rel_dir に split は入りません（互換維持）。
    - ただし、--force-train-subdir-singletons を付けると出力側に train/ を強制付与できます（オプション）。

ログ/TensorBoard:
  ログ:   /data/ucn_prep_cache/logs/ucn_build_conditions.log（回転）
  TB:     /data/ucn_prep_tb/ （--tb 指定時）

GPU厳格モード:
  CUDA必須。CPUフォールバックは明示的にエラー終了。

【本番運転の例（trainのみ/Depth+Edge+Semseg）】
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_conditions.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_prep_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs huggingface_hub>=0.34,<1.0 einops' \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_conditions.py:/app/ucn_build_conditions.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --datasets all --tasks all --semseg-batch-size 1

【サブセット一括チェック（要求の新機能）】
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_conditions.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_prep_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs huggingface_hub>=0.34,<1.0 einops' \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_conditions.py:/app/ucn_build_conditions.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --datasets all --tasks all --subset-check --verbose
"""

import os
import sys
import argparse
import logging
from logging import handlers
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import math
import time
from datetime import datetime
from collections import defaultdict

# ===== 数値・画像 =====
import numpy as np
import cv2

# ===== 進捗 =====
from tqdm import tqdm

# ===== Torch / Transformers =====
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, OneFormerForUniversalSegmentation

# ===== ONNX Runtime（CUDA専用） =====
import onnxruntime as ort


# ==============================
# 0) 既定パス（翔伍さん環境に完全整合）
# ==============================
# 入力データセット（RGB）
BDD10K10K_IMG_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images"  # {train,val}
CITYSCAPES_IMG_ROOT = "/home/shogo/coding/datasets/cityscapes/leftImg8bit"                    # {train,val,test}
GTA5_IMG_ROOT       = "/home/shogo/coding/datasets/GTA5/images/images"                        # 直下に *.png
NUIMAGES_FRONT_ROOT = "/home/shogo/coding/datasets/nuimages/samples/CAM_FRONT"                # 直下or多階層 *.jpg
BDD100K_IMG_ROOT    = "/home/shogo/coding/datasets/BDD_100K_pure100k"                         # {train,val,test}

# Metric3Dv2（ONNX）
DEFAULT_METRIC3D_ONNX = "/home/shogo/coding/Metric3D/onnx/onnx/model.onnx"

# 出力（デフォルト /data 配下）
DEFAULT_OUT_ROOT      = "/data/ucn_condmaps"
DEFAULT_CACHE_ROOT    = "/data/ucn_prep_cache"
DEFAULT_TB_DIR        = "/data/ucn_prep_tb"

# OneFormer (Cityscapes)
ONEFORMER_ID = "shi-labs/oneformer_cityscapes_swin_large"

# 画像拡張子
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ==============================
# 1) ロガー
# ==============================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def setup_logger(log_dir: str, verbose: bool) -> logging.Logger:
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "ucn_build_conditions.log")
    logger = logging.getLogger("ucn_prep")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=20*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))
    logger.addHandler(ch); logger.addHandler(fh)
    logging.raiseExceptions = False
    return logger

def log_env(logger: logging.Logger) -> None:
    logger.info("torch=%s | torch.cuda(build)=%s | cuda_available=%s",
                torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
        if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
            logger.warning("CUDAビルド表示が 12.8 以外です（%s）。既存環境に合わせて続行します。", getattr(torch.version, "cuda", None))


# ==============================
# 2) 汎用I/O
# ==============================
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_jpg_rgb(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    ensure_dir(os.path.dirname(path))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])

def to_gray3(u8: np.ndarray) -> np.ndarray:
    """1ch uint8 → 3ch Gray"""
    if u8.ndim == 2:
        return np.stack([u8, u8, u8], axis=2)
    return u8


# ==============================
# 3) Cityscapes カラーマップ
# ==============================
_CITYSCAPES_TRAINID_COLORS_BGR = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]

def colorize_cityscapes_trainId(trainId: np.ndarray) -> np.ndarray:
    h, w = trainId.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, (b,g,r) in enumerate(_CITYSCAPES_TRAINID_COLORS_BGR):
        if tid >= 19: break
        out[trainId == tid] = (r, g, b)  # 最終はRGBで返す
    return out


# ==============================
# 4) Metric3Dv2 (ONNXRuntime CUDA)
# ==============================
IN_H, IN_W = 512, 1088  # 既存評価と同じ

def build_metric3d_session(onnx_path: str) -> Tuple[ort.InferenceSession, str, str]:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNXRuntime: CUDAExecutionProvider が見つかりません。GPU専用です。")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name

def infer_metric3d_single(sess: ort.InferenceSession, in_name: str, out_name: str, rgb: np.ndarray) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ten = np.transpose(rgb_resized, (2,0,1))[None, ...]  # (1,3,H,W)
    y = sess.run([out_name], {in_name: ten})[0]
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:,0]
    depth = np.squeeze(y).astype(np.float32)
    depth_up = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth_up

def depth_to_ucn_gray3(depth: np.ndarray, clip_low_ptile: float = 2.0, clip_high_ptile: float = 98.0) -> np.ndarray:
    """
    Metric3Dv2 の出力（相対奥行, 値が大→遠い傾向）を
    - ロバストなパーセンタイルで [0,1] 正規化（奥が白寄りになりがち）
    - 反転（1 - norm）して「奥=黒 / 手前=白」にする
    - 0..255 の uint8 にして 3ch Gray に拡張
    """
    d = depth.astype(np.float32)
    lo = float(np.percentile(d, clip_low_ptile))
    hi = float(np.percentile(d, clip_high_ptile))
    if hi <= lo:
        lo, hi = float(np.min(d)), float(np.max(d))
        if hi <= lo:
            g = np.zeros_like(d, dtype=np.uint8)
            return to_gray3(g)
    dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    inv = 1.0 - dn  # 奥=黒 / 手前=白
    u8 = (inv * 255.0 + 0.5).astype(np.uint8)
    return to_gray3(u8)


# ==============================
# 5) OneFormer (Cityscapes)
# ==============================
def build_oneformer(model_id: str, fp16: bool = True):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(CUDA)が必須です（OneFormer）。CPUフォールバックは禁止。")
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = OneFormerForUniversalSegmentation.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if fp16 else torch.float32)
        ).eval()
    except TypeError:
        model = OneFormerForUniversalSegmentation.from_pretrained(
            model_id,
            dtype=(torch.float16 if fp16 else torch.float32)
        ).eval()
    model = model.to("cuda")
    return processor, model

@torch.inference_mode()
def oneformer_semseg_batch(processor, model, rgbs: List[np.ndarray], use_amp: bool = True) -> List[np.ndarray]:
    """
    メモリ安全志向の簡易バッチ。既定はBS=1想定。呼び出し側でサイズを制御。
    """
    sizes = [(im.shape[0], im.shape[1]) for im in rgbs]
    enc = processor(images=rgbs, task_inputs=["semantic"]*len(rgbs), return_tensors="pt")
    pv = enc.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
    else:
        pv = torch.from_numpy(np.array(pv))
    pv = pv.to(model.device, dtype=model.dtype)
    ti = enc.get("task_inputs")
    if isinstance(ti, torch.Tensor):
        ti = ti.to(model.device)

    if (model.device.type == "cuda") and (model.dtype == torch.float16) and use_amp:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=pv, task_inputs=ti)
    else:
        out = model(pixel_values=pv, task_inputs=ti)

    segs = processor.post_process_semantic_segmentation(out, target_sizes=sizes)
    # 明示的にテンソルを解放してフラグメンテーションを抑止
    del pv, ti, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [s.to("cpu").numpy().astype(np.uint8) for s in segs]


# ==============================
# 6) Canny
# ==============================
def canny_rgb_to_gray3(rgb: np.ndarray, t1: float = 100.0, t2: float = 200.0, blur_ksize: int = 3) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_ksize > 0:
        g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    e = cv2.Canny(g, t1, t2)
    return to_gray3(e)


# ==============================
# 7) データセット列挙
# ==============================
def list_images_under(root: str, allowed: set = ALLOWED_IMG_EXT) -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for r,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in allowed:
                out.append(os.path.join(r,f))
    out.sort()
    return out

def list_bdd10k10k(split: str) -> List[str]:
    base = os.path.join(BDD10K10K_IMG_ROOT, split)
    return list_images_under(base)

def list_cityscapes(split: str) -> List[str]:
    base = os.path.join(CITYSCAPES_IMG_ROOT, split)
    return list_images_under(base)

def list_gta5_train() -> List[str]:
    return list_images_under(GTA5_IMG_ROOT)

def list_nuimages_front() -> List[str]:
    return list_images_under(NUIMAGES_FRONT_ROOT)

def list_bdd100k(split: str) -> List[str]:
    base = os.path.join(BDD100K_IMG_ROOT, split)
    return list_images_under(base)


# ==============================
# 8) 出力パス生成
# ==============================
def stem_without_cityscapes_suffix(filename: str) -> str:
    # aachen_000000_000019_leftImg8bit.png → aachen_000000_000019
    s = os.path.splitext(os.path.basename(filename))[0]
    if s.endswith("_leftImg8bit"):
        s = s[:-len("_leftImg8bit")]
    return s

def out_path(out_root: str, dataset_key: str, task: str, rel_dir: str, stem: str) -> str:
    return os.path.join(out_root, dataset_key, task, rel_dir, f"{stem}_{task}.jpg")


# ==============================
# 9) 引数
# ==============================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet 学習用条件マップ一括生成（Metric3Dv2/Canny/OneFormer）")
    ap.add_argument("--datasets", type=str, nargs="+",
                    choices=["all","bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k"],
                    default=["all"])
    ap.add_argument("--tasks", type=str, nargs="+",
                    choices=["all","depth","edge","semseg"],
                    default=["all"])
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--metric3d-onnx", type=str, default=DEFAULT_METRIC3D_ONNX)

    # 本番運転時の使用 split（GTA5/nuImages は入力に split がないため無関係）
    ap.add_argument("--cityscapes-use-splits", type=str, nargs="+", default=["train"], help="{train,val,test} から選択")
    ap.add_argument("--bdd10k10k-use-splits", type=str, nargs="+", default=["train"], help="{train,val}")
    ap.add_argument("--bdd100k-use-splits", type=str, nargs="+", default=["train"], help="{train,val,test}")

    ap.add_argument("--limit", type=int, default=-1, help="各データセットの処理枚数上限（デバッグ用）")
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--tb-dir", type=str, default=DEFAULT_TB_DIR)
    ap.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--verbose", action="store_true")

    # ===== OOM対策：オートバッチ廃止・既定は常にBS=1 =====
    ap.add_argument("--auto-batch", dest="auto_batch", action="store_true", help="（非推奨）OneFormer の自動バッチ推定を使用")
    ap.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", help="OneFormer の自動バッチ推定を使用しない（推奨）")
    ap.set_defaults(auto_batch=False)  # 既定オフ＝常に明示 batch size
    ap.add_argument("--semseg-batch-size", type=int, default=1, help="OneFormer の明示バッチサイズ（既定1）")
    ap.add_argument("--semseg-amp", dest="semseg_amp", action="store_true", help="OneFormer 推論で autocast(FP16) を使用（既定ON）")
    ap.add_argument("--no-semseg-amp", dest="semseg_amp", action="store_false")
    ap.set_defaults(semseg_amp=True)

    ap.add_argument("--clip-low-ptile", type=float, default=2.0, help="Depth 正規化 下側パーセンタイル")
    ap.add_argument("--clip-high-ptile", type=float, default=98.0, help="Depth 正規化 上側パーセンタイル")
    ap.add_argument("--jpg-quality", type=int, default=95)

    # ===== サブセット一括チェックモード =====
    ap.add_argument("--subset-check", action="store_true",
                    help="all_dataset_subset_checkmode: 各データセット×split（存在するもののみ）から1枚ずつ処理して通しチェックを行う。既存生成物はスキップ。")
    ap.add_argument("--subset-report", type=str, default=None,
                    help="サブセットチェックの結果を保存するJSONのパス。未指定時は {cache_root}/reports/subset_check_YYYYMMDD-HHMMSS.json")
    ap.add_argument("--force-train-subdir-singletons", action="store_true",
                    help="GTA5/nuImages のように入力に split が無いデータセットでも、出力側に train/ を強制付与する（互換維持のためデフォルトは付与しない）。")

    return ap.parse_args()


# ==============================
# 10) メイン
# ==============================
def main() -> None:
    args = parse_args()
    logger = setup_logger(os.path.join(args.cache_root, "logs"), args.verbose)

    if not torch.cuda.is_available():
        logger.error("❌ GPU(CUDA) が見つかりません。GPU専用スクリプトです。")
        sys.exit(2)
    log_env(logger)

    # 追加の安全策：断片化緩和（環境変数が未設定なら通知）
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        logger.warning("PYTORCH_CUDA_ALLOC_CONF が未設定です。entrypoint か docker -e で "
                       "expandable_segments:True,max_split_size_mb:128 を推奨。")

    sw: Optional[SummaryWriter] = SummaryWriter(args.tb_dir) if args.tb else None
    if sw:
        logger.info("TensorBoard: %s", args.tb_dir)

    # タスク選択
    do_depth = ("all" in args.tasks) or ("depth" in args.tasks)
    do_edge  = ("all" in args.tasks) or ("edge" in args.tasks)
    do_seg   = ("all" in args.tasks) or ("semseg" in args.tasks)

    # ===== モデル構築 =====
    sess_m3d = None; in_m3d = None; out_m3d = None
    if do_depth:
        try:
            sess_m3d, in_m3d, out_m3d = build_metric3d_session(args.metric3d_onnx)
            logger.info("Metric3Dv2 ONNX loaded: %s", args.metric3d_onnx)
        except Exception as e:
            logger.error("Metric3Dv2 ONNX 構築失敗: %s", repr(e)); sys.exit(1)

    onef_proc = None; onef_model = None
    onef_bs = max(1, int(args.semseg_batch_size))

    # ============ サブセット一括チェック ============
    if args.subset_check:
        logger.info("=== サブセット一括チェックモード開始（--subset-check）===")

        # split 候補は train/val/test（存在するもののみ採用）
        SPLITS = ["train","val","test"]

        # 収集先
        datasets: List[Tuple[str, List[str], str]] = []  # (key, image_list, base_for_rel)
        split_by_path: Dict[str, str] = {}  # レポート用

        def _add_one_sample(ds_key: str, base_root: str, split_root: Optional[str], split_name: str):
            if split_root is None:
                imgs = list_images_under(base_root)
            else:
                imgs = list_images_under(os.path.join(base_root, split_root))
            if not imgs:
                return []
            pick = imgs[0]
            split_by_path[pick] = split_name
            return [pick]

        # BDD10K(10K)
        if "all" in args.datasets or "bdd10k10k" in args.datasets:
            imgs=[]
            for sp in SPLITS:
                base = os.path.join(BDD10K10K_IMG_ROOT, sp)
                if os.path.isdir(base):
                    imgs += _add_one_sample("bdd10k10k", BDD10K10K_IMG_ROOT, sp, sp)
            if imgs:
                datasets.append(("bdd10k10k", imgs, BDD10K10K_IMG_ROOT))

        # Cityscapes
        if "all" in args.datasets or "cityscapes" in args.datasets:
            imgs=[]
            for sp in SPLITS:
                base = os.path.join(CITYSCAPES_IMG_ROOT, sp)
                if os.path.isdir(base):
                    imgs += _add_one_sample("cityscapes", CITYSCAPES_IMG_ROOT, sp, sp)
            if imgs:
                datasets.append(("cityscapes", imgs, CITYSCAPES_IMG_ROOT))

        # GTA5（split なし）→ 擬似 "train"
        if "all" in args.datasets or "gta5" in args.datasets:
            imgs = _add_one_sample("gta5", GTA5_IMG_ROOT, None, "train")
            if imgs:
                datasets.append(("gta5", imgs, GTA5_IMG_ROOT))

        # nuImages(front)（split なし）→ 擬似 "train"
        if "all" in args.datasets or "nuimages_front" in args.datasets:
            imgs = _add_one_sample("nuimages_front", NUIMAGES_FRONT_ROOT, None, "train")
            if imgs:
                datasets.append(("nuimages_front", imgs, NUIMAGES_FRONT_ROOT))

        # BDD100K
        if "all" in args.datasets or "bdd100k" in args.datasets:
            imgs=[]
            for sp in SPLITS:
                base = os.path.join(BDD100K_IMG_ROOT, sp)
                if os.path.isdir(base):
                    imgs += _add_one_sample("bdd100k", BDD100K_IMG_ROOT, sp, sp)
            if imgs:
                datasets.append(("bdd100k", imgs, BDD100K_IMG_ROOT))

        total_imgs = sum(len(x[1]) for x in datasets)
        if total_imgs == 0:
            logger.error("❌ サブセットチェック対象が見つかりません。データセットパスを確認してください。")
            sys.exit(1)
        logger.info("チェック対象（総枚数）: %d", total_imgs)

        # OneFormer 準備（チェックは小規模なので BS=1 固定）
        if do_seg:
            onef_proc, onef_model = build_oneformer(ONEFORMER_ID, fp16=True)
            onef_bs = 1

        # レポート
        report: List[Dict[str, Any]] = []
        any_error = False

        for ds_key, img_list, base_rel in datasets:
            logger.info("=== [CHECK %s] サンプル枚数: %d ===", ds_key, len(img_list))
            for p in img_list:
                split = split_by_path.get(p, "train")

                # rel_dir の構築
                rel_dir = os.path.relpath(os.path.dirname(p), base_rel)
                if args.force_train_subdir_singletons and ds_key in ("gta5","nuimages_front"):
                    rel_dir = os.path.join("train", rel_dir) if rel_dir != "." else "train"

                if ds_key == "cityscapes":
                    stem = stem_without_cityscapes_suffix(p)
                else:
                    stem = os.path.splitext(os.path.basename(p))[0]

                out_depth = out_path(args.out_root, ds_key, "depth", rel_dir, stem)
                out_edge  = out_path(args.out_root, ds_key, "edge",  rel_dir, stem)
                out_seg   = out_path(args.out_root, ds_key, "semseg",rel_dir, stem)

                entry = {
                    "dataset": ds_key,
                    "split": split,
                    "input": p,
                    "outputs": {"depth": out_depth, "edge": out_edge, "semseg": out_seg},
                    "status": {}
                }

                try:
                    rgb = imread_rgb(p)

                    # Edge
                    try:
                        if do_edge:
                            if os.path.exists(out_edge):
                                entry["status"]["edge"] = "skipped_existing"
                            else:
                                e3 = canny_rgb_to_gray3(rgb, 100.0, 200.0, 3)
                                save_jpg_rgb(out_edge, e3, quality=args.jpg_quality)
                                entry["status"]["edge"] = "created"
                        else:
                            entry["status"]["edge"] = "disabled"
                    except Exception as e:
                        any_error = True
                        entry["status"]["edge"] = f"error:{repr(e)}"
                        logger.exception("[CHECK %s] Edge 失敗: %s", ds_key, p)

                    # Depth
                    try:
                        if do_depth:
                            if os.path.exists(out_depth):
                                entry["status"]["depth"] = "skipped_existing"
                            else:
                                d = infer_metric3d_single(sess_m3d, in_m3d, out_m3d, rgb)
                                g3 = depth_to_ucn_gray3(d, args.clip_low_ptile, args.clip_high_ptile)
                                save_jpg_rgb(out_depth, g3, quality=args.jpg_quality)
                                entry["status"]["depth"] = "created"
                        else:
                            entry["status"]["depth"] = "disabled"
                    except Exception as e:
                        any_error = True
                        entry["status"]["depth"] = f"error:{repr(e)}"
                        logger.exception("[CHECK %s] Depth 失敗: %s", ds_key, p)

                    # Semseg
                    try:
                        if do_seg:
                            if os.path.exists(out_seg):
                                entry["status"]["semseg"] = "skipped_existing"
                            else:
                                seg = oneformer_semseg_batch(onef_proc, onef_model, [rgb], use_amp=args.semseg_amp)[0]
                                col = colorize_cityscapes_trainId(seg)
                                save_jpg_rgb(out_seg, col, quality=args.jpg_quality)
                                entry["status"]["semseg"] = "created"
                        else:
                            entry["status"]["semseg"] = "disabled"
                    except Exception as e:
                        any_error = True
                        entry["status"]["semseg"] = f"error:{repr(e)}"
                        logger.exception("[CHECK %s] Semseg 失敗: %s", ds_key, p)

                except Exception as e:
                    any_error = True
                    entry["status"]["fatal"] = f"error:{repr(e)}"
                    logger.exception("[CHECK %s] 入力読み込み失敗: %s", ds_key, p)

                report.append(entry)

        # レポート保存
        ensure_dir(os.path.join(args.cache_root, "reports"))
        if args.subset_report is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            rep_path = os.path.join(args.cache_root, "reports", f"subset_check_{ts}.json")
        else:
            rep_path = args.subset_report
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("📝 サブセットチェックレポート: %s", rep_path)

        # サマリ
        ok_cnt = 0; skip_cnt = 0; err_cnt = 0
        for r in report:
            ds = r["dataset"]; sp = r["split"]; ip = r["input"]
            st = r["status"]
            def tag(k): return st.get(k, "n/a")
            line = f"[{ds}/{sp}] edge={tag('edge')} depth={tag('depth')} semseg={tag('semseg')} :: {ip}"
            if any((isinstance(v, str) and v.startswith("error:")) for v in st.values()):
                err_cnt += 1
                logger.error(line)
            else:
                if any(v == "created" for v in st.values()):
                    ok_cnt += 1
                else:
                    skip_cnt += 1
                logger.info(line)
        logger.info("=== サマリ: OK=%d, SKIP_ONLY=%d, ERR=%d ===", ok_cnt, skip_cnt, err_cnt)

        if sw:
            sw.add_scalar("check/ok", ok_cnt, 0)
            sw.add_scalar("check/skip_only", skip_cnt, 0)
            sw.add_scalar("check/error", err_cnt, 0)
            sw.close()

        sys.exit(0 if not any_error else 3)

    # ============ 通常フル実行 ============
    datasets: List[Tuple[str, List[str], str]] = []  # (key, image_list, base_for_rel)

    # BDD10K(10K)
    if "all" in args.datasets or "bdd10k10k" in args.datasets:
        imgs=[]
        for sp in args.bdd10k10k_use_splits:
            base = os.path.join(BDD10K10K_IMG_ROOT, sp)
            cur = list_images_under(base)
            if args.limit>0: cur = cur[:args.limit]
            imgs += cur
        datasets.append(("bdd10k10k", imgs, BDD10K10K_IMG_ROOT))

    # Cityscapes
    if "all" in args.datasets or "cityscapes" in args.datasets:
        imgs=[]
        for sp in args.cityscapes_use_splits:
            base = os.path.join(CITYSCAPES_IMG_ROOT, sp)
            cur = list_images_under(base)
            if args.limit>0: cur = cur[:args.limit]
            imgs += cur
        datasets.append(("cityscapes", imgs, CITYSCAPES_IMG_ROOT))

    # GTA5
    if "all" in args.datasets or "gta5" in args.datasets:
        imgs = list_gta5_train()
        if args.limit>0: imgs = imgs[:args.limit]
        datasets.append(("gta5", imgs, GTA5_IMG_ROOT))

    # nuImages(front)
    if "all" in args.datasets or "nuimages_front" in args.datasets:
        imgs = list_nuimages_front()
        if args.limit>0: imgs = imgs[:args.limit]
        datasets.append(("nuimages_front", imgs, NUIMAGES_FRONT_ROOT))

    # BDD100K
    if "all" in args.datasets or "bdd100k" in args.datasets:
        imgs=[]
        for sp in args.bdd100k_use_splits:
            base = os.path.join(BDD100K_IMG_ROOT, sp)
            cur = list_images_under(base)
            if args.limit>0: cur = cur[:args.limit]
            imgs += cur
        datasets.append(("bdd100k", imgs, BDD100K_IMG_ROOT))

    total_imgs = sum(len(x[1]) for x in datasets)
    if total_imgs == 0:
        logger.warning("対象画像が見つかりません。データセットパスと --*splits を確認してください。")
        sys.exit(0)
    logger.info("対象総枚数: %d", total_imgs)

    # OneFormer 準備（オートバッチ廃止、既定BS=1）
    if do_seg:
        onef_proc, onef_model = build_oneformer(ONEFORMER_ID, fp16=True)
        if args.auto_batch:
            # どうしても使いたい人向けの後方互換。既定は False。
            try:
                sample_for_auto = None
                for _, imgs, _ in datasets:
                    if imgs:
                        sample_for_auto = imread_rgb(imgs[0]); break
                if sample_for_auto is None:
                    raise RuntimeError("AutoBatch 用サンプル取得失敗")
                onef_bs = 1
                # 以前のような大きな自動バッチは OOM の温床。最大でも 2 まで許容する。
                from math import ceil
                # ここでは安全のため 2 を上限として探索（簡易）
                for b in [2, 1]:
                    try:
                        _ = oneformer_semseg_batch(onef_proc, onef_model, [sample_for_auto]*b, use_amp=args.semseg_amp)
                        onef_bs = b
                        break
                    except RuntimeError:
                        continue
                logger.info("OneFormer Batch Size (auto-capped): %d", onef_bs)
            except Exception as e:
                logger.warning("AutoBatch 無効化（理由: %s）→ batch=1", repr(e))
                onef_bs = 1
        else:
            logger.info("OneFormer Batch Size (manual): %d", onef_bs)

    # 進捗
    processed = 0
    t0 = time.time()

    for ds_key, img_list, base_rel in datasets:
        logger.info("=== [%s] 画像枚数: %d ===", ds_key, len(img_list))

        # セマンティクスのバッチ用ワーク
        pend_rgbs: List[np.ndarray] = []
        pend_meta: List[Tuple[str,str,str]] = []  # (rel_dir, stem, out_path)

        for p in tqdm(img_list, desc=f"{ds_key}", dynamic_ncols=True):
            rel_dir = os.path.relpath(os.path.dirname(p), base_rel)
            if args.force_train_subdir_singletons and ds_key in ("gta5","nuimages_front"):
                rel_dir = os.path.join("train", rel_dir) if rel_dir != "." else "train"

            if ds_key == "cityscapes":
                stem = stem_without_cityscapes_suffix(p)
            else:
                stem = os.path.splitext(os.path.basename(p))[0]

            out_depth = out_path(args.out_root, ds_key, "depth", rel_dir, stem)
            out_edge  = out_path(args.out_root, ds_key, "edge",  rel_dir, stem)
            out_seg   = out_path(args.out_root, ds_key, "semseg",rel_dir, stem)

            rgb = imread_rgb(p)

            # Edge
            if do_edge and (not os.path.exists(out_edge)):
                e3 = canny_rgb_to_gray3(rgb, 100.0, 200.0, 3)
                save_jpg_rgb(out_edge, e3, quality=args.jpg_quality)

            # Depth
            if do_depth and (not os.path.exists(out_depth)):
                d = infer_metric3d_single(sess_m3d, in_m3d, out_m3d, rgb)
                g3 = depth_to_ucn_gray3(d, args.clip_low_ptile, args.clip_high_ptile)
                save_jpg_rgb(out_depth, g3, quality=args.jpg_quality)

            # Semseg（BS=onef_bs、既定1）— 常に都度 flush でフラグメント抑止
            if do_seg:
                if not os.path.exists(out_seg):
                    pend_rgbs.append(rgb)
                    pend_meta.append((rel_dir, stem, out_seg))
                    if len(pend_rgbs) >= onef_bs:
                        segs = oneformer_semseg_batch(onef_proc, onef_model, pend_rgbs, use_amp=args.semseg_amp)
                        for (rel_dir2, stem2, outp), seg in zip(pend_meta, segs):
                            col = colorize_cityscapes_trainId(seg)
                            save_jpg_rgb(outp, col, quality=args.jpg_quality)
                        pend_rgbs.clear(); pend_meta.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            processed += 1
            if sw and (processed % 50 == 0):
                sw.add_scalar("prep/processed", processed, processed)

        # 端数 flush
        if do_seg and pend_rgbs:
            segs = oneformer_semseg_batch(onef_proc, onef_model, pend_rgbs, use_amp=args.semseg_amp)
            for (rel_dir2, stem2, outp), seg in zip(pend_meta, segs):
                col = colorize_cityscapes_trainId(seg)
                save_jpg_rgb(outp, col, quality=args.jpg_quality)
            pend_rgbs.clear(); pend_meta.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("=== [%s] 完了 ===", ds_key)

    dt = time.time() - t0
    logger.info("✅ 全データセット完了: processed=%d, time=%.1fs (%.2f img/s)",
                processed, dt, (processed/max(1.0, dt)))
    if sw:
        sw.add_scalar("prep/total_images", processed, 0)
        sw.add_scalar("prep/throughput_img_per_s", processed/max(1.0, dt), 0)
        sw.close()


if __name__ == "__main__":
    main()

"""

#!/usr/bin/env bash
set -euo pipefail

# ============= GPU / Torch 情報 =============
nvidia-smi || true
python3 - <<'PY'
import sys
try:
    import torch
    print(f"[entrypoint] torch={torch.__version__}, torch.version.cuda(build)={getattr(torch.version,'cuda',None)}")
    print(f"[entrypoint] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[entrypoint] device0={torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[entrypoint] FATAL: import torch failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ============= 永続 pip オーバーレイ（再ビルド不要/ハッシュ差分適用） =============
PIP_OVERLAY_DIR="${PIP_OVERLAY_DIR:-/mnt/hdd/ucn_eval_cache/pip-overlay}"
REQS_OVERLAY_PATH="${REQS_OVERLAY_PATH:-/mnt/hdd/ucn_eval_cache/requirements.overlay.txt}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$PIP_OVERLAY_DIR"
export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"

_overlay_inputs=""
if [ -n "${PIP_INSTALL:-}" ]; then _overlay_inputs="${PIP_INSTALL}"; fi
if [ -f "$REQS_OVERLAY_PATH" ]; then
  _overlay_inputs="${_overlay_inputs}"$'\n'"$(cat "$REQS_OVERLAY_PATH")"
fi

if [ -n "${_overlay_inputs}" ]; then
  need_hash="$(printf "%s" "${_overlay_inputs}" | sha1sum | awk '{print $1}')"
  prev_hash="$(cat "${PIP_OVERLAY_DIR}/.overlay_hash" 2>/dev/null || echo "")"
  if [ "${need_hash}" != "${prev_hash}" ]; then
    echo "[entrypoint] Installing/updating overlay packages into: ${PIP_OVERLAY_DIR}"

    SAFE_REQ="${PIP_OVERLAY_DIR}/.reqs.safe.txt"
    NODEPS_REQ="${PIP_OVERLAY_DIR}/.reqs.nodeps.txt"
    rm -f "$SAFE_REQ" "$NODEPS_REQ"

    # --- REQS ファイルの取り込み（先頭末尾のクォートを除去してから分類） ---
    if [ -f "$REQS_OVERLAY_PATH" ]; then
      while IFS= read -r line; do
        trimmed="$(printf '%s' "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [ -z "$trimmed" ] && continue
        # 先頭末尾の " と ' を除去（中間にあるクォートは保持）
        case "$trimmed" in
          \"*\" ) trimmed="${trimmed%\"}"; trimmed="${trimmed#\"}";;
        esac
        case "$trimmed" in
          \'*\' ) trimmed="${trimmed%\'}"; trimmed="${trimmed#\'}";;
        esac

        printf '%s\n' "$trimmed" | grep -Eqi '^(torch|torchvision|torchaudio)($|[<=>])' && continue
        printf '%s\n' "$trimmed" | grep -Eqi '^(thop|timm)($|[<=>])' && { echo "$trimmed" >> "$NODEPS_REQ"; continue; }
        echo "$trimmed" >> "$SAFE_REQ"
      done < "$REQS_OVERLAY_PATH"
    fi

    # 1) 依存ありで安全群をインストール
    if [ -s "$SAFE_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" -r "$SAFE_REQ"
    fi

    # 2) 依存なしで NoDeps 群（thop, timm）
    if [ -s "$NODEPS_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" --no-deps -r "$NODEPS_REQ"
    fi

    # 3) 環境変数 PIP_INSTALL の取り込み（トークンからクォートを除去）
    if [ -n "${PIP_INSTALL:-}" ]; then
      _pi_sanitized="$(
        printf '%s\n' "$PIP_INSTALL" \
          | tr ' ' '\n' \
          | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
          | sed 's/^"//;s/"$//' \
          | sed "s/^'//;s/'$//" \
          | grep -Evi '^(torch|torchvision|torchaudio)($|[<=>])' \
          | tr '\n' ' '
      )"
      if [ -n "$_pi_sanitized" ]; then
        # shellcheck disable=SC2086
        python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" $_pi_sanitized
      fi
    fi

    # 4) 念のため overlay から torch 系を物理削除（誤混入の保険）
    rm -rf "${PIP_OVERLAY_DIR}/torch" "${PIP_OVERLAY_DIR}/torchvision" "${PIP_OVERLAY_DIR}/torchaudio" || true
    rm -rf "${PIP_OVERLAY_DIR}"/nvidia_* "${PIP_OVERLAY_DIR}"/nvidia* || true

    # 5) transformers との互換確保：huggingface_hub が 1.x なら 0.44.1 にダウングレード
    python3 - <<'PY'
import sys
try:
    import transformers, huggingface_hub
    from packaging.version import Version
    tv = Version(transformers.__version__)
    hv = Version(huggingface_hub.__version__)
    print(f"[entrypoint] versions: transformers={tv}, huggingface_hub={hv}")
    if hv.major >= 1:
        sys.exit(42)
except Exception as e:
    print(f"[entrypoint] version check note: {e}")
    sys.exit(0)
PY
    ret=$?
    if [ "$ret" -eq 42 ]; then
      echo "[entrypoint] Downgrading huggingface_hub to 0.44.1 for transformers compatibility (<1.0)"
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" "huggingface_hub==0.44.1"
    fi

    echo "${need_hash}" > "${PIP_OVERLAY_DIR}/.overlay_hash"
  else
    echo "[entrypoint] Overlay unchanged; skip pip."
  fi
else
  echo "[entrypoint] No overlay inputs (PIP_INSTALL/requirements.overlay.txt)."
fi

# 可視化: 重要依存が読めるか即確認（バージョンも表示）
python3 - <<'PY'
import importlib.util, os
def status(m): return "OK" if importlib.util.find_spec(m) else "MISSING"
mods = [
  "prefetch_generator","easydict","thop","skimage",
  "timm","einops","transformers","huggingface_hub","safetensors",
  "onnxruntime","cv2","numpy","scipy","PIL","yaml","torchvision"
]
print("[entrypoint] PYTHONPATH head:", os.environ.get("PYTHONPATH","").split(":")[0])
for m in mods:
    print(f"[entrypoint] import {m}:", status(m))
try:
    import transformers, huggingface_hub
    print(f"[entrypoint] versions summary: transformers={transformers.__version__}, huggingface_hub={huggingface_hub.__version__}")
except Exception:
    pass
PY

# ============= 評価スクリプト起動 =============
# -------------- ここから挿入してください --------------
# MAIN_PY が指定されていればそれを実行、未指定なら従来通り eval_unicontrol_waymo.py を実行
MAIN_PY="${MAIN_PY:-/app/eval_unicontrol_waymo.py}"
exec python3 -u "${MAIN_PY}" "$@"
# -------------- ここまで挿入してください --------------
"""