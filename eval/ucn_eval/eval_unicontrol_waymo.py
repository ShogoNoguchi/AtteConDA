#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#コード内に実行コマンドや/home/shogo/coding/eval/ucn_eval/docker/Dockerfileと/home/shogo/coding/eval/ucn_eval/docker/entrypoint.shの内容も記しておくことでわかりやすく（重要）。
"""/home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py
[我々の目的]：
本研究では、自動運転ドメインの機械学習用のデータを、画像に付けられたアノテーション（正解ラベル）を保ちながらスタイル変換することで、画像とアノテーションをセットで”水増し”する手法の提案​を目的としています。
従来の画像処理的な水増し（幾何的変換等）は、未だに深層学習分野でも広く使用されており、有効な手段です。しかし、スタイルの多様性が増えないのでこれに頼りすぎるとモデルが過学習（overfitting）してしまい、汎化能力が低下します。
そこで、近年新たな試みとして、画像生成モデルで訓練データを増やす試みがあります。具体的には、我々の場合、何らかの軽量なモデルで元の画像の天候・時間帯を判断し、それを変えることによって画像を増やします。このコードは水増し後のデータセットに対して網羅的な評価を行うコードです。
[コード説明]
Uni-ControlNet を “自動運転用画像水増し装置” として網羅的に定量評価する単一スクリプト（完全版）
- 入力: 元RGB(X) と 生成画像 F(X)（Uni-ControlNet_offline 産物）を WaymoV2 の相対パスでペアリング
- 評価（--tasks で切替）:
  (1) リアリティ指標:
      - CLIP-CMMD（推奨; CLIP embedding + RBF-MMD）
      - CLIP-FID（CLIP embedding での Fréchet 距離）
      - Inception-FID（警告表示のみ; 実装外）
  (2) 構造・意味忠実度:
      - Edge(Canny): L1/RMSE/IoU/F1（X の既存 Canny と F(X) の Canny 再推論を比較）
      - Depth(Metric3Dv2, ONNX): RMSE / scale-invariant RMSE / 相対誤差（X: 既存 NPY, F(X): ONNX 再推論）
      - Semseg(OneFormer Cityscapes): 19クラス混同行列→mIoU
      - すべて /mnt/hdd/ucn_eval_cache にキャッシュ（npz/npy/png）して再利用
  (3) 物体保持・ハルシネーション:
      - Grounding DINO（オープン語彙検出; Transformers 実装）
      - （任意）YOLOP（drivable ROI によるフィルタは既定OFF・必要時のみ --roi-filter）
      - OCR(Tesseract) による標識読み取り
      - 指標: 保持再現率(PR) / 保持適合率(PP) / F1 / ハルシネーション率(HAL)
              幾何安定性（中心誤差/サイズ比誤差/IoU中央値）
              カウント整合性（平均差 / 擬EMD）
  (4) 走行可能領域の保持（新規: --tasks drivable または all）
      - YOLOP の drivable マスク（X vs F(X)）を直接比較
      - 指標: IoU / Dice(F1) / Precision / Recall / L1 / RMSE / Boundary-IoU（境界的一致）
      - 後処理: 2値化閾値 (--drivable-thr), モルフォロジー閉処理 (--drivable-morph-k), 境界許容 (--drivable-edge-tol)

- アノテーション可視化（--annotation-mode）:
  objects: X/FX の検出ボックス（+drivable ROI 任意）を描画
  structure: FX の Edge/Depth/Semseg を横並び保存
  drivable: X/FX の走行可能領域オーバレイを並列保存（新規）
  all: 両方
  off: なし
- 既定パスは翔伍さんの資産に合わせて固定。CLI で上書き可。
- 再現性: 乱数固定（--seed）、TensorBoard（--tb）対応。

注意:
- Docker 側で torch==2.7.0+cu128 / torchvision==0.22.0+cu128 を固定。
- 追加依存（timm, thop, prefetch_generator, scikit-image）は overlay で導入。
- onnxruntime は CPU/GPU どちらでも動作可（CUDAExecutionProvider があれば GPU 使用）。

実行例：
## All（Reality＋Structure＋Objects＋Drivable）一括評価＋可視化（推奨・論文用）
- **OneFormer(Swin) 用に `timm` を overlay 導入**（事前に「0) リセット」実行推奨）
- アノテ：`all` を各split 24枚保存

Uni-ControlNet を “自動運転用画像水増し装置” として網羅的に定量評価する単一スクリプト（完全版）
- 入力: 元RGB(X) と 生成画像 F(X)（Uni-ControlNet_offline 産物）を WaymoV2 の相対パスでペアリング
- 評価（--tasks で切替）:
  (1) リアリティ指標:
      - CLIP-CMMD（推奨; CLIP embedding + RBF-MMD）
      - CLIP-FID（CLIP embedding での Fréchet 距離）
      - Inception-FID（警告表示のみ; 実装外）
  (2) 構造・意味忠実度:
      - Edge(Canny): L1/RMSE/IoU/F1（X の既存 Canny と F(X) の Canny 再推論を比較）
      - Depth(Metric3Dv2, ONNX): RMSE / scale-invariant RMSE / 相対誤差（X: 既存 NPY, F(X): ONNX 再推論）
      - Semseg(OneFormer Cityscapes): 19クラス混同行列→mIoU
      - すべて /mnt/hdd/ucn_eval_cache にキャッシュ（npz/npy/png）して再利用
  (3) 物体保持・ハルシネーション:
      - Grounding DINO（オープン語彙検出; Transformers 実装）
      - （任意）YOLOP（drivable ROI によるフィルタは既定OFF・必要時のみ --roi-filter）
      - OCR(Tesseract) による標識読み取り
      - 指標: 保持再現率(PR) / 保持適合率(PP) / F1 / ハルシネーション率(HAL)
              幾何安定性（中心誤差/サイズ比誤差/IoU中央値）
              カウント整合性（平均差 / 擬EMD）
  (4) 走行可能領域の保持（新規: --tasks drivable または all）
      - YOLOP の drivable マスク（X vs F(X)）を直接比較
      - 指標: IoU / Dice(F1) / Precision / Recall / L1 / RMSE / Boundary-IoU（境界的一致）
      - 後処理: 2値化閾値 (--drivable-thr), モルフォロジー閉処理 (--drivable-morph-k), 境界許容 (--drivable-edge-tol)

- アノテーション可視化（--annotation-mode）:
  objects: X/FX の検出ボックス（+drivable ROI 任意）を描画
  structure: FX の Edge/Depth/Semseg を横並び保存
  drivable: X/FX の走行可能領域オーバレイを並列保存（新規）
  all: 両方
  off: なし
- 既定パスは翔伍さんの資産に合わせて固定。CLI で上書き可。
- 再現性: 乱数固定（--seed）、TensorBoard（--tb）対応。

注意:
- Docker 側で torch==2.7.0+cu128 / torchvision==0.22.0+cu128 を固定。
- 追加依存（timm, thop, prefetch_generator, scikit-image）は overlay で導入。
- onnxruntime は CPU/GPU どちらでも動作可（CUDAExecutionProvider があれば GPU 使用）。

実行例：docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR=/mnt/hdd/ucn_eval_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/mnt/hdd/ucn_eval_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs prefetch_generator pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /mnt/hdd/ucn_eval_cache:/mnt/hdd/ucn_eval_cache \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/hdd/ucn_eval_cache/torch_hub:/root/.cache/torch/hub \
  ucn-eval \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --gdino-box-thr 0.25 \
  --gdino-text-thr 0.20 \
  --use-yolop \
  --yolop-roi-filter \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root  /home/shogo/coding/datasets/WaymoV2/UniControlNet_offline \
  --annotation-mode all \
  --annotate-limit 24 \
  --tb --tb-dir /mnt/hdd/ucn_eval_cache/tensorboard \
  --verbose

"""

import os
import sys
import argparse
import json
import logging
from logging import handlers
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import hashlib
import time
import math
import traceback
import shutil
import multiprocessing as mp
import warnings


# YOLOP/Transformers の将来変更に伴う FutureWarning を既定で抑止
warnings.filterwarnings(
    "ignore",
    message=".*GroundingDinoProcessor.*Use `text_labels` instead.*",
    category=FutureWarning
)

# 数値・画像
import numpy as np
import cv2

# 科学計算
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance, spearmanr  # ★clip-corrで使用


import random

# 進捗
from tqdm import tqdm


# Torch/Transformers
import torch
import torch.nn.functional as F  # ★追加：YOLOP出力の補間とsoftmax/argmaxに使用
from torch.utils.tensorboard import SummaryWriter

# HF models（必要時に遅延ロード）
from transformers import AutoProcessor, OneFormerForUniversalSegmentation, CLIPModel, CLIPImageProcessor
from transformers import AutoProcessor as GDinoProcessor
from transformers import AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor  # ★追加：テキストもエンコード可能な Processor

# ★ここから挿入（新規依存：LPIPS / MS-SSIM）========================
try:
    import lpips  # Learned Perceptual Image Patch Similarity (Zhang+2018)
except Exception as _e:
    lpips = None
try:
    from pytorch_msssim import ms_ssim  # MS-SSIM 実装（VainF）
except Exception as _e:
    ms_ssim = None
# ★ここまで挿入======================================================

# ========== 既定パス（環境に完全準拠） ==========
DEFAULT_ORIG_IMAGE_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_GEN_ROOT        = "/home/shogo/coding/datasets/WaymoV2/UniControlNet_offline"

# 既存の X 側予測（再利用）
DEFAULT_CANNY_ROOT_X     = "/home/shogo/coding/datasets/WaymoV2/CannyEdge"
DEFAULT_DEPTH_NPY_ROOT_X = "/home/shogo/coding/datasets/WaymoV2/Metricv2DepthNPY"
DEFAULT_SEMSEG_ROOT_X    = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"

# F(X) 側の推論キャッシュ（HDD or SSD）
#DEFAULT_HDD_CACHE_ROOT   = "/mnt/hdd/ucn_eval_cache"
DEFAULT_HDD_CACHE_ROOT   = "/data/ucn_eval_cache"  # ★SSD(2nd)へ移設：以後は /data を既定に
# Metric3Dv2 ONNX
DEFAULT_METRIC3D_ONNX    = "/home/shogo/coding/Metric3D/onnx/onnx/model.onnx"  # 既存

# OneFormer / Grounding DINO / CLIP
DEFAULT_ONEFORMER_ID     = "shi-labs/oneformer_cityscapes_swin_large"
DEFAULT_GDINO_ID         = "IDEA-Research/grounding-dino-base"
DEFAULT_CLIP_ID          = "openai/clip-vit-large-patch14-336"
# ★ここから挿入（プロンプトルート：メタ欠落時のフォールバック用）========
DEFAULT_PROMPT_ROOT      = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss"
# ★ここまで挿入======================================================
# 評価対象クラス（OVD: Grounding DINO）
DEFAULT_DET_PROMPTS = [
    "car", "truck", "bus", "motorcycle", "bicycle", "person", "pedestrian",
    "traffic light", "traffic sign", "stop sign", "speed limit sign",
    "crosswalk sign", "construction sign", "traffic cone",
]

# Cityscapes trainId（0..18）— OneFormer と整合
CITYSCAPES_TRAINID = list(range(19))

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# OneFormer/深度の既定入力サイズ等
IN_H, IN_W = 512, 1088  # Metric3D 推奨（既存の資産に整合）


# ========== ユーティリティ（ロガー/環境） ==========
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def setup_logger(log_dir: str, verbose: bool) -> logging.Logger:
    """
    - 外部で注入されたハンドラ/フォーマッタ（%(timestamp)s など）を完全排除
    - 自前の Stream & RotatingFile ハンドラのみ使用
    """
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "eval.log")

    logger = logging.getLogger("ucn_eval")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    # 外部で付いたハンドラを完全クリア（%(timestamp)s を要求する異種フォーマッタ対策）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))

    fh = handlers.RotatingFileHandler(log_path, maxBytes=20*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))

    logger.addHandler(ch)
    logger.addHandler(fh)

    # 例外時にロギングが止まらないように
    logging.raiseExceptions = False
    return logger




def log_env(logger: logging.Logger) -> None:
    try:
        logger.info("torch: %s | build_cuda: %s | cuda_available: %s",
                    torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
            # CUDA 12.8 確認（警告のみ）
            if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
                logger.warning("警告: この PyTorch ビルドの CUDA 表示は %s です（12.8 以外）。既存環境に合わせて続行します。",
                               getattr(torch.version, "cuda", None))
    except Exception:
        pass


# ========== データ列挙 / ペアリング ==========
def _ext_lower(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def list_images(root: str, split: str, camera: str) -> List[str]:
    base = os.path.join(root, split, camera)
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if _ext_lower(f) in ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    out.sort()
    return out

def rel_dir_and_stem(image_path: str, split_root: str) -> Tuple[str, str]:
    # rel_dir: front/{segment} / stem: e.g., 1510593619939663_first
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

def path_ucn_png(gen_root: str, split: str, rel_dir: str, stem: str) -> str:
    # 生成結果は {stem}_ucn.png（num_samples>1 は _ucn-000.png などに対応）
    p = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn.png")
    if os.path.exists(p):
        return p
    cand = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn-000.png")
    return cand

def enumerate_pairs(orig_root: str, gen_root: str, split: str, camera: str, limit: int) -> List[Tuple[str, str, str, str]]:
    """
    戻り値: List[(image_path_X, image_path_FX, rel_dir, stem)]
    """
    items = list_images(orig_root, split, camera)
    if limit > 0:
        items = items[:limit]
    pairs: List[Tuple[str, str, str, str]] = []
    for p in items:
        rel_dir, stem = rel_dir_and_stem(p, os.path.join(orig_root, split))
        fx = path_ucn_png(gen_root, split, rel_dir, stem)
        if os.path.exists(fx):
            pairs.append((p, fx, rel_dir, stem))
    return pairs


class Cache:
    """
    キャッシュ構造 (root = --cache-root):
      - clip/{split}/{rel_dir}/{stem}_{x|fx}.npz          : CLIP 画像埋め込み
      - clip_text/{split}/{rel_dir}/{stem}_text.npz       : CLIP テキスト埋め込み
      - canny_fx/{split}/{rel_dir}/{stem}_edge.png        : F(X) 側の Canny
      - depth_fx/{split}/{rel_dir}/{stem}_depth.npy       : F(X) 側の Metric3D 深度
      - semseg_fx/{split}/{rel_dir}/{stem}_predTrainId.npy: F(X) 側の OneFormer セマセグ trainId
      - yolop_x / yolop_fx                                : YOLOP (drivable/lane) 結果 JSON
      - gdino_x / gdino_fx                                : GroundingDINO 検出結果 JSON
      - ocr_x / ocr_fx                                    : OCR 結果 JSON
      - diversity/{split}_diversity.json                  : 多様性指標(LPIPS,1-MSSSIM)
      - experiments/{experiment_id}.eval.json             : 実験ごとのまとめ JSON
      - experiments/experiments.index.json                : 実験一覧
      - logs/eval.log, 各種 NPY など
    """
    def __init__(self, root: str):
        self.root = root
        self.d_clip   = os.path.join(root, "clip")
        self.d_canny  = os.path.join(root, "canny_fx")
        self.d_depth  = os.path.join(root, "depth_fx")
        self.d_semseg = os.path.join(root, "semseg_fx")
        self.d_yolo_x = os.path.join(root, "yolop_x")
        self.d_yolo_f = os.path.join(root, "yolop_fx")
        self.d_gd_x   = os.path.join(root, "gdino_x")
        self.d_gd_f   = os.path.join(root, "gdino_fx")
        self.d_ocr_x  = os.path.join(root, "ocr_x")
        self.d_ocr_f  = os.path.join(root, "ocr_fx")
        self.d_logs   = os.path.join(root, "logs")
        # 新規キャッシュ: テキスト埋め込み / 多様性 / 実験メタ
        self.d_clip_txt     = os.path.join(root, "clip_text")
        self.d_diversity    = os.path.join(root, "diversity")
        self.d_experiments  = os.path.join(root, "experiments")
        # -------------- ここから挿入してください（SOTA Drivable キャッシュ）--------------
        self.d_sota_x = os.path.join(root, "drivable_sota_x")
        self.d_sota_f = os.path.join(root, "drivable_sota_fx")
        # -------------- ここまで挿入してください（SOTA Drivable キャッシュ）--------------
        for d in [
            self.d_clip, self.d_canny, self.d_depth, self.d_semseg,
            self.d_yolo_x, self.d_yolo_f, self.d_gd_x, self.d_gd_f,
            self.d_ocr_x, self.d_ocr_f, self.d_logs,
            self.d_clip_txt, self.d_diversity, self.d_experiments,
            # -------------- ここから挿入してください（SOTA Drivable キャッシュ）--------------
            self.d_sota_x, self.d_sota_f,
            # -------------- ここまで挿入してください（SOTA Drivable キャッシュ）--------------            
        ]:
            ensure_dir(d)

    # ---- CLIP / Diversity / Experiments 用のパスユーティリティ ----
    def clip_text_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_clip_txt, split, rel_dir, f"{stem}_text.npz")

    def diversity_result_path(self, split: str) -> str:
        return os.path.join(self.d_diversity, f"{split}_diversity.json")

    def experiment_eval_path(self, experiment_id: str) -> str:
        safe = experiment_id.replace("/", "_")
        return os.path.join(self.d_experiments, f"{safe}.eval.json")

    def experiment_index_path(self) -> str:
        return os.path.join(self.d_experiments, "experiments.index.json")

    # ---- 既存のパスユーティリティ（そのまま） ----
    def clip_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        # which in {"x","fx"}
        return os.path.join(self.d_clip, split, rel_dir, f"{stem}_{which}.npz")

    def canny_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_canny, split, rel_dir, f"{stem}_edge.png")

    def depth_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_depth, split, rel_dir, f"{stem}_depth.npy")

    def semseg_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_semseg, split, rel_dir, f"{stem}_predTrainId.npy")

    def yolo_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_yolo_x if which == "x" else self.d_yolo_f
        return os.path.join(d, split, rel_dir, f"{stem}_yolop.json")

    def gdino_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_gd_x if which == "x" else self.d_gd_f
        return os.path.join(d, split, rel_dir, f"{stem}_gdino.json")

    def ocr_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_ocr_x if which == "x" else self.d_ocr_f
        return os.path.join(d, split, rel_dir, f"{stem}_ocr.json")
    # -------------- ここから挿入してください（SOTA Drivable パス）--------------
    def sota_drv_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_sota_x if which == "x" else self.d_sota_f
        return os.path.join(d, split, rel_dir, f"{stem}_sota_drv.json")
    # -------------- ここまで挿入してください（SOTA Drivable パス）--------------

# ========== 画像 I/O と注釈 ==========
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imread_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Failed to read image(gray): {path}")
    return g

def save_indexed_png(path: str, arr_u8: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr_u8)

# ---- 注釈描画ユーティリティ ----
_CITYSCAPES_TRAINID_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]  # BGR(OpenCV)

def colorize_trainId(trainId: np.ndarray) -> np.ndarray:
    h, w = trainId.shape[:2]
    out = np.zeros((h,w,3), dtype=np.uint8)
    for idx, (b,g,r) in enumerate(_CITYSCAPES_TRAINID_COLORS):
        if idx >= 19: break
        out[trainId==idx] = (b,g,r)
    return out

def draw_boxes(rgb: np.ndarray, dets: List[Dict[str,Any]], color=(0,255,0)) -> np.ndarray:
    img = rgb.copy()
    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        cls = d.get("cls","?")
        score = d.get("score",0.0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{cls}:{score:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
# ここから挿入：OneFormer の trainId から「道路系 Drivable マスク」を生成する関数
def semseg_to_drivable_onefroad(trainId: np.ndarray,
                                road_ids: Optional[List[int]] = None) -> np.ndarray:
    """
    OneFormer / Cityscapes の trainId マップから Drivable マスクを生成する。

    引数:
        trainId : np.ndarray, dtype=uint8 or int
            Cityscapes の trainId (0..18) を持つ HxW 配列。
        road_ids : Optional[List[int]]
            Drivable とみなす trainId のリスト。
            None の場合は [0]（純粋な road のみ）を既定とする。

    戻り値:
        mask : np.ndarray, dtype=uint8
            0/1 の Drivable マスク（1=drivable）。
    """
    if road_ids is None:
        # Cityscapes trainId=0 は "road"
        # 必要に応じて sidewalk(1) を足すなどの変更はここに集約する。
        road_ids = [0]

    mask = np.zeros_like(trainId, dtype=np.uint8)
    for rid in road_ids:
        mask[trainId == rid] = 1
    return mask
# ここまで挿入

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float=0.4, color=(0,255,0)) -> np.ndarray:
    col = np.zeros_like(rgb); col[mask>0] = color
    out = cv2.addWeighted(rgb, 1.0, col, alpha, 0.0)
    return out

def save_annotations_objects(outdir: str, split: str, rel_dir: str, stem: str,
                             rgbx: np.ndarray, rgbf: np.ndarray,
                             G: List[Dict[str,Any]], D: List[Dict[str,Any]],
                             drv_x: Optional[np.ndarray]=None, drv_f: Optional[np.ndarray]=None) -> None:
    """
    Objects 可視化（GroundingDINO＋任意でdrivable overlay）。
    重要：
      - X(1920x1280) と F(X)(通常512x512) は解像度が異なるため、
        可視化の公平性のため F(X) 側の画像・bbox を X の解像度に揃えてから描画する。
    """
    ensure_dir(os.path.join(outdir, split, rel_dir))

    # --- X の解像度 ---
    Hx, Wx = rgbx.shape[:2]

    # --- F(X) RGB を X 解像度へアップサンプル ---
    if rgbf.shape[:2] != (Hx, Wx):
        rgbf_vis = cv2.resize(rgbf, (Wx, Hx), interpolation=cv2.INTER_LINEAR)
    else:
        rgbf_vis = rgbf.copy()

    # --- F(X) 側の bbox 座標を X 解像度へスケール変換 ---
    # GroundingDINO 出力の bbox は「元 F(X) の解像度」基準
    D_scaled = _rescale_dets_to(D, src_size=rgbf.shape[:2], dst_size=(Hx, Wx))

    # --- X 側 bbox 描画 ---
    x_vis = draw_boxes(rgbx, G, color=(0,255,0))

    # --- F(X) 側 bbox 描画（解像度揃え後） ---
    f_vis = draw_boxes(rgbf_vis, D_scaled, color=(0,0,255))

    # --- drivable overlay がある場合は必ず X サイズに揃える ---
    if drv_x is not None:
        if drv_x.shape[:2] != (Hx, Wx):
            drv_x_resized = cv2.resize(drv_x.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
        else:
            drv_x_resized = drv_x
        x_vis = overlay_mask(x_vis, drv_x_resized > 0, alpha=0.35, color=(0,255,255))

    if drv_f is not None:
        if drv_f.shape[:2] != (Hx, Wx):
            drv_f_resized = cv2.resize(drv_f.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
        else:
            drv_f_resized = drv_f
        f_vis = overlay_mask(f_vis, drv_f_resized > 0, alpha=0.35, color=(255,255,0))

    # --- 保存（2枚別々：concat なしなので安全 & 今後 concat 拡張にも対応可能） ---
    p1 = os.path.join(outdir, split, rel_dir, f"{stem}_x_objs.jpg")
    p2 = os.path.join(outdir, split, rel_dir, f"{stem}_fx_objs.jpg")
    cv2.imwrite(p1, cv2.cvtColor(x_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p2, cv2.cvtColor(f_vis, cv2.COLOR_RGB2BGR))


def save_annotations_structure(outdir: str, split: str, rel_dir: str, stem: str,
                               rgbx: np.ndarray, rgbf: np.ndarray,
                               edge_x: np.ndarray, edge_fx: np.ndarray,
                               depth_x: np.ndarray, depth_fx: np.ndarray,
                               seg_x: np.ndarray, seg_fx: np.ndarray) -> None:
    """
    構造系の比較可視化:
      上段: X 側  [RGB_X, Edge_X, Depth_X, Semseg_X]
      下段: F(X) 側 [RGB_F, Edge_F, Depth_F, Semseg_F]

    注意:
      - Waymo 元画像 X は 1920x1280、F(X) は 512x512 など解像度が異なる。
      - ここでは「見やすさ」を優先して、すべて rgbx の解像度にリサイズしてからモザイクを組む。
    """
    ensure_dir(os.path.join(outdir, split, rel_dir))

    def _vis_edge(e: np.ndarray) -> np.ndarray:
        # 2値エッジ → 3ch BGR
        return cv2.cvtColor((e > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    def _vis_depth(d: np.ndarray) -> np.ndarray:
        # min-max 正規化 + カラーマップ (INFERNO)
        dd = d.astype(np.float32).copy()
        dd -= np.min(dd)
        dd /= (np.max(dd) + 1e-9)
        dd = (dd * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(dd, cv2.COLORMAP_INFERNO)

    # 可視化タイルを作成（まだ解像度はバラバラ）
    ex3 = _vis_edge(edge_x)
    ef3 = _vis_edge(edge_fx)
    dx3 = _vis_depth(depth_x)
    df3 = _vis_depth(depth_fx)
    sx3 = colorize_trainId(seg_x)
    sf3 = colorize_trainId(seg_fx)

    # === ここから追加: すべてを rgbx の解像度にそろえる =========================
    Hx, Wx = rgbx.shape[:2]

    def _resize_to_x(img: np.ndarray) -> np.ndarray:
        """rgbx と同じ (Hx, Wx) に最近傍補間で揃える（構造可視化用なので最近傍で十分）"""
        if img.shape[0] == Hx and img.shape[1] == Wx:
            return img
        return cv2.resize(img, (Wx, Hx), interpolation=cv2.INTER_NEAREST)

    # X 側タイル
    ex3 = _resize_to_x(ex3)
    dx3 = _resize_to_x(dx3)
    sx3 = _resize_to_x(sx3)

    # F(X) 側タイルも X 側の解像度に合わせる（上下で幅を揃えるため）
    ef3 = _resize_to_x(ef3)
    df3 = _resize_to_x(df3)
    sf3 = _resize_to_x(sf3)

    # F(X) の RGB も X と同じ解像度に揃える
    if rgbf.shape[0] != Hx or rgbf.shape[1] != Wx:
        rgbf_vis = cv2.resize(rgbf, (Wx, Hx), interpolation=cv2.INTER_LINEAR)
    else:
        rgbf_vis = rgbf.copy()
    # === ここまで追加 ==========================================================

    # 横に 4 枚並べた 1 行を 2 段重ねる
    row_x = np.concatenate([rgbx, ex3, dx3, sx3], axis=1)
    row_f = np.concatenate([rgbf_vis, ef3, df3, sf3], axis=1)
    cat   = np.concatenate([row_x, row_f], axis=0)

    out_path = os.path.join(outdir, split, rel_dir, f"{stem}_struct_pair.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))



def save_annotations_drivable(outdir: str, split: str, rel_dir: str, stem: str,
                              rgbx: np.ndarray, rgbf: np.ndarray,
                              drv_x: np.ndarray, drv_f: np.ndarray) -> None:
    """
    走行可能領域（Drivable）の可視化。
    X(1920x1280) と F(X)(通常512x512) が異解像度なので、
    必ず両者を X の解像度に揃えてから横連結する。
    """
    ensure_dir(os.path.join(outdir, split, rel_dir))

    # --- X 解像度を基準にする ---
    Hx, Wx = rgbx.shape[:2]

    # --- F(X) RGB を X 解像度へ揃える ---
    if rgbf.shape[:2] != (Hx, Wx):
        rgbf_vis = cv2.resize(rgbf, (Wx, Hx), interpolation=cv2.INTER_LINEAR)
    else:
        rgbf_vis = rgbf.copy()

    # --- Drivable マスクも X 解像度へ揃える ---
    if drv_x.shape[:2] != (Hx, Wx):
        drv_x_resized = cv2.resize(drv_x.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
    else:
        drv_x_resized = drv_x

    if drv_f.shape[:2] != (Hx, Wx):
        drv_f_resized = cv2.resize(drv_f.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
    else:
        drv_f_resized = drv_f

    # --- オーバレイ可視化（色を少し変える） ---
    x_vis = overlay_mask(rgbx, drv_x_resized > 0, alpha=0.35, color=(0,255,255))
    f_vis = overlay_mask(rgbf_vis, drv_f_resized > 0, alpha=0.35, color=(255,255,0))

    # --- 横連結（解像度が揃っているので必ず成功） ---
    cat = np.concatenate([x_vis, f_vis], axis=1)

    out_path = os.path.join(outdir, split, rel_dir, f"{stem}_drivable.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))

# ここから挿入：Drivable を複数ソースで可視化する Multi 版
def save_annotations_drivable_multi(outdir: str, split: str, rel_dir: str, stem: str,
                                    rgbx: np.ndarray, rgbf: np.ndarray,
                                    masks: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    複数の Drivable 定義（例: YOLOP, OneFormer-road, SOTA-model）を
    一度に可視化する。

    引数:
        outdir : ベース出力ディレクトリ
        split, rel_dir, stem : Waymo の split / front/{segment} / stem
        rgbx, rgbf : 元画像 X, 生成画像 F(X) (RGB, HxWx3)
        masks : Dict[source_name, (mask_x, mask_f)]
            - source_name は "yolop", "onefroad", "sota" など任意のキー
            - 各 mask は 0/1 の np.ndarray (HxW) or 任意解像度
              → 関数内で X の解像度に揃えてから overlay する。

    出力:
        {stem}_drivable_{source_name}.jpg をソースごとに保存する。
    """
    base_dir = os.path.join(outdir, split, rel_dir)
    ensure_dir(base_dir)

    Hx, Wx = rgbx.shape[:2]
    # F(X) も X サイズに揃えたものを 1 度作って使い回す
    if rgbf.shape[:2] != (Hx, Wx):
        rgbf_vis = cv2.resize(rgbf, (Wx, Hx), interpolation=cv2.INTER_LINEAR)
    else:
        rgbf_vis = rgbf.copy()

    for src_name, (drv_x, drv_f) in masks.items():
        if drv_x is None or drv_f is None:
            continue

        # マスクを X 解像度に揃える
        if drv_x.shape[:2] != (Hx, Wx):
            drv_x_resized = cv2.resize(drv_x.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
        else:
            drv_x_resized = drv_x

        if drv_f.shape[:2] != (Hx, Wx):
            drv_f_resized = cv2.resize(drv_f.astype(np.uint8), (Wx, Hx), interpolation=cv2.INTER_NEAREST)
        else:
            drv_f_resized = drv_f

        # オーバレイ（色は source ごとに変えてもよいが、まずは固定2色で）
        x_vis = overlay_mask(rgbx, drv_x_resized > 0, alpha=0.35, color=(0,255,255))
        f_vis = overlay_mask(rgbf_vis, drv_f_resized > 0, alpha=0.35, color=(255,255,0))

        cat = np.concatenate([x_vis, f_vis], axis=1)
        out_path = os.path.join(base_dir, f"{stem}_drivable_{src_name}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))
# ここまで挿入


# ========== (1) リアリティ指標：CLIP特徴, FID/CMMD ==========
class ClipEmbedder:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        # GPU必須：なければ即エラー
        if not torch.cuda.is_available():
            raise RuntimeError("GPU(CUDA)が必須です（CLIP）。CPUフォールバックは禁止。")
        self.device = "cuda"
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.max_batch: Optional[int] = None

    def load(self, logger: logging.Logger):
        if self.model is None:
            logger.info("[CLIP] loading model: %s (GPU only)", self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def autotune(self, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 1024) -> int:
        self.max_batch = autotune_clip_bs(self, logger, sample_rgb, cap=cap)
        return self.max_batch

    @torch.inference_mode()
    def embed_batch(self, images: List[np.ndarray]) -> np.ndarray:
        assert self.model is not None and self.processor is not None
        inputs = self.processor(images=images, return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(pix)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)

    # ★ここから挿入（テキスト埋め込み）=============================
    @torch.inference_mode()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        CLIP テキスト埋め込み（L2 正規化済み）
        """
        assert self.model is not None and self.processor is not None
        enc = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(input_ids=ids, attention_mask=attn)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)
    # ★ここまで挿入==================================================
def feats_to_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    CLIP-FID 用: 平均ベクトルと共分散行列を返す
    feats: (N, D)
    """
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma
# ★ここから挿入（距離・ペアサンプル・プロンプト読み）====================
def pairwise_sqeuclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    L2 正規化済み特徴に対する二乗ユークリッド距離行列（a: [N,D], b: [M,D]）
    d^2 = ||a-b||^2 = 2*(1 - cos(a,b)) と一致
    """
    aa = np.sum(a*a, axis=1, keepdims=True)   # (N,1) == 1
    bb = np.sum(b*b, axis=1, keepdims=True)   # (M,1) == 1
    return np.maximum(aa + bb.T - 2.0*(a @ b.T), 0.0)

def sample_unique_pairs(n: int, m: int, rng: np.random.RandomState) -> List[Tuple[int,int]]:
    """
    n 個から重複なし順不同のペアを m 本だけランダム抽出
    """
    if n < 2 or m <= 0:
        return []
    # 総組合せ
    tot = n*(n-1)//2
    m = min(m, tot)
    # reservoir sampling 相当（高速）
    # インデックス空間 [0, tot) の番号→(i,j) へ射影
    # ここでは単純化して乱択→辞書で重複排除
    chosen = set()
    while len(chosen) < m:
        k = int(rng.randint(0, tot))
        chosen.add(k)
    def unrank(k):
        # 辞書式順序の組合せ番号→ペア (i,j)
        i = int((1+np.sqrt(1+8*k))//2)
        while i*(i-1)//2 > k: i -= 1
        base = i*(i-1)//2
        j = k - base
        return j, i
    return [unrank(k) for k in chosen]

def ucn_meta_path(gen_root: str, split: str, rel_dir: str, stem: str) -> str:
    """生成 PNG {stem}_ucn.png に対するメタ JSON パス"""
    p = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn.meta.json")
    if os.path.exists(p):
        return p
    # num_samples>1 の場合
    p2 = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn-000.meta.json")
    return p2

def ts_from_stem(stem: str) -> str:
    return stem.split('_')[0] if '_' in stem else stem

def prompt_from_meta_or_files(gen_root: str, prompt_root: str, split: str, rel_dir: str, stem: str) -> Optional[str]:
    """
    1) メタ JSON の prompt_used
    2) {stem}_prompt.txt
    3) {timestamp}_prompt.txt
    の順に探索
    """
    meta = ucn_meta_path(gen_root, split, rel_dir, stem)
    if os.path.exists(meta):
        try:
            obj = json.load(open(meta, "r", encoding="utf-8"))
            p = obj.get("prompt_used", None)
            if isinstance(p, str) and len(p.strip()) > 0:
                return p.strip()
        except Exception:
            pass
    cand1 = os.path.join(prompt_root, split, rel_dir, f"{stem}_prompt.txt")
    if os.path.exists(cand1):
        return open(cand1, "r", encoding="utf-8").read().strip()
    cand2 = os.path.join(prompt_root, split, rel_dir, f"{ts_from_stem(stem)}_prompt.txt")
    if os.path.exists(cand2):
        return open(cand2, "r", encoding="utf-8").read().strip()
    return None
# ★ここまで挿入========================================================
def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """FID 距離（Fréchet distance）"""
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def gaussian_mmd(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    CLIP 埋め込みに対する MMD^2（RBF）
    - sigma が None のときは median heuristic（自己距離の 0 を除外）
    - 計算は特徴の内積からの二乗距離で行い、gamma=1/(2*sigma^2)
    """
    def pdist_sq(a: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)          # (N,1)
        return aa + aa.T - 2.0 * (a @ a.T)                 # (N,N)

    def cdist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)          # (N,1)
        bb = np.sum(b * b, axis=1, keepdims=True)          # (M,1)
        return aa + bb.T - 2.0 * (a @ b.T)                 # (N,M)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xx = pdist_sq(x)
    yy = pdist_sq(y)
    xy = cdist_sq(x, y)

    if sigma is None:
        n = xx.shape[0]; m = yy.shape[0]
        # 自己距離を除外
        xx_off = xx[~np.eye(n, dtype=bool)]
        yy_off = yy[~np.eye(m, dtype=bool)]
        pool = np.concatenate([xx_off, yy_off, xy.ravel()])
        # 数値安定性のための下限
        med = float(np.median(pool)) if pool.size > 0 else 1.0
        med = max(med, 1e-12)
        sigma = math.sqrt(0.5 * med)
    sigma = max(float(sigma), 1e-9)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * xx)
    k_yy = np.exp(-gamma * yy)
    k_xy = np.exp(-gamma * xy)
    m = x.shape[0]; n = y.shape[0]
    mmd2 = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1) + 1e-9) \
         + (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1) + 1e-9) \
         - 2.0 * np.sum(k_xy) / (m * n + 1e-9)
    return float(mmd2)



# ========== (2) 構造・意味忠実度 ==========
# --- Edge(Canny) ---
def canny_cpu(rgb: np.ndarray, t1: float = 100.0, t2: float = 200.0, blur_ksize: int = 3) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_ksize > 0:
        g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    e = cv2.Canny(g, t1, t2)
    return e

def edge_metrics(edge_x: np.ndarray, edge_fx: np.ndarray) -> Dict[str, float]:
    """
    2値エッジの一致度（L1/RMSE/IoU/F1）。

    仕様変更（Augmentation 強度評価と整合）:
      - Canny 自体は画像全体に対して計算するが、
      - 指標計算は「画像の下半分」（道路・横断歩道などが主に写る領域）のみを使う。
        => 上半分の空や標識などは Edge 指標で縛らない
        => 標識については GroundingDINO ベースの Objects 指標で評価する。
    """
    # まず解像度を揃える（edge_x を edge_fx のサイズに合わせる）
    if edge_x.shape != edge_fx.shape:
        edge_x = cv2.resize(edge_x, (edge_fx.shape[1], edge_fx.shape[0]), interpolation=cv2.INTER_NEAREST)

    a = (edge_x > 0).astype(np.uint8)
    b = (edge_fx > 0).astype(np.uint8)

    # 下半分だけマスク
    h, w = a.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2 :, :] = True  # 下半分

    a_m = a[mask]
    b_m = b[mask]

    # 念のため保険（極端なケースで mask が空なら全体を使う）
    if a_m.size == 0 or b_m.size == 0:
        a_m = a.reshape(-1)
        b_m = b.reshape(-1)

    a_f = a_m.astype(np.float32)
    b_f = b_m.astype(np.float32)

    l1   = float(np.mean(np.abs(a_f - b_f)))
    rmse = float(np.sqrt(np.mean((a_f - b_f) ** 2)))

    # IoU / F1 用に bool に戻す
    a_b = (a_m > 0)
    b_b = (b_m > 0)

    inter = float(np.sum(a_b & b_b))
    union = float(np.sum(a_b | b_b) + 1e-9)
    iou   = inter / union

    tp = inter
    fp = float(np.sum(~a_b & b_b))
    fn = float(np.sum(a_b & ~b_b))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return {
        "edge_l1":   l1,
        "edge_rmse": rmse,
        "edge_iou":  float(iou),
        "edge_f1":   float(f1),
    }
# -------------- ここから挿入してください（Edge ROI / Traffic Sign）--------------
# 対象クラス（_canonicalize_label() 後の正規化名で判定）
SIGN_CLASSES_DEFAULT = {
    "traffic sign", "stop sign", "speed limit sign", "crosswalk sign", "construction sign"
}

def edge_metrics_roi(edge_x: np.ndarray, edge_fx: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:
    """
    ROI（True領域）に限定した Edge 指標（L1/RMSE/IoU/F1）。
    - edge_x は edge_fx に合わせてサイズを揃える
    - roi_mask は edge_fx のサイズに最近傍リサイズして利用
    - 平均は ROI 内画素でとる（ROIが空ならゼロ返し）
    """
    if edge_x.shape != edge_fx.shape:
        edge_x = cv2.resize(edge_x, (edge_fx.shape[1], edge_fx.shape[0]), interpolation=cv2.INTER_NEAREST)

    rm = roi_mask
    if rm.shape[:2] != edge_fx.shape[:2]:
        rm = cv2.resize(rm.astype(np.uint8), (edge_fx.shape[1], edge_fx.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (rm > 0)

    if np.sum(m) == 0:
        return {"edge_l1": 0.0, "edge_rmse": 0.0, "edge_iou": 0.0, "edge_f1": 0.0, "empty_roi": 1.0}

    a = (edge_x > 0) & m
    b = (edge_fx > 0) & m

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    l1   = float(np.mean(np.abs(a_f - b_f)))
    rmse = float(np.sqrt(np.mean((a_f - b_f) ** 2)))

    inter = float(np.sum(a & b))
    union = float(np.sum(a | b) + 1e-9)
    iou   = inter / union

    tp = inter
    fp = float(np.sum(~a & b))
    fn = float(np.sum(a & ~b))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"edge_l1": l1, "edge_rmse": rmse, "edge_iou": float(iou), "edge_f1": float(f1), "empty_roi": 0.0}

def build_roi_mask_from_dets(
    dets_x: List[Dict[str,Any]],
    dets_f: List[Dict[str,Any]],
    src_size_x: Tuple[int,int],
    src_size_f: Tuple[int,int],
    dst_size: Tuple[int,int],
    sign_classes: set,
) -> np.ndarray:
    """
    X側/F(X)側のDINO検出（ピクセル座標）から、対象クラスの矩形を
    目標解像度(dst_size=(H,W))にスケールして塗りつぶした ROI マスク(0/1)を作る。
    """
    Hd, Wd = dst_size
    mask = np.zeros((Hd, Wd), dtype=np.uint8)

    def _paint(dets: List[Dict[str,Any]], src_size: Tuple[int,int]) -> None:
        scaled = _rescale_dets_to(
            [d for d in dets if d.get("cls","") in sign_classes],
            src_size, (Hd, Wd)
        )
        for d in scaled:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
            x1 = max(0, min(Wd-1, x1)); x2 = max(0, min(Wd-1, x2))
            y1 = max(0, min(Hd-1, y1)); y2 = max(0, min(Hd-1, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2+1, x1:x2+1] = 1

    _paint(dets_x, src_size_x)
    _paint(dets_f, src_size_f)
    return mask
# -------------- ここまで挿入してください（Edge ROI / Traffic Sign）--------------
# --- Depth(Metric3Dv2; si-RMSE 等) ---
import onnxruntime as ort

def build_metric3d_session(onnx_path: str) -> Tuple[ort.InferenceSession, str, str, List[str]]:
    """
    ONNXRuntime を CUDAExecutionProvider のみで作成（CPUExecutionProvider を与えない）。
    CUDA が無ければ明示的に失敗させる（CPUフォールバック禁止）。
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNXRuntime: CUDAExecutionProvider が見つかりません。CPUフォールバックは禁止です。")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider"]  # ← ここが肝：CPU を渡さない
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name, providers


def infer_metric3d_np(session: ort.InferenceSession, in_name: str, out_name: str, rgb: np.ndarray) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    t = np.transpose(rgb_resized, (2,0,1))[None, ...]  # (1,3,H,W)
    y = session.run([out_name], {in_name: t})[0]
    depth = np.squeeze(y).astype(np.float32)
    depth_up = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth_up

def depth_metrics(depth_x: np.ndarray, depth_fx: np.ndarray) -> Dict[str, float]:
    """
    Depth の RMSE / scale-invariant RMSE / 相対誤差。サイズが異なる場合は depth_x を depth_fx サイズにリサイズして比較。
    """
    if depth_x.shape != depth_fx.shape:
        depth_x = cv2.resize(depth_x, (depth_fx.shape[1], depth_fx.shape[0]), interpolation=cv2.INTER_LINEAR)

    eps = 1e-6
    dx = np.maximum(depth_x, eps); df = np.maximum(depth_fx, eps)
    rmse = float(np.sqrt(np.mean((dx - df)**2)))
    d = np.log(dx) - np.log(df)
    si_rmse = float(np.sqrt(np.mean(d**2) - (np.mean(d)**2)))
    rel = np.abs(dx - df) / (dx + eps)
    rel_mean = float(np.mean(rel))
    return {"depth_rmse": rmse, "depth_si_rmse": si_rmse, "depth_rel": rel_mean}

def build_oneformer(model_id: str, device: str = "cuda", fp16: bool = True):
    """
    OneFormer (Cityscapes) を GPU のみでロード（CPUフォールバック禁止）。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(CUDA)が必須です（OneFormer）。CPUフォールバックは禁止。")

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if fp16 else torch.float32

    try:
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id, torch_dtype=dtype).eval()
    except TypeError:
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id, dtype=dtype).eval()

    model = model.to("cuda")
    return processor, model




def oneformer_semseg(processor, model, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    enc = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")
    pv = enc.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        if pv.ndim == 3: pv = pv.unsqueeze(0)
    else:
        pv = torch.from_numpy(np.array(pv))
    pv = pv.to(model.device, dtype=model.dtype)
    ti = enc.get("task_inputs")
    if isinstance(ti, torch.Tensor): ti = ti.to(model.device)
    with torch.inference_mode():
        if (model.dtype == torch.float16) and (model.device.type == "cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=pv, task_inputs=ti)
        else:
            out = model(pixel_values=pv, task_inputs=ti)
    seg = processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
    return seg.to("cpu").numpy().astype(np.uint8)

def confusion_19(gt: np.ndarray, pr: np.ndarray, ncls: int = 19) -> np.ndarray:
    mask = (gt >= 0) & (gt < ncls)
    hist = np.bincount(ncls * gt[mask].astype(int) + pr[mask].astype(int), minlength=ncls**2).reshape(ncls, ncls)
    return hist

def miou_from_conf(hist: np.ndarray) -> Tuple[float, np.ndarray]:
    # IoU_c = TP / (TP + FP + FN)
    tp = np.diag(hist).astype(np.float64)
    fp = np.sum(hist, axis=0) - tp
    fn = np.sum(hist, axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    miou = float(np.nanmean(iou))
    return miou, iou

# ======== Auto-Batch Utilities: 各モデルの最大バッチを OOM を避けて自動探索 ========

def _chunks(lst, n):
    for i in range(0, len(lst), max(1, n)):
        yield lst[i:i+n]

def _is_cuda_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return ("cuda out of memory" in msg) or ("cudnn" in msg and "error" in msg) or ("allocator" in msg and "memory" in msg)

def _torch_sync_empty_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

# ---------- CLIP: 自動バッチ計測 ----------
def autotune_clip_bs(clipper, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 1024) -> int:
    """
    CLIP の get_image_features で使える最大バッチを探索。
    - 二分探索（指数増加→失敗→範囲内二分）
    """
    if not torch.cuda.is_available() or clipper.device == "cpu":
        logger.info("[AutoBatch][CLIP] CPU 推論のため batch=1 固定")
        return 1
    clipper.load(logger)
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            imgs = [sample_rgb] * b
            enc = clipper.processor(images=imgs, return_tensors="pt")
            pix = enc["pixel_values"].to(clipper.device)
            with torch.inference_mode():
                _ = clipper.model.get_image_features(pix)
            del enc, pix, _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            imgs = [sample_rgb] * mid
            enc = clipper.processor(images=imgs, return_tensors="pt")
            pix = enc["pixel_values"].to(clipper.device)
            with torch.inference_mode():
                _ = clipper.model.get_image_features(pix)
            del enc, pix, _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][CLIP] batch=%d", lo)
    return max(1, int(lo))

# ---------- OneFormer: 一括推論 + 自動バッチ ----------
def oneformer_semseg_batch(processor, model, rgbs: List[np.ndarray]) -> List[np.ndarray]:
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
    with torch.inference_mode():
        if (model.dtype == torch.float16) and (model.device.type == "cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=pv, task_inputs=ti)
        else:
            out = model(pixel_values=pv, task_inputs=ti)
    segs = processor.post_process_semantic_segmentation(out, target_sizes=sizes)
    out_list = [s.to("cpu").numpy().astype(np.uint8) for s in segs]
    return out_list

def autotune_oneformer_bs(processor, model, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 64) -> int:
    if not (torch.cuda.is_available() and model.device.type == "cuda"):
        logger.info("[AutoBatch][OneFormer] CPU/非CUDA のため batch=1 固定")
        return 1
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            _ = oneformer_semseg_batch(processor, model, [sample_rgb]*b)
            del _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1)//2
        try:
            _ = oneformer_semseg_batch(processor, model, [sample_rgb]*mid)
            del _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][OneFormer] batch=%d", lo)
    return max(1, int(lo))

# ---------- YOLOP: 一括推論 + 自動バッチ ----------
@torch.inference_mode()
def yolop_infer_batch(model, rgbs: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    YOLOP バッチ推論（正しい2値化）
      - 入力を 640x640, [0,1] 正規化
      - da_seg_out / ll_seg_out のロジット→argmaxで2値化
      - 2値マスクを最近傍補間で 640x640 に揃えて返す
      - 返り値 'drivable' / 'lane' は 0/1 の np.uint8 マスク
    """
    dev = next(model.parameters()).device
    size = 640
    imgs = [cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    ten = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float().to(dev) / 255.0

    det_out, da_seg_out, ll_seg_out = model(ten)  # da/ll: [B,2,h,w]（ロジット）

    # 2値化（argmax）→ 最近傍で (size,size) にアップサンプル
    da_pred = torch.argmax(da_seg_out, dim=1).float().unsqueeze(1)   # [B,1,h,w]
    ll_pred = torch.argmax(ll_seg_out, dim=1).float().unsqueeze(1)   # [B,1,h,w]
    da_pred = F.interpolate(da_pred, size=(size, size), mode="nearest").squeeze(1)  # [B,size,size]
    ll_pred = F.interpolate(ll_pred, size=(size, size), mode="nearest").squeeze(1)

    da_np = da_pred.detach().cpu().numpy().astype(np.uint8)  # 0/1
    ll_np = ll_pred.detach().cpu().numpy().astype(np.uint8)

    out: List[Dict[str, Any]] = []
    for i in range(len(rgbs)):
        out.append({
            "det": None,
            "drivable": da_np[i],  # 0/1 (uint8), size×size
            "lane":     ll_np[i],  # 0/1 (uint8), size×size
        })
    return out


def autotune_yolop_bs(model, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 64) -> int:
    if not (torch.cuda.is_available() and next(model.parameters()).is_cuda):
        logger.info("[AutoBatch][YOLOP] CPU/非CUDA のため batch=1 固定")
        return 1
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            _ = yolop_infer_batch(model, [sample_rgb]*b)
            del _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1)//2
        try:
            _ = yolop_infer_batch(model, [sample_rgb]*mid)
            del _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][YOLOP] batch=%d", lo)
    return max(1, int(lo))
# -------------- ここから挿入してください（HF SemanticSeg → Drivable 代替）--------------
# 任意の HuggingFace セマンティックセグモデルから「drivable」相当クラスを抽出する汎用実装
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation  # 既に import 済の場合は重複OK

def load_hf_semseg(logger: logging.Logger, model_id: str, device: str = "cuda", fp16: bool = True):
    """
    任意のセグモデル（HF）をロード。id2label からクラス名を参照できる。
    """
    if not model_id:
        raise ValueError("sota-seg-model が空です。")
    logger.info("[HF-SemSeg] loading: %s", model_id)
    proc = AutoImageProcessor.from_pretrained(model_id)
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    mdl  = AutoModelForSemanticSegmentation.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    id2label = getattr(mdl.config, "id2label", None)
    if not isinstance(id2label, dict) or len(id2label)==0:
        logger.warning("[HF-SemSeg] id2label が見つかりません。ラベル名判定が使えない可能性があります。")
    return proc, mdl, id2label

@torch.inference_mode()
def hf_semseg_drivable_mask(
    processor, model, rgb: np.ndarray, include_label_keywords: List[str]
) -> np.ndarray:
    """
    HFセグモデルの予測から、「ラベル名に include_label_keywords のいずれかを含む」IDを1、それ以外0の2値マスク。
    """
    H, W = rgb.shape[:2]
    enc = processor(images=rgb, return_tensors="pt")
    enc = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
    out = model(**enc).logits  # [B,C,h,w]
    up = torch.nn.functional.interpolate(out, size=(H,W), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

    # id2label から名前を取り、曖昧一致（小文字化）で対象IDを決める
    id2label = getattr(model.config, "id2label", {})
    kws = [k.lower() for k in include_label_keywords]
    include_ids = set()
    if isinstance(id2label, dict) and len(id2label)>0:
        for k, v in id2label.items():
            name = str(v).lower()
            if any(kw in name for kw in kws):
                include_ids.add(int(k))

    # ヒューリスティク: id2label が無い場合は road=0 前提にはしない（安全にゼロマスク）
    mask = np.zeros((H,W), dtype=np.uint8)
    for cid in include_ids:
        mask[pred==cid] = 1
    return mask
# -------------- ここまで挿入してください（HF SemanticSeg → Drivable 代替）--------------

# ---------- Metric3D(ONNXRuntime): 一括推論(対応時) + 自動バッチ ----------
def infer_metric3d_batch(session: ort.InferenceSession, in_name: str, out_name: str, rgbs: List[np.ndarray]) -> List[np.ndarray]:
    # 入力を [B,3,H,W] に積む（ダイナミックバッチ非対応モデルなら例外）
    h0, w0 = rgbs[0].shape[:2]
    resized = [cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) for rgb in rgbs]
    arr = np.stack([np.transpose(r, (2,0,1)) for r in resized], axis=0)
    y = session.run([out_name], {in_name: arr})[0]  # [B,H,W] or [B,1,H,W]
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:,0]
    out_list = []
    for i in range(y.shape[0]):
        depth = y[i].astype(np.float32)
        depth_up = cv2.resize(depth, (rgbs[i].shape[1], rgbs[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        out_list.append(depth_up)
    return out_list

def autotune_metric3d_bs(session: ort.InferenceSession, in_name: str, out_name: str, logger: logging.Logger,
                         sample_rgb: np.ndarray, cap: int = 16) -> int:
    # ONNX のダイナミックバッチに対応していない場合は 1 にフォールバック
    try:
        _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb, sample_rgb])
        del _
        # 2 が通るなら探索開始
        best, b, failed = 1, 2, False
        while b <= cap:
            try:
                _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb]*b)
                del _
                best = b
                b *= 2
            except Exception:
                failed = True
                break
        lo = best
        hi = min(cap, b-1) if failed else best
        while lo < hi:
            mid = (lo + hi + 1)//2
            try:
                _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb]*mid)
                del _
                lo = mid
            except Exception:
                hi = mid - 1
        logger.info("[AutoBatch][Metric3D] batch=%d", lo)
        return max(1, int(lo))
    except Exception:
        logger.info("[AutoBatch][Metric3D] 動的バッチ非対応 → batch=1")
        return 1

def _recanonize_dets(dets: List[Dict[str,Any]], prompts: List[str]) -> List[Dict[str,Any]]:
    out = []
    for d in dets:
        nd = dict(d)
        nd["cls"] = _canonicalize_label(d.get("cls",""), prompts)
        out.append(nd)
    return out

# ========== (3) 物体保持・ハルシネーション ==========



# YOLOP: PyTorch Hub（hustvl/YOLOP）で BDD100K の det+drivable+lane を同時出力
# 依存: prefetch_generator, thop, scikit-image（overlay で導入）。torch は Docker ベース層固定。
def load_yolop(logger: logging.Logger):
    """
    YOLOP を GPU専用でロード。CPUフォールバックは禁止。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(CUDA)が必須です（YOLOP）。CPUフォールバックは禁止。")

    logger.info("[YOLOP] load via torch.hub hustvl/yolop (pretrained=True, trust_repo=True) [GPU only]")
    try:
        import prefetch_generator  # noqa: F401
    except Exception as e:
        logger.error("YOLOP 依存 'prefetch_generator' が見つかりません。overlay を確認: %s", e); raise
    try:
        import thop  # noqa: F401
    except Exception as e:
        logger.warning("thop が見つかりません（続行可能）: %s", e)

    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
    model.eval().to("cuda")
    return model



@torch.inference_mode()
def yolop_infer(model, rgb: np.ndarray) -> Dict[str, Any]:
    """
    YOLOP 単枚推論（正しい2値化）
      - 入力を 640x640, [0,1] 正規化に変換
      - da_seg_out / ll_seg_out は [B, C=2, h, w] のロジットを返す
      - 2値化は argmax(dim=1) で行い、入力解像度(640)に最近傍でリサイズ
      - 返り値の 'drivable' / 'lane' は 0/1 の np.uint8 マスク（640x640）
    """
    img = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ten = ten.to(next(model.parameters()).device)

    det_out, da_seg_out, ll_seg_out = model(ten)  # da/ll: [B,2,h,w] （ロジット）

    # 2値マスク（argmaxでクラス予測 → 最近傍で 640x640 に揃える）
    da_pred = torch.argmax(da_seg_out, dim=1).float().unsqueeze(1)   # [B,1,h,w]
    ll_pred = torch.argmax(ll_seg_out, dim=1).float().unsqueeze(1)   # [B,1,h,w]
    da_pred = F.interpolate(da_pred, size=(640, 640), mode="nearest").squeeze(1)  # [B,640,640]
    ll_pred = F.interpolate(ll_pred, size=(640, 640), mode="nearest").squeeze(1)

    da = da_pred[0].detach().cpu().numpy().astype(np.uint8)  # 0/1
    ll = ll_pred[0].detach().cpu().numpy().astype(np.uint8)

    return {
        "det": None,
        "drivable": da,   # 0/1 (uint8), 640x640
        "lane": ll        # 0/1 (uint8), 640x640
    }


# ---- ラベル正規化ユーティリティ ----
_CANON_PRIOR = [
    # より「具体的」→「汎用」の順で優先（同長の場合はこの順を優先）
    "speed limit sign", "stop sign", "crosswalk sign", "construction sign",
    "traffic light", "traffic cone", "traffic sign",
    "motorcycle", "bicycle", "truck", "bus", "car",
    "pedestrian", "person",
]

# ---- 検出ボックス座標の座標系変換（src_size=(H,W) → dst_size=(H,W)） ----
def _rescale_dets_to(dets: List[Dict[str,Any]],
                     src_size: Tuple[int,int],
                     dst_size: Tuple[int,int]) -> List[Dict[str,Any]]:
    """
    dets: [{'cls': str, 'score': float, 'bbox': [x1,y1,x2,y2]}, ...]
    src_size: (H_src, W_src)  ← dets の現在の座標系
    dst_size: (H_dst, W_dst)  ← 変換先（ここでは X 側画像の解像度）

    画像サイズが異なる場合、D 側（F(X)）のボックスを X 側の解像度に合わせてスケーリングする。
    """
    Hs, Ws = src_size
    Hd, Wd = dst_size
    if (Hs == Hd) and (Ws == Wd):
        return dets

    sx = float(Wd) / max(1.0, float(Ws))
    sy = float(Hd) / max(1.0, float(Hs))

    out: List[Dict[str,Any]] = []
    for d in dets:
        x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
        x1 = x1 * sx; x2 = x2 * sx
        y1 = y1 * sy; y2 = y2 * sy
        # 範囲クリップ（境界外は切り詰め）
        x1 = float(np.clip(x1, 0.0, max(0.0, Wd - 1)))
        x2 = float(np.clip(x2, 0.0, max(0.0, Wd - 1)))
        y1 = float(np.clip(y1, 0.0, max(0.0, Hd - 1)))
        y2 = float(np.clip(y2, 0.0, max(0.0, Hd - 1)))
        nd = dict(d); nd["bbox"] = [x1, y1, x2, y2]
        out.append(nd)
    return out

def _canonicalize_label(raw: str, prompts: List[str]) -> str:
    """
    Grounding DINO が返す複合ラベル（例: 'car bus', 'person pedestrian'）を
    事前に与えた prompts 内の「最も具体的」な1語に正規化する。
    - 長い語を優先、同長は _CANON_PRIOR の順でタイブレーク
    - pedestrian は 'person' に吸収
    """
    if not isinstance(raw, str):
        return str(raw)

    s = raw.lower().strip()
    # 候補抽出（部分一致）
    cands = [p for p in prompts if p in s]
    if not cands:
        return s

    # 長さ降順で並べ、同長は _CANON_PRIOR の順で優先
    def _key(p):
        return (len(p), -_CANON_PRIOR.index(p) if p in _CANON_PRIOR else 0)
    cands.sort(key=_key, reverse=True)
    canon = cands[0]

    # 同義吸収
    if canon == "pedestrian":
        return "person"
    return canon

class GroundingDINO:
    def __init__(self, model_id: str):
        self.model_id = model_id
        if not torch.cuda.is_available():
            raise RuntimeError("GPU(CUDA)が必須です（GroundingDINO）。CPUフォールバックは禁止。")
        self.processor = None
        self.model = None
        self.device = "cuda"
        self.max_batch: Optional[int] = None

    def load(self, logger: logging.Logger):
        logger.info("[GroundingDINO] loading: %s (GPU only)", self.model_id)
        self.processor = GDinoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()


    def autotune(self, logger: logging.Logger, sample_rgb: np.ndarray, prompts: List[str], cap: int = 64) -> int:
        if not torch.cuda.is_available():
            self.max_batch = 1
            logger.info("[AutoBatch][GDINO] CPU のため batch=1 固定")
            return 1
        # バッチ探索
        best, b, failed = 1, 1, False
        H, W = sample_rgb.shape[:2]
        while b <= cap:
            try:
                text_labels = [prompts] * b
                inputs = self.processor(images=[sample_rgb]*b, text=text_labels, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs)
                _ = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H,W)]*b
                )
                del inputs, outputs, _
                _torch_sync_empty_cache()
                best = b
                b *= 2
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    failed = True
                    _torch_sync_empty_cache()
                    break
                raise
        lo = best
        hi = min(cap, b-1) if failed else best
        while lo < hi:
            mid = (lo + hi + 1)//2
            try:
                text_labels = [prompts] * mid
                inputs = self.processor(images=[sample_rgb]*mid, text=text_labels, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs)
                _ = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H,W)]*mid
                )
                del inputs, outputs, _
                _torch_sync_empty_cache()
                lo = mid
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    _torch_sync_empty_cache()
                    hi = mid - 1
                else:
                    raise
        logger.info("[AutoBatch][GDINO] batch=%d", lo)
        self.max_batch = max(1, int(lo))
        return self.max_batch

    @torch.inference_mode()
    def detect(self, rgb: np.ndarray, prompts: List[str],
               box_thr: float = 0.35, txt_thr: float = 0.25) -> List[Dict[str, Any]]:
        text_labels = [prompts]
        inputs = self.processor(images=rgb, text=text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        H, W = rgb.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr, target_sizes=[(H, W)]
        )[0]
        raw_labels = results.get("text_labels", results.get("labels", []))
        boxes  = results.get("boxes", [])
        scores = results.get("scores", [])
        out = []
        for box, score, label in zip(boxes, scores, raw_labels):
            x1,y1,x2,y2 = [float(v) for v in box.tolist()]
            raw = str(label)
            cls = _canonicalize_label(raw, prompts)
            out.append({"cls": cls, "score": float(score), "bbox": [x1,y1,x2,y2]})
        return out

    @torch.inference_mode()
    def detect_batch(self, rgbs: List[np.ndarray], prompts: List[str],
                     box_thr: float = 0.35, txt_thr: float = 0.25) -> List[List[Dict[str, Any]]]:
        sizes = [im.shape[:2] for im in rgbs]
        text_labels = [prompts] * len(rgbs)
        inputs = self.processor(images=rgbs, text=text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr, target_sizes=sizes
        )
        outs: List[List[Dict[str,Any]]] = []
        for res, (H, W) in zip(results, sizes):
            raw_labels = res.get("text_labels", res.get("labels", []))
            boxes  = res.get("boxes", [])
            scores = res.get("scores", [])
            dets=[]
            for box, score, label in zip(boxes, scores, raw_labels):
                x1,y1,x2,y2 = [float(v) for v in box.tolist()]
                raw = str(label)
                cls = _canonicalize_label(raw, prompts)
                dets.append({"cls": cls, "score": float(score), "bbox": [x1,y1,x2,y2]})
            outs.append(dets)
        return outs


# ========== 走行可能領域（Drivable）メトリクス ==========
def binarize_mask(arr: np.ndarray, thr: float=0.5, morph_k: int=0) -> np.ndarray:
    """連続値マスクを 0/1 に。必要ならモルフォロジー閉処理で穴埋め。"""
    m = (arr > thr).astype(np.uint8)
    if morph_k and morph_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def mask_metrics(mx: np.ndarray, mf: np.ndarray) -> Dict[str, float]:
    """
    2値マスクの一致度（IoU / Dice(F1) / Precision / Recall / L1 / RMSE）。
    形状が異なる場合は mx を mf のサイズに最近傍でリサイズ。
    """
    if mx.shape != mf.shape:
        mx = cv2.resize(mx, (mf.shape[1], mf.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = (mx > 0).astype(np.uint8)
    b = (mf > 0).astype(np.uint8)
    tp = float(np.sum((a==1) & (b==1)))
    fp = float(np.sum((a==0) & (b==1)))
    fn = float(np.sum((a==1) & (b==0)))
    tn = float(np.sum((a==0) & (b==0)))
    union = tp + fp + fn + 1e-9
    inter = tp
    iou = inter / union
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    l1   = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    rmse = float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32))**2)))
    return {
        "drv_iou": float(iou),
        "drv_f1": float(f1),
        "drv_precision": float(prec),
        "drv_recall": float(rec),
        "drv_l1": float(l1),
        "drv_rmse": float(rmse),
    }

def boundary_iou(mx: np.ndarray, mf: np.ndarray, tol: int=1) -> float:
    """
    境界一致度（Boundary IoU）。境界をモルフォロジー勾配で抽出し、tol ピクセルだけ膨張して相互 IoU。
    """
    if mx.shape != mf.shape:
        mx = cv2.resize(mx, (mf.shape[1], mf.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = (mx > 0).astype(np.uint8)
    b = (mf > 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    ea = cv2.morphologyEx(a, cv2.MORPH_GRADIENT, k3)
    eb = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k3)
    if tol and tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        ea = cv2.dilate(ea, k)
        eb = cv2.dilate(eb, k)
    inter = float(np.sum((ea>0) & (eb>0)))
    union = float(np.sum((ea>0) | (eb>0))) + 1e-9
    return float(inter / union)



def upsize_to(rgb: np.ndarray, m640: np.ndarray) -> np.ndarray:
    """YOLOP(640x640) の 2値マスクを RGB の元解像度に最近傍でアップサンプル"""
    h0, w0 = rgb.shape[:2]
    return cv2.resize((m640>0).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)

# OCR（Tesseract 優先。未導入ならスキップ可 → 空文字）
def run_ocr_tesseract(rgb_crop: np.ndarray) -> str:
    """
    Tesseract バイナリ未導入・実行時例外でも落ちないように完全ガード。
    未導入時は空文字 "" を返す。
    """
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except Exception:
        return ""
    try:
        g = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        txt = pytesseract.image_to_string(g, lang="eng", config="--psm 7")
        return (txt or "").strip()
    except TesseractNotFoundError:
        # バイナリが無い場合
        return ""
    except Exception:
        # OCR 失敗時は評価を継続するため空文字で返す
        return ""


# ==== bbox_iou: 完全置換（安全な正規化＋正しい面積計算）====================
def bbox_iou(a: List[float], b: List[float]) -> float:
    """
    IoU between two boxes [x1,y1,x2,y2]. 座標が入れ替わっていても安全に補正。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 座標の正規化（x1<=x2, y1<=y2 を保証）
    if ax2 < ax1: ax1, ax2 = ax2, ax1
    if ay2 < ay1: ay1, ay2 = ay2, ay1
    if bx2 < bx1: bx1, bx2 = bx2, bx1
    if by2 < by1: by1, by2 = by2, by1

    # 交差
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    # 面積（必ず非負）
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / (union + 1e-9)
# ===========================================================================

def match_by_iou(G: List[Dict[str,Any]], D: List[Dict[str,Any]], thr: float) -> List[Tuple[int,int,float]]:
    if len(G)==0 or len(D)==0:
        return []
    C = np.zeros((len(G), len(D)), dtype=np.float32)
    for i,g in enumerate(G):
        for j,d in enumerate(D):
            if g["cls"] != d["cls"]:
                C[i,j] = 1.0  # 不一致クラスにはペナルティ
            else:
                iou = bbox_iou(g["bbox"], d["bbox"])
                C[i,j] = 1.0 - (iou if iou>=thr else 0.0)
    row_ind, col_ind = linear_sum_assignment(C)
    matches = []
    for i,j in zip(row_ind, col_ind):
        iou = bbox_iou(G[i]["bbox"], D[j]["bbox"]) if G[i]["cls"]==D[j]["cls"] else 0.0
        if iou >= thr:
            matches.append((i,j,iou))
    return matches

def det_metrics(
    G_all: List[Dict[str,Any]],
    D_all: List[Dict[str,Any]],
    iou_thr: float,
    img_wh_x: Tuple[int,int],
    img_wh_f: Tuple[int,int],
) -> Dict[str, Any]:
    """
    X と F(X) の bbox は元々ピクセル座標だが、解像度が異なる可能性が高い。
    比較の直前に [0,1] に正規化してから IoU / 中心誤差 / 面積比を計算する。
    """
    Wx, Hx = float(max(1, img_wh_x[0])), float(max(1, img_wh_x[1]))
    Wf, Hf = float(max(1, img_wh_f[0])), float(max(1, img_wh_f[1]))

    def _to_norm(dets: List[Dict[str,Any]], W: float, H: float) -> List[Dict[str,Any]]:
        out = []
        for d in dets:
            x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
            # [0,1] 正規化
            x1 /= W; x2 /= W; y1 /= H; y2 /= H
            # 並び保証 + [0,1] クリップ
            x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
            y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            nd = dict(d); nd["bbox"] = [x1, y1, x2, y2]
            out.append(nd)
        return out

    # 正規化
    Gn = _to_norm(G_all, Wx, Hx)
    Dn = _to_norm(D_all, Wf, Hf)

    classes = sorted(list({g["cls"] for g in Gn} | {d["cls"] for d in Dn}))
    if len(classes) == 0:
        return {
            "per_class": {},
            "center_error_median": 0.0,
            "size_log_ratio_median": 0.0,
            "iou_median": 0.0,
            "count_absdiff_mean": 0.0,
            "count_wasserstein": 0.0,
            "classes": [],
        }

    # 正規化座標での IoU マッチング（座標系に依存しない）
    def _match_by_iou(G: List[Dict[str,Any]], D: List[Dict[str,Any]], thr: float) -> List[Tuple[int,int,float]]:
        if len(G)==0 or len(D)==0:
            return []
        C = np.zeros((len(G), len(D)), dtype=np.float32)
        for i,g in enumerate(G):
            for j,d in enumerate(D):
                if g["cls"] != d["cls"]:
                    C[i,j] = 1.0
                else:
                    iou = bbox_iou(g["bbox"], d["bbox"])
                    C[i,j] = 1.0 - (iou if iou>=thr else 0.0)
        row_ind, col_ind = linear_sum_assignment(C)
        matches = []
        for i,j in zip(row_ind, col_ind):
            iou = bbox_iou(G[i]["bbox"], D[j]["bbox"]) if G[i]["cls"]==D[j]["cls"] else 0.0
            if iou >= thr:
                matches.append((i,j,iou))
        return matches

    per_cls: Dict[str, Dict[str, float]] = {}
    center_errs=[]; size_ratio=[]; iou_list=[]

    for c in classes:
        Gc = [g for g in Gn if g["cls"]==c]
        Dc = [d for d in Dn if d["cls"]==c]
        M  = _match_by_iou(Gc, Dc, iou_thr)

        nG, nD, nM = len(Gc), len(Dc), len(M)
        PR = nM/(nG+1e-9)                 # Preservation-Recall
        PP = nM/(nD+1e-9)                 # Preservation-Precision
        F1 = 2*PR*PP/(PR+PP+1e-9)
        HAL = (nD - nM)/(nD+1e-9)         # ハルシネーション率

        # 幾何安定性（正規化座標）
        c_err=[]; s_err=[]; ious=[]
        for (ii,jj,ij_iou) in M:
            gx1,gy1,gx2,gy2 = Gc[ii]["bbox"]; dx1,dy1,dx2,dy2 = Dc[jj]["bbox"]
            gcx,gcy = (gx1+gx2)/2.0, (gy1+gy2)/2.0
            dcx,dcy = (dx1+dx2)/2.0, (dy1+dy2)/2.0
            cdist   = math.sqrt((gcx-dcx)**2 + (gcy-dcy)**2)       # すでに無次元
            garea   = max(1e-8, (gx2-gx1)*(gy2-gy1))
            darea   = max(1e-8, (dx2-dx1)*(dy2-dy1))
            sratio  = abs(math.log(darea/garea))
            c_err.append(cdist); s_err.append(sratio); ious.append(ij_iou)

        center_errs += c_err
        size_ratio  += s_err
        iou_list    += ious
        per_cls[c]   = {"PR":float(PR), "PP":float(PP), "F1":float(F1), "HAL":float(HAL),
                        "count_G":nG, "count_D":nD}

    # カウント整合性
    cntG = np.array([sum(1 for g in Gn if g["cls"]==c) for c in classes], dtype=np.float64)
    cntD = np.array([sum(1 for d in Dn if d["cls"]==c) for c in classes], dtype=np.float64)
    abs_diff = float(np.mean(np.abs(cntG - cntD)))

    x = np.arange(len(classes), dtype=np.float64)
    sumG = float(cntG.sum()); sumD = float(cntD.sum())
    if sumG <= 0.0 and sumD <= 0.0:
        emd = 0.0
    elif sumG > 0.0 and sumD > 0.0:
        wG = cntG / sumG; wD = cntD / sumD
        emd = float(wasserstein_distance(x, x, u_weights=wG, v_weights=wD))
    else:
        wG = cntG / (sumG + 1e-12)
        wD = cntD / (sumD + 1e-12)
        cdfG = np.cumsum(wG); cdfD = np.cumsum(wD)
        emd = float(np.sum(np.abs(cdfG - cdfD)))

    return {
        "per_class": per_cls,
        "center_error_median": float(np.median(center_errs)) if center_errs else 0.0,
        "size_log_ratio_median": float(np.median(size_ratio)) if size_ratio else 0.0,
        "iou_median": float(np.median(iou_list)) if iou_list else 0.0,
        "count_absdiff_mean": abs_diff,
        "count_wasserstein": emd,
        "classes": classes,
    }



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet: WaymoV2 定量評価スクリプト（単一スクリプト完結）")
    ap.add_argument("--suppress-gdino-futurewarning", action="store_true",
                help="GroundingDINO の FutureWarning を抑止（デフォルト: ON）")
    ap.add_argument("--orig-root", type=str, default=DEFAULT_ORIG_IMAGE_ROOT)
    ap.add_argument("--gen-root", type=str, default=DEFAULT_GEN_ROOT)
    ap.add_argument("--canny-x-root", type=str, default=DEFAULT_CANNY_ROOT_X)
    ap.add_argument("--depth-x-npy-root", type=str, default=DEFAULT_DEPTH_NPY_ROOT_X)
    ap.add_argument("--semseg-x-root", type=str, default=DEFAULT_SEMSEG_ROOT_X)
    ap.add_argument("--cache-root", type=str, default=DEFAULT_HDD_CACHE_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=["training","validation","testing"])
    ap.add_argument("--camera", type=str, default="front")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--prompt-root", type=str, default=DEFAULT_PROMPT_ROOT,
                    help="メタ欠落時のプロンプト探索用ルート")    
    ap.add_argument("--tasks", type=str,
                    choices=["all","reality","structure","objects","drivable","diversity","text"],
                    default="all")
    ap.add_argument("--reality-metric", type=str, choices=["clip-cmmd","clip-fid","inception-fid"], default="clip-cmmd")
    ap.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_ID)
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--metric3d-onnx", type=str, default=DEFAULT_METRIC3D_ONNX)
    ap.add_argument("--use-yolop", action="store_true")
    ap.add_argument("--yolop-roi-filter", action="store_true",
                    help="YOLOP の走行可能マスクで検出中心をフィルタ（デフォルト無効）")
    ap.add_argument("--gdinomodel", type=str, default=DEFAULT_GDINO_ID)
    ap.add_argument("--det-prompts", type=str, nargs="*", default=DEFAULT_DET_PROMPTS)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--ocr-engine", type=str, choices=["none","tesseract"], default="tesseract")
    ap.add_argument("--annotation-mode", type=str, choices=["off","objects","structure","all","drivable"], default="off",
                    help="注釈を保存: objects=検出, structure=Edge/Depth/Semseg, drivable=YOLOP マスク, all=両方, off=なし")
    ap.add_argument("--annotate-limit", type=int, default=32, help="注釈を保存する最大枚数（split毎）")
    ap.add_argument("--annotate-out", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "viz"),
                    help="注釈画像の保存先（HDD）")
    ap.add_argument("--seed", type=int, default=0)
    # ★ 実験管理用
    ap.add_argument(
        "--experiment-id",
        type=str,
        default="EX1",
        help="実験ID（EX1, EX2 など）。結果と設定を cache-root/experiments に JSON 保存します。",
    )
    ap.add_argument(
        "--experiment-note",
        type=str,
        default="",
        help="任意メモ（fine-tune 条件やプロンプト設定などの説明用）。例：Finetune後のUnicontrolNetをQwen版の新promptで推論したF(X)を評価する実験。",
    )

    ap.add_argument("--tb", action="store_true")    
    ap.add_argument("--tb-dir", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "tensorboard"))
    ap.add_argument("--verbose", action="store_true")
    # ---- Auto-Batch 制御 ----
    ap.add_argument("--max-batch-cap", type=int, default=1024, help="自動探索の上限（CLIP/GDINO）。OneFormer/YOLOP は 64 まで。")
    ap.add_argument("--auto-batch", dest="auto_batch", action="store_true", help="自動バッチ探索を有効化（既定）")
    ap.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", help="自動バッチ探索を無効化（手動 clip-batch のみ使用）")
    ap.set_defaults(auto_batch=True)
    # ---- ★ 追加: GDINO のしきい値を CLI から調整 ----
    ap.add_argument("--gdino-box-thr", type=float, default=0.35,
                    help="GroundingDINO の box しきい値（既定 0.35。小物体の Recall を上げるには 0.25 前後に下げる）")
    ap.add_argument("--gdino-text-thr", type=float, default=0.25,
                    help="GroundingDINO の text しきい値（既定 0.25。標識などは 0.20 前後に下げると Recall↑/HAL↑）")
    # ---- ★ 多様性・テキスト追従パラメータ ----
    ap.add_argument("--div-pairs", type=int, default=10000, help="LPIPS/MS-SSIM のランダム評価ペア数（split毎の上限）")
    ap.add_argument("--div-resize", type=int, default=256, help="多様性指標用リサイズ解像度（-1: 元解像度、同一サイズ化必須）")
    ap.add_argument("--lpips-net", type=str, choices=["alex","vgg","squeeze"], default="alex", help="LPIPS の backbone")
    ap.add_argument("--pr-k", type=int, default=3, help="Improved Precision/Recall の k（k-NN 半径）")
    ap.add_argument("--rprecision-k", type=int, default=1, help="R-Precision@K の K（既定=1）")
    ap.add_argument("--rprecision-negatives", type=int, default=99, help="R-Precision の負例数（既定=99）")
    ap.add_argument("--interprompt-method", type=str, choices=["clip-corr","clip-spread"], default="clip-corr",
                    help="inter-prompt 指標: clip-corr=距離相関, clip-spread=セントロイド距離平均")    
    # ★追記: アスペクト比維持のリサイズ方式
    ap.add_argument("--div-resize-mode", type=str,
                    choices=["square","letterbox","center-crop"],
                    default="square",
                    help="多様性指標の前処理方式。square=SIZE×SIZEへ歪みリサイズ（既定）, "
                         "letterbox=長辺をSIZEに合わせ短辺はパディング, "
                         "center-crop=短辺をSIZEに合わせた後に長辺を中央クロップでSIZEに揃える。")

    # ★★★ ここから Drivable 用の 3 パラメータを追加 ★★★
    ap.add_argument(
        "--drivable-thr",
        type=float,
        default=0.5,
        help="走行可能マスクの 2値化しきい値（0〜1, 既定=0.5）。YOLOP/他モデルの出力が連続値のときに有効。",
    )
    ap.add_argument(
        "--drivable-morph-k",
        type=int,
        default=0,
        help="走行可能マスクに対するモルフォロジー閉処理のカーネルサイズ（ピクセル単位, 0 で無効）。"
    )
    ap.add_argument(
        "--drivable-edge-tol",
        type=int,
        default=1,
        help="Boundary IoU 計算時に境界を何ピクセル膨張させるか（既定=1）。"
    )
    # ★★★ Drivable 追加ここまで ★★★    
    # -------------- ここから挿入してください（Edge改革 & Drivable方式選択）--------------
    # Edge 改革: traffic sign ROI と和集合（下半分 ∪ 標識ROI）
    ap.add_argument("--edge-traffic-sign", dest="edge_ts", action="store_true",
                    help="DINO標識BBox内のEdge比較を有効化（デフォルトON）")
    ap.add_argument("--no-edge-traffic-sign", dest="edge_ts", action="store_false",
                    help="traffic sign edge を無効化")
    ap.set_defaults(edge_ts=True)
    ap.add_argument("--edge-sign-classes", type=str, nargs="*",
                    default=["traffic sign","stop sign","speed limit sign","crosswalk sign","construction sign"],
                    help="traffic sign edge に使うクラス名（GroundingDINO正規化名）")
    ap.add_argument("--edge-sign-box-thr", type=float, default=0.25, help="traffic sign 用 DINO box閾値（既定0.25）")
    ap.add_argument("--edge-sign-text-thr", type=float, default=0.20, help="traffic sign 用 DINO text閾値（既定0.20）")

    # Drivable 評価の方式（3通り）: yolop / onefroad / sota（任意HFセグ）
    ap.add_argument("--drivable-methods", type=str, nargs="+",
                    choices=["yolop","onefroad","sota"], default=["yolop","onefroad"],
                    help="走行可能領域の比較方式を列挙（既定: yolop onefroad）")
    ap.add_argument("--onefroad-road-ids", type=int, nargs="+", default=[0],
                    help="OneFormer trainId のうち Drivable とみなすID（既定: [0]=road）")
    ap.add_argument("--sota-seg-model", type=str, default="",
                    help="任意のHFセグモデルID（未指定なら sota 方式はスキップ）")
    ap.add_argument("--sota-drivable-labels", type=str, nargs="+",
                    default=["drivable", "drivable area", "road", "roadway", "lane"],
                    help="HFセグの id2label に含まれる 'drivable' 相当ラベル名（部分一致, 小文字化）")
    # -------------- ここまで挿入してください（Edge改革 & Drivable方式選択）--------------

    return ap.parse_args()



def main() -> None:
    args = parse_args()
    cache = Cache(args.cache_root)
    logger = setup_logger(cache.d_logs, args.verbose)

    # ======= GPU 厳格モード：CUDA 無しは即エラー =======
    if not torch.cuda.is_available():
        logger.error("❌ GPU(CUDA) が見つかりません。CPUフォールバックは禁止です。Docker の --gpus all と NVIDIA ドライバを確認してください。")
        sys.exit(2)

    log_env(logger)

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sw: Optional[SummaryWriter] = SummaryWriter(args.tb_dir) if args.tb else None
    if sw:
        logger.info("TensorBoard: %s", args.tb_dir)
    # 実験単位で指標をまとめるための入れ物（split ごとに dict で格納）
    exp_metrics: Dict[str, Dict[str, Any]] = {
        "reality": {},
        "edge": {},
        "depth": {},
        "semseg": {},
        "objects": {},
        "drivable": {},
        "diversity": {},
        "aug_strength": {},  # ★ X vs F(X) の LPIPS / 1-MS-SSIM
        "pr": {},            # Improved Precision/Recall
        "clipscore": {},
        "rprecision": {},
        "interprompt": {},
    }
    # ★追加: split ごとのペア数（Waymo 全体集計の重みとして使用）
    split_pair_counts: Dict[str, int] = {}
    # モデルハンドル
    clipper = None
    if args.tasks in ("all","reality"):
        clipper = ClipEmbedder(args.clip_model)

    sess_m3d = None; in_m3d = None; out_m3d = None; m3d_bs = 1
    # Metric3D(ONNX) は Structure 系タスクでのみ必要
    need_metric3d = (args.tasks in ("all", "structure"))
    if need_metric3d:
        try:
            sess_m3d, in_m3d, out_m3d, providers = build_metric3d_session(args.metric3d_onnx)
            logger.info("Metric3Dv2 ORT providers: %s", providers)
        except Exception as e:
            logger.error("Metric3Dv2 ONNX Runtime 構築失敗: %s", repr(e))
            sys.exit(1)


    onef_proc = None; onef_model = None; onef_bs = 1
    if args.tasks in ("all","structure"):
        onef_proc, onef_model = build_oneformer(DEFAULT_ONEFORMER_ID, device="cuda", fp16=True)
        logger.info("OneFormer loaded: %s", DEFAULT_ONEFORMER_ID)

    yolop_model = None; yolop_bs = 1
    if args.tasks in ("all","objects","drivable") and args.use_yolop:
        yolop_model = load_yolop(logger)

    gdino = None; gdino_bs = 1
    if args.tasks in ("all","objects"):
        gdino = GroundingDINO(args.gdinomodel); gdino.load(logger)

    ann_counter = {"objects": defaultdict(int), "structure": defaultdict(int), "drivable": defaultdict(int)}
    total_pairs = 0

    for split in args.splits:
        pairs = enumerate_pairs(args.orig_root, args.gen_root, split, args.camera, args.limit)
        pair_count = len(pairs)
        if not pairs:
            logger.warning("[%s] ペアが見つかりません（split=%s, camera=%s）", split, split, args.camera)
            continue
        logger.info("[%s] 評価ペア数: %d", split, pair_count)
        # ★ FIX: split ごとに TS / LH-TS のバッファをクリアする
        edge_scores_ts = []
        edge_scores_lh_ts = []        
        total_pairs += pair_count
        split_pair_counts[split] = pair_count


        # サンプル画像（自動バッチ探索用）
        sample_rgb_x = imread_rgb(pairs[0][0])
        sample_rgb_f = imread_rgb(pairs[0][1])
        logger.info("[%s] sample resolution: X=%dx%d | F=%dx%d | scale=(%.3f, %.3f)",
                    split,
                    sample_rgb_x.shape[1], sample_rgb_x.shape[0],
                    sample_rgb_f.shape[1], sample_rgb_f.shape[0],
                    sample_rgb_f.shape[1] / max(1, sample_rgb_x.shape[1]),
                    sample_rgb_f.shape[0] / max(1, sample_rgb_x.shape[0]))


        # ---------- Reality: CLIP ----------
        if args.tasks in ("all","reality"):
            clipper.load(logger)
            if args.auto_batch:
                cap = int(args.max_batch_cap)
                try:
                    _ = clipper.autotune(logger, sample_rgb_f, cap=cap)
                except Exception as e:
                    logger.warning("CLIP AutoBatch 失敗: %s → 既定 clip-batch=%d を継続", repr(e), args.clip_batch)
            clip_bs = int(clipper.max_batch or args.clip_batch or 16)

            feats_x = []; feats_fx = []
            pbar = tqdm(pairs, desc=f"{split}-clip")
            batch_imgs_x=[]; batch_imgs_fx=[]; batch_indices=[]
            for (px, pfx, rel_dir, stem) in pbar:
                cp_x = cache.clip_path(split, rel_dir, stem, "x")
                cp_f = cache.clip_path(split, rel_dir, stem, "fx")
                if os.path.exists(cp_x) and os.path.exists(cp_f):
                    ex = np.load(cp_x)["feat"]; ef = np.load(cp_f)["feat"]
                    feats_x.append(ex[None,:] if ex.ndim==1 else ex)
                    feats_fx.append(ef[None,:] if ef.ndim==1 else ef)
                    pbar.set_postfix_str("cache")
                    continue
                imgx = imread_rgb(px); imgf = imread_rgb(pfx)
                batch_imgs_x.append(imgx); batch_imgs_fx.append(imgf); batch_indices.append((rel_dir, stem))
                if len(batch_imgs_x) >= clip_bs:
                    fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                    for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                        ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                    feats_x.append(fx); feats_fx.append(ff)
                    batch_imgs_x=[]; batch_imgs_fx=[]; batch_indices=[]
                    pbar.set_postfix_str(f"embed{clip_bs}")
            if batch_imgs_x:
                fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                    ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                feats_x.append(fx); feats_fx.append(ff)
            feats_x = np.concatenate(feats_x, axis=0) if feats_x else np.zeros((0,512),dtype=np.float32)
            feats_fx = np.concatenate(feats_fx, axis=0) if feats_fx else np.zeros((0,512),dtype=np.float32)

            if feats_x.shape[0] > 0 and feats_fx.shape[0] > 0:
                real_val: Optional[float] = None
                metric_name = args.reality_metric

                if args.reality_metric == "clip-fid":
                    mu1, s1 = feats_to_stats(feats_x); mu2, s2 = feats_to_stats(feats_fx)
                    fid = compute_frechet_distance(mu1, s1, mu2, s2)
                    real_val = float(fid)
                    logger.info("[%s][CLIP-FID] = %.4f", split, fid)
                    if sw: sw.add_scalar(f"reality/clip_fid/{split}", fid, 0)
                elif args.reality_metric == "clip-cmmd":
                    mmd2 = gaussian_mmd(feats_x, feats_fx, sigma=None)
                    real_val = float(mmd2)
                    logger.info("[%s][CLIP-CMMD] (MMD^2) = %.4f", split, mmd2)
                    if sw: sw.add_scalar(f"reality/clip_cmmd/{split}", mmd2, 0)
                else:
                    logger.warning("inception-fid は本スクリプトでは非推奨。必要なら別計算に切替。")

                if real_val is not None:
                    exp_metrics["reality"][split] = {
                        "metric": metric_name,
                        "value": real_val,
                        "n_images": int(feats_x.shape[0]),
                    }


        # ---------- Structure: Edge/Depth/Semseg（OneFormer をバッチ化、Metric3D は可能ならバッチ） ----------
        if args.tasks in ("all","structure"):
            if args.auto_batch and (onef_proc is not None):
                try:
                    onef_bs = autotune_oneformer_bs(onef_proc, onef_model, logger, sample_rgb_f, cap=64)
                except Exception as e:
                    logger.warning("OneFormer AutoBatch 失敗: %s → batch=1", repr(e)); onef_bs = 1
            if args.auto_batch and (sess_m3d is not None):
                try:
                    m3d_bs = autotune_metric3d_bs(sess_m3d, in_m3d, out_m3d, logger, sample_rgb_f, cap=16)
                except Exception as e:
                    logger.warning("Metric3D AutoBatch 失敗: %s → batch=1", repr(e)); m3d_bs = 1

            conf_mat = np.zeros((19,19), dtype=np.int64)
            edge_scores = []; depth_scores = []

            # セマンティクス・深度は「必要なものだけ」バッチ推論してキャッシュ
            # ここでは 1 パスで処理しつつ、semseg/depth は pending を溜めて吐く
            pend_for_seg = []; pend_meta_seg = []
            pend_for_dep = []; pend_meta_dep = []

            def _flush_seg():
                nonlocal pend_for_seg, pend_meta_seg
                if not pend_for_seg: return
                segs = oneformer_semseg_batch(onef_proc, onef_model, pend_for_seg) if onef_proc is not None else []
                for (split_, rd, st, pfx_), seg in zip(pend_meta_seg, segs):
                    seg_fx_path = cache.semseg_fx_path(split_, rd, st)
                    ensure_dir(os.path.dirname(seg_fx_path))
                    np.save(seg_fx_path, seg)
                pend_for_seg=[]; pend_meta_seg=[]

            def _flush_dep():
                nonlocal pend_for_dep, pend_meta_dep
                if not pend_for_dep: return
                if m3d_bs > 1:
                    depths = infer_metric3d_batch(sess_m3d, in_m3d, out_m3d, pend_for_dep)
                    for (split_, rd, st, pfx_), depth_fx in zip(pend_meta_dep, depths):
                        depth_fx_path = cache.depth_fx_path(split_, rd, st)
                        ensure_dir(os.path.dirname(depth_fx_path)); np.save(depth_fx_path, depth_fx)
                else:
                    for (split_, rd, st, pfx_) in pend_meta_dep:
                        rgbf = imread_rgb(pfx_)
                        depth_fx = infer_metric3d_np(sess_m3d, in_m3d, out_m3d, rgbf)
                        depth_fx_path = cache.depth_fx_path(split_, rd, st)
                        ensure_dir(os.path.dirname(depth_fx_path)); np.save(depth_fx_path, depth_fx)
                pend_for_dep=[]; pend_meta_dep=[]

            pbar = tqdm(pairs, desc=f"{split}-struct")
            for (px, pfx, rel_dir, stem) in pbar:
                # Edge
                edge_x_path = os.path.join(args.canny_x_root, split, rel_dir, f"{stem}_edge.png")
                if not os.path.exists(edge_x_path):
                    pbar.set_postfix_str("miss-edgeX"); continue
                edge_x = imread_gray(edge_x_path)
                edge_fx_path = cache.canny_fx_path(split, rel_dir, stem)
                if os.path.exists(edge_fx_path):
                    edge_fx = imread_gray(edge_fx_path)
                else:
                    rgbf = imread_rgb(pfx)
                    edge_fx = canny_cpu(rgbf, 100, 200, 3)
                    ensure_dir(os.path.dirname(edge_fx_path)); cv2.imwrite(edge_fx_path, edge_fx)
                e_met = edge_metrics(edge_x, edge_fx); edge_scores.append(e_met)
       # -------------- ここから挿入してください（traffic sign edge / 和集合）--------------
                if args.edge_ts:
                    # DINO結果の確保（X / F）
                    gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                    gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")

                    # 必要に応じて最小限ロード
                    need_detect = (not os.path.exists(gd_x_path)) or (not os.path.exists(gd_f_path))
                    if need_detect:
                        # 軽量ロード（signクラスだけ）
                        try:
                            gd_edge = GroundingDINO(args.gdinomodel); gd_edge.load(logger)
                        except Exception as _e:
                            logger.error("GroundingDINO load failed for edge_ts: %s", repr(_e))
                            gd_edge = None

                        if gd_edge is not None:
                            # 画像の読み出し（元解像度）
                            rgbx_full = imread_rgb(px)
                            rgbf_full = imread_rgb(pfx)
                            dets_x = gd_edge.detect(rgbx_full, args.edge_sign_classes,
                                                    box_thr=args.edge_sign_box_thr, txt_thr=args.edge_sign_text_thr)
                            dets_f = gd_edge.detect(rgbf_full, args.edge_sign_classes,
                                                    box_thr=args.edge_sign_box_thr, txt_thr=args.edge_sign_text_thr)
                            ensure_dir(os.path.dirname(gd_x_path)); json.dump(dets_x, open(gd_x_path,"w"), indent=2)
                            ensure_dir(os.path.dirname(gd_f_path)); json.dump(dets_f, open(gd_f_path,"w"), indent=2)

                    # キャッシュから取得
                    try:
                        dets_x_all = json.load(open(gd_x_path,"r"))
                        dets_f_all = json.load(open(gd_f_path,"r"))
                    except Exception:
                        dets_x_all, dets_f_all = [], []

                    # signクラスのみに絞る（正規化名で一致）
                    signset = set([s.lower() for s in args.edge_sign_classes])
                    dets_x = [d for d in dets_x_all if str(d.get("cls","")).lower() in signset]
                    dets_f = [d for d in dets_f_all if str(d.get("cls","")).lower() in signset]

                    # ROIマスク作成（エッジ比較サイズ＝edge_fx基準）
                    Hf_e, Wf_e = edge_fx.shape[:2]
                    # 元画像サイズ（DINOは元解像度座標）
                    rgbx_full = imread_rgb(px)
                    rgbf_full = imread_rgb(pfx)
                    Hx0, Wx0 = rgbx_full.shape[:2]
                    Hf0, Wf0 = rgbf_full.shape[:2]
                    roi_sign = build_roi_mask_from_dets(dets_x, dets_f,
                                                        (Hx0, Wx0), (Hf0, Wf0),
                                                        (Hf_e, Wf_e),
                                                        signset)

                    # traffic sign edge（ROI＝標識BBoxの和）
                    em_ts = edge_metrics_roi(edge_x, edge_fx, roi_sign)

                    # Edge_of_LowerHalf_and_TrafficSign（下半分 ∪ 標識ROI）
                    mask_lh = np.zeros_like(roi_sign, dtype=np.uint8)
                    mask_lh[Hf_e//2:, :] = 1
                    roi_union = ((roi_sign > 0) | (mask_lh > 0)).astype(np.uint8)
                    em_lh_ts = edge_metrics_roi(edge_x, edge_fx, roi_union)

                    # バッファへ
                    #edge_scores_ts = locals().setdefault("edge_scores_ts", [])
                    #edge_scores_lh_ts = locals().setdefault("edge_scores_lh_ts", [])

                    edge_scores_ts.append(em_ts)
                    edge_scores_lh_ts.append(em_lh_ts)

                # -------------- ここまで挿入してください（traffic sign edge / 和集合）--------------                

                # Depth
                depth_x_path = os.path.join(args.depth_x_npy_root, split, rel_dir, f"{stem}_depth.npy")
                if not os.path.exists(depth_x_path):
                    pbar.set_postfix_str("miss-depthX"); continue
                if not os.path.exists(cache.depth_fx_path(split, rel_dir, stem)):
                    if m3d_bs > 1:
                        pend_for_dep.append(imread_rgb(pfx))
                        pend_meta_dep.append((split, rel_dir, stem, pfx))
                        if len(pend_for_dep) >= m3d_bs: _flush_dep()
                    else:
                        rgbf = imread_rgb(pfx)
                        depth_fx = infer_metric3d_np(sess_m3d, in_m3d, out_m3d, rgbf)
                        ensure_dir(os.path.dirname(cache.depth_fx_path(split, rel_dir, stem))); np.save(cache.depth_fx_path(split, rel_dir, stem), depth_fx)
                depth_x = np.load(depth_x_path).astype(np.float32)
                depth_fx = np.load(cache.depth_fx_path(split, rel_dir, stem)).astype(np.float32)
                d_met = depth_metrics(depth_x, depth_fx); depth_scores.append(d_met)

                # Semseg
                seg_x_path = os.path.join(args.semseg_x_root, split, rel_dir, f"{stem}_predTrainId.npy")
                if not os.path.exists(seg_x_path):
                    pbar.set_postfix_str("miss-segX"); continue
                if not os.path.exists(cache.semseg_fx_path(split, rel_dir, stem)):
                    if onef_bs > 1:
                        pend_for_seg.append(imread_rgb(pfx))
                        pend_meta_seg.append((split, rel_dir, stem, pfx))
                        if len(pend_for_seg) >= onef_bs: _flush_seg()
                    else:
                        rgbf = imread_rgb(pfx)
                        seg_fx = oneformer_semseg(onef_proc, onef_model, rgbf)
                        ensure_dir(os.path.dirname(cache.semseg_fx_path(split, rel_dir, stem))); np.save(cache.semseg_fx_path(split, rel_dir, stem), seg_fx)
                seg_x = np.load(seg_x_path).astype(np.uint8)
                seg_fx = np.load(cache.semseg_fx_path(split, rel_dir, stem)).astype(np.uint8)
                if seg_x.shape != seg_fx.shape:
                    seg_x = cv2.resize(seg_x, (seg_fx.shape[1], seg_fx.shape[0]), interpolation=cv2.INTER_NEAREST)
                conf_mat += confusion_19(seg_x, seg_fx, ncls=19)

                if args.annotation_mode in ("structure","all"):
                    if ann_counter["structure"][split] < args.annotate_limit:
                        rgbx_vis = imread_rgb(px)
                        rgbf_vis = imread_rgb(pfx)
                        save_annotations_structure(
                            args.annotate_out, split, rel_dir, stem,
                            rgbx_vis, rgbf_vis,
                            edge_x, edge_fx,
                            depth_x, depth_fx,
                            seg_x, seg_fx,
                        )
                        ann_counter["structure"][split] += 1

                pbar.set_postfix_str("ok")

            _flush_seg(); _flush_dep()
            if edge_scores:
                # 各指標の平均値 + 何サンプルで平均したか（n_samples）を記録
                es_values = {k: float(np.mean([d[k] for d in edge_scores])) for k in edge_scores[0].keys()}
                es_values["n_samples"] = len(edge_scores)
                logger.info("[%s][Edge] %s", split, json.dumps(es_values, ensure_ascii=False))
                exp_metrics["edge"][split] = es_values
                if sw:
                    for k, v in es_values.items():
                        if k == "n_samples":
                            continue
                        sw.add_scalar(f"struct/edge_{k}/{split}", v, 0)
            # -------------- ここから挿入してください（Edge TS/Union 集計）--------------
            if edge_scores_ts:

                ts_values = {k: float(np.mean([d[k] for d in edge_scores_ts])) for k in edge_scores_ts[0].keys() if k!="empty_roi"}
                ts_values["n_samples_ts"] = len(edge_scores_ts)
                # 名前は明示的に traffic-sign 系と分かるように付与
                exp_metrics["edge"][split].update({
                    "edge_ts_l1": ts_values.get("edge_l1", 0.0),
                    "edge_ts_rmse": ts_values.get("edge_rmse", 0.0),
                    "edge_ts_iou": ts_values.get("edge_iou", 0.0),
                    "edge_ts_f1": ts_values.get("edge_f1", 0.0),
                    "n_samples_ts": ts_values.get("n_samples_ts", 0),
                })
                if sw:
                    for k,v in [("l1","edge_ts_l1"),("rmse","edge_ts_rmse"),("iou","edge_ts_iou"),("f1","edge_ts_f1")]:
                        sw.add_scalar(f"struct/edge_ts_{k}/{split}", exp_metrics["edge"][split][v], 0)

            if edge_scores_lh_ts:

                u_values = {k: float(np.mean([d[k] for d in edge_scores_lh_ts])) for k in edge_scores_lh_ts[0].keys() if k!="empty_roi"}
                u_values["n_samples_lh_ts"] = len(edge_scores_lh_ts)
                exp_metrics["edge"][split].update({
                    "edge_lh_ts_l1": u_values.get("edge_l1", 0.0),
                    "edge_lh_ts_rmse": u_values.get("edge_rmse", 0.0),
                    "edge_lh_ts_iou": u_values.get("edge_iou", 0.0),
                    "edge_lh_ts_f1": u_values.get("edge_f1", 0.0),
                    "n_samples_lh_ts": u_values.get("n_samples_lh_ts", 0),
                })
                if sw:
                    for k,v in [("l1","edge_lh_ts_l1"),("rmse","edge_lh_ts_rmse"),("iou","edge_lh_ts_iou"),("f1","edge_lh_ts_f1")]:
                        sw.add_scalar(f"struct/edge_lh_ts_{k}/{split}", exp_metrics["edge"][split][v], 0)
            # -------------- ここまで挿入してください（Edge TS/Union 集計）--------------
            if depth_scores:
                ds_values = {k: float(np.mean([d[k] for d in depth_scores])) for k in depth_scores[0].keys()}
                ds_values["n_samples"] = len(depth_scores)
                logger.info("[%s][Depth] %s", split, json.dumps(ds_values, ensure_ascii=False))
                exp_metrics["depth"][split] = ds_values
                if sw:
                    for k, v in ds_values.items():
                        if k == "n_samples":
                            continue
                        sw.add_scalar(f"struct/depth_{k}/{split}", v, 0)

            if np.sum(conf_mat) > 0:
                miou, ious = miou_from_conf(conf_mat)
                logger.info("[%s][Semseg] mIoU=%.4f", split, miou)
                exp_metrics["semseg"][split] = {
                    "miou": float(miou),
                    "per_class_iou": [float(x) for x in ious.tolist()],
                    "n_pixels": int(np.sum(conf_mat)),
                }
                if sw:
                    sw.add_scalar(f"struct/semseg_mIoU/{split}", miou, 0)
                    for cid, val in enumerate(ious):
                        sw.add_scalar(f"struct/semseg_IoU_c{cid}/{split}", float(val), 0)
                out_cm = os.path.join(cache.d_logs, f"{split}_confusion.npy")
                np.save(out_cm, conf_mat)


        # ---------- Objects: GDINO/YOLOP をバッチ化（2パス：キャッシュ→集計） ----------
        if args.tasks in ("all","objects"):
            # AutoBatch
            if args.auto_batch and (gdino is not None):
                try:
                    gdino_bs = gdino.autotune(logger, sample_rgb_f, args.det_prompts, cap=64)
                except Exception as e:
                    logger.warning("GDINO AutoBatch 失敗: %s → batch=1", repr(e)); gdino_bs = 1
            if args.auto_batch and (yolop_model is not None):
                try:
                    yolop_bs = autotune_yolop_bs(yolop_model, logger, sample_rgb_f, cap=64)
                except Exception as e:
                    logger.warning("YOLOP AutoBatch 失敗: %s → batch=1", repr(e)); yolop_bs = 1

            # ---- (1) 検出キャッシュの構築（X と FX を別々に、足りないものだけバッチ推論） ----
            # X 側
            miss_x = []
            for (px, pfx, rel_dir, stem) in pairs:
                gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                if not os.path.exists(gd_x_path):
                    miss_x.append((px, rel_dir, stem, gd_x_path))
            pbar = tqdm(list(_chunks(miss_x, gdino_bs if gdino_bs>0 else 1)), desc=f"{split}-obj-gdX")
            for chunk in pbar:
                imgs = [imread_rgb(px) for (px,_,_,_) in chunk]
                # ★ しきい値を CLI から反映
                outs = gdino.detect_batch(
                    imgs, args.det_prompts,
                    box_thr=args.gdino_box_thr, txt_thr=args.gdino_text_thr
                )
                for (px, rel_dir, stem, outp), dets in zip(chunk, outs):
                    ensure_dir(os.path.dirname(outp)); json.dump(dets, open(outp,"w"), indent=2)


            # FX 側
            miss_f = []
            for (px, pfx, rel_dir, stem) in pairs:
                gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")
                if not os.path.exists(gd_f_path):
                    miss_f.append((pfx, rel_dir, stem, gd_f_path))
            pbar = tqdm(list(_chunks(miss_f, gdino_bs if gdino_bs>0 else 1)), desc=f"{split}-obj-gdF")
            for chunk in pbar:
                imgs = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                # ★ しきい値を CLI から反映
                outs = gdino.detect_batch(
                    imgs, args.det_prompts,
                    box_thr=args.gdino_box_thr, txt_thr=args.gdino_text_thr
                )
                for (pfx, rel_dir, stem, outp), dets in zip(chunk, outs):
                    ensure_dir(os.path.dirname(outp)); json.dump(dets, open(outp,"w"), indent=2)


            # YOLOP（drivable マスク）キャッシュ（必要時）
            if args.use_yolop and yolop_model is not None:
                # X
                yx_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    if not os.path.exists(yx_path):
                        yx_miss.append((px, rel_dir, stem, yx_path))
                pbar = tqdm(list(_chunks(yx_miss, yolop_bs if yolop_bs>0 else 1)), desc=f"{split}-obj-yolopX")
                for chunk in pbar:
                    imgs = [imread_rgb(px) for (px,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (px, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv_x = (np.array(y["drivable"], dtype=np.uint8) > 0).astype(np.uint8)  # 既に0/1なので形式統一

                        ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv_x.tolist()}, open(outp,"w"))

                # F
                yf_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")
                    if not os.path.exists(yf_path):
                        yf_miss.append((pfx, rel_dir, stem, yf_path))
                pbar = tqdm(list(_chunks(yf_miss, yolop_bs if yolop_bs>0 else 1)), desc=f"{split}-obj-yolopF")
                for chunk in pbar:
                    imgs = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (pfx, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv_f = (np.array(y["drivable"], dtype=np.uint8) > 0).astype(np.uint8)

                        ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv_f.tolist()}, open(outp,"w"))

            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            per_image_results = []
            pbar = tqdm(pairs, desc=f"{split}-obj")

            for (px, pfx, rel_dir, stem) in pbar:

                rgbx = imread_rgb(px); rgbf = imread_rgb(pfx)
                H, W   = rgbx.shape[:2]
                Hf, Wf = rgbf.shape[:2]

                gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")
                G = json.load(open(gd_x_path,"r")); G = _recanonize_dets(G, args.det_prompts)
                D = json.load(open(gd_f_path,"r")); D = _recanonize_dets(D, args.det_prompts)

                # --- ROIフィルタ（drivable）: ★640→元解像度へアップサイズしてから使用する★
                drv_x = None
                drv_f = None
                if args.use_yolop and yolop_model is not None:
                    # JSONから640x640の2値マスクを読み出し → 元画像サイズにアップサイズ
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")

                    yx = json.load(open(yx_path, "r"))  # {"drivable": [[...]]}
                    yf = json.load(open(yf_path, "r"))

                    m640_x = (np.array(yx["drivable"], dtype=np.uint8) > 0).astype(np.uint8)
                    m640_f = (np.array(yf["drivable"], dtype=np.uint8) > 0).astype(np.uint8)

                    # ★ここが肝：元解像度に合わせる
                    drv_x = upsize_to(rgbx, m640_x)  # shape=(Hx,Wx)
                    drv_f = upsize_to(rgbf, m640_f)  # shape=(Hf,Wf)

                    if args.yolop_roi_filter:
                        def keep_roi(dets: List[Dict[str,Any]], drv_u8: np.ndarray) -> List[Dict[str,Any]]:
                            kept = []
                            h_, w_ = drv_u8.shape[:2]
                            for d in dets:
                                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                                cx = int(np.clip((x1 + x2) / 2, 0, w_ - 1))
                                cy = int(np.clip((y1 + y2) / 2, 0, h_ - 1))
                                if drv_u8[cy, cx] > 0:
                                    kept.append(d)
                            return kept
                        G = keep_roi(G, drv_x)
                        D = keep_roi(D, drv_f)
                # --- ここまで ROI フィルタ修正（アノテの重ね描きは後段で渡す）


                if args.ocr_engine == "tesseract":
                    ocrX = []; ocrF = []
                    for d in G:
                        if "sign" in d["cls"]:
                            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                            crop = rgbx[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                            ocrX.append({"bbox":d["bbox"], "txt":run_ocr_tesseract(crop)})
                    for d in D:
                        if "sign" in d["cls"]:
                            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                            crop = rgbf[max(0,y1):min(Hf,y2), max(0,x1):min(Wf,x2)]
                            ocrF.append({"bbox":d["bbox"], "txt":run_ocr_tesseract(crop)})
                    ocr_x_out = cache.ocr_path(split, rel_dir, stem, "x")
                    ocr_f_out = cache.ocr_path(split, rel_dir, stem, "f")
                    ensure_dir(os.path.dirname(ocr_x_out)); ensure_dir(os.path.dirname(ocr_f_out))
                    json.dump(ocrX, open(ocr_x_out,"w"), indent=2)
                    json.dump(ocrF, open(ocr_f_out,"w"), indent=2)

                # ★ 正規化比較：画像サイズ差を吸収
                dm = det_metrics(G, D, iou_thr=args.iou_thr, img_wh_x=(W,H), img_wh_f=(Wf,Hf))
                per_image_results.append(dm)

                # 検出＋(任意で)drivable を重ねた可視化
                if args.annotation_mode in ("objects","all"):
                    if ann_counter["objects"][split] < args.annotate_limit:
                        ann_drv_x = drv_x if drv_x is not None else None
                        ann_drv_f = drv_f if drv_f is not None else None
                        save_annotations_objects(
                            args.annotate_out, split, rel_dir, stem,
                            rgbx, rgbf, G, D, ann_drv_x, ann_drv_f
                        )
                        ann_counter["objects"][split] += 1

                # 走行可能領域そのものの可視化（tasks=all でも有効にする）
                if args.annotation_mode in ("drivable","all") and (drv_x is not None) and (drv_f is not None):
                    if ann_counter["drivable"][split] < args.annotate_limit:
                        save_annotations_drivable(
                            args.annotate_out, split, rel_dir, stem,
                            rgbx, rgbf, drv_x, drv_f
                        )
                        ann_counter["drivable"][split] += 1

                pbar.set_postfix_str("ok")


            if per_image_results:
                classes = sorted(list({c for r in per_image_results for c in r["per_class"].keys()}))
                agg: Dict[str, Dict[str,float]] = {c: {"PR":0.0,"PP":0.0,"F1":0.0,"HAL":0.0,"N":0.0} for c in classes}
                iou_med=[]; cen_med=[]; sz_med=[]; cad=[]; emd=[]
                for r in per_image_results:
                    iou_med.append(r["iou_median"]); cen_med.append(r["center_error_median"]); sz_med.append(r["size_log_ratio_median"])
                    cad.append(r["count_absdiff_mean"]); emd.append(r["count_wasserstein"])
                    for c,dd in r["per_class"].items():
                        for k in ["PR","PP","F1","HAL"]:
                            agg[c][k] += dd[k]
                        agg[c]["N"] += 1.0
                for c in classes:
                    if agg[c]["N"]>0:
                        for k in ["PR","PP","F1","HAL"]:
                            agg[c][k] = float(agg[c][k]/agg[c]["N"])
                summary = {
                    "iou_median": (float(np.median(iou_med)) if iou_med else float("nan")),
                    "center_error_median": (float(np.median(cen_med)) if cen_med else float("nan")),
                    "size_log_ratio_median": (float(np.median(sz_med)) if sz_med else float("nan")),
                    "count_absdiff_mean": (float(np.mean(cad)) if cad else float("nan")),
                    "count_wasserstein_mean": (float(np.mean(emd)) if emd else float("nan")),
                    "per_class": agg,
                    "n_images": len(per_image_results),  # ★追加：この split で何枚評価したか
                }
                logger.info("[%s][Objects] %s", split, json.dumps(summary, ensure_ascii=False))
                exp_metrics["objects"][split] = summary

 


                if sw:
                    sw.add_scalar(f"objects/iou_median/{split}", summary["iou_median"], 0)
                    sw.add_scalar(f"objects/center_err_med/{split}", summary["center_error_median"], 0)
                    sw.add_scalar(f"objects/size_log_ratio_med/{split}", summary["size_log_ratio_median"], 0)
                    sw.add_scalar(f"objects/count_absdiff_mean/{split}", summary["count_absdiff_mean"], 0)
                    sw.add_scalar(f"objects/count_wasserstein/{split}", summary["count_wasserstein_mean"], 0)
                    for c in classes:
                        for k in ["PR","PP","F1","HAL"]:
                            sw.add_scalar(f"objects/{k}/{split}/{c}", agg[c][k], 0)

        # ---------- Drivable（保持評価：--tasks drivable だけでなく all でも実行） ----------
        if args.tasks in ("all", "drivable"):
            # === 方式選択 ===
            use_yolop    = ("yolop"   in args.drivable_methods)
            use_onefroad = ("onefroad" in args.drivable_methods)
            use_sota     = ("sota"    in args.drivable_methods) and (len(args.sota_seg_model.strip())>0)

            # YOLOP 準備
            yolop_model = None; yolop_bs = 1
            if use_yolop and args.use_yolop:
                yolop_model = load_yolop(logger)
                if args.auto_batch:
                    try:
                        yolop_bs = autotune_yolop_bs(yolop_model, logger, sample_rgb_f, cap=64)
                    except Exception as e:
                        logger.warning("YOLOP AutoBatch 失敗: %s → batch=1", repr(e))
                        yolop_bs = 1

            # OneFormer 準備（F側セグが未キャッシュの場合に使用）
            onef_proc2 = None; onef_model2 = None; onef_bs2 = 1
            if use_onefroad:
                onef_proc2, onef_model2 = build_oneformer(DEFAULT_ONEFORMER_ID, device="cuda", fp16=True)
                if args.auto_batch:
                    try:
                        onef_bs2 = autotune_oneformer_bs(onef_proc2, onef_model2, logger, sample_rgb_f, cap=64)
                    except Exception as e:
                        logger.warning("OneFormer AutoBatch(Drivable) 失敗: %s → batch=1", repr(e))
                        onef_bs2 = 1

            # SOTA(HF) 準備
            hf_proc = None; hf_model = None; hf_id2label = None
            if use_sota:
                try:
                    hf_proc, hf_model, hf_id2label = load_hf_semseg(logger, args.sota_seg_model, device="cuda", fp16=True)
                except Exception as e:
                    logger.error("HF-SemSeg load failed (%s): %s → sota方式はスキップ", args.sota_seg_model, repr(e))
                    use_sota = False

            # --- キャッシュ穴埋め（YOLOP） ---
            if use_yolop and yolop_model is not None:
                # X
                yx_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    if not os.path.exists(yx_path):
                        yx_miss.append((px, rel_dir, stem, yx_path))
                for chunk in tqdm(list(_chunks(yx_miss, yolop_bs)), desc=f"{split}-drvX"):
                    imgs  = [imread_rgb(px) for (px,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (px, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv = (np.array(y["drivable"], dtype=np.uint8) > 0).astype(np.uint8)
                        ensure_dir(os.path.dirname(outp))
                        json.dump({"drivable": drv.tolist()}, open(outp, "w"))
                # F
                yf_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")
                    if not os.path.exists(yf_path):
                        yf_miss.append((pfx, rel_dir, stem, yf_path))
                for chunk in tqdm(list(_chunks(yf_miss, yolop_bs)), desc=f"{split}-drvF"):
                    imgs  = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (pfx, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv = (np.array(y["drivable"], dtype=np.uint8) > 0).astype(np.uint8)
                        ensure_dir(os.path.dirname(outp))
                        json.dump({"drivable": drv.tolist()}, open(outp, "w"))

            # --- 指標集計用バッファ（方式ごと） ---
            per_method_vals: Dict[str, Dict[str, List[float]]] = {}
            def _accum(method: str, m: Dict[str,float]) -> None:
                per_method_vals.setdefault(method, {})
                for k,v in m.items():
                    per_method_vals[method].setdefault(k, []).append(float(v))

            ann_done = 0

            for (px, pfx, rel_dir, stem) in tqdm(pairs, desc=f"{split}-drivable"):
                rgbx = imread_rgb(px)
                rgbf = imread_rgb(pfx)

                masks_for_viz: Dict[str, Tuple[np.ndarray,np.ndarray]] = {}

                # -- yolop --
                if use_yolop and args.use_yolop:
                    yx = json.load(open(cache.yolo_path(split, rel_dir, stem, "x"), "r"))["drivable"]
                    yf = json.load(open(cache.yolo_path(split, rel_dir, stem, "fx"), "r"))["drivable"]
                    a640 = (np.array(yx, dtype=np.uint8) > 0).astype(np.uint8)
                    b640 = (np.array(yf, dtype=np.uint8) > 0).astype(np.uint8)
                    ax = upsize_to(rgbx, a640)
                    bf = upsize_to(rgbf, b640)
                    ax_bin = binarize_mask(ax, thr=args.drivable_thr, morph_k=args.drivable_morph_k)
                    bf_bin = binarize_mask(bf, thr=args.drivable_thr, morph_k=args.drivable_morph_k)
                    m = mask_metrics(ax_bin, bf_bin)
                    bi = boundary_iou(ax_bin, bf_bin, tol=args.drivable_edge_tol)
                    m["drv_boundary_iou"] = bi
                    _accum("yolop", m)
                    masks_for_viz["yolop"] = (ax_bin, bf_bin)

                # -- onefroad --
                if use_onefroad:
                    # X 側は既存 OneFormer_x (trainId) を読み、F 側はキャッシュが無ければ推論
                    seg_x_path = os.path.join(args.semseg_x_root, split, rel_dir, f"{stem}_predTrainId.npy")
                    if not os.path.exists(seg_x_path):
                        logger.warning("semseg X missing (onefroad): %s", seg_x_path)
                        continue
                    seg_x = np.load(seg_x_path).astype(np.uint8)

                    seg_f_path = cache.semseg_fx_path(split, rel_dir, stem)
                    if os.path.exists(seg_f_path):
                        seg_f = np.load(seg_f_path).astype(np.uint8)
                    else:
                        # 単発推論
                        seg_f = oneformer_semseg(onef_proc2, onef_model2, rgbf)
                        ensure_dir(os.path.dirname(seg_f_path)); np.save(seg_f_path, seg_f)

                    drv_x = semseg_to_drivable_onefroad(seg_x, road_ids=args.onefroad-road-ids if hasattr(args,"onefroad-road-ids") else args.onefroad_road_ids)
                    drv_f = semseg_to_drivable_onefroad(seg_f, road_ids=args.onefroad_road_ids)

                    m = mask_metrics(drv_x, drv_f)
                    m["drv_boundary_iou"] = boundary_iou(drv_x, drv_f, tol=args.drivable_edge_tol)
                    _accum("onefroad", m)
                    masks_for_viz["onefroad"] = (drv_x, drv_f)

                # -- sota(HF) --
                if use_sota:
                    sx_path = cache.sota_drv_path(split, rel_dir, stem, "x")
                    sf_path = cache.sota_drv_path(split, rel_dir, stem, "fx")
                    if os.path.exists(sx_path):
                        drv_x = (np.array(json.load(open(sx_path,"r"))["drivable"], dtype=np.uint8)>0).astype(np.uint8)
                    else:
                        drv_x = hf_semseg_drivable_mask(hf_proc, hf_model, rgbx, args.sota_drivable_labels)
                        ensure_dir(os.path.dirname(sx_path)); json.dump({"drivable": drv_x.tolist()}, open(sx_path,"w"))
                    if os.path.exists(sf_path):
                        drv_f = (np.array(json.load(open(sf_path,"r"))["drivable"], dtype=np.uint8)>0).astype(np.uint8)
                    else:
                        drv_f = hf_semseg_drivable_mask(hf_proc, hf_model, rgbf, args.sota_drivable_labels)
                        ensure_dir(os.path.dirname(sf_path)); json.dump({"drivable": drv_f.tolist()}, open(sf_path,"w"))

                    m = mask_metrics(drv_x, drv_f)
                    m["drv_boundary_iou"] = boundary_iou(drv_x, drv_f, tol=args.drivable_edge_tol)
                    _accum("sota", m)
                    masks_for_viz["sota"] = (drv_x, drv_f)

                # 可視化（各方式 3枚＝yolop/onefroad/sota の横並びではなく、方式別に2枚横連結を3ファイル出力）
                if args.annotation_mode in ("drivable","all") and ann_done < args.annotate_limit:
                    if masks_for_viz:
                        save_annotations_drivable_multi(
                            args.annotate_out, split, rel_dir, stem,
                            rgbx, rgbf, masks_for_viz
                        )
                        ann_done += 1

            # --- split平均を方式ごとに保存 ---
            if per_method_vals:
                exp_metrics["drivable"].setdefault(split, {})
                exp_metrics["drivable"][split]["per_method"] = {}
                for method, mm in per_method_vals.items():
                    avg = {k: float(np.mean(vs)) for k,vs in mm.items()}
                    # n_samples は IoU 配列長などの代表
                    any_key = next(iter(mm.keys()))
                    avg["n_samples"] = len(mm[any_key])
                    exp_metrics["drivable"][split]["per_method"][method] = avg
                    logger.info("[%s][Drivable:%s] %s", split, method, json.dumps(avg, ensure_ascii=False))
                    if sw:
                        for k,v in avg.items():
                            if k=="n_samples": continue
                            sw.add_scalar(f"drivable/{method}_{k}/{split}", v, 0)
            else:
                logger.warning("[%s][Drivable] 有効なサンプルがありませんでした。", split)


        # ---------- ★ Diversity（F(X) 内部の多様性）＋ Augmentation Strength（X vs F(X)） ----------
        if args.tasks in ("all", "diversity"):
            if lpips is None or ms_ssim is None:
                logger.error("[Diversity] lpips / pytorch-msssim が見つかりません。entrypoint の PIP_INSTALL / REQS_OVERLAY_PATH を確認してください。")
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                loss_fn = lpips.LPIPS(net=args.lpips_net).to(dev).eval()

                def _prep(img: np.ndarray, size: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
                    """
                    LPIPS / MS-SSIM 共通の前処理
                      - size>0 のときは div-resize-mode に応じて正方形に揃える
                      - 戻り値:
                          - ten    : [0,1] （MS-SSIM 用）
                          - ten_lp : [-1,1]（LPIPS 用）
                    """
                    im = img
                    if size and size > 0:
                        h, w = im.shape[:2]
                        if mode == "square":
                            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                        elif mode == "letterbox":
                            if h >= w:
                                new_h, new_w = size, max(1, int(round(size * w / h)))
                            else:
                                new_w, new_h = size, max(1, int(round(size * h / w)))
                            im_res = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            pad_top  = (size - new_h) // 2
                            pad_bot  = size - new_h - pad_top
                            pad_left = (size - new_w) // 2
                            pad_right= size - new_w - pad_left
                            im = cv2.copyMakeBorder(
                                im_res, pad_top, pad_bot, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                            )
                        elif mode == "center-crop":
                            if h <= w:
                                new_h = size
                                new_w = max(1, int(round(size * w / h)))
                            else:
                                new_w = size
                                new_h = max(1, int(round(size * h / w)))
                            im_res = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            y0 = max(0, (new_h - size) // 2)
                            x0 = max(0, (new_w - size) // 2)
                            im = im_res[y0:y0+size, x0:x0+size]
                        else:
                            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                    ten = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0  # [0,1]
                    ten_lp = ten * 2.0 - 1.0                                      # [-1,1]
                    return ten.to(dev), ten_lp.to(dev)

                # ---- (1) F(X) 内部の多様性: ランダムペアで LPIPS / 1-MS-SSIM ----
                gen_list = [pfx for (_, pfx, _, _) in pairs]
                n = len(gen_list)
                if n < 2:
                    logger.warning("[%s][Diversity] ペア数不足 (n=%d)", split, n)
                else:
                    rs = np.random.RandomState(args.seed + 1234)
                    idx_pairs = sample_unique_pairs(n, args.div_pairs, rs)

                    lpips_vals: List[float] = []
                    msssim_vals: List[float] = []
                    pbar = tqdm(idx_pairs, desc=f"{split}-div")
                    for i, j in pbar:
                        im1 = imread_rgb(gen_list[i])
                        im2 = imread_rgb(gen_list[j])
                        t1, t1_lp = _prep(im1, args.div_resize, args.div_resize_mode)
                        t2, t2_lp = _prep(im2, args.div_resize, args.div_resize_mode)

                        with torch.inference_mode():
                            d = loss_fn(t1_lp.unsqueeze(0), t2_lp.unsqueeze(0)).detach().cpu().numpy().item()
                            lpips_vals.append(float(d))
                            m = ms_ssim(t1.unsqueeze(0), t2.unsqueeze(0), data_range=1.0).detach().cpu().numpy().item()
                            msssim_vals.append(float(1.0 - m))
                        pbar.set_postfix_str("ok")

                    res = {
                        "pairs": len(idx_pairs),
                        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
                        "lpips_std": float(np.std(lpips_vals)) if lpips_vals else float("nan"),
                        "one_minus_ms_ssim_mean": float(np.mean(msssim_vals)) if msssim_vals else float("nan"),
                        "one_minus_ms_ssim_std": float(np.std(msssim_vals)) if msssim_vals else float("nan"),
                    }
                    logger.info("[%s][Diversity] %s", split, json.dumps(res, ensure_ascii=False))
                    exp_metrics["diversity"][split] = res
                    if sw:
                        sw.add_scalar(f"diversity/lpips_mean/{split}", res["lpips_mean"], 0)
                        sw.add_scalar(f"diversity/1-ms-ssim_mean/{split}", res["one_minus_ms_ssim_mean"], 0)
                    ensure_dir(os.path.dirname(cache.diversity_result_path(split)))
                    json.dump(res, open(cache.diversity_result_path(split), "w"), indent=2, ensure_ascii=False)

                # ---- (2) Augmentation Strength: X vs F(X) の LPIPS / 1-MS-SSIM ----
                aug_lpips_vals: List[float] = []
                aug_msssim_vals: List[float] = []
                pbar2 = tqdm(pairs, desc=f"{split}-aug")
                for (px, pfx, rel_dir, stem) in pbar2:
                    imx = imread_rgb(px)
                    imf = imread_rgb(pfx)
                    tx, tx_lp = _prep(imx, args.div_resize, args.div_resize_mode)
                    tf, tf_lp = _prep(imf, args.div_resize, args.div_resize_mode)
                    with torch.inference_mode():
                        d = loss_fn(tx_lp.unsqueeze(0), tf_lp.unsqueeze(0)).detach().cpu().numpy().item()
                        aug_lpips_vals.append(float(d))
                        m = ms_ssim(tx.unsqueeze(0), tf.unsqueeze(0), data_range=1.0).detach().cpu().numpy().item()
                        aug_msssim_vals.append(float(1.0 - m))
                    pbar2.set_postfix_str("ok")

                if aug_lpips_vals:
                    aug_res = {
                        "pairs": len(aug_lpips_vals),
                        "lpips_mean": float(np.mean(aug_lpips_vals)),
                        "lpips_std": float(np.std(aug_lpips_vals)),
                        "one_minus_ms_ssim_mean": float(np.mean(aug_msssim_vals)),
                        "one_minus_ms_ssim_std": float(np.std(aug_msssim_vals)),
                    }
                    logger.info("[%s][AugStrength X_vs_FX] %s", split, json.dumps(aug_res, ensure_ascii=False))
                    exp_metrics["aug_strength"][split] = aug_res
                    if sw:
                        sw.add_scalar(f"aug_strength/lpips_mean/{split}", aug_res["lpips_mean"], 0)
                        sw.add_scalar(f"aug_strength/1-ms-ssim_mean/{split}", aug_res["one_minus_ms_ssim_mean"], 0)
                else:
                    logger.warning("[%s][AugStrength] 有効なペアがありませんでした。", split)


        # ---------- ★ Distribution Coverage: Improved Precision & Recall ----------
        if args.tasks in ("all","diversity"):
            # Reality セクションで CLIP 特徴を計算済みなら再利用したいが、
            # 単独実行にも対応できるよう必要ならここで計算
            if clipper is None:
                clipper = ClipEmbedder(args.clip_model); clipper.load(logger)
                clip_bs = clipper.max_batch or args.clip_batch or 16
            # X / FX の CLIP 特徴を読み込み or 計算
            feats_x = []; feats_fx = []
            for (px, pfx, rel_dir, stem) in pairs:
                cx = cache.clip_path(split, rel_dir, stem, "x")
                cf = cache.clip_path(split, rel_dir, stem, "fx")
                if os.path.exists(cx) and os.path.exists(cf):
                    ex = np.load(cx)["feat"]; ef = np.load(cf)["feat"]
                else:
                    ex = clipper.embed_batch([imread_rgb(px)])[0]
                    ef = clipper.embed_batch([imread_rgb(pfx)])[0]
                    ensure_dir(os.path.dirname(cx)); np.savez(cx, feat=ex)
                    ensure_dir(os.path.dirname(cf)); np.savez(cf, feat=ef)
                feats_x.append(ex); feats_fx.append(ef)
            X = np.stack(feats_x, axis=0).astype(np.float64)
            Y = np.stack(feats_fx, axis=0).astype(np.float64)
            if X.shape[0] >= 5 and Y.shape[0] >= 5:
                # k-NN 半径（各集合の自己距離→k番目）
                k = max(1, int(args.pr_k))
                Dx = pairwise_sqeuclid(X, X)
                np.fill_diagonal(Dx, np.inf)
                rad_x = np.sort(Dx, axis=1)[:, k-1]
                Dy = pairwise_sqeuclid(Y, Y)
                np.fill_diagonal(Dy, np.inf)
                rad_y = np.sort(Dy, axis=1)[:, k-1]
                # 精度: Y が X マニフォールド内に入る割合
                D_yx = pairwise_sqeuclid(Y, X)  # [Ny,Nx]
                in_x = (D_yx <= rad_x[None, :])
                precision = float(np.mean(np.any(in_x, axis=1)))
                # 再現率: X が Y マニフォールド内
                D_xy = pairwise_sqeuclid(X, Y)
                in_y = (D_xy <= rad_y[None, :])
                recall = float(np.mean(np.any(in_y, axis=1)))
                logger.info("[%s][P/R (k=%d)] precision=%.4f | recall=%.4f", split, k, precision, recall)
                exp_metrics["pr"][split] = {
                    "k": int(k),
                    "precision": precision,
                    "recall": recall,
                    "n_samples": int(X.shape[0]),  # ★X/F(X) の CLIP 特徴の点数
                }


                if sw:
                    sw.add_scalar(f"diversity/precision/{split}", precision, 0)
                    sw.add_scalar(f"diversity/recall/{split}", recall, 0)

        # ---------- ★ Text Prompt 追従（CLIPScore / R-Precision / Inter-prompt） ----------
        if args.tasks in ("all","text"):
            if clipper is None:
                clipper = ClipEmbedder(args.clip_model); clipper.load(logger)

            prompts = []
            images_for_text = []
            keys_meta = []

            for (px, pfx, rel_dir, stem) in pairs:
                ptxt = prompt_from_meta_or_files(args.gen_root, args.prompt_root, split, rel_dir, stem)
                if ptxt is None or len(ptxt.strip()) == 0:
                    continue
                prompts.append(ptxt.strip())
                images_for_text.append(pfx)
                keys_meta.append((rel_dir, stem))

            if not prompts:
                logger.warning("[%s][Text] 有効なプロンプトが見つかりませんでした。", split)
            else:
                # 画像埋め込み
                img_feats = []
                for pfx in images_for_text:
                    img_feats.append(clipper.embed_batch([imread_rgb(pfx)])[0])
                V = np.stack(img_feats, axis=0)

                # テキスト埋め込み（キャッシュ）
                txt_feats = []
                for (rel_dir, stem), txt in zip(keys_meta, prompts):
                    cp = cache.clip_text_path(split, rel_dir, stem)
                    if os.path.exists(cp):
                        t = np.load(cp)["feat"]
                    else:
                        t = clipper.embed_texts([txt])[0]
                        ensure_dir(os.path.dirname(cp)); np.savez(cp, feat=t)
                    txt_feats.append(t)
                T = np.stack(txt_feats, axis=0)
                # CLIPScore（Hessel+2021）
                cos = np.sum(V * T, axis=1).clip(-1.0, 1.0)
                clips = 2.5 * np.maximum(cos, 0.0)  # 論文既定 w=2.5
                mean_cs = float(np.mean(clips))
                std_cs  = float(np.std(clips))
                logger.info("[%s][CLIPScore] mean=%.4f | std=%.4f", split, mean_cs, std_cs)
                exp_metrics["clipscore"][split] = {
                    "mean": mean_cs,
                    "std": std_cs,
                    "n_samples": len(clips),  # ★画像数 = プロンプト数
                }
                if sw:
                    sw.add_scalar(f"text/clips_mean/{split}", mean_cs, 0)


                # R-Precision@K（AttnGAN 系）
                K = max(1, int(args.rprecision_k))
                negN = max(1, int(args.rprecision_negatives))
                rs = np.random.RandomState(args.seed + 5678)
                # 各画像 i につき：真のテキスト Ti と、負例 negN 個を抽出 → 類似度順位
                correct = 0; correct_at5 = 0
                for i in range(len(V)):
                    # 自分以外から負例サンプリング
                    cand_idx = [j for j in range(len(T)) if j != i]
                    if len(cand_idx) < negN:
                        neg_idx = cand_idx
                    else:
                        neg_idx = list(rs.choice(cand_idx, size=negN, replace=False))
                    cand = np.stack([T[i]] + [T[j] for j in neg_idx], axis=0)  # [1+negN, D]
                    sim = (cand @ V[i][:,None]).squeeze(-1)  # cos（正規化済み）
                    order = np.argsort(-sim)  # 降順
                    if 0 in order[:K]:
                        correct += 1
                    if 0 in order[:min(5, len(order))]:
                        correct_at5 += 1

                rprec = float(correct / max(1, len(V)))
                rprec5 = float(correct_at5 / max(1, len(V)))
                logger.info("[%s][R-Precision@%d] = %.4f | [@5]=%.4f (negatives=%d)", split, K, rprec, rprec5, negN)
                exp_metrics["rprecision"][split] = {
                    "k": int(K),
                    "negatives": int(negN),
                    "rprecision_at_k": float(rprec),
                    "rprecision_at5": float(rprec5),
                    "n_samples": len(V),  # ★画像数
                }
                if sw:
                    sw.add_scalar(f"text/rprecision_at{K}/{split}", rprec, 0)
                    sw.add_scalar(f"text/rprecision_at5/{split}", rprec5, 0)


                # Inter-prompt（CLIP）
                if len(T) >= 3:

                    if args.interprompt_method == "clip-corr":
                        Dt = pairwise_sqeuclid(T, T)
                        Di = pairwise_sqeuclid(V, V)
                        iu = np.triu_indices_from(Dt, k=1)
                        rho, _ = spearmanr(Dt[iu].ravel(), Di[iu].ravel())
                        rho_f = float(rho)
                        logger.info("[%s][InterPrompt Alignment ρ] = %.4f", split, rho_f)
                        exp_metrics["interprompt"][split] = {
                            "type": "clip-corr",
                            "rho": rho_f,
                            "n_samples": T.shape[0],  # ★プロンプト数
                        }
                        if sw:
                            sw.add_scalar(f"text/interprompt_rho/{split}", rho_f, 0)
                    else:  # clip-spread
                        Dt = pairwise_sqeuclid(T, T)
                        iu = np.triu_indices_from(Dt, k=1)
                        spread = float(np.mean(np.sqrt(Dt[iu])))
                        logger.info("[%s][InterPrompt Spread] = %.4f", split, spread)
                        exp_metrics["interprompt"][split] = {
                            "type": "clip-spread",
                            "spread": spread,
                            "n_samples": T.shape[0],
                        }
                        if sw:
                            sw.add_scalar(f"text/interprompt_spread/{split}", spread, 0)


    # 実験メタ情報を JSON で保存（人間・機械可読）
    # 実験メタ情報を JSON で保存（人間・機械可読）
    # の前に、Waymo 全体 (training + validation + testing) の集約メトリクスを計算する。
    global_summary: Dict[str, Any] = {}

    # split ごとのペア数そのものも記録しておく
    global_summary["total_pairs"] = int(total_pairs)
    global_summary["pairs_per_split"] = {k: int(v) for k, v in split_pair_counts.items()}

    # --- Reality (CLIP-CMMD / CLIP-FID) を画像枚数で重み付き平均 ---
    if exp_metrics["reality"]:
        tot_imgs = 0
        acc_val = 0.0
        for split_name, m in exp_metrics["reality"].items():
            n_i = int(m.get("n_images", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            tot_imgs += n_i
            acc_val += float(m.get("value", 0.0)) * n_i
        if tot_imgs > 0:
            global_summary["reality"] = {
                "metric": args.reality_metric,
                "value": float(acc_val / tot_imgs),
                "total_images": int(tot_imgs),
            }

    def _aggregate_simple(category: str, weight_key: str, skip_keys: Optional[List[str]] = None) -> None:
        """
        split ごとに平均されたスカラー指標を、重み付き平均で Waymo 全体に集約するユーティリティ。
        weight_key には "n_samples" や "pairs" などを渡す。
        """
        cat = exp_metrics.get(category, {})
        if not cat:
            return
        if skip_keys is None:
            skip_set = set()
        else:
            skip_set = set(skip_keys)
        example = next(iter(cat.values()))
        keys = []
        for k, v in example.items():
            if k in skip_set:
                continue
            if isinstance(v, (int, float)):
                keys.append(k)
        if not keys:
            return
        acc = {k: 0.0 for k in keys}
        total_w = 0
        for split_name, m in cat.items():
            w = int(m.get(weight_key, split_pair_counts.get(split_name, 0)))
            if w <= 0:
                continue
            total_w += w
            for k in keys:
                acc[k] += float(m.get(k, 0.0)) * w
        if total_w <= 0:
            return
        out = {k: float(acc[k] / total_w) for k in keys}
        out["total_weight"] = int(total_w)
        global_summary[category] = out

    # Edge / Depth は従来通り
    _aggregate_simple("edge", "n_samples")
    _aggregate_simple("depth", "n_samples")

    # Drivable: per_method を方式ごとに重み付き平均して格納
    if exp_metrics["drivable"]:
        methods = set()
        for split_name, m in exp_metrics["drivable"].items():
            pm = m.get("per_method", {})
            methods |= set(pm.keys())
        global_summary["drivable_per_method"] = {}
        for meth in sorted(list(methods)):
            acc = {}; total_w = 0
            for split_name, m in exp_metrics["drivable"].items():
                pm = m.get("per_method", {})
                if meth not in pm: continue
                w = int(pm[meth].get("n_samples", 0))
                if w <= 0: continue
                total_w += w
                for k,v in pm[meth].items():
                    if k=="n_samples": continue
                    acc[k] = acc.get(k, 0.0) + float(v) * w
            if total_w > 0:
                out = {k: float(v/total_w) for k,v in acc.items()}
                out["total_weight"] = int(total_w)
                global_summary["drivable_per_method"][meth] = out

    # Diversity / Augmentation Strength は "pairs" を重みとして平均（分散は split ごとの値を参照）
    _aggregate_simple("diversity", "pairs", skip_keys=["lpips_std", "one_minus_ms_ssim_std"])
    _aggregate_simple("aug_strength", "pairs", skip_keys=["lpips_std", "one_minus_ms_ssim_std"])

    # Improved Precision / Recall（CLIP manifold P/R）
    if exp_metrics["pr"]:
        tot = 0
        acc_p = 0.0
        acc_r = 0.0
        k_val = None
        for split_name, m in exp_metrics["pr"].items():
            n_i = int(m.get("n_samples", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            tot += n_i
            acc_p += float(m.get("precision", 0.0)) * n_i
            acc_r += float(m.get("recall", 0.0)) * n_i
            if k_val is None:
                k_val = int(m.get("k", 0))
        if tot > 0:
            global_summary["pr"] = {
                "k": int(k_val or 0),
                "precision": float(acc_p / tot),
                "recall": float(acc_r / tot),
                "total_samples": int(tot),
            }

    # CLIPScore 全体平均
    if exp_metrics["clipscore"]:
        tot = 0
        acc_m = 0.0
        for split_name, m in exp_metrics["clipscore"].items():
            n_i = int(m.get("n_samples", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            tot += n_i
            acc_m += float(m.get("mean", 0.0)) * n_i
        if tot > 0:
            global_summary["clipscore"] = {
                "mean": float(acc_m / tot),
                "total_samples": int(tot),
            }

    # R-Precision 全体平均
    if exp_metrics["rprecision"]:
        tot = 0
        acc_k = 0.0
        acc_k5 = 0.0
        k_val = None
        neg_val = None
        for split_name, m in exp_metrics["rprecision"].items():
            n_i = int(m.get("n_samples", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            tot += n_i
            acc_k += float(m.get("rprecision_at_k", 0.0)) * n_i
            acc_k5 += float(m.get("rprecision_at5", 0.0)) * n_i
            if k_val is None:
                k_val = int(m.get("k", 0))
            if neg_val is None:
                neg_val = int(m.get("negatives", 0))
        if tot > 0:
            global_summary["rprecision"] = {
                "k": int(k_val or 0),
                "negatives": int(neg_val or 0),
                "rprecision_at_k": float(acc_k / tot),
                "rprecision_at5": float(acc_k5 / tot),
                "total_samples": int(tot),
            }

    # Inter-prompt: clip-corr / clip-spread
    if exp_metrics["interprompt"]:
        tot = 0
        acc_val = 0.0
        t_type = None
        for split_name, m in exp_metrics["interprompt"].items():
            n_i = int(m.get("n_samples", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            tot += n_i
            if m.get("type") == "clip-corr":
                acc_val += float(m.get("rho", 0.0)) * n_i
                t_type = "clip-corr"
            else:
                acc_val += float(m.get("spread", 0.0)) * n_i
                t_type = m.get("type", "clip-spread")
        if tot > 0 and t_type is not None:
            if t_type == "clip-corr":
                global_summary["interprompt"] = {
                    "type": "clip-corr",
                    "rho": float(acc_val / tot),
                    "total_samples": int(tot),
                }
            else:
                global_summary["interprompt"] = {
                    "type": t_type,
                    "spread": float(acc_val / tot),
                    "total_samples": int(tot),
                }

    # Semseg: confusion 行列を split ごとに保存してあるのでそれらを合算してから mIoU を計算
    conf_total = None
    for split_name in args.splits:
        cm_path = os.path.join(cache.d_logs, f"{split_name}_confusion.npy")
        if not os.path.exists(cm_path):
            continue
        try:
            cm = np.load(cm_path)
        except Exception:
            continue
        if conf_total is None:
            conf_total = cm.astype(np.int64)
        else:
            conf_total = conf_total + cm.astype(np.int64)
    if conf_total is not None:
        g_miou, g_iou = miou_from_conf(conf_total)
        global_summary["semseg"] = {
            "miou": float(g_miou),
            "per_class_iou": [float(x) for x in g_iou.tolist()],
            "total_pixels": int(np.sum(conf_total)),
        }
        try:
            np.save(os.path.join(cache.d_logs, "all_confusion.npy"), conf_total)
        except Exception:
            pass

    # Objects: split ごとの summary を、画像枚数で重み付き平均
    if exp_metrics["objects"]:
        cat = exp_metrics["objects"]
        total_imgs = 0
        keys = ["iou_median", "center_error_median", "size_log_ratio_median",
                "count_absdiff_mean", "count_wasserstein_mean"]
        acc = {k: 0.0 for k in keys}
        cls_acc: Dict[str, Dict[str, float]] = {}
        for split_name, m in cat.items():
            n_i = int(m.get("n_images", split_pair_counts.get(split_name, 0)))
            if n_i <= 0:
                continue
            total_imgs += n_i
            for k in keys:
                acc[k] += float(m.get(k, 0.0)) * n_i
            per_cls = m.get("per_class", {})
            for cls, stats in per_cls.items():
                if cls not in cls_acc:
                    cls_acc[cls] = {"sum_PR": 0.0, "sum_PP": 0.0,
                                    "sum_F1": 0.0, "sum_HAL": 0.0,
                                    "N": 0.0}
                w_cls = float(stats.get("N", 0.0))
                if w_cls <= 0.0:
                    w_cls = float(n_i)
                cls_acc[cls]["sum_PR"]  += float(stats.get("PR", 0.0)) * w_cls
                cls_acc[cls]["sum_PP"]  += float(stats.get("PP", 0.0)) * w_cls
                cls_acc[cls]["sum_F1"]  += float(stats.get("F1", 0.0)) * w_cls
                cls_acc[cls]["sum_HAL"] += float(stats.get("HAL", 0.0)) * w_cls
                cls_acc[cls]["N"]       += w_cls
        if total_imgs > 0:
            obj_out = {k: float(acc[k] / total_imgs) for k in keys}
            obj_out["n_images"] = int(total_imgs)
            pc_out: Dict[str, Dict[str, float]] = {}
            for cls, st in cls_acc.items():
                if st["N"] <= 0.0:
                    continue
                pc_out[cls] = {
                    "PR":  float(st["sum_PR"]  / st["N"]),
                    "PP":  float(st["sum_PP"]  / st["N"]),
                    "F1":  float(st["sum_F1"]  / st["N"]),
                    "HAL": float(st["sum_HAL"] / st["N"]),
                    "N":   float(st["N"]),
                }
            obj_out["per_class"] = pc_out
            global_summary["objects"] = obj_out
    # 実験メタ情報を JSON で保存（人間・機械可読）
    try:
        exp_record = {
            "experiment_id": args.experiment_id,
            "experiment_note": args.experiment_note,
            "timestamp": int(time.time()),
            "orig_root": args.orig_root,
            "gen_root": args.gen_root,
            "prompt_root": args.prompt_root,
            "cache_root": args.cache_root,
            "splits": args.splits,
            "camera": args.camera,
            "tasks": args.tasks,
            "reality_metric": args.reality_metric,
            "total_pairs": int(total_pairs),
            "cmdline": " ".join(sys.argv),
            "metrics": exp_metrics,          # split ごとの詳細
            "metrics_global": global_summary # ★Waymo 全体のまとめ
        }

        exp_path = cache.experiment_eval_path(args.experiment_id)
        ensure_dir(os.path.dirname(exp_path))
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(exp_record, f, ensure_ascii=False, indent=2)

        # 実験一覧インデックスも更新
        idx_path = cache.experiment_index_path()
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

        index[args.experiment_id] = {
            "experiment_id": args.experiment_id,
            "timestamp": exp_record["timestamp"],
            "gen_root": args.gen_root,
            "prompt_root": args.prompt_root,
            "cache_root": args.cache_root,
            "splits": args.splits,
            "camera": args.camera,
            "tasks": args.tasks,
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        logger.info("実験レコードを書き出しました: %s", exp_path)
    except Exception as e:
        logger.error("実験レコードの保存に失敗しました: %s", repr(e))

    logger.info("✅ 全 split 完了。総ペア数: %d | キャッシュ: %s", total_pairs, args.cache_root)
    if sw:
        sw.close()




if __name__ == "__main__":
    main()


