#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/home/shogo/coding/prep/ucn_make_prompts.py

目的:
  SynDiff‑AD(Goel et al., 2025) の “Caption Generation + Subgroup (z / z*)” を
  既存環境 (Ubuntu + RTX5090, CUDA 12.8, Torch 2.7.0+cu128, Docker: ucn-eval) 上で
  一括実行できるユーティリティ。

  - VLM: 既定は Qwen3‑VL‑32B-Instruct (Transformers, enable_thinking=False)
          代替: Qwen2.5‑VL‑32B-Instruct、さらに vLLM で Qwen3‑VL‑32B‑FP8 も選択可
  - CLIP: open_clip_torch (ViT‑SO400M‑14‑SigLIP‑384) で weather × time を分類
  - Train セット (BDD10K/Cityscapes/GTA5/nuImages(front)/BDD100K):
      → z(元サブグループ) を caption に付与して保存
  - Eval セット (Waymo):
      → z*(目標サブグループ; Z\{z} から一様サンプル) を caption に付与し、
         ご指定の「高品質プロンプト構文（形容詞＋装飾文＋Keep the same camera angle...）」で整形
  - simple モード: 学習/評価ともに “簡素プロンプト” での生成を可能に (既定 OFF)

出力:
  /data/ucn_prompts/
    ├─ raw_captions/{DATASET_KEY}/{split}/.../{stem}.txt        # VLM 生キャプション (キャッシュ)
    ├─ prompts_train/{DATASET_KEY}/{split}/.../{stem}.txt        # 学習用 (z を付与)
    ├─ prompts_eval_waymo/{split}/.../{stem}.txt                 # 評価用 (z* を付与; 高品質構文)
    ├─ meta/{DATASET_KEY}/subgroups_source.csv                  # 画像 → z の CSV
    ├─ meta/{DATASET_KEY}/subgroups_target.csv                  # 画像 → z* の CSV (Waymo のみ)
    └─ logs/ucn_make_prompts.log                                 # きわめて詳細な回転ログ

  TensorBoard:
    /data/ucn_prompt_tb/

前提データ (翔伍さん環境):
  - RGB ルート:
      BDD10K(10K)     /home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images/{train,val}
      Cityscapes      /home/shogo/coding/datasets/cityscapes/leftImg8bit/{train,val,test}
      GTA5            /home/shogo/coding/datasets/GTA5/images/images
      nuImages(front) /home/shogo/coding/datasets/nuimages/samples/CAM_FRONT
      BDD100K         /home/shogo/coding/datasets/BDD_100K_pure100k/{train,val,test}
      Waymo RGB       /home/shogo/coding/datasets/WaymoV2/extracted/{training,validation,testing}/front/<segment_id>/...jpg
  - セマンティクス:
      Cityscapes/BDD10K(10K)/GTA5/BDD100K は Cityscapes trainId 互換 PNG (各データセット標準の場所)
      nuImages/Waymo は OneFormer 推論物:
        nuImages  → /data/ucn_condmaps/nuimages_front/semseg/.../{stem}_semseg.jpg (Cityscapes カラーマップ)
        Waymo     → /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes/{training,validation,testing}/front/.../*_predTrainId.npy

重要実装メモ:
  - SynDiff‑AD 3.2節 (CaG): 「セマンティックマスク由来のオブジェクト列」を VLM プロンプトに含め、
    さらに z or z* を “スタイル行” として末尾に付与（論文アルゴリズム1の c* = c ⊔ z* に対応）
  - Waymo の z* ではご要望の「高品質構文」を適用:
      "A realistic {adj} city street scene with {classes}. {decorations}. Keep the same camera angle and composition as the original {src_time} image."
    adj / decorations はサブグループごとに 1 個＋≥3 個をルールベースで割当 (拡張容易)
  - 思考ノイズ(“Thinking…”, `<think>...</think>`) 除去 + 空文字時の再試行 (temperature/top_p 切替)
  - seed 固定 (RUN 間再現; ただし生成多様性は sampling で揺れる設計)

実行例は本ファイル末尾の “CLI & Docker ワンライナー” を参照。
"""

from __future__ import annotations

# ====== 標準 ======
import os
import sys
import re
import csv
import json
import math
import time
import random
import logging
from logging import handlers
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict, Counter

# ====== 数値・画像 ======
import numpy as np
import cv2
from PIL import Image

# ====== Torch / 外部 ======
import torch
from torch.utils.tensorboard import SummaryWriter

# VLM (Qwen)
from transformers import AutoProcessor
try:
    # Qwen3
    from transformers import Qwen3VLForConditionalGeneration as Qwen3Model
except Exception:
    Qwen3Model = None

try:
    # Qwen2.5
    from transformers import Qwen2_5_VLForConditionalGeneration as Qwen25Model
except Exception:
    Qwen25Model = None

# CLIP (open_clip)
import open_clip

# ==============================
# 0) 既定パス（翔伍さん環境と完全整合）
# ==============================
# 入力 (RGB)
BDD10K10K_IMG_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images"  # {train,val}
CITYSCAPES_IMG_ROOT = "/home/shogo/coding/datasets/cityscapes/leftImg8bit"                     # {train,val,test}
GTA5_IMG_ROOT       = "/home/shogo/coding/datasets/GTA5/images/images"                         # 直下 *.png
NUIMAGES_FRONT_ROOT = "/home/shogo/coding/datasets/nuimages/samples/CAM_FRONT"
BDD100K_IMG_ROOT    = "/home/shogo/coding/datasets/BDD_100K_pure100k"                          # {train,val,test}
WAYMO_IMG_ROOT      = "/home/shogo/coding/datasets/WaymoV2/extracted"                          # {training,validation,testing}/front/...

# セマンティクス（trainId / color / npy）
CITYSCAPES_GT_ROOT  = "/home/shogo/coding/datasets/cityscapes/gtFine"                           # {train,val}/.../*_labelIds.png
BDD10K10K_GT_ROOT   = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/labels/sem_seg/masks" # {train,val}
GTA5_GT_ROOT        = "/home/shogo/coding/datasets/GTA5/labels/labels"                          # 直下 *.png (trainId 互換)
BDD100K_GT_ROOT     = "/home/shogo/coding/datasets/BDD_100K_pure100k"                           # {train,val,test} 内に semseg がある場合は優先。なければ color 可視化を利用
NUIMAGES_SEM_COLOR  = "/data/ucn_condmaps/nuimages_front/semseg"                                # *_semseg.jpg (Cityscapes カラー)
WAYMO_SEM_NPY_ROOT  = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"                # {training,validation,testing}/front/.../*_predTrainId.npy

# 出力
DEFAULT_OUT_ROOT    = "/data/ucn_prompts"
DEFAULT_TB_DIR      = "/data/ucn_prompt_tb"
DEFAULT_LOG_DIR     = "/data/ucn_prompts/logs"

# 画像拡張子
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ==============================
# 1) ログ
# ==============================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def setup_logger(log_dir: str, verbose: bool) -> logging.Logger:
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "ucn_make_prompts.log")
    logger = logging.getLogger("ucn_prompt")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=50*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))
    logger.addHandler(ch); logger.addHandler(fh)
    logging.raiseExceptions = False
    return logger

# ==============================
# 2) 共通ユーティリティ
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

def stem_without_cityscapes_suffix(filename: str) -> str:
    # aachen_000000_000019_leftImg8bit.png → aachen_000000_000019
    s = os.path.splitext(os.path.basename(filename))[0]
    if s.endswith("_leftImg8bit"):
        s = s[:-len("_leftImg8bit")]
    return s

def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def to_gray3(u8: np.ndarray) -> np.ndarray:
    if u8.ndim == 2:
        return np.stack([u8,u8,u8], axis=2)
    return u8

# ==============================
# 3) Cityscapes 19 クラスとカラー逆引き
# ==============================
CITYSCAPES_TRAINID_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
# BGR → RGB 注意: 既存可視化は RGB 保存
_CITYSCAPES_TRAINID_COLORS_BGR = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
CITYSCAPES_TRAINID_COLORS_RGB = [(bgr[2],bgr[1],bgr[0]) for bgr in _CITYSCAPES_TRAINID_COLORS_BGR]
COLOR2TID = {tuple(rgb): tid for tid, rgb in enumerate(CITYSCAPES_TRAINID_COLORS_RGB)}

def classes_from_semseg_image(img_rgb: np.ndarray, topk: int = 12) -> List[str]:
    """
    Cityscapes 可視化 (RGB カラー) → trainId 推定 → クラス名頻度順 topk
    未知色は無視。
    """
    h,w = img_rgb.shape[:2]
    arr = img_rgb.reshape(-1,3)
    # 高速化: 256^3 を直接 dict lookup
    # ただしノイズ色はマスク
    tids = []
    for r,g,b in arr:
        tid = COLOR2TID.get((r,g,b), -1)
        if tid >= 0:
            tids.append(tid)
    if not tids:
        return []
    cnt = Counter(tids)
    names = [CITYSCAPES_TRAINID_NAMES[i] for i,_ in cnt.most_common(topk)]
    return names

def classes_from_trainid_png(png_path: str, topk: int = 12) -> List[str]:
    m = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return []
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    tids = m.astype(np.int32).ravel()
    tids = tids[(tids >= 0) & (tids < len(CITYSCAPES_TRAINID_NAMES))]
    if tids.size == 0:
        return []
    cnt = Counter(tids.tolist())
    names = [CITYSCAPES_TRAINID_NAMES[i] for i,_ in cnt.most_common(topk)]
    return names

# ==============================
# 4) Waymo の NPY → クラス名
# ==============================
def classes_from_waymo_npy(npy_path: str, topk: int = 12) -> List[str]:
    if not os.path.isfile(npy_path):
        return []
    try:
        arr = np.load(npy_path)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        tids = arr.astype(np.int32).ravel()
        tids = tids[(tids >= 0) & (tids < len(CITYSCAPES_TRAINID_NAMES))]
        if tids.size == 0:
            return []
        cnt = Counter(tids.tolist())
        names = [CITYSCAPES_TRAINID_NAMES[i] for i,_ in cnt.most_common(topk)]
        return names
    except Exception:
        return []

# ==============================
# 5) データセット列挙 & セマンティクス取得
# ==============================
def relpath_under(base: str, p: str) -> str:
    return os.path.relpath(os.path.dirname(p), base)

def list_bdd10k10k(split: str) -> List[str]:
    return list_images_under(os.path.join(BDD10K10K_IMG_ROOT, split))

def list_cityscapes(split: str) -> List[str]:
    return list_images_under(os.path.join(CITYSCAPES_IMG_ROOT, split))

def list_gta5() -> List[str]:
    return list_images_under(GTA5_IMG_ROOT)

def list_nuimages() -> List[str]:
    return list_images_under(NUIMAGES_FRONT_ROOT)

def list_bdd100k(split: str) -> List[str]:
    return list_images_under(os.path.join(BDD100K_IMG_ROOT, split))

def list_waymo(split_key: str) -> List[str]:
    # split_key in {"training","validation","testing"}
    root = os.path.join(WAYMO_IMG_ROOT, split_key, "front")
    return list_images_under(root)

def try_semseg_classes_for_dataset(dataset_key: str, image_path: str, base_rel: str) -> List[str]:
    """
    入力画像に対応するセマンティクス（trainId or color 可視化 or npy）を探し、
    Cityscapes クラス名の Top-k を返す。
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if dataset_key == "cityscapes":
        # labelIds: gtFine/<split>/.../*_gtFine_labelIds.png
        # stem 特別規則 (_leftImg8bit を落とす)
        stem2 = stem_without_cityscapes_suffix(image_path)
        # split は RGB 側のパスから推定
        rel = os.path.relpath(os.path.dirname(image_path), CITYSCAPES_IMG_ROOT)  # train/aachen
        parts = rel.split(os.sep)
        split = parts[0] if parts else "train"
        png = os.path.join(CITYSCAPES_GT_ROOT, split, *parts[1:], f"{stem2}_gtFine_labelIds.png")
        if os.path.isfile(png):
            return classes_from_trainid_png(png)
    elif dataset_key == "bdd10k10k":
        # BDD10K(10K) は Cityscapes 互換 trainId PNG
        rel = os.path.relpath(os.path.dirname(image_path), os.path.join(BDD10K10K_IMG_ROOT))
        parts = rel.split(os.sep)  # split/...
        split = parts[0] if parts else "train"
        png = os.path.join(BDD10K10K_GT_ROOT, split, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        if os.path.isfile(png):
            return classes_from_trainid_png(png)
    elif dataset_key == "gta5":
        # 00001.png → labels/00001.png
        png = os.path.join(GTA5_GT_ROOT, os.path.splitext(os.path.basename(image_path))[0] + ".png")
        if os.path.isfile(png):
            return classes_from_trainid_png(png)
    elif dataset_key == "bdd100k":
        # 互換ラベルがあれば参照 / なければ color 可視化 (後述)
        # ここでは color 可視化は未定義なのでスキップして /data 側に依存
        pass
    elif dataset_key == "nuimages_front":
        # /data/ucn_condmaps/nuimages_front/semseg/<rel_dir>/<stem>_semseg.jpg
        rel_dir = os.path.relpath(os.path.dirname(image_path), NUIMAGES_FRONT_ROOT)
        color = os.path.join(NUIMAGES_SEM_COLOR, rel_dir, os.path.splitext(os.path.basename(image_path))[0] + "_semseg.jpg")
        if os.path.isfile(color):
            rgb = imread_rgb(color)
            return classes_from_semseg_image(rgb)
    elif dataset_key.startswith("waymo"):
        # Waymo は *_predTrainId.npy
        # 例: /home/.../OneFormer_cityscapes/validation/front/<segid>/<timestamp>_first_predTrainId.npy
        # image: .../extracted/validation/front/<segid>/<timestamp>_first.jpg
        rel = os.path.relpath(image_path, os.path.join(WAYMO_IMG_ROOT))
        parts = rel.split(os.sep)
        # parts: [validation, front, <segid>, <file>.jpg]
        if len(parts) >= 4:
            split = parts[0]
            front = parts[1]
            segid = parts[2]
            fname = os.path.splitext(parts[3])[0]  # 1507..._first
            npy = os.path.join(WAYMO_SEM_NPY_ROOT, split, front, segid, f"{fname}_predTrainId.npy")
            return classes_from_waymo_npy(npy)
    # フォールバック: ない場合は空
    return []

# ==============================
# 6) サブグループ定義 Z（SynDiff‑AD 準拠＋拡張：雨/雪/霧 を追加）
# ==============================
WEATHERS = ["Clear","Cloudy","Rainy","Snowy","Foggy"]  # 拡張
TIMES    = ["Day","Dawn/Dusk","Night"]
Z_ALL = [(w,t) for w in WEATHERS for t in TIMES]  # 5×3=15

# Waymo 高品質プロンプト: 形容詞 / 装飾文（≥3）辞書
WAYMO_STYLE = {
    ("Clear","Day"): {
        "adj": "sunlit",
        "decor": [
            "shadows are crisp along building edges",
            "a clear blue sky gives high contrast",
            "asphalt shows dry, matte texture"
        ]
    },
    ("Clear","Night"): {
        "adj": "moonlit",
        "decor": [
            "streetlights and storefronts emit steady pools of light",
            "neon signage adds saturated accents",
            "long headlight trails reflect off glass surfaces"
        ]
    },
    ("Cloudy","Day"): {
        "adj": "overcast",
        "decor": [
            "soft, diffuse lighting with minimal shadows",
            "muted mid‑tones and compressed contrast",
            "low, uniform cloud cover across the sky"
        ]
    },
    ("Cloudy","Night"): {
        "adj": "hazy",
        "decor": [
            "light bloom around lamps reduces edge clarity",
            "damp air slightly lifts blacks",
            "distant buildings appear subdued"
        ]
    },
    ("Rainy","Day"): {
        "adj": "rain‑soaked",
        "decor": [
            "wet asphalt forms broad mirror‑like reflections",
            "raindrops accumulate along vehicle windows",
            "puddles interrupt lane markings"
        ]
    },
    ("Rainy","Night"): {
        "adj": "rain‑slicked",
        "decor": [
            "headlights and traffic signals glare with halation",
            "streetlight reflections streak across the road",
            "fine spray trails behind moving cars"
        ]
    },
    ("Snowy","Day"): {
        "adj": "snow‑covered",
        "decor": [
            "accumulated snow softens curb lines and rooftops",
            "tracks and slush mark the wheel paths",
            "overall palette shifts slightly cooler and desaturated"
        ]
    },
    ("Snowy","Night"): {
        "adj": "snow‑lit",
        "decor": [
            "orange sodium lights tint the falling snow",
            "ground illumination increases ambient brightness",
            "parked cars carry thin layers of powder"
        ]
    },
    ("Foggy","Day"): {
        "adj": "fog‑bound",
        "decor": [
            "visibility is reduced with distant silhouettes softened",
            "traffic lights appear muted at range",
            "color saturation drops across the scene"
        ]
    },
    ("Foggy","Night"): {
        "adj": "misty",
        "decor": [
            "lamps exhibit strong halation and glow",
            "headlights cut shallow cones into the fog",
            "dark areas lift toward gray due to scatter"
        ]
    },
    # Dawn/Dusk variants
    ("Clear","Dawn/Dusk"): {
        "adj": "golden‑hour",
        "decor": [
            "low‑angle light creates long, warm shadows",
            "sky gradients from orange to blue near the horizon",
            "specular edges appear on windshields"
        ]
    },
    ("Cloudy","Dawn/Dusk"): {
        "adj": "muted‑hour",
        "decor": [
            "cloud layers pick up pastel highlights",
            "soft rim light on signage edges",
            "gentle contrast maintains scene detail"
        ]
    },
    ("Rainy","Dawn/Dusk"): {
        "adj": "drizzling‑hour",
        "decor": [
            "fine droplets bead on reflective surfaces",
            "turn signals shimmer across wet lanes",
            "low sun glints in puddles"
        ]
    },
    ("Snowy","Dawn/Dusk"): {
        "adj": "frost‑tinged",
        "decor": [
            "snow takes on peach and violet tones",
            "breathy haze forms in colder pockets",
            "edges round off where snow accumulates"
        ]
    },
    ("Foggy","Dawn/Dusk"): {
        "adj": "low‑mist",
        "decor": [
            "ground fog hugs the roadway",
            "headlights push gentle light domes",
            "distant structures fade quickly with depth"
        ]
    },
}

# ==============================
# 7) CLIP によるサブグループ推定
# ==============================
class SubgroupCLIP:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # ViT‑SO400M‑14‑SigLIP‑384
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP-384',
            device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # テキストプロンプト（SynDiff‑AD 付録に倣い簡潔な属性文を二段で推定）
        self.weather_prompts = [f"An image taken during {w} weather." for w in WEATHERS]
        self.time_prompts    = [f"An image taken during {t} time."    for t in TIMES]
        with torch.no_grad():
            self.txt_weather = self.model.encode_text(self.tokenizer(self.weather_prompts).to(self.device))
            self.txt_time    = self.model.encode_text(self.tokenizer(self.time_prompts).to(self.device))
            self.txt_weather = self.txt_weather / self.txt_weather.norm(dim=-1, keepdim=True)
            self.txt_time    = self.txt_time    / self.txt_time.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> Tuple[str,str]:
        # PIL 経由で transform → batch=1
        im = Image.fromarray(rgb.astype(np.uint8))
        im_t = self.preprocess(im).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(im_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # 余裕があれば “cos sim → softmax” だが argmax で十分
        w_idx = torch.argmax(img_feat @ self.txt_weather.T, dim=1).item()
        t_idx = torch.argmax(img_feat @ self.txt_time.T, dim=1).item()
        return WEATHERS[w_idx], TIMES[t_idx]

# ==============================
# 8) Qwen VLM ラッパ
# ==============================
@dataclass
class VLMConfig:
    backend: str = "qwen3"  # "qwen3" | "qwen25" | "qwen3_fp8_vllm"
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct"
    device_map: str = "auto"
    dtype: str = "auto"
    attn_impl: Optional[str] = None  # "flash_attention_2" など
    disable_thinking: bool = True
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.8

class QwenVLM:
    def __init__(self, cfg: VLMConfig, device: str, logger: logging.Logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.is_vllm = (cfg.backend == "qwen3_fp8_vllm")
        if self.is_vllm:
            # vLLM で FP8 モデルを使う場合（Transformers 非対応のため）
            try:
                from vllm import LLM, SamplingParams
                self.vllm = LLM(
                    model="Qwen/Qwen3-VL-32B-Instruct-FP8",
                    trust_remote_code=True,
                    tensor_parallel_size=max(1, torch.cuda.device_count()),
                    gpu_memory_utilization=0.80,
                )
                self.sparams = SamplingParams(
                    temperature=0.0 if self.cfg.temperature == 0 else self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_new_tokens
                )
                self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct-FP8")
                self.logger.info("vLLM(Qwen3‑VL‑32B‑FP8) ready.")
            except Exception as e:
                self.logger.error("vLLM 初期化失敗: %s", repr(e))
                raise
        else:
            # Transformers 直ロード (Qwen3 既定 / 代替 Qwen2.5)
            if cfg.backend == "qwen3":
                if Qwen3Model is None:
                    raise RuntimeError("Qwen3VLForConditionalGeneration が見つかりません。transformers 開発版が必要です。")
                kw = dict(dtype=cfg.dtype, device_map=cfg.device_map)
                if cfg.attn_impl:
                    kw.update(attn_implementation=cfg.attn_impl)
                self.model = Qwen3Model.from_pretrained(cfg.model_name, **kw)
                self.processor = AutoProcessor.from_pretrained(cfg.model_name)
                self.logger.info("Qwen3‑VL‑32B loaded with device_map=%s, dtype=%s", cfg.device_map, cfg.dtype)
            elif cfg.backend == "qwen25":
                if Qwen25Model is None:
                    raise RuntimeError("Qwen2_5_VLForConditionalGeneration が見つかりません。")
                name = "Qwen/Qwen2.5-VL-32B-Instruct"
                kw = dict(torch_dtype=cfg.dtype if cfg.dtype != "auto" else "auto", device_map=cfg.device_map)
                if cfg.attn_impl:
                    kw.update(attn_implementation=cfg.attn_impl)
                self.model = Qwen25Model.from_pretrained(name, **kw)
                self.processor = AutoProcessor.from_pretrained(name)
                self.logger.info("Qwen2.5‑VL‑32B loaded with device_map=%s, dtype=%s", cfg.device_map, cfg.dtype)
            else:
                raise ValueError("backend must be qwen3 | qwen25 | qwen3_fp8_vllm")

    def _sanitize(self, s: str) -> str:
        # 思考/解析の痕跡を除去（保守的）
        if not isinstance(s, str):
            return ""
        # <think>…</think> / <analysis>…</analysis>
        s = re.sub(r"<\s*(think|analysis|chain-of-thought)[^>]*>.*?<\s*/\s*\1\s*>", "", s, flags=re.IGNORECASE|re.DOTALL)
        # ```thought ... ```
        s = re.sub(r"```(thought|thinking|analysis)[\s\S]*?```", "", s, flags=re.IGNORECASE)
        # “Thinking...” プレフィクス
        s = re.sub(r"^\s*(Thinking|Thoughts|Reasoning)\s*[:：\-]\s*", "", s, flags=re.IGNORECASE)
        # XML/JSON code fences 全般
        s = re.sub(r"```[\s\S]*?```", "", s)
        # 余分な空白
        return s.strip()

    def caption(self, image_path: str, object_names: List[str], forbid_keywords: List[str]) -> str:
        """
        SynDiff‑AD の CaG: オブジェクト列を含む問い合わせで “サブグループ語を含めない” 詳細 caption を得る。
        """
        # Qwen3 は apply_chat_template に messages の image+text を与える
        # 禁止語: weather/time の露出を避ける
        forbid_clause = "Do not mention weather keywords [{}] or time keywords [{}].".format(
            ", ".join(WEATHERS), ", ".join(TIMES)
        )
        obj_str = ", ".join(object_names) if object_names else "common road scene objects"
        user_text = (
            f"Provide a detailed, single-paragraph caption describing objects [{obj_str}] "
            f"and their spatial relations, background, camera viewpoint, and image quality. "
            f"{forbid_clause} Avoid prefaces; output only the caption."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text",  "text": user_text}
                ]
            }
        ]

        # vLLM 分岐
        if self.is_vllm:
            from qwen_vl_utils import process_vision_info
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size)
            inputs = [{"prompt": text, "multi_modal_data": {"image": image_inputs}}]
            outs = self.vllm.generate(inputs, sampling_params=self.sparams)
            raw = outs[0].outputs[0].text
            return self._sanitize(raw)

        # Transformers
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k,v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                          temperature=self.cfg.temperature, top_p=self.cfg.top_p)
        # Qwen3 “thinking” を抑制（モデル依存; 推論側はテキストのみ出させる）
        if hasattr(self.processor, "tokenizer") and self.cfg.disable_thinking:
            # 一部 Qwen3 系は chat template 引数に enable_thinking を持つが、processor API 統一のため
            pass

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # 入力部をトリム
        in_ids = inputs["input_ids"]
        trimmed = [o[len(i):] for i,o in zip(in_ids, out_ids)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cap = text[0] if isinstance(text, list) else str(text)
        cap = self._sanitize(cap)

        # 空文字→再試行(温度/Top‑p 強化)
        if len(cap) < 5:
            gen_kwargs2 = dict(max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.95)
            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, **gen_kwargs2)
            trimmed = [o[len(i):] for i,o in zip(in_ids, out_ids)]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            cap = self._sanitize(text[0] if isinstance(text, list) else str(text))
        return cap

# ==============================
# 9) プロンプト整形（train / eval）
# ==============================
def prompt_train_final(caption: str, z: Tuple[str,str]) -> str:
    # SynDiff‑AD 3.2に忠実: “caption + サブグループ行”
    weather, time = z
    style_line = f" Image taken in {weather} weather at {time.lower()} time."
    return caption.strip().rstrip(".") + "." + style_line

def prompt_eval_waymo_final(caption: str, z_src: Tuple[str,str], z_tgt: Tuple[str,str], classes: List[str]) -> str:
    """
    Waymo 評価: 高品質構文 + 装飾 (≥3) + カメラアングルの保持
    """
    weather_t, time_t = z_tgt
    weather_s, time_s = z_src
    style = WAYMO_STYLE.get(z_tgt, {"adj":"realistic","decor":["natural lighting","plausible reflections","consistent perspective"]})
    adj = style["adj"]
    decs = style["decor"]
    cls_txt = ", ".join(classes) if classes else "buildings, road, sidewalk, vehicles, traffic signs, and people"
    # 基本骨子
    base = f"A realistic {adj} city street scene with {cls_txt}. "
    # 装飾（3つ以上）
    deco = " ".join([s if s.endswith(".") else (s + ".") for s in decs[:max(3, len(decs))]])
    keep = f"Keep the same camera angle and composition as the original {time_s.lower()} image."
    # 仕上げ
    return (base + deco + " " + keep).strip()

def prompt_simple(classes: List[str], z: Tuple[str,str]) -> str:
    weather, time = z
    cls_txt = ", ".join(classes) if classes else "typical road scene elements"
    return f"A city street scene photo with {cls_txt} at {time.lower()} in {weather.lower()} weather."

# ==============================
# 10) I/O メタ出力
# ==============================
def csv_append(path: str, header: List[str], row: List[Any]) -> None:
    ensure_dir(os.path.dirname(path))
    write_header = (not os.path.exists(path))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

# ==============================
# 11) メイン処理
# ==============================
@dataclass
class Args:
    mode: str                      # "train" or "eval-waymo"
    datasets: List[str]            # ["all"] or keys
    out_root: str
    tb_dir: str
    verbose: bool
    limit: int
    splits_cityscapes: List[str]
    splits_bdd10k10k: List[str]
    splits_bdd100k: List[str]
    waymo_split: str               # "training"|"validation"|"testing"
    simple_mode: bool
    targets_per_image: int
    seed: int
    vlm_backend: str
    vlm_attn_impl: Optional[str]
    vlm_temperature: float
    vlm_top_p: float

def parse_args() -> Args:
    import argparse
    ap = argparse.ArgumentParser(description="SynDiff‑AD 準拠の VLM キャプション＋サブグループ付与（学習/評価プロンプト一括生成）")
    ap.add_argument("--mode", type=str, choices=["train","eval-waymo"], default="train")
    ap.add_argument("--datasets", type=str, nargs="+",
                    choices=["all","bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k","waymo"], default=["all"])
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--tb-dir", type=str, default=DEFAULT_TB_DIR)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)

    ap.add_argument("--splits-cityscapes", nargs="+", default=["train"])
    ap.add_argument("--splits-bdd10k10k", nargs="+", default=["train"])
    ap.add_argument("--splits-bdd100k", nargs="+", default=["train"])
    ap.add_argument("--waymo-split", type=str, choices=["training","validation","testing"], default="validation")

    ap.add_argument("--simple-mode", action="store_true", help="評価/学習とも簡素プロンプトを出す（既定OFF）")
    ap.add_argument("--targets-per-image", type=int, default=1, help="Waymo 評価で各画像に割り当てる z* の個数（既定 1）")

    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--vlm-backend", type=str, choices=["qwen3","qwen25","qwen3_fp8_vllm"], default="qwen3")
    ap.add_argument("--vlm-attn-impl", type=str, default=None, help="例: flash_attention_2")
    ap.add_argument("--vlm-temperature", type=float, default=0.2)
    ap.add_argument("--vlm-top-p", type=float, default=0.8)

    a = ap.parse_args()
    return Args(
        mode=a.mode, datasets=a.datasets, out_root=a.out_root, tb_dir=a.tb_dir,
        verbose=a.verbose, limit=a.limit,
        splits_cityscapes=a.splits_cityscapes, splits_bdd10k10k=a.splits_bdd10k10k, splits_bdd100k=a.splits_bdd100k,
        waymo_split=a.waymo_split, simple_mode=a.simple_mode, targets_per_image=a.targets_per_image,
        seed=a.seed, vlm_backend=a.vlm_backend, vlm_attn_impl=a.vlm_attn_impl,
        vlm_temperature=a.vlm_temperature, vlm_top_p=a.vlm_top_p
    )

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    logger = setup_logger(DEFAULT_LOG_DIR, args.verbose)
    logger.info("=== ucn_make_prompts: start ===")
    logger.info("mode=%s datasets=%s", args.mode, ",".join(args.datasets))
    logger.info("Torch=%s CUDA(build)=%s CUDA_available=%s", torch.__version__, getattr(torch.version,"cuda",None), torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("cuda device0: %s", torch.cuda.get_device_name(0))
        if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
            logger.warning("CUDAビルド表示が12.8以外 (%s) → 既存環境に合わせて続行します。", getattr(torch.version,"cuda",None))

    sw = SummaryWriter(args.tb_dir)
    sw.add_text("run/mode", args.mode, 0)

    # CLIP サブグループ推定器
    clip_sg = SubgroupCLIP(device="cuda" if torch.cuda.is_available() else "cpu")

    # VLM
    vlm_cfg = VLMConfig(
        backend=args.vlm_backend,
        device_map="auto",
        dtype="auto",
        attn_impl=args.vlm_attn_impl,
        disable_thinking=True,
        max_new_tokens=384,
        temperature=args.vlm_temperature,
        top_p=args.vlm_top_p
    )
    vlm = QwenVLM(vlm_cfg, device="cuda" if torch.cuda.is_available() else "cpu", logger=logger)

    # データセット収集
    target_keys = ["bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k"] if args.mode=="train" else ["waymo"]
    if "all" not in args.datasets:
        target_keys = [k for k in target_keys if k in args.datasets]

    datasets: List[Tuple[str, List[str], str, Optional[str]]] = []
    total_imgs = 0

    if args.mode == "train":
        # BDD10K(10K)
        if "bdd10k10k" in target_keys:
            imgs = []
            for sp in args.splits_bdd10k10k:
                cur = list_bdd10k10k(sp)
                if args.limit>0: cur = cur[:args.limit]
                imgs += cur
            datasets.append(("bdd10k10k", imgs, BDD10K10K_IMG_ROOT, None)); total_imgs += len(imgs)

        # Cityscapes
        if "cityscapes" in target_keys:
            imgs = []
            for sp in args.splits_cityscapes:
                cur = list_cityscapes(sp)
                if args.limit>0: cur = cur[:args.limit]
                imgs += cur
            datasets.append(("cityscapes", imgs, CITYSCAPES_IMG_ROOT, None)); total_imgs += len(imgs)

        # GTA5
        if "gta5" in target_keys:
            imgs = list_gta5()
            if args.limit>0: imgs = imgs[:args.limit]
            datasets.append(("gta5", imgs, GTA5_IMG_ROOT, None)); total_imgs += len(imgs)

        # nuImages
        if "nuimages_front" in target_keys:
            imgs = list_nuimages()
            if args.limit>0: imgs = imgs[:args.limit]
            datasets.append(("nuimages_front", imgs, NUIMAGES_FRONT_ROOT, None)); total_imgs += len(imgs)

        # BDD100K
        if "bdd100k" in target_keys:
            imgs = []
            for sp in args.splits_bdd100k:
                cur = list_bdd100k(sp)
                if args.limit>0: cur = cur[:args.limit]
                imgs += cur
            datasets.append(("bdd100k", imgs, BDD100K_IMG_ROOT, None)); total_imgs += len(imgs)

    else:  # eval-waymo
        imgs = list_waymo(args.waymo_split)
        if args.limit>0: imgs = imgs[:args.limit]
        datasets.append((f"waymo_{args.waymo_split}", imgs, os.path.join(WAYMO_IMG_ROOT, args.waymo_split, "front"), args.waymo_split))
        total_imgs = len(imgs)

    logger.info("対象総枚数: %d", total_imgs)
    sw.add_scalar("meta/total_images", total_imgs, 0)
    if total_imgs == 0:
        logger.warning("画像が見つかりません。パスと split を確認してください。")
        sw.close(); return

    processed = 0
    t0 = time.time()

    for ds_key, img_list, base_rel, extra in datasets:
        logger.info("=== [%s] %d images ===", ds_key, len(img_list))
        # 出力パス類
        raw_root     = os.path.join(args.out_root, "raw_captions", ds_key)
        train_root   = os.path.join(args.out_root, "prompts_train", ds_key)
        eval_root    = os.path.join(args.out_root, "prompts_eval_waymo", ds_key)
        meta_root    = os.path.join(args.out_root, "meta", ds_key)
        src_csv      = os.path.join(meta_root, "subgroups_source.csv")
        tgt_csv      = os.path.join(meta_root, "subgroups_target.csv")

        # CSV ヘッダ
        if args.mode == "train":
            csv_append(src_csv, ["image","weather","time"], [])  # header のみ
        else:
            csv_append(src_csv, ["image","weather_src","time_src"], [])
            csv_append(tgt_csv, ["image","weather_tgt","time_tgt"], [])

        for p in img_list:
            try:
                rgb = imread_rgb(p)
            except Exception as e:
                logger.exception("[READ_FAIL] %s", p)
                continue

            # クラス名列（Cityscapes 互換）
            try:
                classes = try_semseg_classes_for_dataset("waymo" if ds_key.startswith("waymo") else ds_key, p, base_rel)
            except Exception:
                classes = []
            if not classes:
                # 何も取れないケースでも最低限の既定語を入れておく（ダミーではなく既存最頻語）
                classes = ["road","sidewalk","building","traffic sign","vegetation","sky","car","person"]

            # サブグループ z（CLIP）
            try:
                weather_src, time_src = clip_sg.predict(rgb)
            except Exception as e:
                logger.exception("[CLIP_FAIL] %s", p)
                weather_src, time_src = "Clear","Day"

            # VLM キャプション（禁止語: weather/time）
            try:
                cap = vlm.caption(p, classes, forbid_keywords=WEATHERS+TIMES)
            except Exception as e:
                logger.exception("[VLM_FAIL] %s", p)
                cap = f"A detailed city street scene with {', '.join(classes)}."

            # キャッシュ: 生キャプション
            rel_dir = os.path.relpath(os.path.dirname(p), base_rel)
            stem = os.path.splitext(os.path.basename(p))[0]
            raw_path = os.path.join(raw_root, rel_dir, f"{stem}.txt")
            write_text(raw_path, cap)

            if args.mode == "train":
                # z を付与（SynDiff‑AD 3.2）
                z = (weather_src, time_src)
                if args.simple_mode:
                    final = prompt_simple(classes, z)
                else:
                    final = prompt_train_final(cap, z)
                outp = os.path.join(train_root, rel_dir, f"{stem}.txt")
                write_text(outp, final)
                csv_append(src_csv, ["image","weather","time"], [p, weather_src, time_src])

            else:
                # z* を一様サンプル（Z\{z} から）
                z_src = (weather_src, time_src)
                cand = [z for z in Z_ALL if z != z_src]
                random.shuffle(cand)
                for k in range(max(1, args.targets_per_image)):
                    z_tgt = cand[k % len(cand)]
                    if args.simple_mode:
                        final = prompt_simple(classes, z_tgt)
                    else:
                        final = prompt_eval_waymo_final(cap, z_src, z_tgt, classes)
                    outp = os.path.join(eval_root, rel_dir, f"{stem}__{z_tgt[0]}_{z_tgt[1].replace('/','-')}.txt")
                    write_text(outp, final)
                    # CSV
                    if k == 0:
                        csv_append(src_csv, ["image","weather_src","time_src"], [p, weather_src, time_src])
                    csv_append(tgt_csv, ["image","weather_tgt","time_tgt"], [p, z_tgt[0], z_tgt[1]])

            processed += 1
            if processed % 25 == 0:
                sw.add_scalar("progress/processed", processed, processed)

        logger.info("=== [%s] done ===", ds_key)

    dt = time.time() - t0
    ips = processed / max(1.0, dt)
    logger.info("✅ 完了: processed=%d, time=%.1fs (%.2f img/s)", processed, dt, ips)
    sw.add_scalar("summary/total_images", processed, 0)
    sw.add_scalar("summary/img_per_s", ips, 0)
    sw.close()

if __name__ == "__main__":
    main()
