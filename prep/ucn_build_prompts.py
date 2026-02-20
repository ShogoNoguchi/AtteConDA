#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/home/shogo/coding/prep/ucn_build_prompts.py

SynDiff-AD 準拠の学習・評価用テキストプロンプト一括生成スクリプト
 - 学習: BDD10K(10K)/Cityscapes/GTA5/nuImages(front)/BDD100K
 - 評価: Waymo (front, first/mid10s/last)
 - VLM: Qwen3-VL 32B (Transformers; AutoModelForImageTextToText)
 - CLIP: open_clip SigLIP (ViT-SO400M-14, 384)
 - RTX5090 + CUDA 12.8 前提（ucn-eval Docker イメージ上で実行）
 - “Thinking...”対策: enable_thinking=False + 出力フィルタ(hidethinking)
 - 生成物: /data/syndiff_prompts/ 配下

 変更点の要旨

sanitize_caption のバグ修正は維持

新規 CLI：--cap-max-new（既定 96）, --cap-words（既定 45）, --resume（既定 ON）, --no-resume

Qwen 生成を 短文化（max_new_tokens 既定 96、冗長抑制※生成時間短縮のための措置）。

既存 JSONL があれば スキップ（レジューム）
TODO：既存prompt・CLIPキャッシュ等があればスキップできるようにもしたい。このスクリプトは実行時に最大2週間くらいかかるので、なるべくなんでもキャッシュを使うようにする（--resumeがデフォルト）。
"""

"""以下実行コマンド

docker run --rm --gpus all \
  --entrypoint /app/entrypoint_prompts_v2.sh \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_prompts.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e HF_HOME=/root/.cache/huggingface \
  -e OPENCLIP_CACHE_DIR=/root/.cache/huggingface/hub \
  -e VLM_LOCAL_DIR=/data/hf_models/Qwen/Qwen3-VL-32B-Instruct \
  -e QWEN_QUANT=4bit \
  -e TRANSFORMERS_OFFLINE=0 -e HF_HUB_OFFLINE=0 \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint_prompts_v2.sh:/app/entrypoint_prompts_v2.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_prompts.py:/app/ucn_build_prompts.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --jobs train eval_waymo \
  --datasets all \
  --cityscapes-use-splits train \
  --bdd10k10k-use-splits train \
  --bdd100k-use-splits train \
  --vlm-local-dir /data/hf_models/Qwen/Qwen3-VL-32B-Instruct \
  --openclip-cache-dir /root/.cache/huggingface/hub \
  --quant 4bit \
  --clip-batch 8 \
  --cap-max-new 96 \
  --cap-words 45 \
  --hidethinking \
  --verbose
"""
# ここから追加: 分析・可視化用の追加ライブラリ
import re
import pandas as pd
import matplotlib.pyplot as plt
# ここまで追加

import os
import sys
import argparse
import logging
from logging import handlers
from typing import Optional
from pathlib import Path
import json
import csv
import random
import time
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch

import open_clip
import torchvision.transforms as T

from transformers import AutoProcessor, AutoModelForImageTextToText

# BitsAndBytesConfig は無い環境でも動くように
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

# ==============================
# 0) 既定パス（翔伍さん環境に整合）
# ==============================
BDD10K10K_IMG_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images"  # {train,val}
CITYSCAPES_IMG_ROOT = "/home/shogo/coding/datasets/cityscapes/leftImg8bit"                       # {train,val,test}
GTA5_IMG_ROOT       = "/home/shogo/coding/datasets/GTA5/images/images"
NUIMAGES_FRONT_ROOT = "/home/shogo/coding/datasets/nuimages/samples/CAM_FRONT"
BDD100K_IMG_ROOT    = "/home/shogo/coding/datasets/BDD_100K_pure100k"

BDD10K10K_GT_LABEL_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/labels/sem_seg/masks"
CITYSCAPES_GT_ROOT      = "/home/shogo/coding/datasets/cityscapes/gtFine"
GTA5_GT_LABEL_ROOT      = "/home/shogo/coding/datasets/GTA5/labels/labels"

WAYMO_RGB_ROOT    = "/home/shogo/coding/datasets/WaymoV2/extracted"
WAYMO_SEMSEG_NPY  = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"

UCN_CONDMAPS_ROOT = "/data/ucn_condmaps"

OUT_ROOT = "/data/syndiff_prompts"
RAW_CAP_ROOT       = os.path.join(OUT_ROOT, "raw_captions")
PROMPTS_TRAIN_ROOT = os.path.join(OUT_ROOT, "prompts_train")
PROMPTS_EVAL_ROOT  = os.path.join(OUT_ROOT, "prompts_eval_waymo")
SUBGROUP_MAP_ROOT  = os.path.join(OUT_ROOT, "subgroup_maps")
LOG_ROOT           = os.path.join(OUT_ROOT, "logs")
TB_ROOT            = os.path.join(OUT_ROOT, "tb")

# ==============================
# 1) ロガー
# ==============================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def setup_logger(name: str, log_dir: str, verbose: bool) -> logging.Logger:
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"{name}.log")
    logger = logging.getLogger(name)
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

def log_env(logger: logging.Logger) -> None:
    logger.info("torch=%s | torch.cuda(build)=%s | cuda_available=%s",
        torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
    if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
        logger.warning("CUDAビルド表示が12.8以外です（%s）。既存環境に合わせて続行します。", getattr(torch.version, "cuda", None))

# ==============================
# 2) Cityscapes
# ==============================
CITYSCAPES_TRAINID_TO_NAME = [
    "road","sidewalk","building","wall","fence",
    "pole","traffic light","traffic sign","vegetation","terrain",
    "sky","person","rider","car","truck",
    "bus","train","motorcycle","bicycle"
]

CITYSCAPES_TRAINID_COLORS_RGB = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
    (0,60,100),(0,80,100),(0,0,230),(119,11,32),
]

def stem_without_cityscapes_suffix(filename: str) -> str:
    s = os.path.splitext(os.path.basename(filename))[0]
    if s.endswith("_leftImg8bit"): s = s[:-len("_leftImg8bit")]
    return s

def unique_trainids_from_label_png(png_path: str) -> list:
    m = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if m is None: raise FileNotFoundError(png_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    u = np.unique(m.astype(np.int32))
    return [int(x) for x in u if 0 <= int(x) <= 18]

def unique_trainids_from_color_semseg_jpg(jpg_path: str) -> list:
    im = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(jpg_path)
    h, w = im.shape[:2]
    scale = max(1, int(max(h,w)//512))
    if scale > 1:
        im = cv2.resize(im, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.int16)
    palette = np.array(CITYSCAPES_TRAINID_COLORS_RGB, dtype=np.int16)
    rgb2 = rgb.reshape(-1,3)[:,None,:]
    pal2 = palette[None,:,:]
    diff = np.sum((rgb2 - pal2)**2, axis=2)
    idx  = np.argmin(diff, axis=1)
    ids  = np.unique(idx)
    return [int(x) for x in ids if 0 <= int(x) <= 18]

def classnames_from_ids(ids: list) -> list:
    return [CITYSCAPES_TRAINID_TO_NAME[i] for i in sorted(set(ids)) if 0 <= i <= 18]

# ==============================
# 3) データ列挙
# ==============================
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def list_images_under(root: str, allowed: set = ALLOWED_IMG_EXT) -> list:
    out=[]
    if not os.path.isdir(root): return out
    for r,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in allowed:
                out.append(os.path.join(r,f))
    out.sort(); return out

def rel_from(path: str, base: str) -> str:
    rp = os.path.relpath(os.path.dirname(path), base)
    return "" if rp == "." else rp

# ==============================
# 4) サブグループ / CLIP
# ==============================
# ==============================
# 4) サブグループ / CLIP
# ==============================
WEATHER = ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy"]
TIME = ["Day", "Twilight", "Night"]

# ここから追加: 文書型プロンプト（sentence モード用）
# WEATHER/TIME の順番は既存のラベルと完全に対応させる
WEATHER_SENTENCE_PROMPTS = [
    "a clear sunny daytime scene",        # Clear
    "a cloudy overcast scene",            # Cloudy
    "a rainy wet road scene",             # Rainy
    "a snowy winter scene",               # Snowy
    "a foggy low-visibility scene",       # Foggy
]

TIME_SENTENCE_PROMPTS = [
    "a daytime driving scene",            # Day
    "a twilight early evening scene",     # Twilight
    "a nighttime dark driving scene",     # Night
]
# ここまで追加

def build_test_prompts(prompt_mode: str = "label") -> list:
    """
    CLIP の text encoder に与えるプロンプト群を構築する。

    prompt_mode:
        - "label"    : 既存どおりの単語ラベル ("Clear", "Cloudy", ..., "Day", "Twilight", "Night")
        - "sentence" : 上の WEATHER_SENTENCE_PROMPTS / TIME_SENTENCE_PROMPTS を使用

    戻り値:
        [weather_prompts, time_prompts] という2要素 list で、
        weather_prompts は WEATHER と同じ順番で 5 個、
        time_prompts    は TIME    と同じ順番で 3 個。
    """
    if prompt_mode == "sentence":
        return [WEATHER_SENTENCE_PROMPTS, TIME_SENTENCE_PROMPTS]

    # 既存仕様: 単語ラベルをそのまま使う
    return [[f"{w}" for w in WEATHER], [t for t in TIME]]


class ClipSubgroupClassifier:
    def __init__(
        self,
        batch_size: int = 8,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        prompt_mode: str = "label",
    ):
        """
        prompt_mode:
            "label"    : WEATHER/TIME の生ラベルを text prompt に使う（既存仕様）
            "sentence" : 文書型の text prompt を使う（分析用）
        """
        self.logger = logger or logging.getLogger("clip")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or os.environ.get("OPENCLIP_CACHE_DIR")
        self.prompt_mode = prompt_mode

        # open_clip v3.2.0 の SigLIP は 'webli' を使用
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384",
            pretrained="webli",
            cache_dir=self.cache_dir,
        )
        self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        self.batch_size = max(1, int(batch_size))

        # WEATHER/TIME に対応した text prompt 群を構築
        self.prompts = build_test_prompts(prompt_mode=self.prompt_mode)
        # self.prompts = [weather_prompts (len=5), time_prompts (len=3)]

        # それぞれをまとめてエンコードして保持（既存仕様を維持）
        self.text_inputs = [self.tokenizer(descs).to(self.device) for descs in self.prompts]
        with torch.no_grad():
            self.text_feats = [self.model.encode_text(toks) for toks in self.text_inputs]

    def classify_batch(self, images: list) -> list:
        pil_images = [Image.fromarray(im) for im in images]
        ims = torch.stack([self.transform(p) for p in pil_images]).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(ims)
        results = []
        for j in range(ims.shape[0]):
            w_best = None
            t_best = None
            for axis, (descs, txtf) in enumerate(zip(self.prompts, self.text_feats)):
                sim = (img_feat[[j]] @ txtf.T).softmax(dim=1)
                idx = int(torch.argmax(sim, dim=1).item())
                if axis == 0:
                    w_best = WEATHER[idx]
                else:
                    t_best = TIME[idx]
            results.append((w_best, t_best))
        return results

    def classify_one(self, im: np.ndarray) -> tuple:
        return self.classify_batch([im])[0]


# ==============================
# 5) Qwen3-VL キャプショナ（量子化・短文化対応）
# ==============================
def sanitize_caption(txt: str) -> str:
    if txt is None:
        return ""
    t = txt.strip()
    # 行頭プレフィックス除去
    bad_prefixes = ("Thinking", "THINKING", "Reasoning", "Chain-of-thought", "<think>", "</think>")
    filtered_lines = []
    for ln in t.splitlines():
        sl = ln.strip()
        if any(sl.startswith(bp) for bp in bad_prefixes):
            continue
        filtered_lines.append(ln)
    t = "\n".join(filtered_lines)
    # <think> ... </think> ブロック除去
    while True:
        s = t.find("<think>")
        if s < 0:
            break
        e = t.find("</think>", s)
        if e < 0:
            break
        t = t[:s] + t[e+8:]
    # 余分な空白整形
    return " ".join(t.split())

def _parse_torch_dtype(s: Optional[str]):
    if not s:
        return None
    ss = str(s).lower()
    if ss in ("auto",):
        return None
    if ss in ("fp16","float16"):
        return torch.float16
    if ss in ("bf16","bfloat16"):
        return torch.bfloat16
    return None

class QwenVLCaptioner:
    def __init__(self,
                 model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
                 torch_dtype: str = "auto",
                 device_map: str = "auto",
                 attn_impl: Optional[str] = None,
                 hidethinking: bool = True,
                 local_dir: Optional[str] = None,
                 quant: str = "none",    # "none" | "8bit" | "4bit"
                 # 追加：短文化のための生成パラメータ
                 cap_max_new: int = 96,
                 cap_min_new: int = 0,
                 cap_words: int = 45,
                 repetition_penalty: float = 1.1,
                 no_repeat_ngram_size: int = 3,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("qwen")
        kw = {}
        if attn_impl:
            kw["attn_implementation"] = attn_impl

        load_id = local_dir if (local_dir and os.path.isdir(local_dir)) else model_id
        self.logger.info(f"Qwen load from: {load_id}")

        # ---- 量子化（env > 引数 > none） ----
        quant_mode = os.environ.get("VLM_QUANT", "").strip().lower() or (quant or "none")
        quant_cfg = None
        if quant_mode in ("4bit","8bit") and BitsAndBytesConfig is not None:
            try:
                import bitsandbytes as _bnb  # noqa: F401
                if quant_mode == "4bit":
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                else:
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.logger.info("BitsAndBytes 量子化を使用: %s", quant_mode)
            except Exception as e:
                self.logger.warning("bitsandbytes 読込失敗: %s → 非量子化で続行", repr(e))
                quant_cfg = None
        elif quant_mode in ("4bit","8bit") and BitsAndBytesConfig is None:
            self.logger.warning("BitsAndBytesConfig が見つからないため量子化を無効化します。")
            quant_cfg = None

        torch_dtype_parsed = None if quant_cfg is not None else _parse_torch_dtype(torch_dtype)

        # from_pretrained：quantization_config は 1 回だけ渡す。dtype は torch_dtype を使用
        self.model = AutoModelForImageTextToText.from_pretrained(
            load_id,
            device_map=device_map,
            quantization_config=quant_cfg,
            torch_dtype=torch_dtype_parsed,
            **kw
        )
        self.processor = AutoProcessor.from_pretrained(load_id)
        self.hidethinking = hidethinking
        # 生成パラメータ保持
        self.cap_max_new = int(cap_max_new)
        self.cap_min_new = int(max(0, cap_min_new))
        self.cap_words = int(cap_words)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)

    def caption(self, rgb: np.ndarray, object_names: list, seed: Optional[int]=None) -> str:
        if seed is not None:
            torch.manual_seed(int(seed)); random.seed(int(seed)); np.random.seed(int(seed))
        prompt = (
            "Provide a concise but semantically dense description of this driving scene. "
            f"Focus on the listed objects and their spatial relations: {object_names}. "
            "Describe the background (urban/rural), camera viewpoint, and image quality (lighting, reflections, motion blur). "
            f"Limit to at most {self.cap_words} words, 1–2 sentences. "
            "Do NOT mention weather/time-of-day keywords such as rain, snow, fog, clear/sunny, day, night, dawn, twilight. "
            "Return a single paragraph."
        )
        image = Image.fromarray(rgb)
        messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cap_max_new,
                min_new_tokens=self.cap_min_new,
                do_sample=False,  # 決定的に
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                use_cache=True
            )
        trimmed = [o[len(i):] for i,o in zip(inputs.input_ids, out_ids)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self.hidethinking:
            text = sanitize_caption(text)
        return text

# ==============================
# 6) Waymo パス
# ==============================
def waymo_paths_to_npy(rgb_path: str) -> str:
    parts = Path(rgb_path).parts
    idx = parts.index("extracted")
    split = parts[idx+1]
    segid = parts[idx+3]
    fname = os.path.splitext(parts[-1])[0]
    return os.path.join(WAYMO_SEMSEG_NPY, split, "front", segid, f"{fname}_predTrainId.npy")

# ==============================
# 7) スタイル辞書
# ==============================
DEFAULT_STYLE_LIB = {
    ("Clear","Day"): {"adj":"sunlit","decor":[
        "crisp visibility with balanced contrast",
        "soft mid‑day shadows on buildings and cars",
        "clean reflections on windshields without glare"]},
    ("Clear","Twilight"): {"adj":"golden‑hour","decor":[
        "warm side‑lighting with long soft shadows",
        "slight orange‑pink sky gradient near the horizon",
        "gentle specular highlights on metallic surfaces"]},
    ("Clear","Night"): {"adj":"nocturnal","decor":[
        "lit by streetlights and storefronts",
        "specular highlights on asphalt from artificial lighting",
        "headlight beams forming soft cones in the distance"]},
    ("Cloudy","Day"): {"adj":"overcast","decor":[
        "diffuse soft lighting with minimal shadows",
        "muted color palette with gentle contrast",
        "uniform skylight giving even exposure across the scene"]},
    ("Cloudy","Twilight"): {"adj":"moody","decor":[
        "low‑contrast ambient light and cool tones",
        "soft horizon glow under a dense cloud layer",
        "subtle reflections on windows and car roofs"]},
    ("Cloudy","Night"): {"adj":"dimly‑lit","decor":[
        "soft pools of light around lamp posts",
        "hazy halos around traffic lights",
        "shadows are weak and edges appear subdued"]},
    ("Rainy","Day"): {"adj":"rain‑soaked","decor":[
        "wet asphalt with mirror‑like reflections",
        "raindrops streaking on car windows",
        "softened edges due to light drizzle"]},
    ("Rainy","Twilight"): {"adj":"rain‑drenched","decor":[
        "cool blue hour reflections along lane markings",
        "headlights elongated into thin streaks on wet surfaces",
        "mist near the ground around moving vehicles"]},
    ("Rainy","Night"): {"adj":"rainy nighttime","decor":[
        "streetlights and headlights reflected as vivid bokeh",
        "glossy road surface with colorful neon reflections",
        "raindrops producing subtle motion blur"]},
    ("Snowy","Day"): {"adj":"snow‑covered","decor":[
        "bright snowpack with slight blue shadows",
        "soft footprints and tire tracks along the roadside",
        "reduced saturation due to snow glare"]},
    ("Snowy","Twilight"): {"adj":"wintry","decor":[
        "blue‑tinted ambient light over snow",
        "soft sky glow contrasting with dark tree lines",
        "breath‑like mist around pedestrians"]},
    ("Snowy","Night"): {"adj":"snowy nighttime","decor":[
        "granular snowflakes lit by lamp posts",
        "sparkling reflections on icy patches",
        "soft muffled ambience with reduced distant visibility"]},
    ("Foggy","Day"): {"adj":"foggy","decor":[
        "reduced contrast and depth cues",
        "buildings fading into low‑lying haze",
        "desaturated palette with soft silhouettes"]},
    ("Foggy","Twilight"): {"adj":"misty","decor":[
        "dim blue ambience and shallow visibility",
        "soft halos around signal lights",
        "vanishing lane markers beyond short distance"]},
    ("Foggy","Night"): {"adj":"hazy nighttime","decor":[
        "headlight beams diffused into cones",
        "sodium lamp halos bleeding into the fog",
        "distant objects reduced to silhouettes"]},
}

# ==============================
# 8) プロンプト合成
# ==============================
def make_train_prompt(caption: str, w: str, t: str) -> str:
    return f"{caption.strip()} Image taken in {w} weather at {t}."

def make_waymo_prompt_adv(caption: str, class_names: list, target_w: str, target_t: str,
                          style_lib: dict) -> str:
    key = (target_w, target_t)
    meta = style_lib.get(key, {"adj":"realistic","decor":["balanced exposure","natural color palette","faithful tonality"]})
    adj = meta.get("adj","realistic")
    decors = meta.get("decor",[])
    while len(decors) < 3:
        decors.append(decors[-1] if decors else "natural appearance with stable tonality")
    decor_text = " ".join(decors[:3])
    cls = ", ".join(class_names) if class_names else "typical urban street elements"
    return (
        f"A realistic {adj} city street scene with {cls}. {caption.strip()} "
        f"{decor_text}. Keep the same camera angle and composition as the original image."
    )

def make_waymo_prompt_simple(class_names: list, target_w: str, target_t: str) -> str:
    cls = ", ".join(class_names) if class_names else "street elements"
    return (
        f"A city street scene photo with {cls} at {target_w} {target_t}. "
        "Keep the same camera angle and composition as the original image."
    )

# ==============================
# 9) I/O
# ==============================
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_jsonl(path: str, rows: list) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_csv(path: str, header: list, rows: list) -> None:
    ensure_dir(os.path.dirname(path))
    newfile = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if newfile: w.writerow(header)
        for r in rows: w.writerow(r)
# ここから追加: 分析用の共通ヘルパー
def classify_paths_with_clip(
    paths: list,
    clip: ClipSubgroupClassifier,
    batch_size: int,
    logger: logging.Logger,
    tag: str,
) -> list:
    """
    画像パスの list を受け取り、CLIP による (weather, time) 推定結果を返す。

    戻り値:
        [(image_path, weather, time), ...]
    """
    results = []
    n = len(paths)
    for i in range(0, n, batch_size):
        chunk = paths[i : i + batch_size]
        ims = []
        good_paths = []
        for p in chunk:
            try:
                rgb = imread_rgb(p)
                ims.append(rgb)
                good_paths.append(p)
            except Exception:
                logger.exception("[ANALYZE][%s] imread failed: %s", tag, p)
        if not ims:
            continue
        wt_list = clip.classify_batch(ims)
        for p, (w, t) in zip(good_paths, wt_list):
            results.append((p, w, t))
    logger.info(
        "[ANALYZE][%s] CLIP classified %d / %d images",
        tag,
        len(results),
        len(paths),
    )
    return results


def build_clip_for_analysis(
    args: argparse.Namespace,
    logger: logging.Logger,
    prompt_mode: str,
    tag: str,
) -> ClipSubgroupClassifier:
    """
    分析 job 用に CLIP SigLIP モデルを構築するヘルパー。
    prompt_mode="label" / "sentence" を指定可能。
    """
    logger.info(
        "[ANALYZE][%s] build ClipSubgroupClassifier (prompt_mode=%s)",
        tag,
        prompt_mode,
    )
    clip = ClipSubgroupClassifier(
        batch_size=args.clip_batch,
        cache_dir=args.openclip_cache_dir,
        logger=logger,
        prompt_mode=prompt_mode,
    )
    return clip


def build_image_montage(
    image_paths: list,
    logger: logging.Logger,
    tag: str,
    num_rows: int = 2,
    num_cols: int = 3,
    tile_width: int = 512,
    tile_height: int = 288,
) -> Optional[np.ndarray]:
    """
    最大 num_rows*num_cols (=6) 枚の画像を読み込み、
    2x3 のタイル状に並べた BGR 画像を返す（cv2 用）。

    画像読み込みに失敗した場合は、その画像はスキップし、
    1枚も読み込めなければ None を返す。
    """
    if not image_paths:
        return None

    paths = image_paths[: num_rows * num_cols]
    tiles = []
    for p in paths:
        try:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(p)
            tile = cv2.resize(bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            tiles.append(tile)
        except Exception:
            logger.exception("[ANALYZE][%s] build_image_montage: load failed: %s", tag, p)

    if not tiles:
        return None

    # 足りない分は真っ黒なタイルで埋める
    num_expected = num_rows * num_cols
    if len(tiles) < num_expected:
        blank = np.zeros_like(tiles[0])
        while len(tiles) < num_expected:
            tiles.append(blank.copy())

    rows = []
    for r in range(num_rows):
        row_tiles = tiles[r * num_cols : (r + 1) * num_cols]
        row = cv2.hconcat(row_tiles)
        rows.append(row)
    montage = cv2.vconcat(rows)
    return montage


def get_clip_text_prompts_for_mode(prompt_mode: str) -> tuple:
    """
    分析用: prompt_mode に応じて CLIP text encoder に渡す文言を返す。

    戻り値:
        (weather_texts, time_texts)
        weather_texts: len=5
        time_texts   : len=3
    """
    if prompt_mode == "sentence":
        return WEATHER_SENTENCE_PROMPTS, TIME_SENTENCE_PROMPTS
    # "label" その他 → 既存のラベル文字列
    return WEATHER, TIME
# ここまで追加



# ==============================
# 10) クラス名抽出など
# ==============================


def list_images_under(root: str, allowed: set = ALLOWED_IMG_EXT) -> list:
    out = []
    if not os.path.isdir(root):
        return out
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in allowed:
                out.append(os.path.join(r, f))
    out.sort()
    return out


def list_dataset_images(ds_key: str, use_splits: Optional[list] = None, limit: int = -1) -> tuple:
    if ds_key == "bdd10k10k":
        base = BDD10K10K_IMG_ROOT
        splits = use_splits or ["train"]
        imgs = []
        for sp in splits:
            imgs.extend(list_images_under(os.path.join(base, sp)))
        if limit > 0:
            imgs = imgs[:limit]
            return imgs, base
        return imgs, base

    if ds_key == "cityscapes":
        base = CITYSCAPES_IMG_ROOT
        splits = use_splits or ["train"]
        imgs = []
        for sp in splits:
            imgs.extend(list_images_under(os.path.join(base, sp)))
        if limit > 0:
            imgs = imgs[:limit]
            return imgs, base
        return imgs, base

    if ds_key == "gta5":
        base = GTA5_IMG_ROOT
        imgs = list_images_under(base)
        if limit > 0:
            imgs = imgs[:limit]
            return imgs, base
        return imgs, base

    if ds_key == "nuimages_front":
        base = NUIMAGES_FRONT_ROOT
        imgs = list_images_under(base)
        if limit > 0:
            imgs = imgs[:limit]
            return imgs, base
        return imgs, base

    if ds_key == "bdd100k":
        base = BDD100K_IMG_ROOT
        splits = use_splits or ["train"]
        imgs = []
        for sp in splits:
            imgs.extend(list_images_under(os.path.join(base, sp)))
        if limit > 0:
            imgs = imgs[:limit]
            return imgs, base
        return imgs, base

    raise ValueError(f"unknown ds_key={ds_key}")


def rel_from(path: str, base: str) -> str:
    rp = os.path.relpath(os.path.dirname(path), base)
    return "" if rp == "." else rp


def stem_without_cityscapes_suffix(filename: str) -> str:
    s = os.path.splitext(os.path.basename(filename))[0]
    if s.endswith("_leftImg8bit"):
        s = s[:-len("_leftImg8bit")]
    return s


CITYSCAPES_TRAINID_TO_NAME = CITYSCAPES_TRAINID_TO_NAME  # lints
CITYSCAPES_TRAINID_COLORS_RGB = CITYSCAPES_TRAINID_COLORS_RGB  # lints


def unique_trainids_from_label_png(png_path: str) -> list:
    m = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(png_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    u = np.unique(m.astype(np.int32))
    return [int(x) for x in u if 0 <= int(x) <= 18]


def unique_trainids_from_color_semseg_jpg(jpg_path: str) -> list:
    im = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(jpg_path)
    h, w = im.shape[:2]
    scale = max(1, int(max(h, w) // 512))
    if scale > 1:
        im = cv2.resize(im, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.int16)
    palette = np.array(CITYSCAPES_TRAINID_COLORS_RGB, dtype=np.int16)
    rgb2 = rgb.reshape(-1, 3)[:, None, :]
    pal2 = palette[None, :, :]
    diff = np.sum((rgb2 - pal2) ** 2, axis=2)
    idx = np.argmin(diff, axis=1)
    ids = np.unique(idx)
    return [int(x) for x in ids if 0 <= int(x) <= 18]


def classnames_from_ids(ids: list) -> list:
    return [CITYSCAPES_TRAINID_TO_NAME[i] for i in sorted(set(ids)) if 0 <= i <= 18]

# ここから追加: caption 内の「天候・時間帯」語検査用パターン
# 英語＋日本語で、晴れ/曇り/雨/雪/霧、および 昼/夕方/夜 などをカバーする
CAPTION_SUBGROUP_PATTERNS = {
    # --- weather ---
    "weather_clear": [
        r"\bclear\b",
        r"\bclear sky\b",
        r"\bclear skies\b",
        r"\bsunny\b",
        r"\bsunlit\b",
        r"\bbright day\b",
        "晴れ",
        "快晴",
        "晴天",
    ],
    "weather_cloudy": [
        r"\bcloudy\b",
        r"\bovercast\b",
        r"\bclouds?\b",
        "曇り",
    ],
    "weather_rainy": [
        r"\brain(y|ing)?\b",
        r"\bdrizzle\b",
        r"\bshowers?\b",
        r"\bwet road\b",
        r"\bpuddles?\b",
        "雨",
        "小雨",
        "大雨",
        "豪雨",
    ],
    "weather_snowy": [
        r"\bsnow(y)?\b",
        r"\bsnowfall\b",
        r"\bslush\b",
        r"\bicy\b",
        r"\bsnow\-covered\b",
        "雪",
        "吹雪",
    ],
    "weather_foggy": [
        r"\bfog(gy)?\b",
        r"\bmist(y)?\b",
        r"\bhaze\b",
        r"\bhazy\b",
        r"\bsmog\b",
        "霧",
        "濃霧",
        "霧雨",
        "靄",
    ],
    # --- time of day ---
    "time_day": [
        r"\bdaytime\b",
        r"\bduring the day\b",
        r"\bmidday\b",
        r"\bnoon\b",
        r"\bmorning\b",
        r"\bafternoon\b",
        "昼",
        "昼間",
        "日中",
        "朝",
        "朝方",
    ],
    "time_twilight": [
        r"\btwilight\b",
        r"\bdusk\b",
        r"\bsunset\b",
        r"\bsunrise\b",
        r"\bgolden hour\b",
        r"\bblue hour\b",
        "夕方",
        "黄昏",
        "日没",
    ],
    "time_night": [
        r"\bnight\b",
        r"\bnighttime\b",
        r"\bat night\b",
        r"\bin the dark\b",
        "夜",
        "夜間",
        "深夜",
    ],
}

CAPTION_SUBGROUP_REGEX = {
    key: [re.compile(pat, flags=re.IGNORECASE) for pat in pats]
    for key, pats in CAPTION_SUBGROUP_PATTERNS.items()
}


def detect_subgroup_tokens_in_caption(text: str) -> dict:
    """
    1つの caption 文字列に対して、
    「天候・時間帯を連想させる表現」が含まれている category を返す。

    戻り値:
        { category_name: [マッチした正規表現パターン, ...], ... }
        何もマッチしなければ空 dict。
    """
    if not text:
        return {}

    hits = {}
    for key, regex_list in CAPTION_SUBGROUP_REGEX.items():
        matched = []
        for rgx in regex_list:
            if rgx.search(text):
                matched.append(rgx.pattern)
        if matched:
            hits[key] = matched
    return hits
# ここまで追加

# -------------- ここから差し替え --------------

def first_existing_path(candidates: list) -> Optional[str]:
    """
    複数候補パスのうち、最初に os.path.exists(...) で見つかったものを返す。
    どれも存在しない場合は None。
    """
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None
# ここから追加: caption 内の「天候・時間帯」語検査用パターン
# 英語＋日本語で、晴れ/曇り/雨/雪/霧、および 昼/夕方/夜 などをカバーする
CAPTION_SUBGROUP_PATTERNS = {
    # --- weather ---
    "weather_clear": [
        r"\bclear\b",
        r"\bclear sky\b",
        r"\bclear skies\b",
        r"\bsunny\b",
        r"\bsunlit\b",
        r"\bbright day\b",
        "晴れ",
        "快晴",
        "晴天",
    ],
    "weather_cloudy": [
        r"\bcloudy\b",
        r"\bovercast\b",
        r"\bclouds?\b",
        "曇り",
    ],
    "weather_rainy": [
        r"\brain(y|ing)?\b",
        r"\bdrizzle\b",
        r"\bshowers?\b",
        r"\bwet road\b",
        r"\bpuddles?\b",
        "雨",
        "小雨",
        "大雨",
        "豪雨",
    ],
    "weather_snowy": [
        r"\bsnow(y)?\b",
        r"\bsnowfall\b",
        r"\bslush\b",
        r"\bicy\b",
        r"\bsnow\-covered\b",
        "雪",
        "吹雪",
    ],
    "weather_foggy": [
        r"\bfog(gy)?\b",
        r"\bmist(y)?\b",
        r"\bhaze\b",
        r"\bhazy\b",
        r"\bsmog\b",
        "霧",
        "濃霧",
        "霧雨",
        "靄",
    ],
    # --- time of day ---
    "time_day": [
        r"\bdaytime\b",
        r"\bduring the day\b",
        r"\bmidday\b",
        r"\bnoon\b",
        r"\bmorning\b",
        r"\bafternoon\b",
        "昼",
        "昼間",
        "日中",
        "朝",
        "朝方",
    ],
    "time_twilight": [
        r"\btwilight\b",
        r"\bdusk\b",
        r"\bsunset\b",
        r"\bsunrise\b",
        r"\bgolden hour\b",
        r"\bblue hour\b",
        "夕方",
        "黄昏",
        "日没",
    ],
    "time_night": [
        r"\bnight\b",
        r"\bnighttime\b",
        r"\bat night\b",
        r"\bin the dark\b",
        "夜",
        "夜間",
        "深夜",
    ],
}#TODO もっと検査語彙を増やす。英語の検査語彙をもっと網羅的に増やす。

CAPTION_SUBGROUP_REGEX = {
    key: [re.compile(pat, flags=re.IGNORECASE) for pat in pats]
    for key, pats in CAPTION_SUBGROUP_PATTERNS.items()
}


def detect_subgroup_tokens_in_caption(text: str) -> dict:
    """
    1つの caption 文字列に対して、
    「天候・時間帯を連想させる表現」が含まれている category を返す。

    戻り値:
        { category_name: [マッチした正規表現パターン, ...], ... }
        何もマッチしなければ空 dict。
    """
    if not text:
        return {}

    hits = {}
    for key, regex_list in CAPTION_SUBGROUP_REGEX.items():
        matched = []
        for rgx in regex_list:
            if rgx.search(text):
                matched.append(rgx.pattern)
        if matched:
            hits[key] = matched
    return hits
# ここまで追加


def unique_trainids_from_any(ds_key: str, img_path: str, base_for_rel: str) -> list:
    """
    各データセットの RGB 画像に対応する trainId マップから、登場クラス ID を抽出する。

    - Cityscapes / BDD10K / GTA5 では可能な限り公式 GT PNG を優先
    - それ以外・欠損時は /data/ucn_condmaps/.../semseg/..._semseg.jpg を使用
    - nuimages_front だけは semseg/{stem}_semseg.jpg と
      semseg/train/{stem}_semseg.jpg の両方に対応（ここが今回の修正ポイント）
    """
    logger = logging.getLogger("ucn_prompts")
    rel_dir = rel_from(img_path, base_for_rel)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    # ---- Cityscapes ----
    if ds_key == "cityscapes":
        stem2 = stem_without_cityscapes_suffix(img_path)
        parts = Path(img_path).parts
        sp, city = parts[-3], parts[-2]

        # 1) 公式 GT を最優先
        gt = os.path.join(CITYSCAPES_GT_ROOT, sp, city, f"{stem2}_gtFine_labelIds.png")
        if os.path.exists(gt):
            return unique_trainids_from_label_png(gt)

        # 2) OneFormer 由来の Semseg JPG
        base = os.path.join(UCN_CONDMAPS_ROOT, "cityscapes", "semseg", sp, city)
        path = first_existing_path([os.path.join(base, f"{stem2}_semseg.jpg")])
        if path is None:
            logger.warning(
                "[unique_trainids_from_any][cityscapes] semseg not found for %s", img_path
            )
            return []
        return unique_trainids_from_color_semseg_jpg(path)

    # ---- BDD10K(10K) ----
    if ds_key == "bdd10k10k":
        sp = Path(img_path).parts[-2]

        gt = os.path.join(BDD10K10K_GT_LABEL_ROOT, sp, f"{stem}.png")
        if os.path.exists(gt):
            return unique_trainids_from_label_png(gt)

        base = os.path.join(UCN_CONDMAPS_ROOT, "bdd10k10k", "semseg", sp)
        path = first_existing_path([os.path.join(base, f"{stem}_semseg.jpg")])
        if path is None:
            logger.warning(
                "[unique_trainids_from_any][bdd10k10k] semseg not found for %s", img_path
            )
            return []
        return unique_trainids_from_color_semseg_jpg(path)

    # ---- GTA5 ----
    if ds_key == "gta5":
        # GTA5 は公式 GT PNG が揃っているのでまずこちら
        gt = os.path.join(GTA5_GT_LABEL_ROOT, f"{stem}.png")
        if os.path.exists(gt):
            return unique_trainids_from_label_png(gt)

        # それでも無い場合だけ semseg JPG を使用（旧仕様 + train/ 付き両対応）
        base = os.path.join(UCN_CONDMAPS_ROOT, "gta5", "semseg")
        candidates = []

        # 旧仕様: semseg/{rel_dir}/{stem}_semseg.jpg
        if rel_dir:
            candidates.append(os.path.join(base, rel_dir, f"{stem}_semseg.jpg"))
        else:
            candidates.append(os.path.join(base, f"{stem}_semseg.jpg"))

        # 新仕様互換: semseg/train/{rel_dir}/{stem}_semseg.jpg
        if rel_dir:
            candidates.append(os.path.join(base, "train", rel_dir, f"{stem}_semseg.jpg"))
        else:
            candidates.append(os.path.join(base, "train", f"{stem}_semseg.jpg"))

        path = first_existing_path(candidates)
        if path is None:
            logger.warning(
                "[unique_trainids_from_any][gta5] semseg not found for %s (candidates=%s)",
                img_path,
                candidates,
            )
            return []
        return unique_trainids_from_color_semseg_jpg(path)

    # ---- nuImages(front) ----
    if ds_key == "nuimages_front":
        base = os.path.join(UCN_CONDMAPS_ROOT, "nuimages_front", "semseg")
        candidates = []

        # 旧仕様（train サブディレクトリ無し）
        if rel_dir:
            candidates.append(os.path.join(base, rel_dir, f"{stem}_semseg.jpg"))
        else:
            candidates.append(os.path.join(base, f"{stem}_semseg.jpg"))

        # 現在の実データ: semseg/train/{stem}_semseg.jpg
        if rel_dir:
            candidates.append(os.path.join(base, "train", rel_dir, f"{stem}_semseg.jpg"))
        else:
            candidates.append(os.path.join(base, "train", f"{stem}_semseg.jpg"))

        path = first_existing_path(candidates)
        if path is None:
            logger.warning(
                "[unique_trainids_from_any][nuimages_front] semseg not found for %s (candidates=%s)",
                img_path,
                candidates,
            )
            # ここでは落とさずに「クラス名なし」として続行
            return []
        return unique_trainids_from_color_semseg_jpg(path)

    # ---- BDD100K ----
    if ds_key == "bdd100k":
        sp = Path(img_path).parts[-2]
        base = os.path.join(UCN_CONDMAPS_ROOT, "bdd100k", "semseg", sp)
        path = first_existing_path([os.path.join(base, f"{stem}_semseg.jpg")])
        if path is None:
            logger.warning(
                "[unique_trainids_from_any][bdd100k] semseg not found for %s", img_path
            )
            return []
        return unique_trainids_from_color_semseg_jpg(path)

    # 予期しないキー
    raise ValueError(f"unknown ds_key={ds_key}")

# -------------- ここまで差し替え --------------


def classnames_for_image(ds_key: str, img_path: str, base_for_rel: str) -> list:
    ids = unique_trainids_from_any(ds_key, img_path, base_for_rel)
    return classnames_from_ids(ids)


def classnames_for_image(ds_key: str, img_path: str, base_for_rel: str) -> list:
    ids = unique_trainids_from_any(ds_key, img_path, base_for_rel)
    return classnames_from_ids(ids)

def list_waymo_images(split: str, limit: int=-1) -> list:
    base = os.path.join(WAYMO_RGB_ROOT, split, "front")
    imgs = list_images_under(base)
    if limit>0: imgs = imgs[:limit]
    return imgs

# ==============================
# 11) メイン
# ==============================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SynDiff-AD: 学習/評価プロンプト一括生成（Qwen3-VL + CLIP）")
    ap.add_argument(
        "--jobs",
        type=str,
        nargs="+",
        choices=[
            "train",
            "eval_waymo",
            # ここから分析系 job
            "analyze_captions",          # (1) caption に天候/時間語が混入していないかチェック
            "analyze_train_subgroups",   # (2) train set の (weather,time) ヒストグラム
            "analyze_eval_mappings",     # (3) Waymo の src→tgt subgroup mapping
            "analyze_clip_gallery",      # (4) 各 subgroup のサンプル画像タイル
            "analyze_clip_similarity",   # (5) CLIP 画像埋め込み vs テキスト埋め込み 類似度行列
            "analyze_all",               # 上の (1)〜(5) をまとめて実行
        ],
        default=["train", "eval_waymo"],
    )
    ap.add_argument("--datasets", type=str, nargs="+", choices=["all","bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k"], default=["all"])
    ap.add_argument("--cityscapes-use-splits", type=str, nargs="+", default=["train"])
    ap.add_argument("--bdd10k10k-use-splits", type=str, nargs="+", default=["train"])
    ap.add_argument("--bdd100k-use-splits", type=str, nargs="+", default=["train"])
    ap.add_argument("--waymo-splits", type=str, nargs="+", default=["training","validation","testing"])
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tb", action="store_true")
    # VLM
    ap.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--vlm-local-dir", type=str, default=os.environ.get("VLM_MODEL_LOCAL_DIR", None))
    ap.add_argument("--vlm-dtype", type=str, default="auto")
    ap.add_argument("--vlm-device-map", type=str, default="auto")
    ap.add_argument("--vlm-attn", type=str, default=None)
    ap.add_argument("--hidethinking", action="store_true"); ap.set_defaults(hidethinking=True)
    # 量子化（nodeps で bitsandbytes を入れておくこと）
    ap.add_argument("--quant", type=str, choices=["none","8bit","4bit"],
                    default=os.environ.get("VLM_QUANT","4bit"),
                    help="Qwen の量子化モード。既定 4bit。")
    # 生成の短文化パラメータ
    ap.add_argument("--cap-max-new", type=int, default=96, help="Qwen の max_new_tokens（短くするほど高速）")
    ap.add_argument("--cap-min-new", type=int, default=0, help="Qwen の min_new_tokens")
    ap.add_argument("--cap-words", type=int, default=45, help="プロンプト内で指示する最大語数（ガイド）")
    # CLIP
    ap.add_argument("--clip-batch", type=int, default=8)
    ap.add_argument("--openclip-cache-dir", type=str, default=os.environ.get("OPENCLIP_CACHE_DIR", None))
    # Waymo
    ap.add_argument("--simplemode", action="store_true")
    ap.add_argument("--style-lib", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # レジューム
    ap.add_argument("--resume", dest="resume", action="store_true", help="既存JSONLがあればスキップ")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    # 分析時に CLIP のテキストプロンプトを文書型にするかどうか
    #   False: 既存どおり "Clear", "Rainy" などのラベル文字列を使用
    #   True : sentence モード ("a clear sunny daytime scene" 等) を使用
    ap.add_argument(
        "--analysis-use-sentence-prompts",
        action="store_true",
        help="分析 job 実行時に、CLIP の text prompt を文書型 (sentence) で評価する",
    )

    ap.set_defaults(resume=True)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = setup_logger("ucn_prompts", LOG_ROOT, args.verbose)
    log_env(logger)

    sw: Optional[SummaryWriter] = SummaryWriter(TB_ROOT) if args.tb else None
    if sw:
        logger.info("TensorBoard: %s", TB_ROOT)

    for p in [RAW_CAP_ROOT, PROMPTS_TRAIN_ROOT, PROMPTS_EVAL_ROOT, SUBGROUP_MAP_ROOT]:
        ensure_dir(p)

    # スタイル辞書
    style_lib = DEFAULT_STYLE_LIB
    if args.style_lib and os.path.isfile(args.style_lib):
        try:
            with open(args.style_lib, "r", encoding="utf-8") as f:
                raw = json.load(f)
            st = {}
            for k, v in raw.items():
                if isinstance(k, str) and "," in k:
                    kk = tuple([x.strip() for x in k.split(",")])
                elif isinstance(k, (list, tuple)) and len(k) == 2:
                    kk = (str(k[0]), str(k[1]))
                else:
                    continue
                st[kk] = v
            if st:
                style_lib = st
                logger.info("外部スタイル辞書を適用: %d entries", len(style_lib))
        except Exception as e:
            logger.warning("外部スタイル辞書の読込失敗: %s → デフォルト使用", repr(e))

    # ====== どの job を動かすか ======
    run_train = "train" in args.jobs
    run_eval_waymo = "eval_waymo" in args.jobs
    run_generation_jobs = run_train or run_eval_waymo
    run_analysis_jobs = any(j.startswith("analyze") for j in args.jobs)

    # 生成ジョブ用の CLIP/Qwen (既存挙動そのまま)
    clip = None
    qwen: Optional[QwenVLCaptioner] = None

    if run_generation_jobs:
        clip = ClipSubgroupClassifier(
            batch_size=args.clip_batch,
            cache_dir=args.openclip_cache_dir,
            logger=logger,
            prompt_mode="label",  # 生成系は既存どおりラベルプロンプト固定
        )

        # 量子化モードを env にも反映（QwenVLCaptioner 内で参照）
        os.environ["VLM_QUANT"] = getattr(args, "quant", "4bit")

        qwen = QwenVLCaptioner(
            model_id=args.vlm_model,
            local_dir=args.vlm_local_dir,
            torch_dtype=args.vlm_dtype,
            device_map=args.vlm_device_map,
            attn_impl=args.vlm_attn,
            hidethinking=args.hidethinking,
            quant=args.quant,
            cap_max_new=args.cap_max_new,
            cap_min_new=args.cap_min_new,
            cap_words=args.cap_words,
            logger=logger,
        )

    # ===== 学習 =====
    if run_train:
        ds_list = (
            ["bdd10k10k", "cityscapes", "gta5", "nuimages_front", "bdd100k"]
            if ("all" in args.datasets)
            else args.datasets
        )
        for ds in ds_list:
            imgs, base_rel = list_dataset_images(
                ds,
                use_splits=(
                    args.bdd10k10k_use_splits
                    if ds == "bdd10k10k"
                    else args.cityscapes_use_splits
                    if ds == "cityscapes"
                    else args.bdd100k_use_splits
                    if ds == "bdd100k"
                    else None
                ),
                limit=args.limit,
            )
            if not imgs:
                logger.warning("[TRAIN][%s] 対象画像なし", ds)
                continue
            logger.info("[TRAIN][%s] 画像枚数: %d", ds, len(imgs))
            out_csv = os.path.join(PROMPTS_TRAIN_ROOT, f"{ds}.csv")
            rows = []
            map_rows = []
            processed = 0
            t0 = time.time()
            for i, p in enumerate(tqdm(imgs, desc=f"train:{ds}", dynamic_ncols=True)):
                try:
                    # レジューム（生成済みJSONLがあればスキップ）
                    raw_jl = os.path.join(RAW_CAP_ROOT, ds, f"{Path(p).stem}.jsonl")
                    if args.resume and os.path.exists(raw_jl):
                        # 読んで rows/map_rows に反映（最小限）
                        try:
                            with open(raw_jl, "r", encoding="utf-8") as f:
                                last = json.loads(list(f)[-1])
                            cap_cached = last.get("caption", "")
                        except Exception:
                            cap_cached = ""
                        rgb = imread_rgb(p)
                        cls_names = classnames_for_image(ds, p, base_rel)
                        (w, t) = clip.classify_one(rgb)  # type: ignore
                        prompt = make_train_prompt(cap_cached, w, t)
                        rows.append([p, w, t, cap_cached, prompt])
                        map_rows.append([p, f"{w},{t}", ""])
                        processed += 1
                        if sw and (processed % 50 == 0):
                            sw.add_scalar(f"train/{ds}_processed", processed, processed)
                        continue

                    rgb = imread_rgb(p)
                    cls_names = classnames_for_image(ds, p, base_rel)
                    (w, t) = clip.classify_one(rgb)  # type: ignore
                    cap = qwen.caption(rgb, cls_names, seed=args.seed + i)  # type: ignore
                    prompt = make_train_prompt(cap, w, t)
                    rows.append([p, w, t, cap, prompt])
                    map_rows.append([p, f"{w},{t}", ""])
                    write_jsonl(
                        raw_jl,
                        [{"image": p, "caption": cap, "time": datetime.now().isoformat()}],
                    )
                    processed += 1
                    if sw and (processed % 50 == 0):
                        sw.add_scalar(f"train/{ds}_processed", processed, processed)
                except Exception:
                    logger.exception("[TRAIN][%s] 失敗: %s", ds, p)
            append_csv(
                out_csv,
                ["image_path", "weather", "time", "raw_caption", "train_prompt"],
                rows,
            )
            append_csv(
                os.path.join(SUBGROUP_MAP_ROOT, f"{ds}_subgroups.csv"),
                ["image_path", "source_subgroup", "target_subgroup"],
                map_rows,
            )
            dt = time.time() - t0
            logger.info(
                "[TRAIN][%s] 完了: %d件, time=%.1fs (%.2f img/s)",
                ds,
                processed,
                dt,
                processed / max(1.0, dt),
            )

    # ===== Waymo 評価 =====
    if run_eval_waymo:
        for sp in args.waymo_splits:
            imgs = list_waymo_images(sp, limit=args.limit)
            if not imgs:
                logger.warning("[EVAL][Waymo/%s] 対象画像なし", sp)
                continue
            logger.info("[EVAL][Waymo/%s] 画像枚数: %d", sp, len(imgs))
            out_csv = os.path.join(PROMPTS_EVAL_ROOT, f"waymo_{sp}.csv")
            rows = []
            map_rows = []
            processed = 0
            t0 = time.time()
            for i, p in enumerate(tqdm(imgs, desc=f"waymo:{sp}", dynamic_ncols=True)):
                try:
                    raw_jl = os.path.join(RAW_CAP_ROOT, f"waymo_{sp}", f"{Path(p).stem}.jsonl")
                    rgb = imread_rgb(p)
                    npy = waymo_paths_to_npy(p)
                    if os.path.exists(npy):
                        seg = np.load(npy)
                        ids = [int(x) for x in np.unique(seg) if 0 <= int(x) <= 18]
                        cls_names = classnames_from_ids(ids)
                    else:
                        cls_names = []
                    (w, t) = clip.classify_one(rgb)  # type: ignore
                    ws = [x for x in WEATHER if x != w]
                    ts = [x for x in TIME if x != t]
                    target_w = random.choice(ws)
                    target_t = random.choice(ts)

                    if args.resume and os.path.exists(raw_jl):
                        try:
                            with open(raw_jl, "r", encoding="utf-8") as f:
                                last = json.loads(list(f)[-1])
                            cap = last.get("caption", "")
                        except Exception:
                            cap = ""
                    else:
                        cap = qwen.caption(rgb, cls_names, seed=args.seed + i)  # type: ignore
                        write_jsonl(
                            raw_jl,
                            [{"image": p, "caption": cap, "time": datetime.now().isoformat()}],
                        )

                    prompt = (
                        make_waymo_prompt_simple(cls_names, target_w, target_t)
                        if args.simplemode
                        else make_waymo_prompt_adv(
                            cap, cls_names, target_w, target_t, style_lib
                        )
                    )
                    rows.append([p, w, t, f"{target_w}", f"{target_t}", cap, prompt])
                    map_rows.append([p, f"{w},{t}", f"{target_w},{target_t}"])
                    processed += 1
                    if sw and (processed % 50 == 0):
                        sw.add_scalar(f"eval_waymo/{sp}_processed", processed, processed)
                except Exception:
                    logger.exception("[EVAL][Waymo/%s] 失敗: %s", sp, p)
            append_csv(
                out_csv,
                [
                    "image_path",
                    "src_weather",
                    "src_time",
                    "tgt_weather",
                    "tgt_time",
                    "raw_caption",
                    "eval_prompt",
                ],
                rows,
            )
            append_csv(
                os.path.join(SUBGROUP_MAP_ROOT, f"waymo_{sp}_subgroups.csv"),
                ["image_path", "source_subgroup", "target_subgroup"],
                map_rows,
            )
            dt = time.time() - t0
            logger.info(
                "[EVAL][Waymo/%s] 完了: %d件, time=%.1fs (%.2f img/s)",
                sp,
                processed,
                dt,
                processed / max(1.0, dt),
            )

    # ===== 分析 job =====
    if run_analysis_jobs:
        analysis_prompt_mode = (
            "sentence" if args.analysis_use_sentence_prompts else "label"
        )
        logger.info(
            "[ANALYZE] analysis jobs=%s, prompt_mode=%s",
            ",".join(args.jobs),
            analysis_prompt_mode,
        )

        if ("analyze_captions" in args.jobs) or ("analyze_all" in args.jobs):
            analyze_captions_forbidden_subgroups(logger)

        if ("analyze_train_subgroups" in args.jobs) or ("analyze_all" in args.jobs):
            analyze_train_subgroups(args, logger, analysis_prompt_mode)

        if ("analyze_eval_mappings" in args.jobs) or ("analyze_all" in args.jobs):
            analyze_eval_mappings(args, logger, analysis_prompt_mode)

        if ("analyze_clip_gallery" in args.jobs) or ("analyze_all" in args.jobs):
            analyze_clip_gallery(args, logger, analysis_prompt_mode)

        if ("analyze_clip_similarity" in args.jobs) or ("analyze_all" in args.jobs):
            analyze_clip_similarity(args, logger, analysis_prompt_mode)

    if sw:
        sw.close()
    logger.info("✅ 全ジョブ完了")
# ここから追加: 分析 job 本体
def analyze_captions_forbidden_subgroups(logger: logging.Logger) -> None:
    """
    (1) caption に天候/時間帯を連想させる語彙が混ざっていないかチェックする。

    - /data/syndiff_prompts/raw_captions/ 配下の全 JSONL を走査
    - 最終行の {"caption": "..."} を読み取り、CAPTION_SUBGROUP_REGEX でマッチを検査
    - マッチした caption について CSV + ログに出力
    """
    root = RAW_CAP_ROOT
    if not os.path.isdir(root):
        logger.warning("[CAPSCAN] RAW_CAP_ROOT not found: %s", root)
        return

    out_dir = os.path.join(OUT_ROOT, "analysis", "caption_scan")
    ensure_dir(out_dir)
    hits_csv = os.path.join(out_dir, "caption_subgroup_hits.csv")
    summary_csv = os.path.join(out_dir, "caption_subgroup_summary.csv")

    # 毎回きれいな結果にしたいので古い結果があれば削除
    for p in [hits_csv, summary_csv]:
        if os.path.exists(p):
            os.remove(p)

    total_captions = 0
    total_hits = 0
    cat_counts = {k: 0 for k in CAPTION_SUBGROUP_PATTERNS.keys()}
    rows = []

    for ds_name in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        jsonl_files = sorted(
            f for f in os.listdir(ds_dir) if f.endswith(".jsonl")
        )
        logger.info("[CAPSCAN] dataset=%s, jsonl files=%d", ds_name, len(jsonl_files))
        for jf in jsonl_files:
            jsonl_path = os.path.join(ds_dir, jf)
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if not lines:
                    continue
                rec = json.loads(lines[-1])
            except Exception:
                logger.exception("[CAPSCAN] JSONL 読み込み失敗: %s", jsonl_path)
                continue

            caption = str(rec.get("caption", ""))
            image_path = str(rec.get("image", ""))
            total_captions += 1

            hits = detect_subgroup_tokens_in_caption(caption)
            if not hits:
                continue

            total_hits += 1
            cats = sorted(hits.keys())
            for cat in cats:
                cat_counts[cat] += 1

            excerpt = caption if len(caption) <= 200 else caption[:200] + "..."
            rows.append(
                [
                    ds_name,
                    jsonl_path,
                    image_path,
                    "|".join(cats),
                    excerpt,
                ]
            )

    if rows:
        append_csv(
            hits_csv,
            ["dataset", "jsonl_path", "image_path", "categories", "caption_excerpt"],
            rows,
        )

    summary_rows = []
    summary_rows.append(["TOTAL_CAPTIONS", total_captions])
    summary_rows.append(["TOTAL_HITS", total_hits])
    hit_rate = (total_hits / total_captions * 100.0) if total_captions > 0 else 0.0
    summary_rows.append(["HIT_RATE_PERCENT", f"{hit_rate:.3f}"])
    summary_rows.append(["", ""])
    for cat, cnt in sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True):
        summary_rows.append([cat, cnt])

    append_csv(summary_csv, ["key", "value"], summary_rows)

    logger.info(
        "[CAPSCAN] total captions=%d, hits=%d (%.3f %%)",
        total_captions,
        total_hits,
        hit_rate,
    )
    logger.info("[CAPSCAN] 詳細: %s, サマリ: %s", hits_csv, summary_csv)


def analyze_train_subgroups(
    args: argparse.Namespace,
    logger: logging.Logger,
    prompt_mode: str,
) -> None:
    """
    (2) 学習セットに対して、CLIP による (weather, time) 分布を再推定し、
        ヒストグラム + ヒートマップを出力する。

    - 入力: /data/syndiff_prompts/prompts_train/*.csv （image_path 列を使用）
    - 出力:
        /data/syndiff_prompts/analysis/train_subgroups/{ds}_subgroup_hist_{prompt_mode}.csv
        /data/syndiff_prompts/analysis/train_subgroups/{ds}_subgroup_heatmap_{prompt_mode}.png
    """
    csv_dir = PROMPTS_TRAIN_ROOT
    if not os.path.isdir(csv_dir):
        logger.warning("[TRAIN-HIST] PROMPTS_TRAIN_ROOT not found: %s", csv_dir)
        return

    out_dir = os.path.join(OUT_ROOT, "analysis", "train_subgroups")
    ensure_dir(out_dir)

    csv_files = sorted(
        Path(csv_dir).glob("*.csv")
    )  # bdd10k10k.csv, cityscapes.csv, ...
    if not csv_files:
        logger.warning("[TRAIN-HIST] no train csv in %s", csv_dir)
        return

    clip = build_clip_for_analysis(args, logger, prompt_mode, tag="train_subgroups")
    try:
        for csv_path in csv_files:
            ds_key = csv_path.stem
            df = pd.read_csv(csv_path)
            if "image_path" not in df.columns:
                logger.warning(
                    "[TRAIN-HIST][%s] image_path 列が見つかりません: %s", ds_key, csv_path
                )
                continue
            paths = df["image_path"].dropna().tolist()
            if not paths:
                logger.warning("[TRAIN-HIST][%s] image_path が空です", ds_key)
                continue

            logger.info(
                "[TRAIN-HIST][%s] CLIP 分類開始: %d images (prompt_mode=%s)",
                ds_key,
                len(paths),
                prompt_mode,
            )
            classified = classify_paths_with_clip(
                paths, clip, args.clip_batch, logger, tag=f"train_{ds_key}"
            )

            counts_pair = {(w, t): 0 for w in WEATHER for t in TIME}
            for _, w, t in classified:
                if (w in WEATHER) and (t in TIME):
                    counts_pair[(w, t)] = counts_pair.get((w, t), 0) + 1

            hist_rows = []
            for w in WEATHER:
                for t in TIME:
                    hist_rows.append([ds_key, w, t, counts_pair.get((w, t), 0)])

            out_csv = os.path.join(
                out_dir, f"{ds_key}_subgroup_hist_{prompt_mode}.csv"
            )
            if os.path.exists(out_csv):
                os.remove(out_csv)
            append_csv(out_csv, ["dataset", "weather", "time", "count"], hist_rows)

            heat = np.array(
                [[counts_pair.get((w, t), 0) for t in TIME] for w in WEATHER],
                dtype=np.float32,
            )
            plt.figure(figsize=(6, 4))
            plt.imshow(heat, aspect="auto")  # デフォルト colormap
            plt.xticks(range(len(TIME)), TIME)
            plt.yticks(range(len(WEATHER)), WEATHER)
            plt.xlabel("time-of-day subgroup")
            plt.ylabel("weather subgroup")
            plt.title(f"{ds_key} subgroup counts (prompts={prompt_mode})")
            plt.colorbar()
            plt.tight_layout()
            out_png = os.path.join(
                out_dir, f"{ds_key}_subgroup_heatmap_{prompt_mode}.png"
            )
            plt.savefig(out_png)
            plt.close()

            logger.info(
                "[TRAIN-HIST][%s] CSV: %s, Heatmap: %s", ds_key, out_csv, out_png
            )
    finally:
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_eval_mappings(
    args: argparse.Namespace,
    logger: logging.Logger,
    prompt_mode: str,
) -> None:
    """
    (3) Waymo 評価データについて、
        source_subgroup → target_subgroup の mapping 分布を集計・可視化する。

    - 入力: /data/syndiff_prompts/prompts_eval_waymo/waymo_{split}.csv
        (image_path, src_weather, src_time, tgt_weather, tgt_time)
    - 出力:
        /data/syndiff_prompts/analysis/eval_mappings/waymo_{split}_mapping_{prompt_mode}.csv
        /data/syndiff_prompts/analysis/eval_mappings/waymo_{split}_mapping_top10_{prompt_mode}.png
    """
    csv_dir = PROMPTS_EVAL_ROOT
    if not os.path.isdir(csv_dir):
        logger.warning("[EVAL-MAP] PROMPTS_EVAL_ROOT not found: %s", csv_dir)
        return

    out_dir = os.path.join(OUT_ROOT, "analysis", "eval_mappings")
    ensure_dir(out_dir)

    csv_files = sorted(Path(csv_dir).glob("waymo_*.csv"))
    if not csv_files:
        logger.warning("[EVAL-MAP] no waymo_*.csv in %s", csv_dir)
        return

    clip = build_clip_for_analysis(args, logger, prompt_mode, tag="eval_mappings")
    try:
        for csv_path in csv_files:
            split = csv_path.stem.replace("waymo_", "")
            df = pd.read_csv(csv_path)
            required = ["image_path", "tgt_weather", "tgt_time"]
            for col in required:
                if col not in df.columns:
                    logger.warning(
                        "[EVAL-MAP][%s] column %s not found in %s",
                        split,
                        col,
                        csv_path,
                    )
                    continue

            paths = df["image_path"].dropna().tolist()
            if not paths:
                logger.warning("[EVAL-MAP][%s] image_path が空です", split)
                continue

            logger.info(
                "[EVAL-MAP][%s] CLIP 再分類開始: %d images (prompt_mode=%s)",
                split,
                len(paths),
                prompt_mode,
            )
            classified = classify_paths_with_clip(
                paths, clip, args.clip_batch, logger, tag=f"waymo_{split}"
            )
            src_map = {p: (w, t) for (p, w, t) in classified}

            counts = {}
            for _, row in df.iterrows():
                p = str(row.get("image_path", ""))
                tgt_w = str(row.get("tgt_weather", ""))
                tgt_t = str(row.get("tgt_time", ""))
                if not p or not tgt_w or not tgt_t:
                    continue

                if p in src_map:
                    src_w, src_t = src_map[p]
                else:
                    # CLIP 再分類に失敗した場合は、元 CSV の src_* をフォールバックとして使う
                    src_w = str(row.get("src_weather", ""))
                    src_t = str(row.get("src_time", ""))
                if not src_w or not src_t:
                    continue

                key = (src_w, src_t, tgt_w, tgt_t)
                counts[key] = counts.get(key, 0) + 1

            if not counts:
                logger.warning("[EVAL-MAP][%s] 有効な mapping がありません", split)
                continue

            summary_rows = []
            for (sw, st, tw, tt), cnt in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                summary_rows.append([split, sw, st, tw, tt, cnt])

            csv_out = os.path.join(
                out_dir, f"waymo_{split}_mapping_{prompt_mode}.csv"
            )
            if os.path.exists(csv_out):
                os.remove(csv_out)
            append_csv(
                csv_out,
                ["split", "src_weather", "src_time", "tgt_weather", "tgt_time", "count"],
                summary_rows,
            )

            # 上位10組 + others を棒グラフで可視化
            top_items = summary_rows[:10]
            others_count = sum(r[5] for r in summary_rows[10:])
            labels = [
                f"{sw}-{st}→{tw}-{tt}" for (_, sw, st, tw, tt, cnt) in top_items
            ]
            values = [cnt for (_, sw, st, tw, tt, cnt) in top_items]
            if others_count > 0:
                labels.append("others")
                values.append(others_count)

            idxs = range(len(labels))
            plt.figure(figsize=(max(8, len(labels) * 0.7), 4))
            plt.bar(idxs, values)
            plt.xticks(idxs, labels, rotation=45, ha="right")
            plt.ylabel("num images")
            plt.title(f"Waymo {split}: subgroup mapping (prompts={prompt_mode})")
            plt.tight_layout()
            png_out = os.path.join(
                out_dir, f"waymo_{split}_mapping_top10_{prompt_mode}.png"
            )
            plt.savefig(png_out)
            plt.close()

            logger.info("[EVAL-MAP][%s] CSV: %s, PNG: %s", split, csv_out, png_out)
    finally:
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_clip_gallery(
    args: argparse.Namespace,
    logger: logging.Logger,
    prompt_mode: str,
) -> None:
    """
    (4) CLIP の source_subgroup 推定の正しさを目視チェックするため、
        各 (weather,time) についてサンプル画像パスと 2x3 モンタージュ画像を出力する。

    - 入力: /data/syndiff_prompts/prompts_train/*.csv (image_path 列を使用)
    - 出力:
        /data/syndiff_prompts/analysis/clip_gallery/{ds}_{weather}_{time}_samples_{prompt_mode}.json
        /data/syndiff_prompts/analysis/clip_gallery/{ds}_{weather}_{time}_gallery6_{prompt_mode}.jpg
    """
    csv_dir = PROMPTS_TRAIN_ROOT
    if not os.path.isdir(csv_dir):
        logger.warning("[GALLERY] PROMPTS_TRAIN_ROOT not found: %s", csv_dir)
        return

    out_dir = os.path.join(OUT_ROOT, "analysis", "clip_gallery")
    ensure_dir(out_dir)

    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        logger.warning("[GALLERY] no train csv in %s", csv_dir)
        return

    clip = build_clip_for_analysis(args, logger, prompt_mode, tag="clip_gallery")
    try:
        for csv_path in csv_files:
            ds_key = csv_path.stem
            df = pd.read_csv(csv_path)
            if "image_path" not in df.columns:
                logger.warning(
                    "[GALLERY][%s] image_path 列が見つかりません: %s", ds_key, csv_path
                )
                continue
            paths = df["image_path"].dropna().tolist()
            if not paths:
                logger.warning("[GALLERY][%s] image_path が空です", ds_key)
                continue

            logger.info(
                "[GALLERY][%s] CLIP 分類開始: %d images (prompt_mode=%s)",
                ds_key,
                len(paths),
                prompt_mode,
            )
            classified = classify_paths_with_clip(
                paths, clip, args.clip_batch, logger, tag=f"gallery_{ds_key}"
            )

            # (weather,time) ごとに最大 10 件蓄える
            buckets = {(w, t): [] for w in WEATHER for t in TIME}
            for p, w, t in classified:
                key = (w, t)
                if key in buckets and len(buckets[key]) < 10:
                    buckets[key].append(p)

            for w in WEATHER:
                for t in TIME:
                    key = (w, t)
                    images = buckets.get(key, [])
                    if not images:
                        continue

                    json_path = os.path.join(
                        out_dir,
                        f"{ds_key}_{w}_{t}_samples_{prompt_mode}.json",
                    )
                    meta = {
                        "dataset": ds_key,
                        "weather": w,
                        "time": t,
                        "prompt_mode": prompt_mode,
                        "sample_image_paths": images,
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    gallery_paths = images[:6]
                    montage = build_image_montage(
                        gallery_paths, logger, tag=f"{ds_key}_{w}_{t}"
                    )
                    if montage is None:
                        continue
                    img_out = os.path.join(
                        out_dir,
                        f"{ds_key}_{w}_{t}_gallery6_{prompt_mode}.jpg",
                    )
                    cv2.imwrite(img_out, montage)

                    logger.info(
                        "[GALLERY][%s][%s,%s] gallery: %s, meta: %s",
                        ds_key,
                        w,
                        t,
                        img_out,
                        json_path,
                    )
    finally:
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_clip_similarity(
    args: argparse.Namespace,
    logger: logging.Logger,
    prompt_mode: str,
) -> None:
    """
    (5) CLIP image/text embedding の類似度行列を可視化する。

    - 手順
        1) train 全画像を CLIP で (weather,time) 推定
        2) 各 weather, time ごとに複数サンプルを保持
        3) set_index=0,1,2 の 3 セットに対して
            - weather 軸: WEATHER それぞれから 1 枚ずつ画像を取り、5x5 類似度行列
            - time   軸: TIME    それぞれから 1 枚ずつ画像を取り、3x3 類似度行列
        4) 類似度行列の min/max / 対角成分などをログに出力
        5) image/text embedding を .npy で保存
    """
    csv_dir = PROMPTS_TRAIN_ROOT
    if not os.path.isdir(csv_dir):
        logger.warning("[SIM] PROMPTS_TRAIN_ROOT not found: %s", csv_dir)
        return

    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        logger.warning("[SIM] no train csv in %s", csv_dir)
        return

    all_paths = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "image_path" not in df.columns:
            continue
        all_paths.extend(df["image_path"].dropna().tolist())
    if not all_paths:
        logger.warning("[SIM] no image_path in any train csv")
        return

    random.shuffle(all_paths)

    clip = build_clip_for_analysis(args, logger, prompt_mode, tag="clip_similarity")
    try:
        logger.info(
            "[SIM] CLIP 分類開始: %d images (prompt_mode=%s)",
            len(all_paths),
            prompt_mode,
        )
        classified = classify_paths_with_clip(
            all_paths, clip, args.clip_batch, logger, tag="sim_train"
        )

        images_by_weather = {w: [] for w in WEATHER}
        images_by_time = {t: [] for t in TIME}
        for p, w, t in classified:
            if w in images_by_weather and len(images_by_weather[w]) < 64:
                images_by_weather[w].append(p)
            if t in images_by_time and len(images_by_time[t]) < 64:
                images_by_time[t].append(p)

        for w in WEATHER:
            logger.info(
                "[SIM] weather=%s, samples=%d", w, len(images_by_weather[w])
            )
        for t in TIME:
            logger.info("[SIM] time=%s, samples=%d", t, len(images_by_time[t]))

        weather_texts, time_texts = get_clip_text_prompts_for_mode(prompt_mode)
        device = clip.device

        with torch.no_grad():
            tok_w = clip.tokenizer(weather_texts).to(device)
            txt_feat_w = clip.model.encode_text(tok_w)
            tok_t = clip.tokenizer(time_texts).to(device)
            txt_feat_t = clip.model.encode_text(tok_t)

            txt_feat_w = txt_feat_w / txt_feat_w.norm(dim=-1, keepdim=True)
            txt_feat_t = txt_feat_t / txt_feat_t.norm(dim=-1, keepdim=True)

        sim_dir = os.path.join(OUT_ROOT, "analysis", "clip_similarity")
        ensure_dir(sim_dir)
        np.save(
            os.path.join(sim_dir, f"weather_text_embeddings_{prompt_mode}.npy"),
            txt_feat_w.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(sim_dir, f"time_text_embeddings_{prompt_mode}.npy"),
            txt_feat_t.detach().cpu().numpy(),
        )
        with open(
            os.path.join(sim_dir, f"text_prompts_{prompt_mode}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "prompt_mode": prompt_mode,
                    "weather_prompts": weather_texts,
                    "time_prompts": time_texts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        def _build_axis_similarity(
            axis_name: str,
            labels: list,
            images_by_label: dict,
            text_feats: torch.Tensor,
            text_labels_for_axis: list,
            set_idx: int,
        ) -> None:
            img_feats_list = []
            selected_labels = []
            samples_meta = []

            for label in labels:
                candidates = images_by_label.get(label, [])
                if not candidates:
                    logger.warning(
                        "[SIM][%s] set=%d, label=%s に対応する画像がありません",
                        axis_name,
                        set_idx + 1,
                        label,
                    )
                    continue
                idx = set_idx % len(candidates)
                img_path = candidates[idx]
                try:
                    rgb = imread_rgb(img_path)
                    img_pil = Image.fromarray(rgb)
                    img_tensor = clip.transform(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = clip.model.encode_image(img_tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                except Exception:
                    logger.exception(
                        "[SIM][%s] 画像埋め込み失敗: %s", axis_name, img_path
                    )
                    continue

                img_feats_list.append(feat)
                selected_labels.append(label)
                samples_meta.append({"label": label, "image_path": img_path})

            if not img_feats_list:
                logger.warning(
                    "[SIM][%s] set=%d で有効な画像がありません", axis_name, set_idx + 1
                )
                return

            img_feats = torch.cat(img_feats_list, dim=0)
            sim = (img_feats @ text_feats.T).detach().cpu().numpy()
            sim_min = float(sim.min())
            sim_max = float(sim.max())
            logger.info(
                "[SIM][%s] set=%d similarity range: min=%.4f, max=%.4f",
                axis_name,
                set_idx + 1,
                sim_min,
                sim_max,
            )

            # 埋め込みとメタ情報を保存
            np.save(
                os.path.join(
                    sim_dir,
                    f"{axis_name}_set{set_idx+1}_image_embeddings_{prompt_mode}.npy",
                ),
                img_feats.cpu().numpy(),
            )
            with open(
                os.path.join(
                    sim_dir,
                    f"{axis_name}_set{set_idx+1}_samples_{prompt_mode}.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "axis": axis_name,
                        "set_index": set_idx,
                        "prompt_mode": prompt_mode,
                        "samples": samples_meta,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 類似度行列をヒートマップとして保存
            plt.figure(figsize=(5, 4))
            plt.imshow(sim, aspect="auto")
            plt.xticks(range(len(text_labels_for_axis)), text_labels_for_axis, rotation=45, ha="right")
            plt.yticks(range(len(selected_labels)), selected_labels)
            plt.xlabel("text subgroup label")
            plt.ylabel("image subgroup (predicted)")
            plt.title(
                f"CLIP {axis_name} similarity set{set_idx+1} (prompts={prompt_mode})"
            )
            plt.colorbar()
            plt.tight_layout()
            png_out = os.path.join(
                sim_dir,
                f"{axis_name}_set{set_idx+1}_similarity_{prompt_mode}.png",
            )
            plt.savefig(png_out)
            plt.close()

        # set_index = 0,1,2 の 3 セット
        for set_idx in range(3):
            _build_axis_similarity(
                axis_name="weather",
                labels=WEATHER,
                images_by_label=images_by_weather,
                text_feats=txt_feat_w,
                text_labels_for_axis=WEATHER,
                set_idx=set_idx,
            )
            _build_axis_similarity(
                axis_name="time",
                labels=TIME,
                images_by_label=images_by_time,
                text_feats=txt_feat_t,
                text_labels_for_axis=TIME,
                set_idx=set_idx,
            )

    finally:
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# ここまで追加


if __name__ == "__main__":
    main()

"""以下、実行時使用するエントリーポイント/home/shogo/coding/eval/ucn_eval/docker/entrypoint_prompts_v2.sh

#!/usr/bin/env bash
set -euo pipefail

# ===== ユーザ調整可能な環境変数 =====
: "${MAIN_PY:=/app/ucn_build_prompts.py}"                  # 実行する Python スクリプト
: "${PIP_OVERLAY_DIR:=/data/ucn_prep_cache/pip-overlay}"   # pip overlay 設置先（torch を壊さない）
: "${HF_HOME:=/root/.cache/huggingface}"                   # HF キャッシュ
: "${OPENCLIP_CACHE_DIR:=/root/.cache/huggingface/hub}"    # open_clip のHFキャッシュ
: "${VLM_LOCAL_DIR:=/data/hf_models/Qwen/Qwen3-VL-32B-Instruct}"  # Qwen3-VL-32B の固定保存先
: "${QWEN_QUANT:=4bit}"  # 4bit|8bit|none

# 依存解決なし（torch を巻き込ませない；nodeps）
: "${PIP_NODEPS:=open_clip_torch==3.2.0 bitsandbytes>=0.48.2 accelerate==1.11.0}"

# 依存解決あり（torch系は絶対入れない）
: "${PIP_INSTALL:=transformers==4.57.1 huggingface_hub==0.36.0 safetensors pandas tqdm pillow opencv-python-headless psutil matplotlib}"


export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
mkdir -p "${PIP_OVERLAY_DIR}" "${HF_HOME}"

echo "[prompts-entry-v2] torch summary (python3):"
python3 - <<'PY'
try:
    import torch
    print(f"  torch={torch.__version__}, torch.version.cuda={getattr(torch.version,'cuda',None)}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device0={torch.cuda.get_device_name(0)}")
    print(f"  torch.__file__={getattr(torch,'__file__',None)}")
except Exception as e:
    print("  torch import failed:", e)
PY

# ===== pip overlay 安全インストール（torchは絶対入れない） =====
pip_base=(python3 -m pip install --upgrade --no-cache-dir --progress-bar off --target "${PIP_OVERLAY_DIR}")

if [[ -n "${PIP_NODEPS}" ]]; then
  echo "[prompts-entry-v2] nodeps install: ${PIP_NODEPS}"
  "${pip_base[@]}" --no-deps ${PIP_NODEPS}
fi
if [[ -n "${PIP_INSTALL}" ]]; then
  echo "[prompts-entry-v2] normal install: ${PIP_INSTALL}"
  "${pip_base[@]}" ${PIP_INSTALL}
fi

export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"
export HF_HOME OPENCLIP_CACHE_DIR QWEN_QUANT

# ===== 事前ダウンロード（オンライン時のみ） =====
if [[ "${TRANSFORMERS_OFFLINE:-0}" != "1" ]]; then
  echo "[prompts-entry-v2] snapshot_download for Qwen (HF) & cache open_clip SigLIP (webli)"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

VLM_LOCAL_DIR = os.environ.get("VLM_LOCAL_DIR","/data/hf_models/Qwen/Qwen3-VL-32B-Instruct")
os.makedirs(VLM_LOCAL_DIR, exist_ok=True)

def ok_dir(p):
    try:
        return any(n.endswith(".safetensors") for n in os.listdir(p))
    except Exception:
        return False

# Qwen 本体
if not ok_dir(VLM_LOCAL_DIR):
    print(f"[snap] downloading Qwen -> {VLM_LOCAL_DIR}")
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-32B-Instruct",
        local_dir=VLM_LOCAL_DIR,
        local_dir_use_symlinks=False,
        max_workers=16
    )
else:
    print(f"[snap] Qwen exists: {VLM_LOCAL_DIR}")
PY

  # open_clip 側の SigLIP (ViT-SO400M-14, 384, webli) をキャッシュ
  python3 - <<'PY'
import os
import open_clip
cache_dir = os.environ.get("OPENCLIP_CACHE_DIR","/root/.cache/huggingface/hub")
print(f"[snap] caching open_clip SigLIP weights (webli) into {cache_dir} …")
_ = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP-384',
        pretrained='webli',
        cache_dir=cache_dir,
    )
print("[snap] open_clip SigLIP cached")
PY
fi

# 以降は（基本）オフラインでもOK
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 主要モジュールの読み取り確認
python3 - <<'PY'
import importlib, os
mods = ["transformers","huggingface_hub","safetensors","open_clip","bitsandbytes","tokenizers","cv2"]
for m in mods:
    try:
        mod = importlib.import_module(m if m!="cv2" else "cv2")
        v = getattr(mod, "__version__", "OK")
        p = getattr(mod, "__file__", "builtin")
        print(f"[prompts-entry-v2] import {m}: OK ({v}) path={p}")
    except Exception as e:
        print(f"[prompts-entry-v2] import {m}: MISSING ({e})")
PY

# 実行（python3 固定）
exec python3 -u "${MAIN_PY}" "$@"

"""

