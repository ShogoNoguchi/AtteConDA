#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/data/coding/DGInStyle/src/tools/infer_waymo_unicontrol_offline.py

WaymoV2(Front) × DGInStyle (SD1.5 + ControlNet(semseg)) オフライン推論（評価互換版）

このスクリプトは、あなたの既存 Uni-ControlNet 推論スクリプト
  /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py
の「実験管理・決定論seed・ログ・per-image meta・eval側との互換（ファイル命名規約）」を
DGInStyle 用に置き換えたもの。

【DGInStyle側の前提】
- 公式 repo は demo.py で HuggingFace の重みを利用する（ControlNet_UNet-S など）
- pipeline_refine.py の StableDiffusionControlNetRefinePipeline を使って推論する
  (DGInStyle repo の controlnet/ ディレクトリ内)

【入出力（Waymo規約・評価互換）】
X(RGB) : {image_root}/{split}/front/{segment}/{stem}.jpg
Semseg(NPY, trainId 0..18) : {semseg_root}/{split}/front/{segment}/{stem}_predTrainId.npy
Prompt(CSV, Qwen) : {prompt_root}/waymo_{split}.csv  (image_path, eval_prompt)
F(X)(PNG) : {out_root}/{split}/front/{segment}/{stem}_ucn.png   ※評価互換
F(X) meta(JSON) : 同 dir {stem}_ucn.meta.json                  ※評価互換

【実験メタ（評価キャッシュと分離）】
{experiments_root}/InferenceOutputing/experiments/
  - {EX}.infer.json
  - {EX}.infer.sh
  - experiments.index.json
  - {EX}.infer.log（ログのコピー）

【決定論seed】
--seed -1 のとき:
  seed = sha256(f"{experiment_id}::{abs_image_path}") から導出（Uni版と同思想）
--seed >= 0 のとき:
  その固定seedを全画像に適用

【DG-config / DG-ckpt の扱い】
- --DG-config : ここで読む“簡易YAML” (key: value のみ)。PyYAML依存なし。
- --DG-ckpt   : 互換のため引数として保持（基本は config の controlnet_id を使う）。
               もし --DG-ckpt が指定され、かつ文字列が空でなければ controlnet_id を上書きする。
               例: --DG-ckpt yurujaja/DGInStyle

【備考】
- Rare Class Sampling / Crop は一切しない（ユーザ指定）
- 条件は semseg のみ（DGInStyle の前提に合わせる）
"""
# ============================================================
# [CRITICAL FIX] Completely disable xformers visibility
# ============================================================
# ============================================================
# [CRITICAL FIX] Safe xformers patch (no CUDA / no C++ usage)
# ============================================================
# ============================================================
# [CRITICAL FIX] xformers FULL stub with valid __spec__
# ============================================================
import sys
import types
import importlib.machinery

def _install_xformers_stub_full():
    """
    Make diffusers believe that xformers exists,
    while disabling all CUDA/C++ functionality safely.

    Key point:
    - __spec__ MUST be a valid ModuleSpec
    - otherwise importlib.util.find_spec() raises ValueError
    """

    # --- create xformers module ---
    xmod = types.ModuleType("xformers")
    xmod.__file__ = "<xformers-stub>"
    xmod.__path__ = []  # mark as package
    xmod.__package__ = "xformers"
    xmod.__spec__ = importlib.machinery.ModuleSpec(
        name="xformers",
        loader=None,
        is_package=True,
    )

    # --- create xformers.ops submodule ---
    ops = types.ModuleType("xformers.ops")
    ops.__file__ = "<xformers.ops-stub>"
    ops.__package__ = "xformers.ops"
    ops.__spec__ = importlib.machinery.ModuleSpec(
        name="xformers.ops",
        loader=None,
        is_package=False,
    )

    def _disabled(*args, **kwargs):
        raise RuntimeError(
            "xformers is DISABLED in DGInStyle inference. "
            "Reason: broken CUDA/C++ build for this environment."
        )

    # diffusers が触る可能性のある関数名を明示的に定義
    ops.memory_efficient_attention = _disabled
    ops.__all__ = ["memory_efficient_attention"]

    # attach
    xmod.ops = ops

    # register into sys.modules
    sys.modules["xformers"] = xmod
    sys.modules["xformers.ops"] = ops

_install_xformers_stub_full()


import os
import sys
import argparse
import json
import logging
from logging import handlers
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import time
import traceback
import csv
import shutil
import shlex
import random

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch

# ==============
# 既定パス（翔伍さん資産に合わせる）
# ==============
DEFAULT_IMAGE_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_SEMSEG_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_PROMPT_ROOT = "/data/syndiff_prompts/prompts_eval_waymo"  # Qwen CSV
DEFAULT_OUT_ROOT = "/data/coding/datasets/WaymoV2/DGInstyle_Pure"
DEFAULT_EXPERIMENTS_ROOT = "/data/ucn_infer_cache"

DEFAULT_SPLITS = ["training", "validation", "testing"]
DEFAULT_CAMERA = "front"
DEFAULT_DG_CONFIG = "/data/coding/DGInStyle/configs/dginstyle_sd15_semseg.yaml"
DEFAULT_DG_CKPT = ""  # optional override (HF id string, local folder path etc.)

_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ===========================
# ロガー
# ===========================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "infer_dginstyle_offline.log")

    logger = logging.getLogger("infer_dginstyle_offline")
    logger.setLevel(logging.DEBUG)

    # 既存ハンドラ排除
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.handlers.clear()
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    )

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def _log_env(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("=== DGInStyle Offline Inference (WaymoV2) ===")
    logger.info("image-root : %s", args.image_root)
    logger.info("semseg-root : %s", args.semseg_root)
    logger.info("prompt-root(CSV): %s", args.prompt_root)
    logger.info("out-root : %s", args.out_root)
    logger.info("experiments-root: %s", args.experiments_root)
    logger.info("splits : %s", " ".join(args.splits))
    logger.info("camera : %s", args.camera)
    logger.info("DG-config : %s", args.DG_config)
    logger.info("DG-ckpt(override) : %s", args.DG_ckpt)
    logger.info("image-resolution: %d", args.image_resolution)
    logger.info("num-samples : %d", args.num_samples)
    logger.info("ddim-steps : %d", args.ddim_steps)
    logger.info("scale(cfg) : %.3f", args.scale)
    logger.info("strength : %.3f", args.strength)
    logger.info("seed : %d (-1→決定論 per-image)", args.seed)
    logger.info("overwrite : %s", args.overwrite)
    logger.info("limit : %d", args.limit)
    logger.info("verbose : %s", args.verbose)
    try:
        logger.info(
            "torch: %s | build_cuda: %s | cuda_available: %s",
            torch.__version__,
            getattr(torch.version, "cuda", None),
            torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
            if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
                logger.warning(
                    "警告: この PyTorch ビルドの CUDA 表示は %s です（12.8 以外）。既存環境で続行します。",
                    getattr(torch.version, "cuda", None),
                )
    except Exception:
        pass


# ===========================
# 実験メタ管理（Uni版互換）
# ===========================
class ExperimentsManager:
    """
    推論実験メタを experiments_root/InferenceOutputing/experiments に保存・索引
    """
    def __init__(self, root: str):
        self.root = os.path.join(root, "InferenceOutputing")
        self.dir_exp = os.path.join(self.root, "experiments")
        _ensure_dir(self.dir_exp)

    def exp_json_path(self, experiment_id: str) -> str:
        safe = experiment_id.replace("/", "_")
        return os.path.join(self.dir_exp, f"{safe}.infer.json")

    def exp_sh_path(self, experiment_id: str) -> str:
        safe = experiment_id.replace("/", "_")
        return os.path.join(self.dir_exp, f"{safe}.infer.sh")

    def index_path(self) -> str:
        return os.path.join(self.dir_exp, "experiments.index.json")

    def write_record(self, experiment_id: str, record: Dict[str, Any]) -> None:
        with open(self.exp_json_path(experiment_id), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def update_index(self, experiment_id: str, summary: Dict[str, Any]) -> None:
        idx_path = self.index_path()
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}
        index[experiment_id] = summary
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def write_repro_sh(self, experiment_id: str, cmdline: str, workdir: Optional[str] = None) -> None:
        sh_path = self.exp_sh_path(experiment_id)
        _ensure_dir(os.path.dirname(sh_path))
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
        ]
        if workdir:
            lines += [f"cd {workdir}"]
        lines += [cmdline]
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass


# ===========================
# データ列挙
# ===========================
def _ext_lower(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def _list_waymo_images(image_root: str, split: str, camera: str) -> List[str]:
    base = os.path.join(image_root, split, camera)
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if _ext_lower(f) in _ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    return sorted(out)

def _rel_dir_and_stem(image_path: str, split_root: str) -> Tuple[str, str]:
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)  # front/{segment}
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

def _path_semseg_npy(semseg_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(semseg_root, split, rel_dir, f"{stem}_predTrainId.npy")

def _out_paths(out_root: str, split: str, rel_dir: str, stem: str) -> Tuple[str, str]:
    out_dir = os.path.join(out_root, split, rel_dir)
    _ensure_dir(out_dir)
    return (
        os.path.join(out_dir, f"{stem}_ucn.png"),
        os.path.join(out_dir, f"{stem}_ucn.meta.json"),
    )


# ===========================
# プロンプトCSV（Uni版互換）
# ===========================
def _load_prompt_table_for_split(prompt_root: str, split: str, logger: logging.Logger) -> Dict[str, str]:
    table: Dict[str, str] = {}
    csv_path = os.path.join(prompt_root, f"waymo_{split}.csv")
    if not os.path.exists(csv_path):
        logger.warning("[prompt] CSV not found for split=%s: %s (→ fallback prompt 使用)", split, csv_path)
        return table

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "image_path" not in fieldnames or "eval_prompt" not in fieldnames:
                logger.error("[prompt] CSV missing required columns image_path/eval_prompt: %s", csv_path)
                return {}
            for row in reader:
                img = (row.get("image_path") or "").strip()
                pr = (row.get("eval_prompt") or "").strip()
                if img and pr:
                    table[img] = pr
    except Exception as e:
        logger.error("[prompt] failed to read CSV for split=%s: %s (path=%s)", split, repr(e), csv_path)
        return {}

    logger.info("[prompt] split=%s : loaded %d prompts from %s", split, len(table), csv_path)
    return table

def _pickup_prompt(px_abs: str, split: str, tbl: Dict[str, str], image_root: str) -> Tuple[str, str, List[str]]:
    """
    優先順：
    1) CSV の image_path が X の絶対パスに一致
    2) CSV の image_path が X の image_root 相対パスに一致
    3) フォールバック
    """
    cands: List[str] = []
    if px_abs in tbl:
        return tbl[px_abs], "csv:image_path(abs)", [px_abs]

    rel_from_root = os.path.relpath(px_abs, os.path.join(image_root, split))
    if rel_from_root in tbl:
        return tbl[rel_from_root], "csv:relpath", [rel_from_root]

    fb = "a high-quality driving scene with changed weather/time conditions"
    return fb, "fallback:default", cands


# ===========================
# 乱数（seed）ユーティリティ（Uni版互換）
# ===========================
def _derive_seed(global_seed: int, key: str) -> int:
    if global_seed >= 0:
        return int(global_seed)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# 簡易YAML（key: value）パーサ（PyYAML不要）
# ===========================
def _parse_simple_yaml(path: str) -> Dict[str, str]:
    """
    対応:
      key: value
    非対応:
      ネスト、配列、複雑なクォートなど（必要なら拡張する）
    """
    cfg: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # クォート除去（単純対応）
            if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                v = v[1:-1]
            cfg[k] = v
    return cfg

def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


# ===========================
# Semseg NPY → one-hot tensor
# ===========================
def _load_semseg_trainid_npy(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    return arr.astype(np.int32)

def _resize_trainid(seg: np.ndarray, size: int) -> np.ndarray:
    # label は最近傍でリサイズ（整数ラベルを壊さない）
    return cv2.resize(seg.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)

def _to_one_hot(seg: np.ndarray, num_classes: int = 19) -> torch.Tensor:
    """
    Cityscapes trainId (0..18) → DGInStyle ControlNet input

    出力:
        torch.FloatTensor shape = [1, 20, H, W]

    内訳:
        - ch 0..18 : Cityscapes trainId one-hot
        - ch 19    : DGInStyle 用 reserved / ignore channel（常に 0）

    NOTE:
        DGInStyle ControlNet は conditioning_channels=20 で学習済み。
        推論時も必ず 20ch を渡す必要がある。
    """
    h, w = seg.shape[:2]

    # --- 19ch one-hot ---
    one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
    valid = (seg >= 0) & (seg < num_classes)
    for c in range(num_classes):
        one_hot[c][valid & (seg == c)] = 1.0

    one_hot = torch.from_numpy(one_hot)  # [19, H, W]

    # --- 1ch ignore / reserved (all zeros) ---
    ignore_ch = torch.zeros((1, h, w), dtype=torch.float32)

    # --- concat → [20, H, W] ---
    out = torch.cat([one_hot, ignore_ch], dim=0)

    # --- batch dimension ---
    out = out.unsqueeze(0)  # [1, 20, H, W]

    return out



# ===========================
# DGInStyle pipeline ロード
# ===========================
def _import_dginstyle_modules(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

def _load_pipe_from_config(
    cfg: Dict[str, str],
    logger: logging.Logger,
) -> Any:
    """
    DGInStyle の ControlNet + SD1.5 pipeline をロードする。
    """
    # repo root を sys.path に入れる（controlnet/ を import できるように）
    repo_root = str(Path(__file__).resolve().parents[2])  # /data/coding/DGInStyle
    _import_dginstyle_modules(repo_root)

    # DGInStyle repo 内実装を import
    from diffusers import DDIMScheduler
    from controlnet.controlnet_model import ControlNetModel
    from controlnet.pipeline_refine import StableDiffusionControlNetRefinePipeline

    controlnet_id = cfg.get("controlnet_id", "yurujaja/DGInStyle")
    controlnet_subfolder = cfg.get("controlnet_subfolder", "ControlNet_UNet-S")
    sd_base_id = cfg.get("sd_base_id", "runwayml/stable-diffusion-v1-5")
    revision = cfg.get("hf_revision", "") or None

    dtype_s = (cfg.get("torch_dtype", "fp16") or "fp16").strip().lower()
    if dtype_s == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    use_cpu_offload = _str2bool(cfg.get("use_cpu_offload", "false"))
    enable_xformers = _str2bool(cfg.get("enable_xformers", "false"))

    # ControlNet
    logger.info("[DG] loading ControlNet: id=%s | subfolder=%s | revision=%s | dtype=%s",
                controlnet_id, controlnet_subfolder, str(revision), str(torch_dtype))
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        subfolder=controlnet_subfolder,
        revision=revision,
    )

    # Pipeline
    logger.info("[DG] loading SD pipeline: base=%s | dtype=%s", sd_base_id, str(torch_dtype))
    pipe = StableDiffusionControlNetRefinePipeline.from_pretrained(
        sd_base_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        revision=revision,
    )
    # scheduler: DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # optional optimizations
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("[DG] xformers memory efficient attention: ENABLED")
        except Exception as e:
            logger.warning("[DG] xformers enable failed: %s (→ disabled)", repr(e))

    # Device
    if use_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            logger.info("[DG] accelerate CPU offload: ENABLED")
        except Exception as e:
            logger.warning("[DG] enable_model_cpu_offload failed: %s (→ fallback to pipe.to(cuda))", repr(e))
            pipe.to("cuda")
    else:
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None  # 念のため（pipeline_refine 側でも None にしている）
    return pipe


# ===========================
# CLI
# ===========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DGInStyle Offline Inference with WaymoV2 (Semseg-only + Qwen prompts) [eval compatible]"
    )
    ap.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--prompt-root", type=str, default=DEFAULT_PROMPT_ROOT,
                    help="Waymo 用プロンプト CSV (waymo_{split}.csv) 群のルート")
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT,
                    help="生成画像と per-image META の保存先（評価互換命名）")
    ap.add_argument("--experiments-root", type=str, default=DEFAULT_EXPERIMENTS_ROOT,
                    help="推論実験メタの保存ルート（この直下に InferenceOutputing/experiments を作成）")

    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)

    # DGInStyle 用
    ap.add_argument("--DG-config", dest="DG_config", type=str, default=DEFAULT_DG_CONFIG)
    ap.add_argument("--DG-ckpt", dest="DG_ckpt", type=str, default=DEFAULT_DG_CKPT,
                    help="互換のための上書き引数（基本はDG-configのcontrolnet_idを使用。非空ならcontrolnet_idを上書き）")

    # generation params（Uni版と同名に寄せる）
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--scale", type=float, default=7.5, help="CFG guidance scale")
    ap.add_argument("--strength", type=float, default=1.0, help="DG pipeline strength (0..1)")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--a-prompt", type=str, default="best quality, extremely detailed")
    ap.add_argument("--n-prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")

    # 実験管理
    ap.add_argument("--experiment-id", type=str, default="EX9")
    ap.add_argument("--experiment-note", type=str, default="")

    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1, help="各 split の先頭N枚のみ。-1 で全件")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


# ===========================
# メイン
# ===========================
def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_root)
    logger = _setup_logger(args.out_root, verbose=args.verbose)
    _log_env(logger, args)

    # CUDA 必須
    if not torch.cuda.is_available():
        logger.error("❌ GPU(CUDA) が見つかりません。RTX5090 + CUDA12.8 + torch+cu128 環境を確認してください。")
        sys.exit(2)

    # DG-config 読み込み
    if not os.path.isfile(args.DG_config):
        logger.error("DG-config not found: %s", args.DG_config)
        sys.exit(1)
    dg_cfg = _parse_simple_yaml(args.DG_config)

    # --DG-ckpt が非空なら controlnet_id を上書き（互換用）
    if isinstance(args.DG_ckpt, str) and len(args.DG_ckpt.strip()) > 0:
        dg_cfg["controlnet_id"] = args.DG_ckpt.strip()

    # optional: rescale_factor / multi_diff_stride
    rescale_factor = int(dg_cfg.get("rescale_factor", "1") or "1")
    multi_diff_stride = int(dg_cfg.get("multi_diff_stride", "64") or "64")

    # pipeline build
    try:
        pipe = _load_pipe_from_config(dg_cfg, logger)
    except Exception as e:
        logger.error("❌ DGInStyle pipeline load failed: %s", repr(e))
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    # 実験メタ管理
    exman = ExperimentsManager(args.experiments_root)
    t0 = int(time.time())
    per_split_counts: Dict[str, Dict[str, int]] = {}
    per_split_fail: Dict[str, int] = {}
    per_split_csvrows: Dict[str, int] = {}

    # prompt tables
    prompt_tables: Dict[str, Dict[str, str]] = {}
    for sp in args.splits:
        tbl = _load_prompt_table_for_split(args.prompt_root, sp, logger)
        prompt_tables[sp] = tbl
        per_split_csvrows[sp] = len(tbl)

    total_gen = 0

    for split in args.splits:
        split_root = os.path.join(args.image_root, split)
        items = _list_waymo_images(args.image_root, split, args.camera)
        if args.limit > 0:
            items = items[:args.limit]
        if not items:
            logger.warning("[%s] 画像が見つかりません。", split)
            continue

        logger.info("[%s] 入力画像: %d 枚", split, len(items))
        gen_cnt = 0
        fail_cnt = 0

        tbl = prompt_tables.get(split, {})

        for px in tqdm(items, desc=f"{split}-infer"):
            rel_dir, stem = _rel_dir_and_stem(px, split_root)
            out_png, out_meta = _out_paths(args.out_root, split, rel_dir, stem)

            if (not args.overwrite) and os.path.exists(out_png):
                continue

            # semseg
            seg_path = _path_semseg_npy(args.semseg_root, split, rel_dir, stem)
            seg = _load_semseg_trainid_npy(seg_path)
            if seg is None:
                fail_cnt += 1
                logger.warning("[%s][%s] semseg missing/invalid: %s", split, stem, seg_path)
                continue

            # resize to resolution
            size = int(args.image_resolution)
            seg_rs = _resize_trainid(seg, size=size)
            cond = _to_one_hot(seg_rs, num_classes=19).to("cuda", dtype=torch.float32)

            # prompt
            prompt, psrc, pcands = _pickup_prompt(px, split, tbl, args.image_root)
            # DGInStyle では prompt に a_prompt を後ろ付け（Uniと同思想）
            prompt_full = f"{prompt}, {args.a_prompt}".strip()

            # per-image deterministic seed
            seed_key = f"{args.experiment_id}::{px}"
            seed_used = _derive_seed(args.seed, seed_key)
            _seed_everything(seed_used)
            generator = torch.Generator(device="cuda").manual_seed(int(seed_used))

            try:
                out = pipe(
                    prompt=prompt_full,
                    cond_image=cond,
                    num_inference_steps=int(args.ddim_steps),
                    guidance_scale=float(args.scale),
                    negative_prompt=str(args.n_prompt),
                    num_images_per_prompt=int(args.num_samples),
                    eta=float(args.eta),
                    generator=generator,
                    output_type="pil",
                    strength=float(args.strength),
                    rescale_factor=int(rescale_factor),
                    multi_diff_stride=int(multi_diff_stride),
                )
                images = out.images  # List[PIL.Image.Image]
            except Exception as e:
                fail_cnt += 1
                logger.error("[%s][%s] 推論失敗: %s", split, stem, repr(e))
                if args.verbose:
                    traceback.print_exc()
                continue

            # save
            if int(args.num_samples) == 1:
                images[0].save(out_png)
                out_png_primary = out_png
            else:
                out_dir = os.path.dirname(out_png)
                out_png_primary = os.path.join(out_dir, f"{stem}_ucn-000.png")
                for k, im in enumerate(images):
                    p = os.path.join(out_dir, f"{stem}_ucn-{k:03d}.png")
                    im.save(p)

            # meta（eval側が prompt_used を読むので必ず入れる）
            meta_obj = {
                "experiment_id": args.experiment_id,
                "experiment_note": args.experiment_note,
                "timestamp": int(time.time()),
                "image_path_x": px,
                "rel_dir": rel_dir,
                "stem": stem,
                "out_png": out_png_primary,
                "prompt_used": prompt,
                "prompt_source": psrc,
                "prompt_candidates": pcands,
                "dg_config_path": args.DG_config,
                "dg_config": dg_cfg,
                "dg_controlnet_override": args.DG_ckpt,
                "image_resolution": int(args.image_resolution),
                "num_samples": int(args.num_samples),
                "ddim_steps": int(args.ddim_steps),
                "scale": float(args.scale),
                "strength": float(args.strength),
                "eta": float(args.eta),
                "seed_mode": ("per-image-deterministic" if args.seed < 0 else "fixed"),
                "seed_used": int(seed_used),
                "rescale_factor": int(rescale_factor),
                "multi_diff_stride": int(multi_diff_stride),
            }
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, ensure_ascii=False, indent=2)

            gen_cnt += 1
            per_split_counts[split] = {"inputs": len(items), "generated": gen_cnt}
            per_split_fail[split] = fail_cnt
            total_gen += 1

        logger.info("[%s] 完了: generated=%d / inputs=%d (fail=%d)", split, gen_cnt, len(items), fail_cnt)

    # 実験サマリ JSON
    def _cmdline_str() -> str:
        # "python -u <file> <args...>" で再現できる形
        parts = ["python", "-u", str(Path(__file__).resolve())] + sys.argv[1:]
        return " ".join(shlex.quote(p) for p in parts)

    record = {
        "kind": "dginstyle_inference",
        "experiment_id": args.experiment_id,
        "experiment_note": args.experiment_note,
        "timestamp": t0,
        "duration_sec": int(max(0, time.time() - t0)),
        "image_root": args.image_root,
        "semseg_root": args.semseg_root,
        "prompt_root": args.prompt_root,
        "out_root": args.out_root,
        "dg_config_path": args.DG_config,
        "dg_config": dg_cfg,
        "dg_ckpt_override": args.DG_ckpt,
        "image_resolution": int(args.image_resolution),
        "num_samples": int(args.num_samples),
        "ddim_steps": int(args.ddim_steps),
        "scale": float(args.scale),
        "strength": float(args.strength),
        "eta": float(args.eta),
        "seed": int(args.seed),
        "splits": args.splits,
        "camera": args.camera,
        "counts_per_split": per_split_counts,
        "failures_per_split": per_split_fail,
        "total_generated": int(total_gen),
        "csv_rows_per_split": per_split_csvrows,
        "cmdline": _cmdline_str(),
    }
    exman.write_record(args.experiment_id, record)

    exman.update_index(
        args.experiment_id,
        {
            "experiment_id": args.experiment_id,
            "timestamp": t0,
            "out_root": args.out_root,
            "prompt_root": args.prompt_root,
            "dg_config_path": args.DG_config,
            "dg_controlnet_id": dg_cfg.get("controlnet_id", ""),
            "splits": args.splits,
            "camera": args.camera,
            "total_generated": int(total_gen),
        },
    )

    # 再現シェル（.sh）
    workdir = str(Path(__file__).resolve().parents[2])  # /data/coding/DGInStyle
    exman.write_repro_sh(args.experiment_id, record["cmdline"], workdir=workdir)

    # out_root のログを experiments 側にもコピー
    try:
        src_log = os.path.join(args.out_root, "infer_dginstyle_offline.log")
        if os.path.exists(src_log):
            dst_log = os.path.join(exman.dir_exp, f"{args.experiment_id}.infer.log")
            shutil.copy2(src_log, dst_log)
    except Exception:
        pass

    logger.info("✅ 推論完了: total_generated=%d | experiments_root=%s", total_gen, exman.root)
    logger.info(" 実験JSON: %s", exman.exp_json_path(args.experiment_id))
    logger.info(" 再現SH : %s", exman.exp_sh_path(args.experiment_id))
    logger.info(" index : %s", exman.index_path())


if __name__ == "__main__":
    main()
