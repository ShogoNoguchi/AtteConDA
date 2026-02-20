# /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py
# -*- coding: utf-8 -*-
"""
WaymoV2(Front) × Uni-ControlNet (SD1.5, uni_v15) オフライン多条件(3条件)推論
- ローカル条件: Canny / Depth / Semseg（3つのみを用いる）
- 使わない条件(MLSD/HED/Sketch/OpenPose)のチャネルはゼロ埋め
- グローバル条件(content/CLIP)は未使用 → 0ベクトル

【本ファイルの拡張（実験管理）】
- --experiment-id / --experiment-note / --experiments-root を追加
- 画像出力（F(X)）は --out-root に保存（従来どおり）
- 実験メタは --experiments-root/InferenceOutputing/experiments へ保存（評価側キャッシュと厳密分離）
  - {EX}.infer.json               : 実験サマリ（人間・機械可読）
  - experiments.index.json        : 全実験のインデックス
  - {EX}.infer.sh                 : 再現用の完全実行コマンド（.sh）
- 各画像の {stem}_ucn.meta.json に experiment_id など詳細を追記

【入出力（Waymo規約）】
X(RGB)            : /home/shogo/coding/datasets/WaymoV2/extracted/{split}/front/{segment}/{stem}.jpg
Canny(PNG)        : /home/shogo/coding/datasets/WaymoV2/CannyEdge/{split}/front/{segment}/{stem}_edge.png
Depth(IMG,PNG)    : /home/shogo/coding/datasets/WaymoV2/Metricv2DepthIMG/{split}/front/{segment}/{stem}_depth.png
Semseg(NPY)       : /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes/{split}/front/{segment}/{stem}_predTrainId.npy
Prompt(CSV, Qwen) : /data/syndiff_prompts/prompts_eval_waymo/waymo_{split}.csv
F(X)(PNG)         : --out-root/{split}/front/{segment}/{stem}_ucn.png
F(X) meta(JSON)   : 同ディレクトリ {stem}_ucn.meta.json

【実行例（EX3: base公開ckpt + Qwenプロンプト, 512px, 1サンプル）】
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/uni_v15.yaml \
  --uni-ckpt /data/coding/Uni-ControlNet/ckpt/uni.ckpt \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/Ucn_fromBasePublicCkpt \
  --experiments-root /data/ucn_infer_cache \
  --experiment-id EX3 \
  --experiment-note "Uni public base ckpt + Qwen prompts (ablation of finetune)" \
  --overwrite --verbose
"""

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

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from pytorch_lightning import seed_everything

# ==== リポジトリ内部（既存） ====
if "./" not in sys.path:
    sys.path.append("./")
import utils.config as config  # save_memory フラグを利用
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from annotator.util import HWC3  # 3ch整形ユーティリティ（依存はこれのみ）

# ===========================
# 既定パス（翔伍さんの資産）
# ===========================
DEFAULT_IMAGE_ROOT     = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_CANNY_ROOT     = "/home/shogo/coding/datasets/WaymoV2/CannyEdge"
DEFAULT_DEPTH_IMGROOT  = "/home/shogo/coding/datasets/WaymoV2/Metricv2DepthIMG"
DEFAULT_SEMSEG_ROOT    = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_PROMPT_ROOT    = "/data/syndiff_prompts/prompts_eval_waymo"  # Qwen CSV
DEFAULT_OUT_ROOT       = "/data/coding/datasets/WaymoV2/Ucn_byPure_Finetune"  # 既定（EX2互換）
DEFAULT_SPLITS         = ["training", "validation", "testing"]
DEFAULT_CAMERA         = "front"
DEFAULT_UNI_CONFIG     = "/data/coding/Uni-ControlNet/configs/uni_v15.yaml"
DEFAULT_UNI_CKPT       = "/data/coding/ckpts/version_4/checkpoints/epoch=0-step=30000.ckpt"  # EX2互換

# 実験メタの保存ルート（評価キャッシュと明確に分離）
DEFAULT_EXPERIMENTS_ROOT = "/data/ucn_infer_cache"  # この直下に InferenceOutputing/experiments を切る

# Cityscapes trainId 0..18 のパレット
_CITYSCAPES_TRAINID_COLORS: List[Tuple[int, int, int]] = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ===========================
# ロガー
# ===========================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "infer_unicontrol_offline.log")

    logger = logging.getLogger("infer_unicontrol_offline")
    logger.setLevel(logging.DEBUG)
    # 既存ハンドラを一掃（異種フォーマッタ混入防止）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = handlers.RotatingFileHandler(log_path, maxBytes=20 * 1024 * 1024,
                                      backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def _log_env(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("=== Uni-ControlNet Offline Inference (WaymoV2) ===")
    logger.info("image-root      : %s", args.image_root)
    logger.info("canny-root      : %s", args.canny_root)
    logger.info("depth-img-root  : %s", args.depth_img_root)
    logger.info("semseg-root     : %s", args.semseg_root)
    logger.info("prompt-root(CSV): %s", args.prompt_root)
    logger.info("out-root        : %s", args.out_root)
    logger.info("experiments-root: %s", args.experiments_root)
    logger.info("splits          : %s", " ".join(args.splits))
    logger.info("camera          : %s", args.camera)
    logger.info("uni-config      : %s", args.uni_config)
    logger.info("uni-ckpt        : %s", args.uni_ckpt)
    logger.info("image-resolution: %d", args.image_resolution)
    logger.info("num-samples     : %d", args.num_samples)
    logger.info("ddim-steps      : %d", args.ddim_steps)
    logger.info("scale(cfg)      : %.3f", args.scale)
    logger.info("strength(local) : %.3f", args.strength)
    logger.info("global-strength : %.3f (content=unused→zeros)", args.global_strength)
    logger.info("seed            : %d (-1→決定論 per-image)", args.seed)
    logger.info("overwrite       : %s", args.overwrite)
    logger.info("limit           : %d", args.limit)
    logger.info("verbose         : %s", args.verbose)
    try:
        logger.info("torch: %s | build_cuda: %s | cuda_available: %s",
                    torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
            if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
                logger.warning("警告: この PyTorch ビルドの CUDA 表示は %s です（12.8 以外）。既存環境で続行します。",
                               getattr(torch.version, "cuda", None))
    except Exception:
        pass

# ===========================
# 実験メタ管理
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
        # そのまま再現できるように python -u + 引数で保存
        lines += [cmdline]
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass

# ===========================
# データ列挙・小物ユーティリティ
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
    # rel_dir: front/{segment} / stem: e.g., 1510593619939663_first
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

def _timestamp_from_stem(stem: str) -> str:
    return stem.split("_")[0] if "_" in stem else stem

def _path_canny(canny_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(canny_root, split, rel_dir, f"{stem}_edge.png")

def _path_depth_png(depth_img_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(depth_img_root, split, rel_dir, f"{stem}_depth.png")

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
# 前処理（オフライン条件 → 3ch画像）
# ===========================
def _resize_hw(img: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

def _load_png_as_hwc3(path: str, W: int, H: int) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    im = _resize_hw(im, W, H)
    return HWC3(im)

def _build_palette_img(index2d: np.ndarray) -> np.ndarray:
    h, w = index2d.shape[:2]
    pal_img = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, (r, g, b) in enumerate(_CITYSCAPES_TRAINID_COLORS):
        pal_img[index2d == tid] = (b, g, r)  # BGR（OpenCV保存向け）
    return pal_img

def _load_semseg_as_hwc3(semseg_npy: str, W: int, H: int) -> Optional[np.ndarray]:
    if not os.path.exists(semseg_npy):
        return None
    seg = np.load(semseg_npy)
    if seg.ndim != 2:
        return None
    pal = _build_palette_img(seg.astype(np.uint8))
    pal = _resize_hw(pal, W, H)
    return HWC3(pal)

def _zero_map(W: int, H: int) -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)

# ===========================
# 乱数（シード）ユーティリティ
# ===========================
def _derive_seed(global_seed: int, key: str) -> int:
    if global_seed >= 0:
        return int(global_seed)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF

# ===========================
# Uni-ControlNet 推論本体
# ===========================
def _build_model_and_sampler(uni_config: str, uni_ckpt: str, device: str = "cuda") -> Tuple[torch.nn.Module, DDIMSampler]:
    model = create_model(uni_config).cpu()
    state = load_state_dict(uni_ckpt, location="cuda" if device.startswith("cuda") else "cpu")
    # 事前学習 ckpt と現在の CLIP 実装差分による余計な key を無視
    state.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)
    missing_unexp = model.load_state_dict(state, strict=False)
    print("load_state_dict(strict=False) ->", missing_unexp)
    if device.startswith("cuda"):
        model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def _sample_one(
    model,
    ddim_sampler: DDIMSampler,
    canny_map: Optional[np.ndarray],
    depth_map: Optional[np.ndarray],
    seg_map: Optional[np.ndarray],
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples: int,
    H: int,
    W: int,
    ddim_steps: int,
    strength: float,
    scale: float,
    seed: int,
    eta: float,
    global_strength: float,
    logger: logging.Logger,
) -> List[np.ndarray]:
    """
    1フレーム分の Uni-ControlNet サンプリング。
    local_control: [canny, mlsd, hed, sketch, openpose, midas(depth), seg] を3chずつ連結 → (H,W,21)
    global_control: 未使用なので 0 ベクトル（768）
    """
    seed_everything(seed)

    # [canny, mlsd, hed, sketch, openpose, midas(depth), seg]
    canny = canny_map if canny_map is not None else _zero_map(W, H)
    mlsd  = _zero_map(W, H)
    hed   = _zero_map(W, H)
    sketch= _zero_map(W, H)
    openp = _zero_map(W, H)
    depth = depth_map if depth_map is not None else _zero_map(W, H)
    seg   = seg_map if seg_map is not None else _zero_map(W, H)

    detected_maps_list = [canny, mlsd, hed, sketch, openp, depth, seg]
    detected_maps = np.concatenate(detected_maps_list, axis=2)  # (H, W, 21)

    with torch.no_grad():
        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)  # (B, H, W, 21)
        local_control = torch.permute(local_control, (0, 3, 1, 2)).contiguous().clone()

        global_emb = torch.zeros((num_samples, 768), dtype=torch.float32, device="cuda")

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_emb)

        cond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
            "global_control": [global_emb],
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
            "global_control": [uc_global_control],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Uni-ControlNet 論文に倣い control_scales を 13 層ぶん strength で揃える
        model.control_scales = [strength] * 13

        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            batch_size=num_samples,
            shape=shape,
            conditioning=cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            global_strength=global_strength,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (torch.permute(x_samples, (0, 2, 3, 1)) * 127.5 + 127.5).clamp(0, 255).byte().cpu().numpy()
        results = [x_samples[i] for i in range(num_samples)]
        return results

# ===========================
# Qwen3-VL プロンプト CSV ローダ
# ===========================
def _load_prompt_table_for_split(prompt_root: str, split: str, logger: logging.Logger) -> Dict[str, str]:
    """
    /data/syndiff_prompts/prompts_eval_waymo/waymo_{split}.csv から
    image_path → eval_prompt の辞書を構築
    """
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
                pr  = (row.get("eval_prompt") or "").strip()
                if img and pr:
                    table[img] = pr
    except Exception as e:
        logger.error("[prompt] failed to read CSV for split=%s: %s (path=%s)", split, repr(e), csv_path)
        return {}

    logger.info("[prompt] split=%s : loaded %d prompts from %s", split, len(table), csv_path)
    return table

def _pickup_prompt(px_abs: str, split: str, rel_dir: str, stem: str, tbl: Dict[str, str], image_root: str) -> Tuple[str, str, List[str]]:
    """
    優先順：
      1) CSV の image_path が X の絶対パスに一致
      2) CSV の image_path が X の image_root 相対パスに一致
      3) フォールバック（デフォルト文字列）
    """
    cands: List[str] = []
    # 1) 絶対パス一致
    if px_abs in tbl:
        return tbl[px_abs], "csv:image_path(abs)", [px_abs]

    # 2) 相対パス一致（image_root からの相対）
    rel_from_root = os.path.relpath(px_abs, os.path.join(image_root, split))
    # 例: front/SEG/151059..._first.jpg
    if rel_from_root in tbl:
        return tbl[rel_from_root], "csv:relpath", [rel_from_root]

    # 3) フォールバック
    fb = "a high-quality driving scene with changed weather/time conditions"
    return fb, "fallback:default", cands

# ===========================
# CLI
# ===========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet Offline Inference with WaymoV2 (Canny/Depth/Semseg + Qwen prompts)")
    ap.add_argument("--image-root",     type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--canny-root",     type=str, default=DEFAULT_CANNY_ROOT)
    ap.add_argument("--depth-img-root", type=str, default=DEFAULT_DEPTH_IMGROOT)
    ap.add_argument("--semseg-root",    type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--prompt-root",    type=str, default=DEFAULT_PROMPT_ROOT, help="Waymo 用プロンプト CSV (waymo_{split}.csv) 群のルート")
    ap.add_argument("--out-root",       type=str, default=DEFAULT_OUT_ROOT,    help="生成画像と per-image META の保存先")
    ap.add_argument("--experiments-root", type=str, default=DEFAULT_EXPERIMENTS_ROOT,
                    help="推論実験メタの保存ルート（この直下に InferenceOutputing/experiments を作成）")

    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--uni-config", type=str, default=DEFAULT_UNI_CONFIG)
    ap.add_argument("--uni-ckpt",   type=str, default=DEFAULT_UNI_CKPT)

    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--num-samples",      type=int, default=1)
    ap.add_argument("--ddim-steps",       type=int, default=50)
    ap.add_argument("--strength",         type=float, default=1.0)
    ap.add_argument("--global-strength",  type=float, default=0.0)
    ap.add_argument("--scale",            type=float, default=7.5)
    ap.add_argument("--seed",             type=int,   default=-1)
    ap.add_argument("--eta",              type=float, default=0.0)
    ap.add_argument("--a-prompt", type=str, default="best quality, extremely detailed")
    ap.add_argument("--n-prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")

    ap.add_argument("--limit", type=int, default=-1, help="各 split の先頭N枚のみ。-1 で全件")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose",   action="store_true")

    # ★ 実験管理
    ap.add_argument("--experiment-id",   type=str, default="EX1", help="実験ID（EX1/EX2/EX3...）")
    ap.add_argument("--experiment-note", type=str, default="",    help="実験のメモ（Ablation条件など）")
    return ap.parse_args()

# ===========================
# メイン
# ===========================
def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_root)
    logger = _setup_logger(args.out_root, verbose=args.verbose)
    _log_env(logger, args)

    # CUDA 必須チェック
    if not torch.cuda.is_available():
        logger.error("❌ GPU(CUDA) が見つかりません。RTX5090 + CUDA12.8 + torch+cu128 環境を確認してください。")
        sys.exit(2)

    # モデル構築
    if not os.path.isfile(args.uni_config):
        logger.error("Uni config not found: %s", args.uni_config); sys.exit(1)
    if not os.path.isfile(args.uni_ckpt):
        logger.error("Uni checkpoint not found: %s", args.uni_ckpt); sys.exit(1)
    model, sampler = _build_model_and_sampler(args.uni_config, args.uni_ckpt, device="cuda")

    # 実験メタ管理
    exman = ExperimentsManager(args.experiments_root)
    t0 = int(time.time())
    per_split_counts: Dict[str, Dict[str, int]] = {}  # {"training": {"inputs":N, "generated":M}, ...}
    per_split_fail:   Dict[str, int] = {}
    per_split_csvrows:Dict[str, int] = {}

    # プロンプトテーブルを split ごとにロード
    prompt_tables: Dict[str, Dict[str,str]] = {}
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

        for idx, px in enumerate(tqdm(items, desc=f"{split}-infer")):
            rel_dir, stem = _rel_dir_and_stem(px, split_root)
            out_png, out_meta = _out_paths(args.out_root, split, rel_dir, stem)

            if (not args.overwrite) and os.path.exists(out_png):
                # 既存結果はスキップ（上書きしない）
                continue

            # 条件マップ（H=W=image-resolution）
            W = H = int(args.image_resolution)

            canny = _load_png_as_hwc3(_path_canny(args.canny_root, split, rel_dir, stem), W, H)
            depth = _load_png_as_hwc3(_path_depth_png(args.depth_img_root, split, rel_dir, stem), W, H)
            seg   = _load_semseg_as_hwc3(_path_semseg_npy(args.semseg_root, split, rel_dir, stem), W, H)

            # プロンプト
            prompt, psrc, pcands = _pickup_prompt(px, split, rel_dir, stem, tbl, args.image_root)

            # シード：-1 のときは「EXID + 絶対パス」に基づく決定論シード
            seed_key = f"{args.experiment_id}::{px}"
            seed_used = _derive_seed(args.seed, seed_key)

            try:
                imgs = _sample_one(
                    model, sampler,
                    canny_map=canny, depth_map=depth, seg_map=seg,
                    prompt=prompt, a_prompt=args.a_prompt, n_prompt=args.n_prompt,
                    num_samples=args.num_samples, H=H, W=W, ddim_steps=args.ddim_steps,
                    strength=args.strength, scale=args.scale, seed=seed_used, eta=args.eta,
                    global_strength=args.global_strength, logger=logger
                )
            except Exception as e:
                fail_cnt += 1
                logger.error("[%s][%s] 推論失敗: %s", split, stem, repr(e))
                if args.verbose:
                    traceback.print_exc()
                continue

            # 保存（num-samples==1 は _ucn.png、>1 は _ucn-000.png ...）
            if args.num_samples == 1:
    # ★FIX: PIL は RGB 前提。ここで [:,:,::-1] をすると RGB→BGR になって色が壊れる。
    #       => 生成品質の主観評価・CLIP評価・GDINO評価すべてに悪影響が出るので絶対に反転しない。
                Image.fromarray(imgs[0]).save(out_png)# RGB→BGRでなく、PILはRGBなのでそのままでOKだがOpenCV変換対策で[::-1]せずにPIL保存に統一
                # PILはRGB前提。ここは cv2 使わない保存なので OK
            else:
                for k, im in enumerate(imgs):
                    stemk = f"{stem}_ucn-{k:03d}.png"
                    Image.fromarray(im).save(os.path.join(os.path.dirname(out_png), stemk))

            # META
            meta_obj = {
                "experiment_id": args.experiment_id,
                "experiment_note": args.experiment_note,
                "timestamp": int(time.time()),
                "image_path_x": px,
                "rel_dir": rel_dir,
                "stem": stem,
                "out_png": out_png if args.num_samples == 1 else os.path.join(os.path.dirname(out_png), f"{stem}_ucn-000.png"),
                "prompt_used": prompt,
                "prompt_source": psrc,
                "prompt_candidates": pcands,
                "uni_config": args.uni_config,
                "uni_ckpt": args.uni_ckpt,
                "image_resolution": int(args.image_resolution),
                "num_samples": int(args.num_samples),
                "ddim_steps": int(args.ddim_steps),
                "scale": float(args.scale),
                "strength": float(args.strength),
                "global_strength": float(args.global_strength),
                "seed_mode": ("per-image-deterministic" if args.seed < 0 else "fixed"),
                "seed_used": int(seed_used),
            }
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, ensure_ascii=False, indent=2)

            gen_cnt += 1

        per_split_counts[split] = {"inputs": len(items), "generated": gen_cnt}
        per_split_fail[split]   = fail_cnt
# ★FIX: total_gen は「今生成した1枚」を足すべき。gen_cnt(累積)を足すと三角数で爆増して壊れる。
        total_gen += 1
        logger.info("[%s] 生成 %d / 入力 %d（失敗 %d）", split, gen_cnt, len(items), fail_cnt)

    # 実験サマリ JSON
    record = {
        "kind": "unicontrol_inference",
        "experiment_id": args.experiment_id,
        "experiment_note": args.experiment_note,
        "timestamp": t0,
        "duration_sec": int(max(0, time.time() - t0)),
        "image_root": args.image_root,
        "canny_root": args.canny_root,
        "depth_img_root": args.depth_img_root,
        "semseg_root": args.semseg_root,
        "prompt_root": args.prompt_root,
        "out_root": args.out_root,
        "uni_config": args.uni_config,
        "uni_ckpt": args.uni_ckpt,
        "image_resolution": int(args.image_resolution),
        "num_samples": int(args.num_samples),
        "ddim_steps": int(args.ddim_steps),
        "scale": float(args.scale),
        "strength": float(args.strength),
        "global_strength": float(args.global_strength),
        "seed": int(args.seed),
        "splits": args.splits,
        "camera": args.camera,
        "counts_per_split": per_split_counts,
        "failures_per_split": per_split_fail,
        "total_generated": int(total_gen),
        "cmdline": "python -u " + " ".join(map(lambda s: s if " " not in s else f"\"{s}\"", [__file__] + sys.argv[1:])),
    }
    exman.write_record(args.experiment_id, record)

    # 実験インデックス（人間可読サマリ）
    exman.update_index(args.experiment_id, {
        "experiment_id": args.experiment_id,
        "timestamp": t0,
        "out_root": args.out_root,
        "prompt_root": args.prompt_root,
        "uni_ckpt": args.uni_ckpt,
        "splits": args.splits,
        "camera": args.camera,
        "total_generated": int(total_gen),
    })

    # 再現シェル（.sh）
    workdir = str(Path(__file__).resolve().parent.parent.parent)  # /data/coding/Uni-ControlNet
    exman.write_repro_sh(args.experiment_id, record["cmdline"], workdir=workdir)

    # 参考としてロガーファイルを experiments 側にもコピーしておく（任意）
    try:
        src_log = os.path.join(args.out_root, "infer_unicontrol_offline.log")
        if os.path.exists(src_log):
            dst_log = os.path.join(exman.dir_exp, f"{args.experiment_id}.infer.log")
            shutil.copy2(src_log, dst_log)
    except Exception:
        pass

    logger.info("✅ 推論完了: total_generated=%d | experiments_root=%s", total_gen, exman.root)
    logger.info("   実験JSON: %s", exman.exp_json_path(args.experiment_id))
    logger.info("   再現SH  : %s", exman.exp_sh_path(args.experiment_id))
    logger.info("   index   : %s", exman.index_path())

if __name__ == "__main__":
    main()
