# /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py
# -*- coding: utf-8 -*-
"""
WaymoV2(Front) × Uni-ControlNet (SD1.5, uni_v15) オフライン多条件(3条件)推論
- 使うローカル条件: Canny / Depth / Semseg（この3つのみ）
- 使わない条件(MLSD/HED/Sketch/OpenPose)のチャネルはゼロ埋め（公式スライドの推奨に準拠）
- グローバル条件(content/CLIP埋め込み)は未使用 → 0 ベクトル

入出力と対応付け（WaymoV2の規約に準拠）:
  画像列挙     : /home/shogo/coding/datasets/WaymoV2/extracted/{split}/front/{segment}/{stem}.jpg
  Canny(PNG)   : /home/shogo/coding/datasets/WaymoV2/CannyEdge/{split}/front/{segment}/{stem}_edge.png
  Depth(PNG)   : /home/shogo/coding/datasets/WaymoV2/Metricv2DepthIMG/{split}/front/{segment}/{stem}_depth.png
                 （近=白 / 遠=黒：MiDaS互換に合わせた可視化済みを使用）
  Semseg(NPY)  : /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes/{split}/front/{segment}/{stem}_predTrainId.npy
  Prompt(TXT)  : /home/shogo/coding/datasets/WaymoV2/Prompts_gptoss/{split}/front/{segment}/{stem}_prompt.txt
                  または {timestamp}_prompt.txt（←本スクリプトは両方探す）
  出力(PNG)    : /home/shogo/coding/datasets/WaymoV2/UniControlNet_offline/{split}/front/{segment}/{stem}_ucn.png
  出力(META)   : 上と同ディレクトリに {stem}_ucn.meta.json

実行例:
  cd /data/coding/Uni-ControlNet
  python -u src/tools/infer_waymo_unicontrol_offline.py \
    --splits training validation testing \
    --camera front \
    --image-resolution 512 \
    --num-samples 1 \
    --ddim-steps 50 \
    --scale 7.5 \
    --strength 1.0 \
    --global-strength 0.0 \
    --seed -1 \
    --overwrite \
    --verbose

注意:
- 既定の SD1.5 + uni_v15 設定/ckpt パスは公式 test.py と同一相対パス（configs/uni_v15.yaml, ckpt/uni.ckpt）。
- CUDA 12.8 + torch+cu128 の既存環境を尊重（インストール操作なし）。起動ログに詳細を記録。
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

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from pytorch_lightning import seed_everything

# Uni-ControlNet の内部 API にアクセス（公式 test.py と同様）
if "./" not in sys.path:
    sys.path.append("./")

import utils.config as config  # save_memory フラグを利用
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from annotator.util import HWC3  # 3ch整形ユーティリティ（依存はこれのみ）

# ===========================
#  既定パス（Shogo さんの資産に合わせた既定値）
# ===========================
DEFAULT_IMAGE_ROOT   = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_CANNY_ROOT   = "/home/shogo/coding/datasets/WaymoV2/CannyEdge"
DEFAULT_DEPTH_IMGROOT= "/home/shogo/coding/datasets/WaymoV2/Metricv2DepthIMG"
DEFAULT_SEMSEG_ROOT  = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_PROMPT_ROOT  = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss"
DEFAULT_OUT_ROOT     = "/home/shogo/coding/datasets/WaymoV2/UniControlNet_offline"
DEFAULT_SPLITS       = ["training", "validation", "testing"]
DEFAULT_CAMERA       = "front"

DEFAULT_UNI_CONFIG   = "./configs/uni_v15.yaml"
DEFAULT_UNI_CKPT     = "./ckpt/uni.ckpt"

# Cityscapes trainId 0..18 のパレット（OneFormer推論スクリプトで使用したものと一致）
_CITYSCAPES_TRAINID_COLORS: List[Tuple[int, int, int]] = [
    (128,  64, 128),  # 0: road
    (244,  35, 232),  # 1: sidewalk
    ( 70,  70,  70),  # 2: building
    (102, 102, 156),  # 3: wall
    (190, 153, 153),  # 4: fence
    (153, 153, 153),  # 5: pole
    (250, 170,  30),  # 6: traffic light
    (220, 220,   0),  # 7: traffic sign
    (107, 142,  35),  # 8: vegetation
    (152, 251, 152),  # 9: terrain
    ( 70, 130, 180),  # 10: sky
    (220,  20,  60),  # 11: person
    (255,   0,   0),  # 12: rider
    (  0,   0, 142),  # 13: car
    (  0,   0,  70),  # 14: truck
    (  0,  60, 100),  # 15: bus
    (  0,  80, 100),  # 16: train
    (  0,   0, 230),  # 17: motorcycle
    (119,  11,  32),  # 18: bicycle
]

_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ===========================
#  ロガー
# ===========================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "infer_unicontrol_offline.log")
    logger = logging.getLogger("infer_unicontrol_offline")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=20*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
    if not logger.handlers:
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

def _log_env(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("=== Uni-ControlNet Offline Inference (WaymoV2) ===")
    logger.info("image-root   : %s", args.image_root)
    logger.info("canny-root   : %s", args.canny_root)
    logger.info("depth-imgroot: %s", args.depth_img_root)
    logger.info("semseg-root  : %s", args.semseg_root)
    logger.info("prompt-root  : %s", args.prompt_root)
    logger.info("out-root     : %s", args.out_root)
    logger.info("splits       : %s", " ".join(args.splits))
    logger.info("camera       : %s", args.camera)
    logger.info("uni-config   : %s", args.uni_config)
    logger.info("uni-ckpt     : %s", args.uni_ckpt)
    logger.info("image-resolution: %d", args.image_resolution)
    logger.info("num-samples  : %d", args.num_samples)
    logger.info("ddim-steps   : %d", args.ddim_steps)
    logger.info("scale(cfg)   : %.3f", args.scale)
    logger.info("strength(local): %.3f", args.strength)
    logger.info("global-strength: %.3f (content=unused→zeros)", args.global_strength)
    logger.info("seed         : %d ( -1 → 各フレーム由来の決定論シード )", args.seed)
    logger.info("overwrite    : %s", args.overwrite)
    logger.info("limit        : %d", args.limit)
    logger.info("verbose      : %s", args.verbose)
    # Torch/CUDA 情報
    try:
        import torch
        logger.info("torch: %s | build_cuda: %s | cuda_available: %s",
                    torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
        if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
            logger.warning("警告: この PyTorch ビルドの CUDA 表示は %s です（12.8 以外）。既存環境でそのまま続行します。",
                           getattr(torch.version, "cuda", None))
    except Exception:
        pass


# ===========================
#  データ列挙・対応付け
# ===========================
def _ext_lower(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def _list_waymo_images(image_root: str, split: str, camera: str) -> List[str]:
    base = os.path.join(image_root, split, camera)
    out: List[str] = []
    if not os.path.isdir(base): return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if _ext_lower(f) in _ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    return sorted(out)

def _rel_dir_and_stem(image_path: str, split_root: str) -> Tuple[str, str]:
    # rel_dir: front/{segment} / stem: e.g., 1510593619939663_last
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

# --- 追加: timestamp 抽出とプロンプト候補パス（フル/タイムスタンプ） ---
def _timestamp_from_stem(stem: str) -> str:
    # '1507678826876435_first' -> '1507678826876435'
    return stem.split('_')[0] if '_' in stem else stem

def _path_prompt_txt_full(prompt_root: str, split: str, rel_dir: str, stem: str) -> str:
    # 旧来の期待（stemに first/mid10s/last を含む形）
    return os.path.join(prompt_root, split, rel_dir, f"{stem}_prompt.txt")

def _path_prompt_txt_ts(prompt_root: str, split: str, rel_dir: str, stem: str) -> str:
    # 生成スクリプトの現行規約（timestamp のみ）
    ts = _timestamp_from_stem(stem)
    return os.path.join(prompt_root, split, rel_dir, f"{ts}_prompt.txt")

def _path_canny(canny_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(canny_root, split, rel_dir, f"{stem}_edge.png")

def _path_depth_png(depth_img_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(depth_img_root, split, rel_dir, f"{stem}_depth.png")

def _path_semseg_npy(semseg_root: str, split: str, rel_dir: str, stem: str) -> str:
    return os.path.join(semseg_root, split, rel_dir, f"{stem}_predTrainId.npy")

def _out_paths(out_root: str, split: str, rel_dir: str, stem: str) -> Tuple[str, str]:
    out_dir = os.path.join(out_root, split, rel_dir)
    _ensure_dir(out_dir)
    return (os.path.join(out_dir, f"{stem}_ucn.png"),
            os.path.join(out_dir, f"{stem}_ucn.meta.json"))


# ===========================
#  前処理（オフライン条件 → 3ch 画像）
# ===========================
def _resize_hw(img: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

def _load_png_as_hwc3(path: str, W: int, H: int, force_gray_to_rgb: bool=True) -> Optional[np.ndarray]:
    if not os.path.exists(path): return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None: return None
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
    if not os.path.exists(semseg_npy): return None
    seg = np.load(semseg_npy)
    if seg.ndim != 2: return None
    pal = _build_palette_img(seg.astype(np.uint8))
    pal = _resize_hw(pal, W, H)
    return HWC3(pal)

def _zero_map(W: int, H: int) -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


# ===========================
#  乱数（シード）ユーティリティ
# ===========================
def _derive_seed(global_seed: int, key: str) -> int:
    h = hashlib.sha256((key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF


# ===========================
#  Uni-ControlNet 推論本体
# ===========================
def _build_model_and_sampler(uni_config: str, uni_ckpt: str, device: str="cuda"):
    model = create_model(uni_config).cpu()
    state = load_state_dict(uni_ckpt, location='cuda' if device.startswith("cuda") else 'cpu')
    state.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)
    missing_unexp = model.load_state_dict(state, strict=False)
    print('load_state_dict(strict=False) ->', missing_unexp)
    if device.startswith("cuda"):
        model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def _sample_one(
    model,
    ddim_sampler: DDIMSampler,
    canny_map: np.ndarray,
    depth_map: np.ndarray,
    seg_map: np.ndarray,
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples: int,
    H: int, W: int,
    ddim_steps: int,
    strength: float,
    scale: float,
    seed: int,
    eta: float,
    global_strength: float,
    logger: logging.Logger,
) -> List[np.ndarray]:

    seed_everything(seed)

    # [canny, mlsd, hed, sketch, openpose, midas(depth), seg] の順で3ch連結（未使用はゼロ）
    canny = canny_map if canny_map is not None else _zero_map(W, H)
    mlsd  = _zero_map(W, H)
    hed   = _zero_map(W, H)
    sketch= _zero_map(W, H)
    openp = _zero_map(W, H)
    depth = depth_map if depth_map is not None else _zero_map(W, H)
    seg   = seg_map   if seg_map   is not None else _zero_map(W, H)

    detected_maps_list = [canny, mlsd, hed, sketch, openp, depth, seg]
    detected_maps = np.concatenate(detected_maps_list, axis=2)  # (H, W, 21)

    with torch.no_grad():
        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = torch.permute(local_control, (0, 3, 1, 2)).contiguous().clone()

        global_emb = torch.zeros((num_samples, 768), dtype=torch.float32, device="cuda")

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_emb)

        cond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
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
#  メイン
# ===========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet Offline Inference with WaymoV2 (Canny/Depth/Semseg)")
    ap.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--canny-root", type=str, default=DEFAULT_CANNY_ROOT)
    ap.add_argument("--depth-img-root", type=str, default=DEFAULT_DEPTH_IMGROOT)
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--prompt-root", type=str, default=DEFAULT_PROMPT_ROOT)
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--uni-config", type=str, default=DEFAULT_UNI_CONFIG)
    ap.add_argument("--uni-ckpt", type=str, default=DEFAULT_UNI_CKPT)

    ap.add_argument("--image-resolution", type=int, default=512, help="生成解像度（H=W）")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--global-strength", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--eta", type=float, default=0.0)

    ap.add_argument("--a-prompt", type=str, default="best quality, extremely detailed")
    ap.add_argument("--n-prompt", type=str,
                   default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")

    ap.add_argument("--limit", type=int, default=-1, help="各 split の先頭N枚のみ。-1 で全件")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_root)
    logger = _setup_logger(args.out_root, verbose=args.verbose)
    _log_env(logger, args)

    # モデル構築
    if not os.path.isfile(args.uni_config):
        logger.error("Uni config not found: %s", args.uni_config); sys.exit(1)
    if not os.path.isfile(args.uni_ckpt):
        logger.error("Uni checkpoint not found: %s", args.uni_ckpt); sys.exit(1)
    model, sampler = _build_model_and_sampler(args.uni_config, args.uni_ckpt, device="cuda")

    H = W = int(args.image_resolution)

    total_ok, total_skip, total_ng = 0, 0, 0

    for split in args.splits:
        split_root = os.path.join(args.image_root, split, args.camera)
        items = _list_waymo_images(args.image_root, split, args.camera)
        if args.limit > 0:
            items = items[:args.limit]
        logger.info("[%s] targets: %d  under %s", split, len(items), split_root)
        if not items: continue

        pbar = tqdm(items, desc=f"{split}")
        for image_path in pbar:
            rel_dir, stem = _rel_dir_and_stem(image_path, os.path.join(args.image_root, split))

            out_png, out_meta = _out_paths(args.out_root, split, rel_dir, stem)
            if (not args.overwrite) and os.path.exists(out_png):
                total_skip += 1; pbar.set_postfix_str("skip"); continue

            # 条件ファイルの推定パス
            canny_png  = _path_canny(args.canny_root, split, rel_dir, stem)
            depth_png  = _path_depth_png(args.depth_img_root, split, rel_dir, stem)
            semseg_npy = _path_semseg_npy(args.semseg_root, split, rel_dir, stem)

            # --- プロンプト探索（フルstem → タイムスタンプ の順で検査） ---
            prompt_txt_full = _path_prompt_txt_full(args.prompt_root, split, rel_dir, stem)
            prompt_txt_ts   = _path_prompt_txt_ts(args.prompt_root, split, rel_dir, stem)
            prompt_path_used: Optional[str] = None
            prompt_source = "fallback:default"
            if os.path.exists(prompt_txt_full):
                prompt_path_used = prompt_txt_full
                prompt_source = "file:full-stem"
            elif os.path.exists(prompt_txt_ts):
                prompt_path_used = prompt_txt_ts
                prompt_source = "file:timestamp"

            try:
                # ローカル条件（存在しない場合はゼロマップ → 公式推奨と整合）
                canny = _load_png_as_hwc3(canny_png, W, H)
                depth = _load_png_as_hwc3(depth_png, W, H)
                seg   = _load_semseg_as_hwc3(semseg_npy, W, H)

                # プロンプト読取 or フォールバック
                if prompt_path_used is not None:
                    with open(prompt_path_used, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                else:
                    prompt = "a realistic driving scene from a front-facing dashcam with consistent geometry"

                logger.info("[prompt] %s | used: %s", prompt_source, prompt_path_used)

                # シード
                seed = args.seed if args.seed >= 0 else _derive_seed(0, image_path)

                # 推論
                results = _sample_one(
                    model=model, ddim_sampler=sampler,
                    canny_map=canny, depth_map=depth, seg_map=seg,
                    prompt=prompt, a_prompt=args.a_prompt, n_prompt=args.n_prompt,
                    num_samples=args.num_samples,
                    H=H, W=W, ddim_steps=args.ddim_steps,
                    strength=args.strength, scale=args.scale,
                    seed=seed, eta=args.eta,
                    global_strength=args.global_strength,
                    logger=logger,
                )

                # 保存（num_samples>1 のときは _ucn-000.png 形式）
                Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
                if args.num_samples == 1:
                    cv2.imwrite(out_png, cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR))
                else:
                    for i, img in enumerate(results):
                        out_i = out_png.replace("_ucn.png", f"_ucn-{i:03d}.png")
                        cv2.imwrite(out_i, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # メタ
                meta = {
                    "split": split, "camera": args.camera,
                    "image_path": image_path,
                    "canny_png": canny_png, "depth_png": depth_png, "semseg_npy": semseg_npy,
                    "prompt_candidates": [prompt_txt_full, prompt_txt_ts],
                    "prompt_path_used": prompt_path_used,
                    "prompt_source": prompt_source,
                    "prompt_used": prompt,
                    "params": {
                        "image_resolution": args.image_resolution,
                        "num_samples": args.num_samples,
                        "ddim_steps": args.ddim_steps,
                        "strength": args.strength,
                        "global_strength": args.global_strength,
                        "scale": args.scale,
                        "seed": seed,
                        "eta": args.eta,
                        "a_prompt": args.a_prompt,
                        "n_prompt": args.n_prompt,
                    },
                    "uni": {
                        "config": args.uni_config,
                        "ckpt": args.uni_ckpt,
                    },
                    "timestamp": int(time.time()),
                }
                with open(out_meta, "w", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False, indent=2))

                total_ok += 1; pbar.set_postfix_str("ok")

            except Exception as e:
                total_ng += 1; pbar.set_postfix_str("err")
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), image_path)
                # 失敗時にもデバッグ JSON を残す
                out_dbg = out_meta.replace("_ucn.meta.json", "_ucn.debug.json")
                dbg = {
                    "image_path": image_path,
                    "canny_png": canny_png, "depth_png": depth_png, "semseg_npy": semseg_npy,
                    "prompt_candidates": [prompt_txt_full, prompt_txt_ts],
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": tb,
                }
                try:
                    with open(out_dbg, "w", encoding="utf-8") as f:
                        f.write(json.dumps(dbg, ensure_ascii=False, indent=2))
                except Exception:
                    pass

    logger.info("✅ 完了 OK:%d  SKIP:%d  NG:%d  出力: %s", total_ok, total_skip, total_ng, args.out_root)

if __name__ == "__main__":
    main()
