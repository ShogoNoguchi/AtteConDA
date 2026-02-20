# -*- coding: utf-8 -*-
"""
WaymoV2(front) を OneFormer(Cityscapes Swin-L) で 1枚ずつ順次推論。
出力:
(1) クラスID .npy（uint8, Cityscapes trainId: 0..18）
(2) Cityscapes配色のインデックスPNG
(3) 失敗時に per-image の .debug.json

要点:
- OneFormer は AutoProcessor を用いて images + task_inputs=["semantic"] を渡す
- pixel_values は (1,C,H,W) Tensor に整形し、model.dtype / device に整合
- 既存出力があればスキップ。--overwrite で上書き
- 強いロギング（コンソール＋ファイル）、失敗時は .debug.json で詳細記録
- Cityscapes パレットは trainId=0..18 の 19色（過去スクリプトの18色不足を修正）

環境注意:
- Ubuntu + RTX5090（CUDA 12.8 固定）を前提に、既存の torch+cu128 をそのまま使用
- 本スクリプトはインストール操作を行わない。起動時に torch / CUDA 情報を詳細ログ

実行例:
python /home/shogo/coding/tools/oneformer/infer_waymo_oneformer_seq.py \
  --input-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --output-root /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes \
  --splits training validation testing \
  --device cuda --fp16 --limit -1

"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import argparse
import logging
from logging import handlers
from typing import List, Tuple, Any, Dict, Optional
import gc
import time
import traceback
import json

import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoProcessor, OneFormerForUniversalSegmentation
from tqdm import tqdm


# ======== 既定パス ========
DEFAULT_INPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_MODEL_ID = "shi-labs/oneformer_cityscapes_swin_large"
DEFAULT_CAMERA = "front"
DEFAULT_SPLITS = ["training", "validation", "testing"]

# ======== Cityscapes trainId パレット（19色, 0..18） ========
# 参考: road, sidewalk, building, wall, fence, pole,
#       traffic light, traffic sign, vegetation, terrain, sky,
#       person, rider, car, truck, bus, train, motorcycle, bicycle
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
    (  0,  80, 100),  # 16: train  ← 過去スクリプトで欠落していた色
    (  0,   0, 230),  # 17: motorcycle
    (119,  11,  32),  # 18: bicycle
]

_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ======== ユーティリティ ========
def _build_palette() -> List[int]:
    pal = [0] * 256 * 3
    for i, (r, g, b) in enumerate(_CITYSCAPES_TRAINID_COLORS):
        pal[i*3:i*3+3] = [r, g, b]
    return pal
_PALETTE: List[int] = _build_palette()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _ext_lower(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run.log")
    logger = logging.getLogger("infer_waymo_oneformer")
    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)

    # File (rotating)
    fh = handlers.RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    fh.setFormatter(fh_fmt)

    # Avoid duplicate handlers when re-run in same session
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def _list_waymo_images(input_root: str, split: str, camera: str) -> List[str]:
    """
    WaymoV2/extracted/{split}/{camera}/{segment_id}/{first|mid10s|last}.jpg を列挙。
    拡張子は .jpg/.jpeg/.png/.bmp を許容（既存資産との互換性のため）。
    """
    base = os.path.join(input_root, split, camera)
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        for f in fs:
            ext = _ext_lower(f)
            if ext in _ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    return sorted(out)

def _out_paths(in_path: str, split_root: str, out_root: str, naming: str) -> Tuple[str, str, str]:
    """
    入力root配下の相対ディレクトリ/ファイル構造を出力側にもそのまま反映する。
    suffix: predTrainId or semantic
    """
    rel_dir = os.path.relpath(os.path.dirname(in_path), split_root)  # e.g., front/{segment_id}
    stem = os.path.splitext(os.path.basename(in_path))[0]            # e.g., first / mid10s / last
    dst_dir = os.path.join(out_root, rel_dir); _ensure_dir(dst_dir)
    suffix = "predTrainId" if naming == "predTrainId" else "semantic"
    return (
        os.path.join(dst_dir, f"{stem}_{suffix}.npy"),
        os.path.join(dst_dir, f"{stem}_{suffix}.png"),
        os.path.join(dst_dir, f"{stem}_{suffix}.debug.json"),
    )

def _save_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr.astype(np.uint8), allow_pickle=False)

def _save_indexed_png(path: str, index_map: np.ndarray) -> None:
    img = Image.fromarray(index_map, mode="P")
    img.putpalette(_PALETTE)
    _ensure_dir(os.path.dirname(path))
    img.save(path, "PNG", optimize=True)

def _type_tree(x: Any, depth: int = 0) -> str:
    indent = "  " * depth
    if isinstance(x, torch.Tensor):
        return f"{indent}Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    if isinstance(x, np.ndarray):
        return f"{indent}ndarray(shape={x.shape}, dtype={x.dtype})"
    if isinstance(x, (list, tuple)):
        s = [_type_tree(xi, depth + 1) for xi in (x[:3] if len(x) > 3 else x)]
        inner = "\n".join(s)
        return f"{indent}{type(x).__name__}(len={len(x)})[\n{inner}\n{indent}]"
    if isinstance(x, dict):
        lines = []
        for k, v in x.items():
            lines.append(f"{indent}{k}:")
            lines.append(_type_tree(v, depth + 1))
        return "\n".join(lines)
    return f"{indent}{type(x).__name__}: {repr(x)[:120]}"

def _normalize_pv_to_1chw(pv: Any) -> torch.Tensor:
    # Tensor -> (1,C,H,W)
    if isinstance(pv, torch.Tensor):
        t = pv
        if t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim == 4:
            pass
        else:
            raise TypeError(f"pixel_values Tensor 次元不正: {t.ndim}")
        return t
    # ndarray -> (1,C,H,W)
    if isinstance(pv, np.ndarray):
        if pv.ndim == 3:
            pv = np.expand_dims(pv, 0)
        elif pv.ndim == 4:
            pass
        else:
            raise TypeError(f"pixel_values ndarray 次元不正: {pv.ndim}")
        return torch.from_numpy(pv)
    # list/tuple -> 先頭を再帰
    if isinstance(pv, (list, tuple)):
        if len(pv) == 0:
            raise TypeError("pixel_values が空の list/tuple")
        return _normalize_pv_to_1chw(pv[0])
    raise TypeError(f"pixel_values 型不正: {type(pv)}")

def _log_env(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("torch: %s", torch.__version__)
    logger.info("torch.version.cuda(build): %s", getattr(torch.version, "cuda", None))
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            logger.info("CUDA device 0: %s", torch.cuda.get_device_name(0))
        except Exception:
            logger.warning("CUDA device name 取得失敗")
    logger.info("dtype: %s", "fp16" if args.fp16 else "fp32")
    logger.info("naming: %s", args.naming)
    logger.info("overwrite: %s", args.overwrite)
    logger.info("model-id: %s", args.model_id)
    logger.info("splits: %s", " ".join(args.splits))
    logger.info("camera: %s", args.camera)
    if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
        logger.warning("警告: ビルド時CUDA (%s) が 12.8 ではありません。既存環境に合わせて実行は続行します。",
                       torch.version.cuda)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WaymoV2 -> Cityscapes(trainId) OneFormer 推論スクリプト")
    ap.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT,
                    help="WaymoV2/extracted のルート")
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
                    help="出力ルート（相対構造をそのまま複製）")
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                    help="HuggingFaceのモデルID")
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS,
                    help="処理する分割（training, validation, testing など）")
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA,
                    help="カメラ名（既定: front）")
    ap.add_argument("--device", type=str, default="cuda",
                    help="推論デバイス（cuda / cpu）")
    ap.add_argument("--limit", type=int, default=-1,
                    help="各splitの先頭N枚のみ処理。-1で全件")
    ap.add_argument("--naming", type=str, choices=["predTrainId", "semantic"], default="predTrainId",
                    help="ファイル名のsuffix")
    ap.add_argument("--fp16", action="store_true",
                    help="半精度(fp16)で実行")
    ap.add_argument("--overwrite", action="store_true",
                    help="既存出力を上書き")
    ap.add_argument("--sleep-ms", type=int, default=0,
                    help="各枚処理後スリープ(ms)")
    ap.add_argument("--verbose", action="store_true",
                    help="ログ詳細（DEBUG）をコンソールにも出す")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    # 出力ルート直下にログをまとめる
    _ensure_dir(args.output_root)
    logger = _setup_logger(args.output_root, verbose=args.verbose)

    # 環境ログ
    _log_env(logger, args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA 未検出。--device cpu での実行も検討してください。")
        sys.exit(1)

    # Processor / Model ロード
    logger.info("Loading processor: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    logger.info("Loading model: %s", args.model_id)
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    model = OneFormerForUniversalSegmentation.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    ).eval().to(args.device)

    # 速度チューニング（安全な範囲）
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    total_ok, total_ng = 0, 0

    for split in args.splits:
        split_in_root = os.path.join(args.input_root, split, args.camera)
        split_out_root = os.path.join(args.output_root, split)
        _ensure_dir(split_out_root)

        items = _list_waymo_images(args.input_root, split, args.camera)
        if args.limit > 0:
            items = items[:args.limit]

        if not items:
            logger.warning("[%s] 対象なし: %s", split, split_in_root)
            continue

        logger.info("[%s] 対象枚数: %d", split, len(items))
        pbar = tqdm(items, desc=f"{split}")

        for p in pbar:
            npy_path, png_path, dbg_path = _out_paths(
                in_path=p,
                split_root=os.path.join(args.input_root, split),
                out_root=split_out_root,
                naming=args.naming,
            )

            if (not args.overwrite) and os.path.exists(npy_path) and os.path.exists(png_path):
                pbar.set_postfix_str("skip")
                continue

            out = None; seg = None; rgb = None; img = None; enc = None; pv = None; ti = None
            try:
                # 画像読み込み（BGR -> RGB）
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("cv2.imread が None を返しました")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 1) AutoProcessor でエンコード（semantic）
                enc = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")

                # 2) pixel_values を (1,C,H,W) Tensor へ正規化し、device+dtype 整合
                pv = _normalize_pv_to_1chw(enc.get("pixel_values")).to(args.device, dtype=model.dtype)
                ti = enc.get("task_inputs")
                if isinstance(ti, torch.Tensor):
                    ti = ti.to(args.device)

                if args.verbose:
                    logger.debug("---- ENCODED INPUTS TREE ----\n%s", _type_tree(enc))
                    logger.debug("pixel_values(normalized): %s", _type_tree(pv))
                    logger.debug("task_inputs: %s", _type_tree(ti))

                # 3) 推論
                h, w = rgb.shape[:2]
                with torch.inference_mode():
                    if model.dtype == torch.float16 and args.device.startswith("cuda"):
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = model(pixel_values=pv, task_inputs=ti)
                    else:
                        out = model(pixel_values=pv, task_inputs=ti)

                seg = processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
                pred = seg.to("cpu").numpy().astype(np.uint8)

                # 4) 保存
                _save_npy(npy_path, pred)
                _save_indexed_png(png_path, pred)
                logger.info("[OK] %s | %s", npy_path, png_path)
                total_ok += 1
                pbar.set_postfix_str("ok")

            except Exception as e:
                total_ng += 1
                tb = traceback.format_exc()
                log: Dict[str, Any] = {
                    "input_path": p,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": tb,
                    "torch_version": torch.__version__,
                    "torch_cuda_build": getattr(torch.version, "cuda", None),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                    "model_dtype": str(model.dtype),
                    "encoded_tree": _type_tree(enc) if enc is not None else "unavailable",
                    "pixel_values_tree": _type_tree(pv) if pv is not None else "unavailable",
                    "task_inputs_tree": _type_tree(ti) if ti is not None else "unavailable",
                }
                logger.error("[ERR] %s: %s | in: %s", log["error_type"], log["error_msg"], p)
                try:
                    with open(dbg_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(log, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            finally:
                out = None; seg = None; rgb = None; img = None; enc = None; pv = None; ti = None
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

    logger.info("✅ 完了 OK:%d NG:%d 出力: %s", total_ok, total_ng, args.output_root)


if __name__ == "__main__":
    main()
