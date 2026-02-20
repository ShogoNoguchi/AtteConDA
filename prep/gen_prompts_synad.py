#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_prompts_synad.py
Train用RGB（BDD10K(10K)/Cityscapes/GTA5/nuImages(front)/BDD100K）と
評価用RGB（Waymo）に対して、SynDiff-AD手順に沿ったテキストプロンプトを一括生成。

- ステージA（VLM専用パス）: LLaVA(既定) でキャプションのみ生成 → 保存
- ステージB（CLIP専用パス）: OllamaモデルをVRAMからアンロード → CLIPをCUDAにロード → 分類 & 最終プロンプト生成

- VLM: LLaVA (llava:13b) /api/generate のみ使用（/api/chat禁止）
- CLIP: open_clip timm/ViT-SO400M-14-SigLIP-384 による Weather×Time 推定
- “Thinking…”や<think>…</think>の除去・再試行・空文字対策
- ログは /data/ucn_prompt_cache/logs/gen_prompts_synad.log（回転）
- TensorBoardは /data/ucn_prompt_tb/
- 既存環境のCUDA12.8 + torch(既存)を尊重、pip overlayで依存を追加
- CPUフォールバックは完全禁止（CUDA未検出なら即停止）
"""

from __future__ import annotations
import os, sys, json, math, time, random, logging
from logging import handlers
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

# ===== Torch / TB =====
import torch
from torch.utils.tensorboard import SummaryWriter

# ===== OpenCLIP(SigLIP) =====
import open_clip
from PIL import Image

# ===== 自作モジュール =====
CURDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURDIR)
from qwen3_vl_client import build_vlm_backend
from subgroups_synad import (
    WEATHERS, TIMES, all_subgroups,
    CITYSCAPES_TRAINID_TO_NAME, CITYSCAPES_NAME_LIST,
    pick_adjective, decorations_for
)

# ===== 既定パス =====
BDD10K10K_IMG_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images" # {train,val}
CITYSCAPES_IMG_ROOT = "/home/shogo/coding/datasets/cityscapes/leftImg8bit"                     # {train,val,test}
GTA5_IMG_ROOT       = "/home/shogo/coding/datasets/GTA5/images/images"
NUIMAGES_FRONT_ROOT = "/home/shogo/coding/datasets/nuimages/samples/CAM_FRONT"
BDD100K_IMG_ROOT    = "/home/shogo/coding/datasets/BDD_100K_pure100k"
WAYMO_ROOT          = "/home/shogo/coding/datasets/WaymoV2/extracted"
UCM_ROOT            = "/data/ucn_condmaps"
WAYMO_ONEFORMER_NPY = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"

DEFAULT_OUT_ROOT   = "/data/ucn_prompts"
DEFAULT_CACHE_ROOT = "/data/ucn_prompt_cache"
DEFAULT_TB_DIR     = "/data/ucn_prompt_tb"

# ===== ロガー =====
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def setup_logger(cache_root: str, verbose: bool) -> logging.Logger:
    log_dir = os.path.join(cache_root, "logs"); ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "gen_prompts_synad.log")
    logger = logging.getLogger("synad_prompter")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers): logger.removeHandler(h)
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S", style='%'))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=50*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S", style='%'))
    logger.addHandler(ch); logger.addHandler(fh)
    logging.raiseExceptions = False
    return logger

# ===== I/O =====
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_text(path: str, txt: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

# ===== データ列挙 =====
ALLOW_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def list_images_under(root: str) -> List[str]:
    out=[]
    if not os.path.isdir(root): return out
    for r,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in ALLOW_EXT:
                out.append(os.path.join(r,f))
    out.sort()
    return out

def stem_cityscapes(filename: str) -> str:
    s = os.path.splitext(os.path.basename(filename))[0]
    return s[:-len("_leftImg8bit")] if s.endswith("_leftImg8bit") else s

def rel_dir(base: str, path: str) -> str:
    rd = os.path.relpath(os.path.dirname(path), base)
    return "." if rd == "." else rd

def enumerate_dataset(dataset_key: str,
                      splits: Optional[List[str]] = None) -> Tuple[List[str], str]:
    if dataset_key == "bdd10k10k":
        assert splits, "splits required"
        ims=[]
        for sp in splits: ims += list_images_under(os.path.join(BDD10K10K_IMG_ROOT, sp))
        return ims, BDD10K10K_IMG_ROOT
    if dataset_key == "cityscapes":
        assert splits, "splits required"
        ims=[]
        for sp in splits: ims += list_images_under(os.path.join(CITYSCAPES_IMG_ROOT, sp))
        return ims, CITYSCAPES_IMG_ROOT
    if dataset_key == "gta5":
        ims = list_images_under(GTA5_IMG_ROOT); return ims, GTA5_IMG_ROOT
    if dataset_key == "nuimages_front":
        ims = list_images_under(NUIMAGES_FRONT_ROOT); return ims, NUIMAGES_FRONT_ROOT
    if dataset_key == "bdd100k":
        assert splits, "splits required"
        ims=[]
        for sp in splits: ims += list_images_under(os.path.join(BDD100K_IMG_ROOT, sp))
        return ims, BDD100K_IMG_ROOT
    if dataset_key == "waymo":
        ims=[]
        for sp in ["training","validation","testing"]:
            d = os.path.join(WAYMO_ROOT, sp, "front")
            if os.path.isdir(d): ims += list_images_under(d)
        return ims, WAYMO_ROOT
    raise ValueError(f"unknown dataset_key: {dataset_key}")

# ===== CLIP (OpenCLIP) =====
class CLIPSubgroupClassifier:
    """
    SigLIP ViT-SO400M-14 (384) を open_clip 経由で使用。
    Weather と Time の2軸を**独立ソフトマックス**で推定し、文字列 "Weather, Time" を返す。
    """
    def __init__(self, device="cuda:0", batch_size: int = 8):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. CPU fallback is forbidden.")
        self.device = device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP-384',
            device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        self.prompts_w = [f"An image taken during {w} weather." for w in WEATHERS]
        self.prompts_t = [f"An image taken during {t} time."    for t in TIMES]
        with torch.no_grad():
            self.text_w = self.model.encode_text(self.tokenizer(self.prompts_w).to(self.device))
            self.text_t = self.model.encode_text(self.tokenizer(self.prompts_t).to(self.device))
        self.bs = int(batch_size)

    def classify_one(self, rgb: np.ndarray) -> Tuple[str, str]:
        pil = Image.fromarray(rgb)
        img = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img)
            sw = (img_feat @ self.text_w.T).softmax(dim=1).squeeze(0)
            st = (img_feat @ self.text_t.T).softmax(dim=1).squeeze(0)
            wi = int(torch.argmax(sw).item()); ti = int(torch.argmax(st).item())
        return WEATHERS[wi], TIMES[ti]

# ===== Semsegからクラス名抽出（上位12件） =====
CITYSCAPES_RGB_TO_TRAINID = {
    (128, 64,128):0,(244,35,232):1,( 70, 70, 70):2,(102,102,156):3,(190,153,153):4,
    (153,153,153):5,(250,170, 30):6,(220,220,  0):7,(107,142, 35):8,(152,251,152):9,
    ( 70,130,180):10,(220, 20, 60):11,(255,  0,  0):12,(  0,  0,142):13,(  0,  0, 70):14,
    (  0, 60,100):15,(  0, 80,100):16,(  0,  0,230):17,(119, 11, 32):18,
}

def try_extract_classes_from_semseg(dataset_key: str, base_rel: str, img_path: str) -> List[str]:
    # Waymo: OneFormer NPY があれば優先
    if dataset_key == "waymo":
        try:
            parts = Path(img_path).parts
            idx = parts.index("extracted")
            split = parts[idx+1]; segment = parts[idx+3]
            npy_path = os.path.join(
                WAYMO_ONEFORMER_NPY, split, "front", segment,
                f"{os.path.splitext(os.path.basename(img_path))[0]}_predTrainId.npy"
            )
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)
                u, c = np.unique(arr, return_counts=True)
                pairs = sorted([(int(ui), int(ci)) for ui,ci in zip(u,c) if int(ui) in CITYSCAPES_TRAINID_TO_NAME], key=lambda x: -x[1])
                names = [CITYSCAPES_TRAINID_TO_NAME[i] for i,_ in pairs[:12]]
                return names
        except Exception:
            pass

    # colorized semseg jpg
    if dataset_key in ("bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k","waymo"):
        base, path = base_rel, img_path
        rd = rel_dir(base, path)
        stem = stem_cityscapes(path) if dataset_key=="cityscapes" else os.path.splitext(os.path.basename(path))[0]
        semjpg = os.path.join(UCM_ROOT, dataset_key, "semseg", rd, f"{stem}_semseg.jpg")
        if os.path.isfile(semjpg):
            col = cv2.cvtColor(cv2.imread(semjpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # 計算コスト緩和（最大長辺 1600 に制限：上位クラスは不変）
            h,w,_ = col.shape
            m = max(h,w)
            if m > 1600:
                scale = 1600.0 / float(m)
                col = cv2.resize(col, (max(1,int(round(w*scale))), max(1,int(round(h*scale)))), interpolation=cv2.INTER_NEAREST)
            flat = col.reshape(-1,3)
            uniq, cnt = np.unique(flat, axis=0, return_counts=True)
            pairs = []
            for (r,g,b), c in zip(uniq, cnt):
                tid = CITYSCAPES_RGB_TO_TRAINID.get((r,g,b), None)
                if tid is not None:
                    pairs.append((tid, int(c)))
            pairs.sort(key=lambda x: -x[1])
            names = [CITYSCAPES_TRAINID_TO_NAME[i] for i,_ in pairs[:12]]
            return names
    return []

# ===== VLMキャプション生成 =====
CAPTION_PROMPT_TEMPLATE = (
    "Provide a concise, semantically dense description of the scene focusing on the following objects and their "
    "relationships: {objects}. Describe the background (urban/rural), layout, and image quality. "
    "Do NOT mention weather or time-of-day keywords such as Clear, Cloudy, Rainy, Snowy, Foggy, Day, Night, Dawn/Dusk. "
    "Write in 2–3 sentences as a single paragraph."
)

def build_synad_training_prompt(vlm_caption: str, weather: str, time_of_day: str) -> str:
    return f"{vlm_caption.strip()} Image taken during {time_of_day} time and {weather} weather."

def build_waymo_improved_prompt(class_names: List[str], vlm_caption: str, target_weather: str, target_time: str) -> str:
    adj = pick_adjective(target_weather, target_time)
    cls_txt = ", ".join(class_names) if class_names else ", ".join(CITYSCAPES_NAME_LIST[:10])
    decos = decorations_for(target_weather, target_time)
    deco_txt = " ".join(decos[:3]) if len(decos)>=3 else " ".join(decos)
    return (
        f"A realistic {adj} city street scene with {cls_txt}, {vlm_caption.strip()} "
        f"{deco_txt} Keep the same camera angle and composition as the original image."
    )

# ===== 出力パス =====
def out_caption_path(out_root: str, dataset_key: str, base_rel: str, img_path: str) -> str:
    rd = rel_dir(base_rel, img_path)
    stem = stem_cityscapes(img_path) if dataset_key=="cityscapes" else os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(out_root, "captions", dataset_key, rd, f"{stem}_caption.txt")

def out_prompt_path(out_root: str, dataset_key: str, base_rel: str, img_path: str) -> str:
    rd = rel_dir(base_rel, img_path)
    stem = stem_cityscapes(img_path) if dataset_key=="cityscapes" else os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(out_root, "prompts", dataset_key, rd, f"{stem}_prompt.txt")

def out_waymo_prompt_path(out_root: str, img_path: str, target_weather: str, target_time: str) -> str:
    p = Path(img_path)
    split = p.parts[p.parts.index("extracted")+1]
    segment = p.parts[p.parts.index("front")+1]
    stem = os.path.splitext(p.name)[0]
    return os.path.join(out_root, "prompts_waymo", split, "front", segment, f"{stem}_target-{target_weather}-{target_time}.txt")

# ===== 判定等 =====
def is_bad_caption(txt: str) -> bool:
    if not txt: return True
    t = txt.strip()
    if len(t) < 10: return True
    bad = ["Thinking...", "(thinking", "I'm unable to see", "I cannot view", "As an AI model"]
    if any(badword.lower() in t.lower() for badword in bad): return True
    return False

def add_tb_scalar(sw: Optional[SummaryWriter], tag: str, val: float, step: int) -> None:
    if sw: sw.add_scalar(tag, val, step)

# ===== メイン =====
def main():
    import argparse
    ap = argparse.ArgumentParser(description="SynDiff-AD準拠の一括プロンプト生成（2パス：VLM→アンロード→CLIP）")
    ap.add_argument("--datasets", nargs="+", default=["bdd10k10k","cityscapes","gta5","nuimages_front","bdd100k","waymo"])
    ap.add_argument("--cityscapes-splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--bdd10k10k-splits", nargs="+", default=["train","val"])
    ap.add_argument("--bdd100k-splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--tb-dir", default=DEFAULT_TB_DIR)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--verbose", action="store_true")
    # VLM backend (Ollama)
    ap.add_argument("--vlm-backend", choices=["ollama"], default="ollama")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="llava:13b")
    ap.add_argument("--hidethinking", action="store_true")
    # Waymo
    ap.add_argument("--targets-per-image", type=int, default=1)
    ap.add_argument("--simplemode", action="store_true")
    ap.add_argument("--force-train-subdir-singletons", action="store_true")
    args = ap.parse_args()

    logger = setup_logger(args.cache_root, args.verbose)
    ensure_dir(args.out_root); ensure_dir(args.cache_root)
    sw = SummaryWriter(args.tb_dir)

    # ===== CUDA 必須（CPUフォールバック禁止） =====
    logger.info("torch=%s cuda=%s available=%s", torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is forbidden.")
    logger.info("device0=%s", torch.cuda.get_device_name(0))

    # ===== 画像リスト収集（1回だけ） =====
    datasets: List[Tuple[str, List[str], str]] = []  # (key, imgs, base_rel)
    for ds_key in args.datasets:
        if ds_key == "bdd10k10k":
            imgs, base_rel = enumerate_dataset("bdd10k10k", args.bdd10k10k_splits)
        elif ds_key == "cityscapes":
            imgs, base_rel = enumerate_dataset("cityscapes", args.cityscapes_splits)
        elif ds_key == "gta5":
            imgs, base_rel = enumerate_dataset("gta5", None)
        elif ds_key == "nuimages_front":
            imgs, base_rel = enumerate_dataset("nuimages_front", None)
        elif ds_key == "bdd100k":
            imgs, base_rel = enumerate_dataset("bdd100k", args.bdd100k_splits)
        elif ds_key == "waymo":
            imgs, base_rel = enumerate_dataset("waymo", None)
        else:
            logger.warning("unknown dataset key: %s", ds_key); continue
        if args.limit > 0:
            imgs = imgs[:args.limit]
        datasets.append((ds_key, imgs, base_rel))

    # ------------------------------------------------------------------------------------
    # ステージA：VLMキャプション専用パス（Ollama起動→ウォームアップ→全画像キャプション→アンロード）
    # ------------------------------------------------------------------------------------
    logger.info("=== [Stage A] VLM captioning (model=%s @ %s) ===", args.ollama_model, args.ollama_host)
    vlm = build_vlm_backend("ollama", hidethinking=args.hidethinking, host=args.ollama_host, model=args.ollama_model)
    # 1) まず全モデルをアンロードしてクリーンスタート
    vlm.ensure_unloaded_all()
    # 2) モデル存在を保証し、ウォームアップ（一度ロード）
    vlm.ensure_model_present()
    vlm.warmup()

    total_A = 0; ok_caps = 0; retried = 0; skipped_A = 0
    for (ds_key, imgs, base_rel) in datasets:
        logger.info("=== [A:%s] %d images ===", ds_key, len(imgs))
        for p in tqdm(imgs, desc=f"A:{ds_key}", dynamic_ncols=True):
            try:
                rgb = imread_rgb(p)
            except Exception:
                logger.exception("[READ ERR] %s", p); skipped_A += 1; continue

            obj_list = CITYSCAPES_NAME_LIST[:10]  # とりあえず objects を与える（CLIPなし段階）
            cap_prompt = CAPTION_PROMPT_TEMPLATE.format(objects=", ".join(obj_list))

            cap = ""
            try:
                cap = vlm.caption(rgb, cap_prompt)
                if is_bad_caption(cap):
                    retried += 1; cap = vlm.caption(rgb, cap_prompt)
                if is_bad_caption(cap):
                    retried += 1
                    cap_prompt2 = CAPTION_PROMPT_TEMPLATE.format(objects=", ".join(obj_list[:6]))
                    cap = vlm.caption(rgb, cap_prompt2)
            except Exception:
                logger.exception("[VLM ERR] %s", p); cap = ""

            if is_bad_caption(cap):
                cap = ("A city street scene with vehicles, pedestrian space, buildings and traffic elements. "
                       "Layout is urban and perspective is consistent. Lighting and exposure are typical for the original capture.")

            cap_out = out_caption_path(args.out_root, ds_key, base_rel, p)
            write_text(cap_out, cap)
            ok_caps += 1
            total_A += 1

    logger.info("=== [Stage A] done: total=%d ok_caps=%d retried=%d skipped=%d ===", total_A, ok_caps, retried, skipped_A)

    # 3) VLMのVRAMを**必ず**解放（アンロード）。CUDAキャッシュも掃除。
    vlm.ensure_unloaded_all()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    # ------------------------------------------------------------------------------------
    # ステージB：CLIP を CUDA にロード → 分類 → 最終プロンプト生成
    # ------------------------------------------------------------------------------------
    logger.info("=== [Stage B] Load CLIP on CUDA and generate prompts ===")
    clf = CLIPSubgroupClassifier(device="cuda:0", batch_size=8)
    logger.info("CLIP ready (SigLIP-384 on CUDA)")

    tables_dir = os.path.join(args.out_root, "tables"); ensure_dir(tables_dir)
    orig_map: Dict[str, Dict[str,str]] = {}
    waymo_targets_records: List[Dict[str,str]] = []
    total_B = 0; skipped_B = 0

    for (ds_key, imgs, base_rel) in datasets:
        logger.info("=== [B:%s] %d images ===", ds_key, len(imgs))
        for p in tqdm(imgs, desc=f"B:{ds_key}", dynamic_ncols=True):
            try:
                rgb = imread_rgb(p)
            except Exception:
                logger.exception("[READ ERR] %s", p); skipped_B += 1; continue

            # 1) CLIP サブグループ
            try:
                w, tday = clf.classify_one(rgb)
            except Exception:
                logger.exception("[CLIP ERR] %s", p); skipped_B += 1; continue
            orig_map[p] = {"weather": w, "time": tday}

            # 2) Semseg からクラス名（WaymoはOneFormerなど）
            try:
                ds_for_sem = "waymo" if ds_key=="waymo" else ds_key
                cls_names = try_extract_classes_from_semseg(ds_for_sem, base_rel, p)
            except Exception:
                cls_names = []
            obj_list = cls_names if cls_names else CITYSCAPES_NAME_LIST[:10]

            # 3) 事前保存済みキャプションをロード
            cap_out = out_caption_path(args.out_root, ds_key, base_rel, p)
            try:
                with open(cap_out, "r", encoding="utf-8") as f:
                    cap = f.read().strip()
            except Exception:
                cap = ("A city street scene with vehicles, pedestrian space, buildings and traffic elements. "
                       "Layout is urban and perspective is consistent. Lighting and exposure are typical for the original capture.")

            # 4) 出力
            if ds_key != "waymo":
                final_prompt = build_synad_training_prompt(cap, w, tday)
                prm_out = out_prompt_path(args.out_root, ds_key, base_rel, p)
                write_text(prm_out, final_prompt)
            else:
                Z = all_subgroups()
                cand = [(ww,tt) for (ww,tt) in Z if not (ww==w and tt==tday)]
                random.shuffle(cand)
                sel = cand[:max(1, int(args.targets_per_image))]
                classes_for_waymo = obj_list
                for (tw, ttod) in sel:
                    if args.simplemode:
                        simple = (f"A city street scene photo with {', '.join(classes_for_waymo)} at {ttod} in {tw} weather. "
                                  f"Keep the same camera angle and composition as the original image.")
                        prm_out = out_waymo_prompt_path(args.out_root, p, tw, ttod)
                        write_text(prm_out, simple)
                    else:
                        improved = build_waymo_improved_prompt(classes_for_waymo, cap, tw, ttod)
                        prm_out = out_waymo_prompt_path(args.out_root, p, tw, ttod)
                        write_text(prm_out, improved)
                    waymo_targets_records.append({
                        "image": p, "orig_weather": w, "orig_time": tday,
                        "target_weather": tw, "target_time": ttod
                    })

            total_B += 1
            if total_B % 50 == 0:
                add_tb_scalar(sw, "progress/processed_B", total_B, total_B)

    # テーブル保存
    orig_csv = os.path.join(args.out_root, "tables", "orig_subgroups.csv")
    with open(orig_csv, "w", encoding="utf-8") as f:
        f.write("image,weather,time\n")
        for k, v in orig_map.items():
            f.write(f"{k},{v['weather']},{v['time']}\n")

    if waymo_targets_records:
        tgt_csv = os.path.join(args.out_root, "tables", "waymo_target_subgroups.csv")
        with open(tgt_csv, "w", encoding="utf-8") as f:
            f.write("image,orig_weather,orig_time,target_weather,target_time\n")
            for r in waymo_targets_records:
                f.write(f"{r['image']},{r['orig_weather']},{r['orig_time']},{r['target_weather']},{r['target_time']}\n")

    logger.info("✅ finished: A_total=%d A_ok=%d A_retried=%d A_skipped=%d | B_total=%d B_skipped=%d",
                total_A, ok_caps, retried, skipped_A, total_B, skipped_B)
    add_tb_scalar(sw, "summary/A_total", total_A, 0)
    add_tb_scalar(sw, "summary/B_total", total_B, 0)
    sw.close()

if __name__ == "__main__":
    main()
