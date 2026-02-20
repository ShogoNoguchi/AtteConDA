# /data/coding/Uni-ControlNet/src/tools/build_anno_syndiff_multi.py
# -*- coding: utf-8 -*-
"""
学習・評価用の anno_syndiff_*.csv を自動生成するツール。

- 学習用:
    /data/coding/Uni-ControlNet/data/anno_syndiff_train.csv

    ← 元ネタ:
        - /data/syndiff_prompts/prompts_train/{ds}.csv
        - /data/ucn_condmaps/{ds}/{depth,edge,semseg}/...

- Waymo validation 用:
    /data/coding/Uni-ControlNet/data/anno_syndiff_waymo_val.csv

    ← 元ネタ:
        - /data/syndiff_prompts/prompts_eval_waymo/waymo_validation.csv
        - /home/shogo/coding/datasets/WaymoV2/{CannyEdge,Metricv2DepthIMG,OneFormer_cityscapes}/...

実行例:

    cd /data/coding/Uni-ControlNet
    python -u src/tools/build_anno_syndiff_multi.py \
        --make-train \
        --make-waymo-val \
        --limit -1 \
        --verbose
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ====== 学習用データセットの RGB ルート ======
BDD10K10K_IMG_ROOT = "/home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images"
CITYSCAPES_IMG_ROOT = "/home/shogo/coding/datasets/cityscapes/leftImg8bit"
GTA5_IMG_ROOT       = "/home/shogo/coding/datasets/GTA5/images/images"
NUIMAGES_FRONT_ROOT = "/home/shogo/coding/datasets/nuimages/samples/CAM_FRONT"
BDD100K_IMG_ROOT    = "/home/shogo/coding/datasets/BDD_100K_pure100k"

# ====== 条件マップルート ======
UCN_CONDMAPS_ROOT   = "/data/ucn_condmaps"

# ====== プロンプト ======
PROMPTS_TRAIN_ROOT  = "/data/syndiff_prompts/prompts_train"
PROMPTS_EVAL_ROOT   = "/data/syndiff_prompts/prompts_eval_waymo"

# ====== Waymo 条件マップ ======
WAYMO_IMAGE_ROOT    = "/home/shogo/coding/datasets/WaymoV2/extracted"
WAYMO_CANNY_ROOT    = "/home/shogo/coding/datasets/WaymoV2/CannyEdge"
WAYMO_DEPTH_IMGROOT = "/home/shogo/coding/datasets/WaymoV2/Metricv2DepthIMG"
WAYMO_SEMSEG_ROOT   = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _list_images(root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    out.sort()
    return out


def _rel_from(path: str, base: str) -> str:
    rp = os.path.relpath(os.path.dirname(path), base)
    return "" if rp == "." else rp


def _stem_without_cityscapes_suffix(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    if s.endswith("_leftImg8bit"):
        s = s[:-len("_leftImg8bit")]
    return s


def _first_existing(cands: List[str]) -> Optional[str]:
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------
# 学習用: 5 データセットの 3 条件パスを組み立てる
# ---------------------------------------------------------

def _cond_paths_for_train(ds: str, image_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    各 dataset に対して (depth_path, edge_path, semseg_path) を返す。
    見つからなければ None が入る。
    """
    if ds == "bdd10k10k":
        base_rel = BDD10K10K_IMG_ROOT
        rel_dir = _rel_from(image_path, base_rel)  # "train" or "val"
        stem = os.path.splitext(os.path.basename(image_path))[0]
        depth = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", rel_dir, f"{stem}_depth.jpg")
        edge  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  rel_dir, f"{stem}_edge.jpg")
        seg   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg",rel_dir, f"{stem}_semseg.jpg")
        return depth if os.path.exists(depth) else None, \
               edge  if os.path.exists(edge)  else None, \
               seg   if os.path.exists(seg)   else None

    if ds == "cityscapes":
        base_rel = CITYSCAPES_IMG_ROOT
        rel_dir = _rel_from(image_path, base_rel)  # "train/aachen" など
        stem2 = _stem_without_cityscapes_suffix(image_path)
        depth = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", rel_dir, f"{stem2}_depth.jpg")
        edge  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  rel_dir, f"{stem2}_edge.jpg")
        seg   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg",rel_dir, f"{stem2}_semseg.jpg")
        return depth if os.path.exists(depth) else None, \
               edge  if os.path.exists(edge)  else None, \
               seg   if os.path.exists(seg)   else None

    if ds == "gta5":
        base_rel = GTA5_IMG_ROOT
        rel_dir = _rel_from(image_path, base_rel)  # 通常 ""（cwd）
        stem = os.path.splitext(os.path.basename(image_path))[0]
        # デフォルト: train サブディレクトリ無し
        depth1 = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", f"{stem}_depth.jpg")
        edge1  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  f"{stem}_edge.jpg")
        seg1   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg",f"{stem}_semseg.jpg")
        # 互換用: semseg/train/...
        depth2 = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", "train", f"{stem}_depth.jpg")
        edge2  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  "train", f"{stem}_edge.jpg")
        seg2   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg","train", f"{stem}_semseg.jpg")
        depth = _first_existing([depth1, depth2])
        edge  = _first_existing([edge1,  edge2])
        seg   = _first_existing([seg1,   seg2])
        return depth, edge, seg

    if ds == "nuimages_front":
        base_rel = NUIMAGES_FRONT_ROOT
        rel_dir = _rel_from(image_path, base_rel)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        depth1 = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", rel_dir, f"{stem}_depth.jpg")
        edge1  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  rel_dir, f"{stem}_edge.jpg")
        seg1   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg",rel_dir, f"{stem}_semseg.jpg")
        # 互換用: semseg/train/...
        depth2 = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", "train", rel_dir, f"{stem}_depth.jpg")
        edge2  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  "train", rel_dir, f"{stem}_edge.jpg")
        seg2   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg","train", rel_dir, f"{stem}_semseg.jpg")
        depth = _first_existing([depth1, depth2])
        edge  = _first_existing([edge1,  edge2])
        seg   = _first_existing([seg1,   seg2])
        return depth, edge, seg

    if ds == "bdd100k":
        base_rel = BDD100K_IMG_ROOT
        rel_dir = _rel_from(image_path, base_rel)  # "train" / "val" / "test"
        stem = os.path.splitext(os.path.basename(image_path))[0]
        depth = os.path.join(UCN_CONDMAPS_ROOT, ds, "depth", rel_dir, f"{stem}_depth.jpg")
        edge  = os.path.join(UCN_CONDMAPS_ROOT, ds, "edge",  rel_dir, f"{stem}_edge.jpg")
        seg   = os.path.join(UCN_CONDMAPS_ROOT, ds, "semseg",rel_dir, f"{stem}_semseg.jpg")
        return depth if os.path.exists(depth) else None, \
               edge  if os.path.exists(edge)  else None, \
               seg   if os.path.exists(seg)   else None

    raise ValueError(f"unknown dataset key: {ds}")


def _build_train_csv(out_path: str, limit: int = -1, verbose: bool = True) -> None:
    """
    /data/syndiff_prompts/prompts_train/{ds}.csv を元に
    anno_syndiff_train.csv を作る。
    """
    datasets = ["bdd10k10k", "cityscapes", "gta5", "nuimages_front", "bdd100k"]
    rows: List[list] = []

    for ds in datasets:
        prom_csv = os.path.join(PROMPTS_TRAIN_ROOT, f"{ds}.csv")
        if not os.path.exists(prom_csv):
            if verbose:
                print(f"[build_train] skip {ds}: prompts csv not found: {prom_csv}")
            continue

        if verbose:
            print(f"[build_train] loading prompts: {prom_csv}")
        with open(prom_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cnt = 0
            for r in reader:
                img_path = r["image_path"]
                txt = r.get("train_prompt", "").strip()
                depth_path, edge_path, semseg_path = _cond_paths_for_train(ds, img_path)
                if (depth_path is None) or (edge_path is None) or (semseg_path is None):
                    # 条件マップが欠けているものはスキップ（ログは控えめに）
                    continue
                rows.append([ds, img_path, depth_path, edge_path, semseg_path, txt])
                cnt += 1
                if (limit > 0) and (cnt >= limit):
                    break
        if verbose:
            print(f"[build_train] {ds}: {len(rows)} rows accumulated so far.")

    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image_path", "depth_path", "edge_path", "semseg_path", "txt"])
        for row in rows:
            writer.writerow(row)

    if verbose:
        print(f"[build_train] wrote {len(rows)} rows to {out_path}")


# ---------------------------------------------------------
# Waymo validation 用 CSV
# ---------------------------------------------------------

def _waymo_rel_dir_and_stem(image_path: str) -> Tuple[str, str, str]:
    """
    Waymo RGB パスから (split, rel_dir, stem) を取り出す。

    image_path 例:
      /home/shogo/coding/datasets/WaymoV2/extracted/validation/front/{segment}/{stem}.jpg

    戻り値:
      split   = "validation"
      rel_dir = "front/{segment}"
      stem    = "1507678826876435_first" など
    """
    p = Path(image_path)
    parts = p.parts
    try:
        idx = parts.index("extracted")
    except ValueError:
        raise ValueError(f"unexpected Waymo path: {image_path}")
    split = parts[idx + 1]
    camera = parts[idx + 2]
    segment = parts[idx + 3]
    rel_dir = os.path.join(camera, segment)  # "front/xxxxx"
    stem = os.path.splitext(parts[-1])[0]
    return split, rel_dir, stem


def _build_waymo_val_csv(out_path: str, limit: int = -1, verbose: bool = True) -> None:
    """
    /data/syndiff_prompts/prompts_eval_waymo/waymo_validation.csv を元に
    anno_syndiff_waymo_val.csv を生成。
    """
    prom_csv = os.path.join(PROMPTS_EVAL_ROOT, "waymo_validation.csv")
    if not os.path.exists(prom_csv):
        raise FileNotFoundError(prom_csv)

    rows: List[list] = []
    if verbose:
        print(f"[build_waymo_val] loading prompts: {prom_csv}")
    with open(prom_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cnt = 0
        for r in reader:
            img_path = r["image_path"]
            txt = r.get("eval_prompt", "").strip()
            split, rel_dir, stem = _waymo_rel_dir_and_stem(img_path)

            edge_path = os.path.join(WAYMO_CANNY_ROOT, split, rel_dir, f"{stem}_edge.png")
            depth_path = os.path.join(WAYMO_DEPTH_IMGROOT, split, rel_dir, f"{stem}_depth.png")
            semseg_path = os.path.join(WAYMO_SEMSEG_ROOT, split, rel_dir, f"{stem}_predTrainId.npy")

            if (not os.path.exists(edge_path)) or (not os.path.exists(depth_path)) or (not os.path.exists(semseg_path)):
                # 欠けているものはスキップ
                continue

            rows.append(["waymo_validation", img_path, depth_path, edge_path, semseg_path, txt])
            cnt += 1
            if (limit > 0) and (cnt >= limit):
                break

    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image_path", "depth_path", "edge_path", "semseg_path", "txt"])
        for row in rows:
            writer.writerow(row)

    if verbose:
        print(f"[build_waymo_val] wrote {len(rows)} rows to {out_path}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build anno_syndiff_*.csv for Uni-ControlNet finetuning")
    ap.add_argument("--make-train", action="store_true", help="Build anno_syndiff_train.csv")
    ap.add_argument("--make-waymo-val", action="store_true", help="Build anno_syndiff_waymo_val.csv")
    ap.add_argument("--limit", type=int, default=-1, help="Max rows per dataset (debug)")
    ap.add_argument(
        "--train-out",
        type=str,
        default="./data/anno_syndiff_train.csv",
        help="Output path for train csv",
    )
    ap.add_argument(
        "--waymo-val-out",
        type=str,
        default="./data/anno_syndiff_waymo_val.csv",
        help="Output path for waymo validation csv",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if (not args.make_train) and (not args.make_waymo_val):
        print("Nothing to do: specify --make-train and/or --make-waymo-val")
        return

    if args.make_train:
        _build_train_csv(args.train_out, limit=args.limit, verbose=args.verbose)
    if args.make_waymo_val:
        _build_waymo_val_csv(args.waymo_val_out, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
