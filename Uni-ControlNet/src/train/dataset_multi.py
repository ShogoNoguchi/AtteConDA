# /data/coding/Uni-ControlNet/src/train/dataset_multi.py
# -*- coding: utf-8 -*-
"""
Multi-dataset + offline conditions + absolute-path Dataset for Uni-ControlNet finetuning.

- 入力: CSV (header 付き)
    dataset,image_path,depth_path,edge_path,semseg_path,txt

- 出力バッチ:
    {
        "jpg":               float32 (H,W,3) in [-1,1],
        "txt":               str,
        "local_conditions":  float32 (H,W,21) in [0,1],
        "global_conditions": float32 (768,)  # ここでは常にゼロベクトル
    }

local_conditions の 21ch は公式の順番に合わせる:
    [canny, mlsd, hed, sketch, openpose, midas(depth), seg] × 3ch = 21ch

ただし実際に非ゼロなのは
    - canny  ← edge_path
    - midas  ← depth_path
    - seg    ← semseg_path（JPG/PNG/Numpy の両方に対応）
その他の条件は毎回ゼロ画像を生成する。
"""

import csv
import os
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from .util import keep_and_drop  # 公式実装を再利用


# Cityscapes trainId → RGB パレット（OneFormer/Waymo と同じ定義に合わせる）
CITYSCAPES_TRAINID_COLORS_RGB = [
    (128, 64, 128),  # 0: road
    (244, 35, 232),  # 1: sidewalk
    (70, 70, 70),    # 2: building
    (102, 102, 156), # 3: wall
    (190, 153, 153), # 4: fence
    (153, 153, 153), # 5: pole
    (250, 170, 30),  # 6: traffic light
    (220, 220, 0),   # 7: traffic sign
    (107, 142, 35),  # 8: vegetation
    (152, 251, 152), # 9: terrain
    (70, 130, 180),  # 10: sky
    (220, 20, 60),   # 11: person
    (255, 0, 0),     # 12: rider
    (0, 0, 142),     # 13: car
    (0, 0, 70),      # 14: truck
    (0, 60, 100),    # 15: bus
    (0, 80, 100),    # 16: train
    (0, 0, 230),     # 17: motorcycle
    (119, 11, 32),   # 18: bicycle
]


def _ensure_hwc3_from_any(arr: np.ndarray) -> np.ndarray:
    """任意のグレイスケール/3ch/4ch画像を HWC3 uint8(BGR) に正規化する。"""
    if arr is None:
        raise ValueError("Input image array is None")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.shape[2] == 3:
            # そのまま BGR とみなす
            pass
        else:
            raise ValueError(f"Unsupported channel dim: {arr.shape}")
    else:
        raise ValueError(f"Unsupported ndim: {arr.ndim}")
    return arr


def _load_rgb_image(path: str, resolution: int) -> np.ndarray:
    """RGB 画像 (H,W,3), float32 [-1,1] を読み込む。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_bgr = cv2.resize(img_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = (img_rgb.astype(np.float32) / 127.5) - 1.0  # [-1,1]
    return img


def _load_cond_image_or_zeros(path: Optional[str], resolution: int) -> np.ndarray:
    """
    条件マップ用の 3ch 画像を読み込む。
    - path が None/空/存在しない場合はゼロ画像 (H,W,3) を返す。
    - 画像フォーマットは JPG/PNG/BMP など OpenCV が読めるもの。
    - 戻り値は float32 [0,1], HWC3(RGB)。
    """
    if not path or (not os.path.exists(path)):
        return np.zeros((resolution, resolution, 3), dtype=np.float32)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # 存在するが読めない → ゼロで代用（ログは呼び出し側で出す想定）
        return np.zeros((resolution, resolution, 3), dtype=np.float32)

    img = _ensure_hwc3_from_any(img)
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img_rgb.astype(np.float32) / 255.0)


def _load_semseg_any(path: Optional[str], resolution: int) -> np.ndarray:
    """
    Semseg 用の条件マップを HWC3 float32 [0,1] で返す。

    - 拡張子 .npy の場合:
        Cityscapes trainId (0..18) の 2D マップをロードし、
        CITYSCAPES_TRAINID_COLORS_RGB に従ってカラーリング。
    - それ以外 (.jpg/.png など) の場合:
        通常の画像として読み込み。
    - path が無い/読めない場合はゼロ画像。
    """
    if not path or (not os.path.exists(path)):
        return np.zeros((resolution, resolution, 3), dtype=np.float32)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        try:
            seg = np.load(path)
        except Exception:
            return np.zeros((resolution, resolution, 3), dtype=np.float32)
        if seg.ndim != 2:
            return np.zeros((resolution, resolution, 3), dtype=np.float32)

        seg = seg.astype(np.int32)
        h, w = seg.shape[:2]
        pal = np.zeros((h, w, 3), dtype=np.uint8)
        for tid, (r, g, b) in enumerate(CITYSCAPES_TRAINID_COLORS_RGB):
            pal[seg == tid] = (r, g, b)  # RGB
        pal = cv2.resize(pal, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        return (pal.astype(np.float32) / 255.0)

    # 画像として読み込む
    return _load_cond_image_or_zeros(path, resolution)


class MultiCondDataset(Dataset):
    """
    絶対パス CSV ベースの Uni-ControlNet 用 Dataset。

    CSV 形式:
        dataset,image_path,depth_path,edge_path,semseg_path,txt
    """

    def __init__(
        self,
        anno_path: str,
        resolution: int = 512,
        drop_txt_prob: float = 0.0,
        keep_all_cond_prob: float = 0.1,
        drop_all_cond_prob: float = 0.1,
        drop_each_cond_prob: Optional[List[float]] = None,
        global_dim: int = 768,
    ) -> None:
        super().__init__()
        if not os.path.exists(anno_path):
            raise FileNotFoundError(anno_path)

        self.resolution = int(resolution)
        self.drop_txt_prob = float(drop_txt_prob)
        self.keep_all_cond_prob = float(keep_all_cond_prob)
        self.drop_all_cond_prob = float(drop_all_cond_prob)
        if drop_each_cond_prob is None:
            # [canny, mlsd, hed, sketch, openpose, midas, seg] の 7 要素
            self.drop_each_cond_prob = [0.5] * 7
        else:
            if len(drop_each_cond_prob) != 7:
                raise ValueError("drop_each_cond_prob must have length 7")
            self.drop_each_cond_prob = list(map(float, drop_each_cond_prob))
        self.global_dim = int(global_dim)

        self.entries: List[Dict[str, Any]] = []
        with open(anno_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = ["dataset", "image_path", "depth_path", "edge_path", "semseg_path", "txt"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise ValueError(f"[MultiCondDataset] column '{col}' is missing in {anno_path}")
            for row in reader:
                self.entries.append(
                    dict(
                        dataset=row["dataset"],
                        image_path=row["image_path"],
                        depth_path=row["depth_path"],
                        edge_path=row["edge_path"],
                        semseg_path=row["semseg_path"],
                        txt=row.get("txt", ""),
                    )
                )

        if len(self.entries) == 0:
            raise RuntimeError(f"[MultiCondDataset] no entries found in {anno_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        e = self.entries[index]

        # 1) RGB 画像（[-1,1]）
        image = _load_rgb_image(e["image_path"], self.resolution)

        # 2) TEXT
        anno_txt = e.get("txt", "")
        if random.random() < self.drop_txt_prob:
            anno_txt = ""

        # 3) ローカル条件（7種 × 3ch = 21ch）
        #    順番: [canny, mlsd, hed, sketch, openpose, midas, seg]
        #    実際に非ゼロにするのは:
        #      - canny  ← edge_path
        #      - midas  ← depth_path
        #      - seg    ← semseg_path
        H = self.resolution
        W = self.resolution

        # canny
        canny = _load_cond_image_or_zeros(e["edge_path"], self.resolution)
        # mlsd / hed / sketch / openpose はゼロマップ
        zeros = np.zeros((H, W, 3), dtype=np.float32)
        mlsd = zeros
        hed = zeros
        sketch = zeros
        openpose = zeros
        # midas(depth)
        depth = _load_cond_image_or_zeros(e["depth_path"], self.resolution)
        # seg
        seg = _load_semseg_any(e["semseg_path"], self.resolution)

        local_list = [canny, mlsd, hed, sketch, openpose, depth, seg]

        # 4) keep_and_drop で条件ドロップ（公式ロジックを再利用）
        local_list = keep_and_drop(
            local_list,
            self.keep_all_cond_prob,
            self.drop_all_cond_prob,
            self.drop_each_cond_prob,
        )

        if len(local_list) != 0:
            local_conditions = np.concatenate(local_list, axis=2)  # (H,W,21)
        else:
            # すべてドロップされた場合はゼロ
            local_conditions = np.zeros((H, W, 21), dtype=np.float32)

        # 5) global_conditions は「毎回ゼロベクトル」
        #    UniControlNet(mode=uni) では global_adapter への入力になるが、
        #    content embedding を持っていないため、ここでは 0 ベクトルで固定する。
        global_conditions = np.zeros((self.global_dim,), dtype=np.float32)

        return dict(
            jpg=image,                 # (H,W,3), float32 [-1,1]
            txt=anno_txt,              # str
            local_conditions=local_conditions,   # (H,W,21), float32 [0,1]
            global_conditions=global_conditions, # (768,)
        )
