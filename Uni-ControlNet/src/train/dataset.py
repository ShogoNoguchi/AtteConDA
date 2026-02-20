import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset

from .util import *


class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        file_ids, self.annos = read_anno(anno_path)
        self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]
        self.local_paths = {}
        for local_type in local_type_list:
            self.local_paths[local_type] = [os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids]
        self.global_paths = {}
        for global_type in global_type_list:
            self.global_paths[global_type] = [os.path.join(condition_root, global_type, file_id + '.npy') for file_id in file_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0

        anno = self.annos[index]
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

        local_conditions = []
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
            local_conditions.append(condition)
        global_conditions = []
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)

        if random.random() < self.drop_txt_prob:
            anno = ''
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.annos)
        

 # ここから挿入してください（AbsPathUniDataset の追加）
class AbsPathUniDataset(Dataset):
    """
    Uni-ControlNet 用の「絶対パス版」データセット。

    - anno_path には 1 行ごとにタブ区切り or カンマ区切りで以下の情報を持たせる想定：
        rgb_path_abs, txt_prompt, depth_path_abs, edge_path_abs, semseg_path_abs

      例（TSV）:
        /home/shogo/.../image.png  <TAB>  a driving scene with cars ...  <TAB>  /data/ucn_condmaps/.../depth.jpg  <TAB>  /data/ucn_condmaps/.../edge.jpg  <TAB>  /data/ucn_condmaps/.../semseg.jpg

    - local_conditions は Uni-ControlNet 公式が想定する 7 種の順序に合わせて 21ch を構成する：
        [canny, mlsd, hed, sketch, openpose, midas(depth), seg]
      ただし、我々は「canny / depth / semseg」のみ事前生成しているので、
        canny  : edge_path_abs
        depth  : depth_path_abs
        seg    : semseg_path_abs
        mlsd, hed, sketch, openpose : すべて 0 埋めの黒画像 (H,W,3)
      として 3ch 画像を連結する。

    - global_conditions は、今回の自動運転 finetune では text embedding ではなく
      既に LatentDiffusion 側の cross-attn 経由でテキストを使うため、
      「使わない」→ 0 ベクトルを 1 個だけ返すか、空 list にする。
      ここでは UniControlNet 側の get_input 実装と整合させるため、
      shape = (1, 768) のゼロを返す。
    """
    def __init__(
        self,
        anno_path,
        resolution,
        drop_txt_prob=0.0,
        keep_all_cond_prob=0.1,
        drop_all_cond_prob=0.1,
        drop_each_cond_prob=None
    ):
        if drop_each_cond_prob is None:
            # canny / depth / seg の 3 条件だが、local_conditions は 7 ブロック持つ
            # ここでは簡単に 7 要素全部 0.0〜1.0 にできるように 0.5 ずつをデフォルトにする。
            drop_each_cond_prob = [0.5] * 7

        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob

        self.rgb_paths = []
        self.txt_annos = []
        self.depth_paths = []
        self.edge_paths = []
        self.semseg_paths = []

        if not os.path.isfile(anno_path):
            raise FileNotFoundError(f"AbsPathUniDataset: anno_path not found: {anno_path}")

        # ログを少し出す（Python 初心者向けデバッグ用）
        print(f"[AbsPathUniDataset] Loading anno from: {anno_path}")

        with open(anno_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # タブ優先、その次にカンマ
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split(",")

                if len(parts) < 5:
                    raise ValueError(
                        f"[AbsPathUniDataset] anno line {line_idx} has {len(parts)} columns (<5): {line}"
                    )

                rgb_path = parts[0].strip()
                txt = parts[1].strip()
                depth_path = parts[2].strip()
                edge_path = parts[3].strip()
                semseg_path = parts[4].strip()

                self.rgb_paths.append(rgb_path)
                self.txt_annos.append(txt)
                self.depth_paths.append(depth_path)
                self.edge_paths.append(edge_path)
                self.semseg_paths.append(semseg_path)

        self.length = len(self.rgb_paths)
        print(f"[AbsPathUniDataset] Loaded {self.length} samples.")

    def __len__(self):
        return self.length

    def _read_rgb_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"[AbsPathUniDataset] image not found or unreadable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution, self.resolution))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return img

    def _read_cond_image_u8(self, path):
        """
        条件マップ JPG (0-255 の 3ch) をそのまま読み、(H,W,3) uint8 で返す。
        読み込み不可の場合はゼロ画像を返して「安全に進む」。
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # ログは training loop 側の logger に任せる。ここでは print のみにとどめる。
            print(f"[AbsPathUniDataset] WARNING: condition image not found, using zeros: {path}")
            img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.resolution, self.resolution))
        return img

    def __getitem__(self, index):
        # 1) RGB 画像
        rgb = self._read_rgb_image(self.rgb_paths[index])

        # 2) テキストアノテーション
        anno = self.txt_annos[index]
        if random.random() < self.drop_txt_prob:
            anno = ""

        # 3) ローカル条件 7 ブロックの構成
        #    canny, mlsd, hed, sketch, openpose, midas(depth), seg
        edge_u8 = self._read_cond_image_u8(self.edge_paths[index])
        depth_u8 = self._read_cond_image_u8(self.depth_paths[index])
        seg_u8 = self._read_cond_image_u8(self.semseg_paths[index])

        H, W = self.resolution, self.resolution
        zero_u8 = np.zeros((H, W, 3), dtype=np.uint8)

        canny = edge_u8
        mlsd = zero_u8
        hed = zero_u8
        sketch = zero_u8
        openpose = zero_u8
        midas = depth_u8
        seg = seg_u8

        local_list = [canny, mlsd, hed, sketch, openpose, midas, seg]

        # 公式 UniDataset と同じ keep_and_drop ロジックを使うため、
        # 一旦 float32 [0,1] にしてから keep_and_drop に渡す。
        local_list_f = []
        for arr in local_list:
            local_list_f.append(arr.astype(np.float32) / 255.0)

        # 4) keep_and_drop による条件のサブサンプリング
        local_list_f = keep_and_drop(
            local_list_f,
            self.keep_all_cond_prob,
            self.drop_all_cond_prob,
            self.drop_each_cond_prob
        )

        if len(local_list_f) != 0:
            local_conditions = np.concatenate(local_list_f, axis=2)  # (H,W, N*3)
        else:
            # 何も残らなかった場合は 0 チャネルを返す（下流で len(local_conditions)==0 判定が効くように）
            local_conditions = np.zeros((H, W, 0), dtype=np.float32)

        # 5) global_conditions はゼロベクトル (1,768) を 1 つだけ持つ
        global_conditions = np.zeros((1, 768), dtype=np.float32)

        return dict(
            jpg=rgb,
            txt=anno,
            local_conditions=local_conditions,
            global_conditions=global_conditions,
        )
# ここまで挿入してください
       