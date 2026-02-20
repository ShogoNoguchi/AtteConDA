# /data/coding/Uni-ControlNet/models/logger.py
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency: int = 2000,
        max_images: int = 4,
        clamp: bool = True,
        increase_log_steps: bool = True,
        rescale: bool = True,
        disabled: bool = False,
        log_on_batch_idx: bool = False,
        log_first_step: bool = False,
        log_images_kwargs=None,
        num_local_conditions: int = 7,
    ) -> None:
        """
        Uni-ControlNet 本家版 ImageLogger。

        - batch_frequency: 何ステップごとに画像を保存するか
        - max_images: 各種画像ごとに保存する最大枚数
        - clamp: [-1, 1] の範囲にクランプするか
        - rescale: [-1, 1] の画像を [0, 1] に線形変換するか
        - disabled: True のとき一切ログしない
        - log_on_batch_idx: True なら batch_idx ベースで頻度を決定
        - log_first_step: 最初のステップでもログするかどうか
        - log_images_kwargs: pl_module.log_images に渡す追加引数
        - num_local_conditions: local_control の中に含まれるローカル条件の数
                                （1 条件あたり 3 チャンネル想定なので、C = 3 * num_local_conditions）
        """
        super().__init__()
        self.rescale = rescale
        self.batch_freq = int(batch_frequency)
        self.max_images = int(max_images)
        if not increase_log_steps:
            # ここは本家そのまま。将来的にステップを増やすロジック用のフック。
            self.log_steps = [self.batch_freq]
        self.clamp = bool(clamp)
        self.disabled = bool(disabled)
        self.log_on_batch_idx = bool(log_on_batch_idx)
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs is not None else {}
        self.log_first_step = bool(log_first_step)
        self.num_local_conditions = int(num_local_conditions)

    @rank_zero_only
    def log_local(
        self,
        save_dir: str,
        split: str,
        images: dict,
        global_step: int,
        current_epoch: int,
        batch_idx: int,
    ) -> None:
        """
        実際に PNG ファイルを書き出す処理。

        保存先ディレクトリ構成:
        save_dir / "image_log" / split / ファイル群
        """
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if k == "local_control":
                # local_control は (N, C, H, W) で C = 3 * num_local_conditions を想定
                # H = W = 1 ならスキップ（実質情報が無い場合）
                _, _, h, w = images[k].shape
                if h == w == 1:
                    continue
                # 1 条件ごとに 3 チャンネルに分割して保存
                for local_idx in range(self.num_local_conditions):
                    c_start = 3 * local_idx
                    c_end = 3 * (local_idx + 1)
                    grid = torchvision.utils.make_grid(
                        images[k][:, c_start:c_end, :, :],
                        nrow=4,
                    )
                    if self.rescale:
                        # [-1, 1] -> [0, 1]
                        grid = (grid + 1.0) / 2.0  # 形状は (C, H, W)
                    # (C, H, W) -> (H, C, W) -> (H, W, C)
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255.0).astype(np.uint8)

                    filename = "gs-{:06}_e-{:06}_b-{:06}_{}_{}.png".format(
                        global_step,
                        current_epoch,
                        batch_idx,
                        k,
                        local_idx,
                    )
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)

            elif k != "global_control":
                # global_control はログしない仕様。本家の elif k != 'global_control' に合わせる。
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    # [-1, 1] -> [0, 1]
                    grid = (grid + 1.0) / 2.0  # (C, H, W)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255.0).astype(np.uint8)

                filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                    global_step,
                    current_epoch,
                    batch_idx,
                    k,
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx: int, split: str = "train") -> None:
        """
        LightningModule 側の log_images(...) を呼び出し、
        その結果を log_local に渡して PNG 保存する。
        """
        # 本家は log_on_batch_idx が False でも check_idx = batch_idx を見ている。
        # コメントにある通り、本来は pl_module.global_step を見ることも想定している。
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step

        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            # logger 型は今は使っていないが、本家のコードに合わせて残す
            logger_type = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch,
                    split=split,
                    **self.log_images_kwargs,
                )

            for k in images:
                # 画像枚数を max_images までに制限
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

            # TensorBoard の SummaryWriter ではなく、PNG を直接保存する
            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx: int) -> bool:
        """
        ログ出力のタイミング判定。
        現状は単純に batch_frequency の剰余だけを見る。
        """
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        PyTorch Lightning の on_train_batch_end フック。

        Lightning 1.8 以降では dataloader_idx 引数が追加されているので、
        デフォルト値付きで受け取って無視する。
        """
        if self.disabled:
            return
        self.log_img(pl_module, batch, batch_idx, split="train")
