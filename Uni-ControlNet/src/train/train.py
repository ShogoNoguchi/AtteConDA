# /data/coding/Uni-ControlNet/src/train/train.py
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import logging

# カレントディレクトリを module path に追加（元のコードと同じ）
if "./" not in sys.path:
    sys.path.append("./")

from omegaconf import OmegaConf

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数のパーサ。
    """
    parser = argparse.ArgumentParser(description="Uni-ControlNet Training")

    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/local_v15.yaml",
        help="OmegaConf 形式の config ファイルパス",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="初期学習率",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="学習バッチサイズ",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=int(1e5),
        help="最大学習ステップ数 (Trainer.max_steps)",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="./ckpt/init_local.ckpt",
        help="初期重みとして読み込む ckpt (state_dict)。新規学習開始時に使用。",
    )
    parser.add_argument(
        "--resume-training-from",
        type=str,
        default="",
        help=(
            "Lightning の学習用 checkpoint (.ckpt) から再開する場合のパス。"
            "空文字列のままなら、新規に --resume-path から初期化して学習を開始する。"
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./log_local/",
        help="ログおよび checkpoint の保存ディレクトリ（例: ./logs/finetune_uni_syndiff）",
    )
    parser.add_argument(
        "--logger-version",
        type=int,
        default=4,
        help=(
            "TensorBoardLogger の version 番号。"
            "logdir/lightning_logs/version_{この番号} にログを書く。"
        ),
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=500,
        help="ImageLogger / checkpoint の出力間隔 (ステップ数)",
    )
    parser.add_argument(
        "--ckpt-every-n-steps",
        type=int,
        default=30000,
        help="★追加: このステップ間隔で“永続”checkpointを保存する（例: 30000）",
    )
    parser.add_argument(
        "--sd-locked",
        type=str,
        default="True",
        help="SD 本体を凍結するかどうか。'True' / 'False' / '1' / '0' などの文字列で指定。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader の num_workers",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="Trainer に渡す gpus 引数 (-1: 全 GPU, 0: CPU)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=1.0,
        help="Lightning Trainer の val_check_interval (今は limit_val_batches=0 なので実質無効)",
    )

    return parser.parse_args()


def str_to_bool(x: str) -> bool:
    """
    'True' / 'true' / '1' などを True、
    'False' / 'false' / '0' などを False と解釈する。
    """
    x = str(x).strip().lower()
    if x in ("1", "true", "t", "yes", "y"):
        return True
    if x in ("0", "false", "f", "no", "n"):
        return False
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = parse_args()

    config_path = args.config_path
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    training_steps = int(args.training_steps)
    resume_path = args.resume_path
    resume_training_from = args.resume_training_from
    default_logdir = args.logdir
    logger_version = int(args.logger_version)
    logger_freq = int(args.log_freq)
    ckpt_every_n = int(args.ckpt_every_n_steps)
    sd_locked = str_to_bool(args.sd_locked)
    num_workers = int(args.num_workers)
    gpus = int(args.gpus)
    val_check_interval = float(args.val_check_interval)

    # ===== Config & Model =====
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config["model"])

    if resume_training_from:
        print(f"[Trainer] Resume training from Lightning checkpoint: {resume_training_from}")
    else:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume-path ckpt not found: {resume_path}")
        state_dict = load_state_dict(resume_path, location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(unexpected_keys) > 0:
            print("[UniControlNet] WARNING: unexpected keys in state_dict (ignored):")
            for k in unexpected_keys:
                print(f" - {k}")
        if len(missing_keys) > 0:
            print("[UniControlNet] WARNING: missing keys in state_dict (left at init):")
            for k in missing_keys[:50]:
                print(f" - {k}")
            if len(missing_keys) > 50:
                print(f" ... and {len(missing_keys) - 50} more")

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    # ===== Dataset / DataLoader =====
    if "data_train" in config:
        dataset_train = instantiate_from_config(config["data_train"])
    else:
        dataset_train = instantiate_from_config(config["data"])

    dataloader_train = DataLoader(
        dataset_train,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    if "data_val" in config:
        dataset_val = instantiate_from_config(config["data_val"])
        dataloader_val = DataLoader(
            dataset_val,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )
    else:
        dataloader_val = None

    # ===== Logger / Callback =====
    tb_logger = TensorBoardLogger(
        save_dir=default_logdir,
        name="lightning_logs",
        version=logger_version,
    )

    # checkpoint 保存先を固定（実験で探しやすくする）
    ckpt_dir = os.path.join(default_logdir, "lightning_logs", f"version_{logger_version}", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    image_logger = ImageLogger(batch_frequency=logger_freq)

    # (a) 従来：log_freqごと + last
    checkpoint_callback_logfreq = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=logger_freq,
        save_last=True,
        save_top_k=-1,
        filename="logfreq-epoch{epoch:02d}-step{step:09d}",
    )

    # (b) ★追加：30000stepごとに“永続”保存（全部残す）
    checkpoint_callback_30k = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=ckpt_every_n,
        save_last=False,
        save_top_k=-1,
        filename="periodic-step{step:09d}",
    )

    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[image_logger, checkpoint_callback_logfreq, checkpoint_callback_30k],
        logger=tb_logger,
        default_root_dir=default_logdir,
        max_steps=training_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=0,
    )

    fit_kwargs = dict(
        model=model,
        train_dataloaders=dataloader_train,
        ckpt_path=resume_training_from or None,
    )
    if dataloader_val is not None:
        fit_kwargs["val_dataloaders"] = dataloader_val

    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
