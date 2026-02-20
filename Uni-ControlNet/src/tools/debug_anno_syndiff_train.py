# /data/coding/Uni-ControlNet/src/tools/debug_anno_syndiff_train.py
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="anno_syndiff_train.csv の統計を出して、実際に何枚学習に使われているか確認するツール"
    )
    parser.add_argument(
        "--anno-path",
        type=str,
        default="./data/anno_syndiff_train.csv",
        help="解析する anno_syndiff_train.csv のパス",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="train.py で使っている batch_size（イテレーション数の計算に使用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anno_path = Path(args.anno_path)

    if not anno_path.is_file():
        raise FileNotFoundError(f"anno_syndiff_train.csv が見つかりません: {anno_path}")

    print(f"[debug] 読み込む CSV: {anno_path}")

    total_rows = 0
    per_dataset = Counter()
    missing_depth = 0
    missing_edge = 0
    missing_semseg = 0
    missing_any = 0

    # CSV フォーマット:
    # header: ["dataset", "image_path", "depth_path", "edge_path", "semseg_path", "txt"]
    with anno_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            ds = row.get("dataset", "").strip()
            per_dataset[ds] += 1

            depth_path = row.get("depth_path", "").strip()
            edge_path = row.get("edge_path", "").strip()
            semseg_path = row.get("semseg_path", "").strip()

            # 一応、存在確認もしてみる（全部ファイルを stat すると時間はかかるが、1 回限りのデバッグ用途なので良しとする）
            depth_exists = os.path.exists(depth_path)
            edge_exists = os.path.exists(edge_path)
            semseg_exists = os.path.exists(semseg_path)

            if not depth_exists:
                missing_depth += 1
            if not edge_exists:
                missing_edge += 1
            if not semseg_exists:
                missing_semseg += 1
            if not (depth_exists and edge_exists and semseg_exists):
                missing_any += 1

    print("===============================================")
    print("【anno_syndiff_train.csv の統計】")
    print(f"総サンプル数（=行数）: {total_rows}")
    print("データセット別サンプル数:")
    for ds, cnt in sorted(per_dataset.items()):
        print(f"  - {ds:15s}: {cnt}")

    if missing_any > 0:
        print("-----------------------------------------------")
        print("※ CSV 上のパスだが、実際のファイルが存在しないもの（debug 用）")
        print(f"  depth_path が存在しない行数 : {missing_depth}")
        print(f"  edge_path  が存在しない行数 : {missing_edge}")
        print(f"  semseg_path が存在しない行数: {missing_semseg}")
        print(f"  3 種のうち 1 つでも欠けている行数: {missing_any}")
        print("  （通常は 0 が望ましい。 >0 の場合は ucn_build_conditions.py か build_anno_syndiff_multi.py のロジックを要確認）")
    else:
        print("-----------------------------------------------")
        print("depth/edge/semseg のファイルパスは、すべて実在しているようです。")

    # 学習時のイテレーション数との関係も出しておく
    bs = max(1, int(args.batch_size))
    iters_per_epoch = (total_rows + bs - 1) // bs

    print("===============================================")
    print(f"想定 DataLoader 設定: batch_size = {bs}")
    print(f"1 エポックあたりのイテレーション数 ≒ ceil({total_rows} / {bs}) = {iters_per_epoch}")
    print("→ Lightning のログに出ている 'Epoch 0: ... /XXXXX' の XXXXX と一致するはずです。")
    print("===============================================")


if __name__ == "__main__":
    main()
