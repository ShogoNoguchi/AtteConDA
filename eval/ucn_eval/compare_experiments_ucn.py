#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_experiments_ucn.py

完全リライト版。
Uni-ControlNet / SynDiff-AD 系の EX○○.eval.json と EX○○.eval.json を比較し、
「どの指標でどちらが優位か」を正しく算出・可視化する研究用途スクリプト。

主な改良点:
    - Reality metric (clip-cmmd, cmmd, mmd) を direction="lower" として強制扱い。
    - “方向推定” を完全に再設計し、確定ルール → ヒューリスティック → fallback の三段階で推論。
    - flatten_metrics をより堅牢にし、辞書・リストの安定走査を保証。
    - 相対改善率計算の科学的安定化 (a=0 の場合の分母取り扱い)。
    - ログの構造を整理し、研究用途向けに一貫した INFO 出力。
    - コード全体の構造を関数ごとに明確化し、依存木も docstring に記述。
    - 全行省略なし。コピペでそのまま実行可能。

依存関係（木構造）:
    compare_experiments_ucn.py
    ├─ main()
    │   ├─ parse_args()
    │   ├─ setup_logger()
    │   ├─ load_json()
    │   ├─ infer_experiment_name()
    │   ├─ flatten_metrics()
    │   ├─ determine_direction()
    │   ├─ compare_two_values()
    │   ├─ build_summary()
    │   ├─ save_outputs()
    │   └─ logger
    ├─ logging, json, os, argparse, datetime, math
    └─ 標準ライブラリのみ
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

Number = Union[int, float]

# ============================================================
# ロガー
# ============================================================

def setup_logger(log_path: str | None) -> logging.Logger:
    """
    ロガーをセットアップする。
    出力:
        - 標準出力
        - ファイル (log_path が指定されている場合)

    注意:
        - 重複ログ回避のため、handlers が空の場合のみ追加。
    """
    logger = logging.getLogger("ucn_ex_compare")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ============================================================
# JSON 読み込み
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    """UTF-8 JSON ロード。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 実験名を JSON から推定
# ============================================================

def infer_experiment_name(js: Dict[str, Any], default_name: str) -> str:
    """
    実験名を JSON から推定する。
    優先順位:
        1. ルートのよくあるキー
        2. meta/info 内のよくあるキー
        3. default_name
    """
    for key in ["experiment_id", "experiment", "name", "id", "ex_name", "ex_id"]:
        if key in js and isinstance(js[key], str):
            return js[key]

    for section in ["meta", "info"]:
        meta = js.get(section)
        if isinstance(meta, dict):
            for key in ["experiment_id", "name", "id"]:
                if key in meta and isinstance(meta[key], str):
                    return meta[key]

    return default_name


# ============================================================
# メトリクスのフラット化
# ============================================================

def flatten_metrics(obj: Any, prefix: str = "") -> Dict[str, Number]:
    """
    ネストされた JSON から「完全パス → 数値(float)」の dict に変換する。

    例:
        {"training": {"Edge": {"f1": 0.98}}}
        → {"training/Edge/f1": 0.98}

    数値以外は無視。
    """
    flat: Dict[str, Number] = {}

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            new_prefix = f"{prefix}/{k}" if prefix else k
            flat.update(flatten_metrics(obj[k], new_prefix))

    elif isinstance(obj, list):
        # list の中身が全部数値なら index 付きで追加
        if all(isinstance(x, (int, float)) for x in obj):
            for i, v in enumerate(obj):
                flat[f"{prefix}[{i}]"] = float(v)
        else:
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                flat.update(flatten_metrics(v, new_prefix))

    elif isinstance(obj, (int, float)):
        if prefix:
            flat[prefix] = float(obj)

    return flat


# ============================================================
# metric direction の決定（完全リライト版）
# ============================================================

# --- 確定ルール（最上位で適用される） ---
DIRECTION_FORCE_LOWER = [
    "mmd",
    "cmmd",
    "clip-cmmd",
    "wasserstein",
    "absdiff",
    "l1",
    "l2",
    "rmse",
    "si_rmse",
    "rel",
    "error",
    "loss",
    "distance",
    "divergence",
    "mae",
]

DIRECTION_FORCE_HIGHER = [
    "iou",
    "f1",
    "precision",
    "recall",
    "score",
    "rho",
    "alignment",
    "lpips",
    "one_minus_ms_ssim",
    "diversity",
    "strength",
    "spread",  # -------------- ここから挿入してください（追加）--------------
]

def determine_direction(metric_path: str) -> str:
    key = metric_path.lower()

    # --- Reality metric: lower is better ---
    if "reality" in key:
        return "lower"

    # --- HAL: lower is better ---
    if key.endswith("/hal") or "/hal" in key:
        return "lower"

    # --- PR: higher is better ---
    if key.endswith("/pr") and not key.endswith("/n_samples") and not key.endswith("/n"):
        return "higher"

    # --- PP: higher is better ---
    if key.endswith("/pp"):
        return "higher"

    # --- F1 は既に higher に入る（f1 ∈ DIRECTION_FORCE_HIGHER）---

    # --- size_log_ratio_median: lower is better ---
    if "size_log_ratio_median" in key:
        return "lower"

    # --- force-lower keywords ---
    for kw in DIRECTION_FORCE_LOWER:
        if kw in key:
            return "lower"

    # --- force-higher keywords ---
    for kw in DIRECTION_FORCE_HIGHER:
        if kw in key:
            return "higher"

    return "unknown"





# ============================================================
# A と B の値比較
# ============================================================

def compare_two_values(a: float, b: float, direction: str) -> Tuple[str, float]:
    """
    EX_A の値 a と EX_B の値 b を比較。
    relative improvement は:
        direction="higher" → (b - a) / denom
        direction="lower"  → (a - b) / denom
        unknown             → (b - a) / denom （とりあえず b > a で B）
    denom:
        原則 abs(a)。a=0 の場合 abs(b) or 1e-9。
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return "tie", 0.0

    if a == b:
        return "tie", 0.0

    denom = abs(a)
    if denom < 1e-12:
        denom = abs(b)
        if denom < 1e-12:
            denom = 1.0  # 両者ほぼゼロの時の安全策

    if direction == "higher":
        winner = "B" if b > a else "A"
        rel = (b - a) / denom

    elif direction == "lower":
        winner = "B" if b < a else "A"
        rel = (a - b) / denom

    else:  # unknown
        winner = "B" if b > a else "A"
        rel = (b - a) / denom

    return winner, rel


# ============================================================
# split の推定
# ============================================================

def detect_split(path: str) -> str:
    path_low = path.lower()
    for sp in ["training", "validation", "testing"]:
        if path_low.startswith(sp + "/") or f"/{sp}/" in path_low:
            return sp
    return "overall"


# ============================================================
# サマリー構築
# ============================================================

def build_summary(
    flat_a: Dict[str, float],
    flat_b: Dict[str, float],
    name_a: str,
    name_b: str,
    logger: logging.Logger,
    verbose: bool,
    top_k: int = 25,
) -> Dict[str, Any]:
    """
    A/B メトリクス比較のメイン処理。
    """
    rows: List[Dict[str, Any]] = []

    # 共通キーのみ比較
    common_keys = sorted(set(flat_a.keys()) & set(flat_b.keys()))

    for path in common_keys:
        a = flat_a[path]
        b = flat_b[path]
        direction = determine_direction(path)
        winner, rel = compare_two_values(a, b, direction)
        split = detect_split(path)

        rows.append(
            {
                "metric": path,
                "split": split,
                "direction": direction,
                "value_A": a,
                "value_B": b,
                "winner": winner,
                "rel_improvement_B_vs_A": rel,
            }
        )

    # 改善率絶対値順でソート
    rows_sorted = sorted(rows, key=lambda r: abs(r["rel_improvement_B_vs_A"]), reverse=True)

    if verbose:
        logger.info("===== Comparison Summary (Top %d diffs) =====", top_k)
        for r in rows_sorted[:top_k]:
            logger.info(
                "[%s] %s | dir=%s | A=%.6f | B=%.6f | winner=%s | rel=%+.2f%%",
                r["split"],
                r["metric"],
                r["direction"],
                r["value_A"],
                r["value_B"],
                r["winner"],
                r["rel_improvement_B_vs_A"] * 100.0,
            )

    # split ごとにまとめる
    per_split: Dict[str, List[Dict[str, Any]]] = {
        "training": [],
        "validation": [],
        "testing": [],
        "overall": [],
    }
    for r in rows_sorted:
        per_split[r["split"]].append(r)

    summary = {
        "experiment_A": name_a,
        "experiment_B": name_b,
        "generated_at": datetime.now().isoformat(),
        "metrics": per_split,
    }
    return summary


# ============================================================
# ファイル出力
# ============================================================

def save_outputs(summary: Dict[str, Any], out_json: str, out_md: str, logger: logging.Logger) -> None:
    """comparison_{A}_vs_{B}.json および md を保存。"""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Markdown
    with open(out_md, "w", encoding="utf-8") as f:
        A = summary["experiment_A"]
        B = summary["experiment_B"]

        f.write("# EX comparison report\n\n")
        f.write(f"- Experiment A: {A}\n")
        f.write(f"- Experiment B: {B}\n")
        f.write(f"- Generated at: {summary['generated_at']}\n\n")

        for split, rows in summary["metrics"].items():
            f.write(f"## Split: {split}\n\n")
            f.write("| metric | direction | A | B | winner | ΔB_vs_A (%) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(
                    f"| {r['metric']} | {r['direction']} "
                    f"| {r['value_A']:.6f} | {r['value_B']:.6f} "
                    f"| {r['winner']} | {r['rel_improvement_B_vs_A']*100:.2f} |\n"
                )
            f.write("\n")

    logger.info("Saved JSON: %s", out_json)
    logger.info("Saved MD  : %s", out_md)


# ============================================================
# argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Uni-ControlNet evaluation JSONs."
    )

    parser.add_argument(
        "--Comparison_between_EX_and_report_for_researchpresentation_Mode",
        "--compare-ex",
        dest="compare_ex",
        nargs=2,
        metavar=("EX_A_JSON", "EX_B_JSON"),
        help="EX1.eval.json EX2.eval.json を渡すと比較モードが有効になる。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力先ディレクトリ。",
    )

    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="ログ出力を抑制し、可視化用ファイルのみ生成。",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="ログファイルパス (省略時: output-dir/compare_ex.log)",
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.compare_ex:
        raise SystemExit(
            "Error: 必ず --Comparison_between_EX_and_report_for_researchpresentation_Mode "
            "EX1.eval.json EX2.eval.json を指定してください。"
        )

    ex_a_json, ex_b_json = args.compare_ex

    if not os.path.isfile(ex_a_json):
        raise FileNotFoundError(ex_a_json)
    if not os.path.isfile(ex_b_json):
        raise FileNotFoundError(ex_b_json)

    # output-dir の決定
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(ex_b_json)) or "."

    log_path = args.log_file or os.path.join(args.output_dir, "compare_ex.log")
    logger = setup_logger(log_path)

    logger.info("Experiment A JSON: %s", ex_a_json)
    logger.info("Experiment B JSON: %s", ex_b_json)

    js_a = load_json(ex_a_json)
    js_b = load_json(ex_b_json)

    name_a = infer_experiment_name(js_a, os.path.basename(ex_a_json))
    name_b = infer_experiment_name(js_b, os.path.basename(ex_b_json))

    logger.info("Experiment A name: %s", name_a)
    logger.info("Experiment B name: %s", name_b)

    flat_a = flatten_metrics(js_a)
    flat_b = flatten_metrics(js_b)

    logger.info("Flattened metrics count: A=%d, B=%d", len(flat_a), len(flat_b))

    summary = build_summary(
        flat_a,
        flat_b,
        name_a,
        name_b,
        logger,
        verbose=not args.viz_only,
    )

    out_json = os.path.join(args.output_dir, f"comparison_{name_a}_vs_{name_b}.json")
    out_md = os.path.join(args.output_dir, f"comparison_{name_a}_vs_{name_b}.md")

    save_outputs(summary, out_json, out_md, logger)


if __name__ == "__main__":
    main()
