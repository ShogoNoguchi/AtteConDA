#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_caption_time_weather.py

Qwen が生成したキャプション(JSONL)を再帰走査し、
「時刻・天候の表現」を正規表現で検出。
ヒットしたサンプルの “プロンプト（キャプション）パス” を CSV に出力し、
件数サマリも表示するユーティリティ。

既定:
  - 入力ルート: /data/syndiff_prompts/raw_captions
  - 出力CSV   : /data/syndiff_prompts/checks/caption_time_weather_hits.csv
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ========= 可変パラメータ =========
DEFAULT_RAW_CAP_ROOT = "/data/syndiff_prompts/raw_captions"
DEFAULT_OUT_CSV      = "/data/syndiff_prompts/checks/caption_time_weather_hits.csv"

# raw_captions/ の 1階層目名から prompts_* のCSVパスを推定するために使う
def infer_prompt_csv_path(raw_cap_root: Path, caption_jsonl: Path) -> str:
    """
    raw_captions/{key}/xxx.jsonl -> prompts_train/{key}.csv  (通常データセット)
    raw_captions/waymo_{split}/xxx.jsonl -> prompts_eval_waymo/waymo_{split}.csv
    """
    try:
        rel = caption_jsonl.relative_to(raw_cap_root)
    except Exception:
        return ""
    key = rel.parts[0] if len(rel.parts) > 0 else ""
    out_root = raw_cap_root.parent  # /data/syndiff_prompts
    if key.startswith("waymo_"):
        return str(out_root / "prompts_eval_waymo" / f"{key}.csv")
    elif key != "":
        return str(out_root / "prompts_train" / f"{key}.csv")
    return ""


def build_weather_patterns() -> List[str]:
    # ASCII hyphen と各種ダッシュに対応
    HYP = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
    p: List[str] = []

    # 降水系
    p.append(rf"\brain(?:{HYP}\w+|\w*)\b")     # rain, rainy, rainfall, rain-soaked, ...
    p.append(r"\bdrizzle\w*\b")
    p.append(r"\bshower(?:s)?\b")
    p.append(r"\bdownpour(?:s)?\b")
    p.append(r"\brainstorm(?:s)?\b")
    p.append(r"\bthunderstorm(?:s)?\b")
    p.append(r"\bthunder\w*\b")
    p.append(r"\blightning\b")
    p.append(r"\bwet\s+(?:road|roads|asphalt|street|streets|pavement|surface|ground)s?\b")
    p.append(r"\bpuddle(?:s)?\b")
    p.append(rf"\brain(?:{HYP})?slick(?:ed)?\b")  # rain-slick, rain‑slicked
    p.append(r"\braindrop(?:s)?\b")

    # 雪・氷
    p.append(rf"\bsnow(?:{HYP}\w+|\w*)\b")    # snow, snowy, snow-covered, ...
    p.append(r"\bsleet\w*\b")
    p.append(r"\bhail\w*\b")
    p.append(r"\bblizzard\w*\b")
    p.append(r"\bflurr(?:y|ies)\b")
    p.append(r"\bicy\b")                      # 'icy' 単語として
    p.append(r"\bblack\s+ice\b")

    # 空模様・日射
    p.append(r"\bovercast\b")
    p.append(r"\bcloud(?:y|s)\b")
    p.append(r"\bcloud\s+cover\b")
    # 'clear' は曖昧なので文脈限定
    p.append(r"\bclear\s+(?:sky|skies|weather|day|night|conditions)\b")
    p.append(r"\bsunny\b")
    p.append(r"\bsun(?:light|shine|lit)\b")

    # 視程・大気質
    p.append(r"\bfog(?:gy)?\b")
    p.append(r"\bmist(?:y)?\b")
    p.append(r"\bhaze(?:y)?\b")
    p.append(r"\bsmog\b")
    p.append(r"\bsmok(?:y|ey)\b")

    # 風・湿度
    p.append(r"\bwind(?:y|s)?\b")
    p.append(r"\bbreeze(?:y)?\b")
    p.append(r"\bgust(?:s)?\b")
    p.append(r"\bgale(?:s)?\b")
    p.append(r"\bhumid(?:ity)?\b")
    return p


def build_time_patterns() -> List[str]:
    HYP = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
    p: List[str] = []

    # 時間帯
    p.append(r"\bmorning\b")
    p.append(r"\bafternoon\b")
    p.append(r"\bevening\b")
    p.append(r"\bnight(?:time)?\b")
    p.append(r"\bnoon\b")
    p.append(rf"\bmid(?:{HYP})?day\b")   # ★修正: rfr -> rf  （midday / mid-day）
    p.append(r"\bmidnight\b")
    p.append(r"\bdaylight\b")
    p.append(r"\bdaytime\b")
    p.append(r"\bduring\s+the\s+day\b")
    p.append(r"\bin\s+daylight\b")

    # 薄明や関連
    p.append(r"\bdawn\b")
    p.append(r"\bdusk\b")
    p.append(r"\btwilight\b")
    p.append(rf"\bgolden(?:{HYP})?hour\b")  # ★修正: rfr -> rf
    p.append(rf"\bblue(?:{HYP})?hour\b")    # ★修正: rfr -> rf
    p.append(rf"\bpre(?:{HYP})?dawn\b")     # ★修正: rfr -> rf  （pre-dawn / predawn）
    p.append(r"\bdaybreak\b")
    p.append(r"\bfirst\s+light\b")
    p.append(r"\bsunrise\b")
    p.append(r"\bsunset\b")
    p.append(r"\bsundown\b")
    p.append(r"\bafter\s+dark\b")
    p.append(r"\bnocturnal\b")
    p.append(r"\bmoon(?:lit|light)\b")

    # 定型句
    p.append(r"\bat\s+night\b")
    p.append(r"\bin\s+the\s+morning\b")
    p.append(r"\bin\s+the\s+afternoon\b")
    p.append(r"\bin\s+the\s+evening\b")

    # 数値時刻（12時間）
    p.append(r"\b(?:[1-9]|1[0-2])\s?(?:a\.?m\.?|p\.?m\.?)\b")                 # 7 pm / 7pm / 7 p.m.
    p.append(r"\b(?:[1-9]|1[0-2])[:：h][0-5]\d\s?(?:a\.?m\.?|p\.?m\.?)\b")   # 7:30 pm / 7h30 pm
    # 数値時刻（24時間）
    p.append(r"\b(?:[01]?\d|2[0-3])[:：][0-5]\d\b")                          # 19:30
    return p


def compile_union_regex(patterns: List[str]) -> re.Pattern:
    """
    個々のパターンを OR で束ねて単一の正規表現にする。
    IGNORECASE + Unicode 対応。
    """
    union = "|".join(f"(?:{pat})" for pat in patterns)
    return re.compile(union, flags=re.IGNORECASE)


def read_last_caption_from_jsonl(jsonl_path: Path) -> Tuple[str, str]:
    """
    JSONL の最後の行を読んで "caption" と "image" を返す（なければ空文字）。
    """
    cap, img = "", ""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = line
            if last is not None:
                obj = json.loads(last)
                cap = str(obj.get("caption", "") or "")
                img = str(obj.get("image", "") or "")
    except Exception:
        pass
    return cap, img


def scan(root: Path, out_csv: Path) -> None:
    root = root.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    weather_re = compile_union_regex(build_weather_patterns())
    time_re    = compile_union_regex(build_time_patterns())

    rows: List[List[str]] = []
    total = 0
    has_weather = 0
    has_time = 0
    has_any = 0
    none_any = 0

    jsonl_files = sorted(root.rglob("*.jsonl"))
    for jpath in jsonl_files:
        total += 1
        caption, image_path = read_last_caption_from_jsonl(jpath)

        # マッチ抽出（重複排除）
        w_hits = [m.group(0) for m in weather_re.finditer(caption)]
        t_hits = [m.group(0) for m in time_re.finditer(caption)]
        w_uniq = sorted(set([h.strip() for h in w_hits]))
        t_uniq = sorted(set([h.strip() for h in t_hits]))

        hw = int(len(w_uniq) > 0)
        ht = int(len(t_uniq) > 0)
        any_flag = 1 if (hw or ht) else 0

        has_weather += hw
        has_time    += ht
        has_any     += any_flag
        none_any    += (0 if any_flag else 1)

        # データセット推定 & prompts CSV の場所
        ds_key = ""
        try:
            rel = jpath.relative_to(root)
            ds_key = rel.parts[0] if len(rel.parts) > 0 else ""
        except Exception:
            ds_key = ""

        prompt_csv_path = infer_prompt_csv_path(root, jpath)

        # 出力行
        rows.append([
            str(jpath),                           # caption_path (プロンプト=キャプションパス)
            image_path,                           # 画像元
            ds_key,                               # データセットキー
            prompt_csv_path,                      # 集約プロンプトCSVのパス（参考）
            str(hw), str(ht),                     # has_weather, has_time
            ";".join(w_uniq),                     # weather_terms
            ";".join(t_uniq),                     # time_terms
            caption.replace("\n", " ").strip(),   # caption 本文（確認用）
        ])

    # CSV 書き出し
    header = [
        "caption_path","image_path","dataset","prompt_csv_path",
        "has_weather","has_time","weather_terms","time_terms","caption"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # サマリ表示
    print("=== Scan Summary ===")
    print(f"root            : {root}")
    print(f"out_csv         : {out_csv}")
    print(f"jsonl files     : {total}")
    print(f"has_weather     : {has_weather}")
    print(f"has_time        : {has_time}")
    print(f"has_any (either): {has_any}")
    print(f"no_hits         : {none_any}")
    print("====================")


def main():
    ap = argparse.ArgumentParser(description="Scan Qwen captions for time/weather terms and emit CSV of hits.")
    ap.add_argument("--raw-cap-root", type=str, default=DEFAULT_RAW_CAP_ROOT,
                    help="raw captions のルート (/data/syndiff_prompts/raw_captions)")
    ap.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV,
                    help="検出結果CSVの出力先 (/data/syndiff_prompts/checks/caption_time_weather_hits.csv)")
    args = ap.parse_args()

    scan(Path(args.raw_cap_root), Path(args.out_csv))


if __name__ == "__main__":
    main()
