# -*- coding: utf-8 -*-
"""
WaymoV2(Front) セマンティック(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール v3.1（改）〔フォールバック禁止／成功まで再試行／写実優先・芸術抑制／整合性自己修正／Negative Prompt出力〕
固定パス: /home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v3.1.py

主な変更点（本改版）:
- HTTPフォールバック撤廃: requests 必須（urllib フォールバック削除）。失敗は例外で明示。
- 強化ウォームアップ: /api/chat を最大 N 回（指数バックオフ）叩き、空レスを起動待ちとして吸収。
- テーマ語彙の見直し: 日中テーマから街灯系語彙を撤廃。夜限定語を日中へ流入させない。
- System へ整合性ルールを明文化: 「昼/晴/太陽なら街灯/ヘッドライトを主要光源としない」等。
- 矛盾検出→Refine へ矛盾リストを明示: 日/夜/街灯/雨/雪/霧の論理フラグを提示し、どちらか一貫へ修正。
- 失敗時サルベージ強化: (1) テーマ無効 → (2) 温度↓ → (3) 温度↑ → (4) 短文化 → (5) think=none
- 生成物: prompt.txt / negprompt.txt / prompt.meta.json / split別追記jsonl（決定的seedと救済段階を保存）

参考（設計根拠）:
- ControlNet: 凍結SDにゼロ畳み込みで条件を付与（空間整合性に有利）。ICCV 2023. arXiv:2302.05543
- T2I-Adapter: 軽量Adapterで制御信号を付与し、複数条件も合成可能。arXiv:2302.08453
- DriveDreamer: 実運転データ由来の世界モデル/拡張生成。ECCV 2024.
- Diffusers Docs: negative_prompt は望まない要素の抑制に有効（実務知見）。
- gpt-oss: Harmony 形式の chat 応答、thinking トレース分離。
"""

import os
import sys
import argparse
import json
import logging
from logging import handlers
from typing import List, Dict, Any, Tuple, Optional
import time
import traceback
import hashlib
import random
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

# ======== HTTP(フォールバック禁止) ========
try:
    import requests  # フォールバック禁止: 無ければ即エラー
except Exception as e:
    raise ImportError(
        "requests が必要です（フォールバック禁止）。`pip install requests` を実行してください。"
    ) from e

# ===== 既定パス =====
DEFAULT_SEMSEG_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_IMAGE_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss_v3"
DEFAULT_CAMERA = "front"
DEFAULT_SPLITS = ["training", "validation", "testing"]
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# 命名
NAMING_CHOICES = ["predTrainId", "semantic"]
ALLOWED_IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# ===== Cityscapes trainId 0..18 -> 英語名 =====
TRAINID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# ===== 写実優先: 禁止/抑制したい語彙（芸術調/過度な表現） =====
BANNED_STYLE_TOKENS = [
    "cinematic", "bokeh", "HDR", "ultra-detailed", "hyper-detailed", "8k", "film grain",
    "studio lighting", "dramatic lighting", "volumetric light", "epic", "fantasy", "illustration",
    "anime", "cartoon", "oil painting", "watercolor", "digital art", "CGI", "render",
    "tilt-shift", "Dutch angle", "soft focus", "glowing", "lens flare"
]

# ===== 整合テーマ（昼夜・天候のみ。街灯語は除外） =====
STYLE_THEMES = [
    # day
    {"id": "clear_day",      "soft_hint": "on a clear day with natural daylight and realistic shadows", "tags": ["day", "clear"]},
    {"id": "overcast_day",   "soft_hint": "under an overcast sky with soft diffuse daylight", "tags": ["day", "overcast"]},
    {"id": "rainy_day",      "soft_hint": "during light daytime rain with damp roads and mild reflections", "tags": ["day", "rain"]},
    {"id": "foggy_day",      "soft_hint": "in light daytime fog with reduced contrast in the distance", "tags": ["day", "fog"]},
    {"id": "snowy_day",      "soft_hint": "in daytime snow with soft ambient daylight", "tags": ["day", "snow"]},
    # night / twilight（夜限定語は street/led/sodium を含めない）
    {"id": "dry_night",      "soft_hint": "at night with stable visibility on dry asphalt", "tags": ["night"]},
    {"id": "wet_night",      "soft_hint": "at night after light rain with subtle road reflections", "tags": ["night", "wet"]},
    {"id": "dawn",           "soft_hint": "at dawn with low sun and long soft shadows", "tags": ["dawn"]},
    {"id": "dusk",           "soft_hint": "at dusk with warm sky glow as lights begin to appear", "tags": ["dusk"]},
]

# ===== ネガティブプロンプト（写実寄りの不具合/芸術調を抑制） =====
NEGATIVE_PROMPT_BASE = [
    "low quality", "blurry", "oversaturated", "overexposed", "underexposed",
    "cartoon", "anime", "illustration", "cgi", "render", "digital painting",
    "unrealistic colors", "glowing edges", "lens flare", "bokeh", "tilt-shift", "soft focus",
    "duplicate objects", "distorted geometry", "warped perspective", "text watermark"
]

# ===== ロガー =====
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run.log")
    logger = logging.getLogger("promptgen_gptoss_v3_1")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
    if not logger.handlers:
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

def _log_env(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("=== PromptGen v3.1 (Waymo -> gpt-oss:20b) ===")
    logger.info("semseg-root: %s", args.semseg_root)
    logger.info("image-root : %s", args.image_root)
    logger.info("output-root: %s", args.output_root)
    logger.info("splits : %s", " ".join(args.splits))
    logger.info("camera : %s", args.camera)
    logger.info("naming : %s", args.naming)
    logger.info("ollama-url : %s", args.ollama_base_url)
    logger.info("model : %s", args.model)
    logger.info("min-area-ratio: %.6f, min-pixels: %d", args.min_area_ratio, args.min_pixels)
    logger.info("num-predict: %d", args.num_predict)
    logger.info("run-seed : %d", args.run_seed)
    logger.info("overwrite : %s", args.overwrite)
    logger.info("limit : %d", args.limit)
    logger.info("retries (HTTP) : %d", args.retries)
    logger.info("warmup : %s", args.warmup_check)
    logger.info("think : %s", args.think)
    logger.info("top-k-labels : %d", args.top_k_labels)
    logger.info("theme-mode : %s", args.theme_mode)
    logger.info("refine-pass : %s", args.refine_pass)
    logger.info("retry-until-success : %s", args.retry_until_success)
    logger.info("max-attempts-per-item : %d", args.max_attempts_per_item)
    logger.info("requests : required (no fallback)")

# ===== 走査 =====
def _list_semseg_files(semseg_split_root: str, camera: str, naming: str) -> List[str]:
    suffix = "predTrainId" if naming == "predTrainId" else "semantic"
    base = os.path.join(semseg_split_root, camera)
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(f"_{suffix}.npy"):
                out.append(os.path.join(r, f))
    return sorted(out)

def _infer_image_path_for_npy(npy_path: str, semseg_split_root: str, image_split_root: str) -> Optional[str]:
    d = os.path.dirname(npy_path)
    try:
        rel_dir = os.path.relpath(d, semseg_split_root)  # front/{segment}
    except Exception:
        return None
    stem = Path(npy_path).name.split("_")[0]  # first / mid10s / last
    for ext in ALLOWED_IMG_EXTS:
        cand = os.path.join(image_split_root, rel_dir, f"{stem}{ext}")
        if os.path.exists(cand):
            return cand
    return None

# ===== 乱数 / 再現性 =====
def _derive_seed_for_item(run_seed: int, key: str, attempt_idx: int = 0) -> int:
    base = f"{run_seed}@{key}@attempt{attempt_idx}"
    h = hashlib.sha256(base.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF

def _rng_from_seed(seed: int) -> random.Random:
    rng = random.Random(); rng.seed(seed); return rng

# ===== ラベル抽出 =====
def extract_present_labels(seg: np.ndarray, min_area_ratio: float, min_pixels: int) -> List[Tuple[int, float]]:
    if seg.ndim != 2:
        raise ValueError(f"seg ndim expected 2, got {seg.ndim}")
    h, w = seg.shape; total = float(h*w)
    cnt = np.bincount(seg.flatten(), minlength=19).astype(np.int64)
    present: List[Tuple[int, float]] = []
    for cls_id in range(19):
        px = int(cnt[cls_id]); ratio = px/total if total>0 else 0.0
        if px >= min_pixels or ratio >= min_area_ratio:
            present.append((cls_id, ratio))
    present.sort(key=lambda x: x[1], reverse=True)
    return present

# ===== 英語化ユーティリティ =====
def pluralize_noun(name: str) -> str:
    irregular = {
        "person": "people",
        "traffic light": "traffic lights",
        "traffic sign": "traffic signs",
    }
    if name in irregular:
        return irregular[name]
    if name.endswith("y") and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"

def nouns_to_phrase(names: List[str]) -> str:
    names = [pluralize_noun(n) for n in names]
    if not names: return ""
    if len(names) == 1: return names[0]
    if len(names) == 2: return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def scene_hint_from_labels(label_names: List[str]) -> str:
    s = set(label_names)
    if "building" in s and "road" in s and ("sidewalk" in s or "traffic sign" in s):
        return "a city street scene captured from a front-facing dashcam"
    if "terrain" in s and "road" in s and "vegetation" in s and "building" not in s:
        return "a suburban or rural road scene seen from a moving vehicle"
    if "road" in s and ("truck" in s or "bus" in s) and "building" not in s:
        return "a highway scene viewed from a vehicle"
    return "a driving scene viewed from a vehicle's front camera"

# ===== テーマ選択（再現性あり／Soft Hint or Force） =====
def choose_theme(rng: random.Random, mode: str = "soft") -> Optional[Dict[str, Any]]:
    if mode == "none":
        return None
    idx = rng.randrange(len(STYLE_THEMES))
    return STYLE_THEMES[idx]

# ===== ネガティブプロンプト =====
def build_negative_prompt() -> str:
    return ", ".join(NEGATIVE_PROMPT_BASE)

# ===== System / User プロンプト =====
def build_system_prompt(max_words: int) -> str:
    banned = ", ".join(BANNED_STYLE_TOKENS)
    # 昼夜・天候・照明の整合性を厳密に：昼/晴/日光→街灯/ヘッドライトを主要光源にしない
    return (
        "You are an expert prompt writer for photorealistic text-to-image diffusion used in autonomous driving datasets. "
        f"Write exactly one natural English prompt in 1–2 sentences (<= {max_words} words). "
        "Keep a realistic dashcam viewpoint and plausible road geometry. "
        "Base descriptions only on the provided object categories (use natural synonyms only; do not invent new categories). "
        "Ensure internal consistency among time-of-day, weather, and lighting. "
        "If daytime/clear/sunlight terms are used, do not describe streetlights or car headlights as the main illumination; "
        "if nighttime terms are used, such lights are acceptable. "
        f"Avoid artistic or cinematic style words such as: {banned}. "
        "Output ONLY the final prompt text—no labels, bullet points, or JSON."
    )

def build_user_prompt(objects_phrase: str, scene_hint: str, theme_hint: Optional[str], theme_mode: str) -> str:
    soft_line = (f"- Soft style preference: {theme_hint}\n" if (theme_hint and theme_mode != "none") else "")
    return (
        "Context:\n"
        f"- Scene: {scene_hint}\n"
        f"- Objects present: {objects_phrase}\n"
        f"{soft_line}"
        "Instruction:\n"
        "- Describe naturally in 1–2 sentences suitable for a diffusion model.\n"
        "- Prefer the soft style if coherent; otherwise choose a fully consistent alternative.\n"
        "- Avoid rigid lists or meta language.\n"
    )

# ===== HTTP =====
def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, data=data, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()

def _extract_text_from_message_obj(m: Any) -> str:
    if isinstance(m, dict):
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts: List[str] = []
            for part in c:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    for k in ("text", "content", "value"):
                        v = part.get(k)
                        if isinstance(v, str): parts.append(v)
            text = " ".join(parts).strip()
            if text: return text
    return ""

# ===== Ollama 呼び出し（Harmony: /api/chat 固定） =====
def call_ollama_chat(
    base_url: str, model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
    timeout_sec: int=120, retries: int=2, retry_backoff_ms: int=800, think: Optional[str]="low"
) -> Tuple[str, Dict[str, Any]]:
    """
    戻り値: (text, last_obj)
    - gpt-oss 前提: chat のみ使用。generate へのフォールバックはしない（禁止）。
    """
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    last_obj: Dict[str, Any] = {}
    for i in range(max(1, retries+1)):
        try:
            payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": options}
            if think and think != "none":
                payload["think"] = str(think)
            last_obj = _http_post_json(chat_url, payload, timeout_sec)
            if isinstance(last_obj, dict) and "message" in last_obj:
                txt = _extract_text_from_message_obj(last_obj.get("message"))
                if isinstance(txt, str) and txt.strip():
                    return txt.strip(), last_obj
            raise RuntimeError("Ollama /api/chat returned no usable text")
        except Exception:
            if i < retries:
                time.sleep(max(0, retry_backoff_ms)/1000.0)
                continue
            raise
    return "", last_obj  # 到達しない想定

# ===== 矛盾スキャン（強化版） =====
_DAY_WORDS    = r"\b(day|clear|sunny|daylight|noon|midday|afternoon|morning)\b"
_NIGHT_WORDS  = r"\b(night|nighttime|moonlit|stars|streetlights?)\b"
_LIGHTWORDS   = r"\b(streetlights?|headlights?|taillights?)\b"
_WET_WORDS    = r"\b(wet|after (light )?rain|damp|rainy|drizzle|showers)\b"
_SNOW_WORDS   = r"\b(snow|snowy|snowfall|snowing)\b"
_FOG_WORDS    = r"\b(fog|foggy|mist|haze|hazy)\b"

def consistency_scan(text: str) -> Dict[str, bool]:
    t = text.lower()
    flags = {
        "day_and_streetlight": bool(re.search(_DAY_WORDS, t) and re.search(_LIGHTWORDS, t)),
        "day_and_night":       bool(re.search(_DAY_WORDS, t) and re.search(_NIGHT_WORDS, t)),
        "snow_and_rain":       bool(re.search(_SNOW_WORDS, t) and re.search(_WET_WORDS, t)),
        "overlength":          len(text.split()) > 70,  # ゆるめ
        "looks_instr":         looks_like_instruction(text),
    }
    return flags

def looks_like_instruction(s: str) -> bool:
    t = s.lower()
    bad_subs = [
        "we need to", "return only", "do not include", "bullet points", "no brand names",
        "avoid", "instruction:", "objects:", "scene:", "atmosphere:", "lighting:", "should", "must",
    ]
    return any(k in t for k in bad_subs)

# ===== 精緻化（Pass2）：矛盾修正・自然化・語数制限 =====
def refine_prompt(
    base_url: str, model: str, raw_prompt: str, options: Dict[str, Any],
    timeout_sec: int, retries: int, think: Optional[str], max_words: int, diag_flags: Dict[str, bool]
) -> Tuple[str, Dict[str, Any]]:
    # 具体的な矛盾フラグを列挙して修正を要求
    issues = []
    if diag_flags.get("day_and_streetlight"): issues.append("daytime terms conflict with streetlights/headlights")
    if diag_flags.get("day_and_night"):       issues.append("daytime and nighttime terms are both present")
    if diag_flags.get("snow_and_rain"):       issues.append("snow and rain are both present")
    issue_text = ("; ".join(issues)) if issues else "No explicit contradictions detected; just polish and shorten."
    checklist = (
        f"Fix: {issue_text}. Keep <= {max_words} words, dashcam viewpoint, plausible road geometry. "
        "If daytime/clear/sunlight terms are present, remove streetlights/headlights as main illumination. "
        "Do not introduce new object categories beyond those already implied. Output only the final prompt."
    )
    messages = [
        {"role": "system", "content": "You are a precise copy editor for prompts. Keep meaning, fix contradictions, and polish."},
        {"role": "user", "content": f"Original:\n{raw_prompt}\n\nInstruction:\n{checklist}"},
    ]
    return call_ollama_chat(base_url, model, messages, options, timeout_sec, retries, think)

# ===== 保存 =====
def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_txt(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ===== バリデーション =====
def is_valid_prompt(text: str, max_words: int) -> bool:
    if not isinstance(text, str): return False
    t = " ".join(text.replace("\n", " ").split()).strip()
    if not t: return False
    if looks_like_instruction(t): return False
    words = [w for w in t.split(" ") if w]
    if len(words) > max_words + 10:
        return False
    return True

# ===== 1フレーム生成（成功まで再試行） =====
def generate_for_item_until_success(
    logger: logging.Logger,
    args: argparse.Namespace,
    npy_path: str,
    semseg_split_root: str,
    image_split_root: str,
    out_dir: str,
    rel_dir: str,
    stem: str
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    out_txt  = os.path.join(out_dir, f"{stem}_prompt.txt")
    out_neg  = os.path.join(out_dir, f"{stem}_negprompt.txt")
    out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
    out_dbg  = os.path.join(out_dir, f"{stem}_prompt.debug.json")
    jsonl_path = os.path.join(args.output_root, f"prompts_{rel_dir.replace(os.sep,'_')}.jsonl")

    seg = np.load(npy_path)
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)

    label_pairs_all = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
    if not label_pairs_all:
        raise RuntimeError("no valid labels after thresholding")
    label_pairs_top = label_pairs_all[:max(1, args.top_k_labels)]
    label_names_top = [TRAINID_TO_NAME[i] for (i, _) in label_pairs_top]
    scene_hint = scene_hint_from_labels([TRAINID_TO_NAME[i] for (i, _) in label_pairs_all])

    image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)
    negative_prompt_text = build_negative_prompt()

    attempt = 0
    success = False
    final_text: Optional[str] = None

    def salvage_stage(attempt_idx: int) -> int:
        # 0:標準 →1:テーマ無効 →2:温度低 →3:温度高 →4:短文化 →5:think=none →(戻る)
        return attempt_idx % 6

    while True:
        attempt += 1
        stage = salvage_stage(attempt - 1)

        if (args.max_attempts_per_item >= 0) and (attempt > args.max_attempts_per_item):
            logger.error("[FAIL] attempts exceeded for %s", npy_path)
            break

        frame_seed = _derive_seed_for_item(args.run_seed, npy_path, attempt_idx=attempt-1)
        rng = _rng_from_seed(frame_seed)

        # テーマ（夜/昼を跨ぐ事故を減らす）
        theme_mode_eff = args.theme_mode
        if stage == 1:
            theme_mode_eff = "none"
        theme_obj = choose_theme(rng, mode=theme_mode_eff)
        theme_hint = (theme_obj or {}).get("soft_hint") if theme_obj else None

        # オブジェクト列挙
        objects_phrase = nouns_to_phrase(label_names_top)

        system_msg = build_system_prompt(args.max_words)
        user_msg = build_user_prompt(objects_phrase, scene_hint, theme_hint, theme_mode_eff)
        messages = [{"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}]

        temperature = float(args.temperature)
        num_predict = int(args.num_predict)
        think_level = None if (args.think == "none") else args.think
        if stage == 2:
            temperature = max(0.2, temperature * 0.7)
        elif stage == 3:
            temperature = min(1.2, temperature * 1.2)
        elif stage == 4:
            num_predict = max(60, int(num_predict * 0.7))
        elif stage == 5:
            think_level = None  # think フィールドを外す

        options = {
            "temperature": float(temperature),
            "top_p": float(args.top_p),
            "repeat_penalty": float(args.repeat_penalty),
            "num_predict": int(num_predict),
            "seed": int(frame_seed),
        }

        try:
            # === Pass1 ===
            raw_text, last_obj1 = call_ollama_chat(
                base_url=args.ollama_base_url, model=args.model, messages=messages, options=options,
                timeout_sec=args.timeout_sec, retries=args.retries, retry_backoff_ms=args.retry_backoff_ms,
                think=think_level
            )
            clean1 = " ".join(raw_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
            flags1 = consistency_scan(clean1)

            # === Pass2（Refine with explicit contradictions） ===
            final_candidate = clean1
            last_obj2 = None
            if args.refine_pass:
                ref_text, last_obj2 = refine_prompt(
                    base_url=args.ollama_base_url, model=args.model, raw_prompt=clean1, options=options,
                    timeout_sec=args.timeout_sec, retries=args.retries, think=think_level,
                    max_words=args.max_words, diag_flags=flags1
                )
                final_candidate = " ".join(ref_text.replace("\n", " ").replace("\"", "").replace("```", "").split())

            flags2 = consistency_scan(final_candidate)

            if not is_valid_prompt(final_candidate, args.max_words) or any([
                flags2.get("day_and_streetlight"),
                flags2.get("day_and_night"),
                flags2.get("snow_and_rain"),
                flags2.get("looks_instr"),
            ]):
                raise RuntimeError("final prompt failed validation or consistency checks")

            meta = {
                "split_dir": rel_dir, "camera": args.camera,
                "npy_path": npy_path, "image_path": image_path,
                "labels_top": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs_top],
                "labels_all": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs_all],
                "scene_hint": scene_hint,
                "theme": theme_obj,
                "ollama": {"base_url": args.ollama_base_url, "model": args.model, "options": options},
                "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "attempt": int(attempt),
                "timestamp": int(time.time()),
                "diagnostics": {"pass1_flags": flags1, "final_flags": flags2, "salvage_stage": stage},
                "pass1_text": clean1 if args.refine_pass else None,
                "negative_prompt": negative_prompt_text,
            }

            save_txt(out_txt, final_candidate)
            save_txt(out_neg, negative_prompt_text)
            safe_write_json(out_meta, meta)
            append_jsonl(jsonl_path, {"prompt": final_candidate, "negative_prompt": negative_prompt_text, **meta})

            final_text = final_candidate
            success = True
            break

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("[RETRY] %s at stage=%d attempt=%d | %s", type(e).__name__, stage, attempt, npy_path)
            dbg = {
                "npy_path": npy_path, "split_dir": rel_dir, "attempt": attempt,
                "error_type": type(e).__name__, "error_msg": str(e),
                "traceback": tb, "salvage_stage": stage
            }
            try: safe_write_json(out_dbg, dbg)
            except Exception: pass

            if not args.retry_until_success:
                break

            time.sleep(max(0.2, args.retry_backoff_ms/1000.0))
            continue

    return success, final_text, None

# ===== CLI / ウォームアップ / main =====
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator (Harmony/chat, v3.1 改)")
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--naming", type=str, choices=NAMING_CHOICES, default="predTrainId")
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument("--min-pixels", type=int, default=4000)
    ap.add_argument("--top-k-labels", type=int, default=7)
    ap.add_argument("--run-seed", type=int, default=42)
    ap.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    ap.add_argument("--repeat-penalty", type=float, default=1.12, dest="repeat_penalty")
    ap.add_argument("--num-predict", type=int, default=120, dest="num_predict")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--retries", type=int, default=2)  # HTTPリトライ
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")
    # v3.1 改
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--theme-mode", type=str, choices=["none", "soft", "force"], default="soft")
    ap.add_argument("--refine-pass", dest="refine_pass", action="store_true")
    ap.add_argument("--no-refine-pass", dest="refine_pass", action="store_false")
    ap.set_defaults(refine_pass=True)
    ap.add_argument("--retry-until-success", dest="retry_until_success", action="store_true")
    ap.add_argument("--no-retry-until-success", dest="retry_until_success", action="store_false")
    ap.set_defaults(retry_until_success=True)
    ap.add_argument("--max-attempts-per-item", type=int, default=-1)
    # ウォームアップ回数
    ap.add_argument("--warmup-retries", type=int, default=12, help="warmup 最大リトライ回数（指数バックオフ）")
    return ap.parse_args()

def _ollama_warmup(logger: logging.Logger, base_url: str, model: str, timeout_sec: int, think: Optional[str], warmup_retries: int) -> None:
    base = base_url.rstrip("/")
    url = base + "/api/chat"
    attempt = 0
    sleep_ms = 250
    while attempt < max(1, warmup_retries):
        attempt += 1
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK in one word."}],
            "stream": False,
            "options": {"num_predict": 8},
        }
        if think and think != "none":
            payload["think"] = str(think)
        try:
            obj = _http_post_json(url, payload, timeout_sec)
            msg = (obj.get("message") or {}).get("content", "")
            if isinstance(msg, str) and msg.strip():
                logger.info("[warmup] chat responded at attempt=%d", attempt)
                return
            logger.warning("[warmup] empty text at attempt=%d; model may be mid-load.", attempt)
        except Exception as e:
            logger.warning("[warmup] chat failed at attempt=%d: %s", attempt, repr(e))
        time.sleep(max(0, sleep_ms)/1000.0)
        sleep_ms = min(int(sleep_ms * 1.7), 5000)  # 指数バックオフ
    logger.warning("Warmup(chat): gave up after %d attempts (will continue and rely on per-item retries).", attempt)

def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)
    logger = _setup_logger(args.output_root, verbose=args.verbose)
    _log_env(logger, args)

    think_level = None if (args.think == "none") else args.think
    if args.warmup_check:
        _ollama_warmup(logger, args.ollama_base_url, args.model, args.timeout_sec, think=think_level, warmup_retries=args.warmup_retries)

    total_ok_all = 0
    total_ng_all = 0

    for split in args.splits:
        semseg_split_root = os.path.join(args.semseg_root, split)
        image_split_root = os.path.join(args.image_root, split)

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0:
            files = files[:args.limit]
        if not files:
            logger.warning("[%s] no targets under: %s", split, semseg_split_root)
            continue

        logger.info("[%s] target files: %d", split, len(files))
        ok_count = 0
        ng_count = 0

        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            stem = Path(npy_path).name.split("_")[0]
            out_dir = os.path.join(args.output_root, split, rel_dir)

            try:
                if (not args.overwrite):
                    out_txt = os.path.join(out_dir, f"{stem}_prompt.txt")
                    out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
                    if os.path.exists(out_txt) and os.path.exists(out_meta):
                        pbar.set_postfix_str("skip")
                        ok_count += 1
                        continue

                success, final_text, _ = generate_for_item_until_success(
                    logger, args, npy_path, semseg_split_root, image_split_root, out_dir, rel_dir, stem
                )
                if success and final_text:
                    ok_count += 1
                    pbar.set_postfix_str("ok")
                else:
                    ng_count += 1
                    pbar.set_postfix_str("ng")

            except Exception as e:
                ng_count += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), npy_path)
                out_dbg = os.path.join(out_dir, f"{stem}_prompt.debug.json")
                dbg = {"npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e), "traceback": tb, "split": split}
                try: safe_write_json(out_dbg, dbg)
                except Exception: pass

        logger.info("[%s] done: OK=%d NG=%d -> %s", split, ok_count, ng_count, args.output_root)
        total_ok_all += ok_count
        total_ng_all += ng_count

    if total_ng_all == 0:
        print(f"✅ ALL DONE: OK={total_ok_all}, NG=0, out={args.output_root}")
    else:
        print(f"⚠️ DONE WITH ERRORS: OK={total_ok_all}, NG={total_ng_all}, out={args.output_root}")

if __name__ == "__main__":
    main()
