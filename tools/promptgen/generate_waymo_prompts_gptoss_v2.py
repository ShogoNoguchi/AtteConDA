# -*- coding: utf-8 -*-
"""
WaymoV2(Front) セマンティックラベル(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール v2（自由度強化＋スタイル整合性自動修復 / Harmony前提 / thinkingは保存しない）

設計要点:
- スタイルテーマを "一意→整合な語彙群" にマップ（昼×街灯のような矛盾を事前に排除）。
- few-shot夜例を削除し、LLMの言語自由度を拡大（ただし最低限の制約: 構図維持/過度な新カテゴリ追加の抑制）。
- 自動整合性チェック＆自己修復（最大試行回数）で出力の一貫性を担保。
- RUN間の再現性（run_seed + ファイルパス → frame_seed）、RUN内の多様性（seedからの軽い揺らぎ）を両立。
- Ollama /api/chat を強制（gpt-ossはHarmony形式が前提）。
- thinking(trace) はトップレベル "think" を渡せるが、thinking出力は保存しない。

既知の前提:
- OneFormer Cityscapes trainId (0..18) .npy が存在する。
- WaymoV2の front カメラ構造およびRGBパスは既存のレイアウトに準拠。

出力:
- prompt.txt（1行）
- prompt.meta.json（メタ）
- prompts_{split}.jsonl（集約）
- 矛盾や例外時は prompt.debug.json に詳細を保存

使用例:
$ python /home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v2.py \
    --semseg-root /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes \
    --image-root  /home/shogo/coding/datasets/WaymoV2/extracted \
    --output-root /home/shogo/coding/datasets/WaymoV2/Prompts_gptoss_v2 \
    --splits training validation testing \
    --camera front --naming predTrainId \
    --run-seed 20251029 --warmup-check \
    --repair --repair-max-attempts 2 --max-words 64 \
    --verbose

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

# requests があれば使用、無ければ urllib にフォールバック
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request  # type: ignore
    _HAS_REQUESTS = False

# ===== 既定パス =====
DEFAULT_SEMSEG_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_IMAGE_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss_v2"
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

# ===== スタイルテーマ（矛盾の無い語彙をテーマ別に定義） =====
#   各テーマは atmosphere の短句（天候/時間帯）と、互換性のある lighting / surface / ambience 語彙の候補を持つ。
STYLE_THEMES = {
    "clear_day": {
        "atmosphere": ["on a clear day", "under a clear blue sky", "in bright daylight"],
        "lighting": [
            "crisp sunlight with clean shadows", "bright midday light", "clear visibility",
            "dry asphalt and high clarity"
        ],
        "negative_markers": ["night", "streetlight", "moon", "headlight", "taillight", "neon"],
    },
    "overcast": {
        "atmosphere": ["in overcast weather", "under a gray sky", "with cloud-covered daylight"],
        "lighting": [
            "soft diffuse light", "muted contrast", "even illumination without harsh shadows"
        ],
        "negative_markers": ["night", "streetlight", "moon", "neon"],
    },
    "rain": {
        "atmosphere": ["in rainy weather", "during light rain", "under passing showers"],
        "lighting": [
            "wet asphalt with gentle reflections", "raindrops softening highlights", "overcast diffuse light"
        ],
        "negative_markers": ["dry asphalt", "dusty", "sun-baked"],
    },
    "after_rain": {
        "atmosphere": ["after light rain", "shortly after rainfall"],
        "lighting": [
            "damp asphalt with subtle reflections", "soft post-rain glow", "clearing skies with residual moisture"
        ],
        "negative_markers": ["heavy snowfall", "blizzard"],
    },
    "fog": {
        "atmosphere": ["in foggy weather", "in light fog", "in morning mist"],
        "lighting": [
            "low contrast and diffused light", "haze softening distant details"
        ],
        "negative_markers": ["crisp high-contrast shadows", "deep black shadows"],
    },
    "snow": {
        "atmosphere": ["in snowy weather", "amid light snowfall", "with fresh snow on the ground"],
        "lighting": [
            "bright overcast glow", "muted colors with snow reflection", "soft light scattering"
        ],
        "negative_markers": ["heavy rain", "wet asphalt (without snow)"],
    },
    "night": {
        "atmosphere": ["at night", "during nighttime"],
        "lighting": [
            "streetlights and car headlights shaping the scene",
            "cool LED streetlights", "warm sodium streetlights",
            "reflections from headlights on the asphalt"
        ],
        "negative_markers": ["bright daylight", "blue sky", "direct sunlight", "sunlit"],
    },
    "golden_hour": {
        "atmosphere": ["at golden hour", "in late-afternoon golden light"],
        "lighting": [
            "warm low-angle sunlight with long shadows", "backlit glow near sunset"
        ],
        "negative_markers": ["neon night glow", "deep midnight"],
    },
    "dawn": {
        "atmosphere": ["at dawn", "in early morning light"],
        "lighting": [
            "cool pre-sunrise tones", "soft low-angle light", "quiet streets before rush hour"
        ],
        "negative_markers": ["noon sun overhead", "intense midday glare"],
    },
    "dusk": {
        "atmosphere": ["at dusk", "during blue hour"],
        "lighting": [
            "fading ambient light with early streetlights", "subtle twilight contrast"
        ],
        "negative_markers": ["harsh midday sun"],
    },
    "hazy_summer": {
        "atmosphere": ["in hazy conditions", "on a humid summer afternoon"],
        "lighting": [
            "slightly reduced clarity", "softened distance cues", "gentle glare"
        ],
        "negative_markers": ["crystal-clear winter air"],
    },
}

# ===== ロガー =====
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run_v2.log")
    logger = logging.getLogger("promptgen_gptoss_v2")
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
    logger.info("=== PromptGen v2 (Waymo -> gpt-oss:20b) ===")
    logger.info("semseg-root: %s", args.semseg_root)
    logger.info("image-root : %s", args.image_root)
    logger.info("output-root: %s", args.output_root)
    logger.info("splits : %s", " ".join(args.splits))
    logger.info("camera : %s", args.camera)
    logger.info("naming : %s", args.naming)
    logger.info("ollama-url : %s", args.ollama_base_url)
    logger.info("model : %s", args.model)
    logger.info("min-area-ratio: %.6f, min-pixels: %d", args.min_area_ratio, args.min_pixels)
    logger.info("max-words: %d", args.max_words)
    logger.info("num-predict: %d", args.num_predict)
    logger.info("run-seed : %d", args.run_seed)
    logger.info("overwrite : %s", args.overwrite)
    logger.info("limit : %d", args.limit)
    logger.info("requests : %s", "available" if _HAS_REQUESTS else "fallback to urllib")
    logger.info("retries : %d", args.retries)
    logger.info("warmup : %s", args.warmup_check)
    logger.info("think : %s", args.think)
    logger.info("repair : %s (max-attempts=%d)", args.repair, args.repair_max_attempts)

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

# ===== 乱数 =====
def _derive_seed_for_item(run_seed: int, key: str) -> int:
    h = hashlib.sha256((str(run_seed) + "@" + key).encode("utf-8")).digest()
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

def _group_label_names_by_importance(label_pairs: List[Tuple[int, float]], max_count: int, rng: random.Random) -> List[str]:
    """
    面積上位を優先しつつランダム性もわずかに導入（再現性はframe_seedに依存）。
    roadやsky等の頻出要素は自然に残りやすいが、全列挙は避ける。
    """
    names_sorted = [TRAINID_TO_NAME[i] for (i, _) in label_pairs]
    if not names_sorted:
        return []
    # 上位を強めに残しつつ、尾部を少しシャッフルして短く
    head = names_sorted[:max(1, max_count//2)]
    tail = names_sorted[max(1, max_count//2):max_count+2]  # 少し余裕を見てサンプル
    rng.shuffle(tail)
    picked = list(dict.fromkeys(head + tail))  # 重複除去
    return picked[:max_count]

def _to_natural_phrase(objs: List[str]) -> str:
    # 英語として自然な並べ方に整える（厳密なOxfordは必須ではない）
    if not objs:
        return ""
    if len(objs) == 1:
        return objs[0]
    if len(objs) == 2:
        return f"{objs[0]} and {objs[1]}"
    return ", ".join(objs[:-1]) + f", and {objs[-1]}"

def _scene_hint_from_labels(label_names: List[str]) -> str:
    s = set(label_names)
    if "building" in s and "road" in s and ("sidewalk" in s or "traffic sign" in s):
        return "a city street scene viewed from a front-facing dashcam"
    if "terrain" in s and "road" in s and "vegetation" in s and "building" not in s:
        return "a suburban or rural roadway seen from a moving vehicle"
    if "road" in s and ("truck" in s or "bus" in s) and "building" not in s:
        return "a highway scene seen from a vehicle"
    return "a driving scene captured from the vehicle's front camera"

# ===== テーマ選択（再現性あり） =====
def _sample_theme_id(rng: random.Random, allowed: List[str]) -> str:
    if not allowed:
        allowed = list(STYLE_THEMES.keys())
    return rng.choice(allowed)

def _sample_theme_pack(rng: random.Random, theme_id: str) -> Dict[str, Any]:
    theme = STYLE_THEMES[theme_id]
    atm = rng.choice(theme["atmosphere"])
    # lighting は 1～2 個程度を自然に混ぜる（語彙を軽めに抑える）
    lighting_choices = theme["lighting"][:]
    rng.shuffle(lighting_choices)
    picked = lighting_choices[: rng.randint(1, min(2, len(lighting_choices)))]
    return {
        "theme_id": theme_id,
        "atmosphere": atm,
        "lighting_list": picked,
        "negative_markers": theme.get("negative_markers", []),
    }

# ===== HTTP =====
def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8"); headers = {"Content-Type": "application/json"}
    if _HAS_REQUESTS:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout_sec)  # type: ignore
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, data=data, headers=headers)  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:  # type: ignore
            return json.loads(r.read().decode("utf-8"))

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
                        if isinstance(v, str):
                            parts.append(v)
            text = " ".join(parts).strip()
            if text:
                return text
    return ""

# ===== Ollama 呼び出し（Harmony前提: /api/chat 強制） =====
def call_ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                     timeout_sec: int = 120, retries: int = 2, retry_backoff_ms: int = 800,
                     think: Optional[str] = "low") -> Tuple[str, Dict[str, Any]]:
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    last_obj: Dict[str, Any] = {}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options
    }
    if think and think != "none":
        payload["think"] = str(think)

    for i in range(max(1, retries + 1)):
        try:
            last_obj = _http_post_json(chat_url, payload, timeout_sec)
            if isinstance(last_obj, dict) and "message" in last_obj:
                txt = _extract_text_from_message_obj(last_obj.get("message"))
                if isinstance(txt, str) and txt.strip():
                    return txt.strip(), last_obj
            raise RuntimeError("Ollama /api/chat returned no usable text")
        except Exception:
            if i < retries:
                time.sleep(max(0, retry_backoff_ms) / 1000.0)
                continue
            raise

# ===== メッセージ構築（few-shot廃止・自由度重視・最小制約） =====
def build_initial_messages(scene_hint: str, objects_phrase: str, theme_pack: Dict[str, Any], max_words: int) -> List[Dict[str, str]]:
    # lighting は「候補」扱いで、LLMが自然に統合できるよう 'may include' のニュアンスを与える
    lighting_hint = "; ".join(theme_pack.get("lighting_list", []))
    atmosphere_hint = theme_pack.get("atmosphere", "")
    sys = (
        "You are an expert prompt writer for text-to-image diffusion models in autonomous driving datasets. "
        f"Write one photorealistic English prompt in 1–2 sentences (<= {max_words} words). "
        "Be natural and coherent. Keep the same camera angle and composition as a front-facing dashcam. "
        "Do not add object categories that contradict the provided context; minor paraphrasing is fine. "
        "Return ONLY the prompt text."
    )
    user = (
        f"Context:\n"
        f"- scene: {scene_hint}\n"
        f"- present categories (not exhaustive): {objects_phrase}\n"
        f"- theme: {atmosphere_hint}\n"
        f"- lighting hints (optional, keep consistent with the theme): {lighting_hint}\n"
        f"Write a concise, natural description that would guide a diffusion model."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

# ===== オプション（温度等は seed に基づく軽い揺らぎ） =====
def build_llm_options(rng: random.Random, base_temperature: float, base_top_p: float,
                      base_repeat_penalty: float, num_predict: int, seed: int) -> Dict[str, Any]:
    def jitter(x: float, low: float, high: float, scale: float = 0.12) -> float:
        d = (rng.random() * 2 - 1) * scale
        v = x * (1.0 + d)
        return float(max(low, min(high, v)))
    return {
        "temperature": float(jitter(base_temperature, 0.05, 1.50, 0.12)),
        "top_p": float(jitter(base_top_p, 0.50, 1.00, 0.10)),
        "repeat_penalty": float(jitter(base_repeat_penalty, 1.00, 2.00, 0.10)),
        "num_predict": int(num_predict),
        "seed": int(seed),
    }

# ===== 整合性チェック & 自動修復 =====
_CONTRA_PATTERNS = {
    "night": re.compile(r"\b(day( |time)|sun(light|lit)?|blue sky|midday)\b", re.IGNORECASE),
    "daytime": re.compile(r"\b(night|streetlight|headlight|taillight|moon|neon)\b", re.IGNORECASE),
    "rain": re.compile(r"\b(dry asphalt|dusty)\b", re.IGNORECASE),
    "snow": re.compile(r"\b(heavy rain|wet asphalt)\b", re.IGNORECASE),
    "overcast": re.compile(r"\b(crisp (sun)?light|harsh (midday )?sun|strong shadows)\b", re.IGNORECASE),
    "fog": re.compile(r"\b(high[- ]?contrast|deep black shadows)\b", re.IGNORECASE),
    "after_rain": re.compile(r"\b(dry asphalt)\b", re.IGNORECASE),
    "hazy_summer": re.compile(r"\b(crystal[- ]?clear (air|sky))\b", re.IGNORECASE),
    "dusk": re.compile(r"\b(harsh midday (sun|light))\b", re.IGNORECASE),
    "dawn": re.compile(r"\b(noon sun|midday)\b", re.IGNORECASE),
    "golden_hour": re.compile(r"\b(deep midnight|neon)\b", re.IGNORECASE),
}

def _theme_to_guard_key(theme_id: str) -> str:
    if theme_id == "night":
        return "night"
    if theme_id in ("clear_day", "overcast", "rain", "after_rain", "fog", "snow", "hazy_summer", "golden_hour", "dawn", "dusk"):
        return "daytime" if theme_id == "clear_day" else theme_id
    return theme_id

def detect_contradictions(theme_id: str, text: str, negative_markers: List[str]) -> List[str]:
    issues: List[str] = []
    key = _theme_to_guard_key(theme_id)
    pat = _CONTRA_PATTERNS.get(key)
    if pat and pat.search(text or ""):
        issues.append(f"pattern_violation::{key}")
    # 追加のNG語を機械的に検出
    tlow = (text or "").lower()
    neg_hits = [w for w in (negative_markers or []) if w.lower() in tlow]
    if neg_hits:
        issues.append("negative_marker::" + ",".join(neg_hits))
    return issues

def build_repair_messages(original_text: str, theme_pack: Dict[str, Any], max_words: int) -> List[Dict[str, str]]:
    atmosphere_hint = theme_pack.get("atmosphere", "")
    lighting_hint = "; ".join(theme_pack.get("lighting_list", []))
    sys = (
        "You fix prompts for diffusion models. Ensure strict consistency between theme and lighting. "
        f"Return ONE coherent prompt in 1–2 sentences (<= {max_words} words). Return ONLY the prompt text."
    )
    user = (
        f"Theme: {atmosphere_hint}\n"
        f"Compatible lighting: {lighting_hint}\n"
        f"Rewrite the following prompt to strictly match the theme while preserving its intent:\n"
        f"---\n{original_text}\n---"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

# ===== 保存 =====
def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_txt_json_and_jsonl(out_txt: str, out_meta_json: str, jsonl_path: str,
                            prompt: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(out_txt))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")
    safe_write_json(out_meta_json, meta)
    _ensure_dir(os.path.dirname(jsonl_path))
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps({"prompt": prompt, **meta}, ensure_ascii=False) + "\n")

# ===== CLI / ウォームアップ / main =====
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator v2 (consistency + freedom)")
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--naming", type=str, choices=NAMING_CHOICES, default="predTrainId")
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument("--min-pixels", type=int, default=4000)
    ap.add_argument("--run-seed", type=int, default=42)
    ap.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    ap.add_argument("--repeat-penalty", type=float, default=1.10, dest="repeat_penalty")
    ap.add_argument("--num-predict", type=int, default=120, dest="num_predict")
    ap.add_argument("--max-words", type=int, default=64)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")

    # v2 追加オプション
    ap.add_argument("--objects-max", type=int, default=8, help="プロンプトに示唆するカテゴリの最大数（自然文に溶かす）")
    ap.add_argument("--themes", type=str, nargs="*", default=list(STYLE_THEMES.keys()),
                    help="使用を許可するテーマID（例: clear_day night rain ...）")
    ap.add_argument("--repair", action="store_true", help="整合性チェック後に自動修復を行う")
    ap.add_argument("--repair-max-attempts", type=int, default=2)
    return ap.parse_args()

def _ollama_warmup(logger: logging.Logger, base_url: str, model: str, timeout_sec: int, think: Optional[str]) -> None:
    base = base_url.rstrip("/")
    url = base + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "stream": False,
        "options": {"num_predict": 8},
    }
    if think and think != "none":
        payload["think"] = str(think)
    try:
        obj = _http_post_json(url, payload, timeout_sec)
        msg = (obj.get("message") or {}).get("content", "")
        if isinstance(msg, str) and msg.strip():
            return
        logger.warning("Warmup(chat): returned empty text. Model may be mid-load.")
    except Exception as e:
        logger.warning("Warmup(chat) failed: %s", repr(e))

def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)
    logger = _setup_logger(args.output_root, verbose=args.verbose)
    _log_env(logger, args)

    think_level = None if (args.think == "none") else args.think

    if args.warmup_check:
        _ollama_warmup(logger, args.ollama_base_url, args.model, args.timeout_sec, think=think_level)

    total_ok = 0
    total_ng = 0

    for split in args.splits:
        semseg_split_root = os.path.join(args.semseg_root, split)
        image_split_root = os.path.join(args.image_root, split)
        jsonl_path = os.path.join(args.output_root, f"prompts_{split}.jsonl")

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0:
            files = files[:args.limit]
        if not files:
            logger.warning("[%s] no targets under: %s", split, semseg_split_root)
            continue

        logger.info("[%s] target files: %d", split, len(files))
        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            stem = Path(npy_path).name.split("_")[0]  # first / mid10s / last
            out_dir = os.path.join(args.output_root, split, rel_dir)
            out_txt = os.path.join(out_dir, f"{stem}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
            out_dbg = os.path.join(out_dir, f"{stem}_prompt.debug.json")

            try:
                if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                    pbar.set_postfix_str("skip"); 
                    continue

                # 1) セマンティック読み込み
                seg = np.load(npy_path)
                if seg.dtype != np.uint8:
                    seg = seg.astype(np.uint8)

                label_pairs = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
                label_names_all = [TRAINID_TO_NAME[i] for (i, _) in label_pairs]
                scene_hint = _scene_hint_from_labels(label_names_all)

                # 2) 再現性用シード
                frame_seed = _derive_seed_for_item(args.run_seed, npy_path)
                rng = _rng_from_seed(frame_seed)

                # 3) オブジェクトの自然言語ヒント
                label_names = _group_label_names_by_importance(label_pairs, max(2, args.objects_max), rng)
                objects_phrase = _to_natural_phrase(label_names)

                # 4) テーマ選択（整合語彙）
                theme_id = _sample_theme_id(rng, args.themes)
                theme_pack = _sample_theme_pack(rng, theme_id)

                # 5) LLM呼び出し（初回）
                messages = build_initial_messages(scene_hint, objects_phrase, theme_pack, args.max_words)
                options = build_llm_options(rng, args.temperature, args.top_p, args.repeat_penalty,
                                            args.num_predict, frame_seed)
                prompt_text, last_obj = call_ollama_chat(
                    base_url=args.ollama_base_url, model=args.model,
                    messages=messages, options=options, timeout_sec=args.timeout_sec,
                    retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                )

                clean = " ".join(prompt_text.replace("\n", " ").replace("\"", "").replace("```", "").split())

                # 6) 整合性チェック
                issues = detect_contradictions(theme_id, clean, theme_pack.get("negative_markers", []))
                repaired = False
                attempts = 0

                while args.repair and issues and attempts < max(0, int(args.repair_max_attempts)):
                    attempts += 1
                    logger.debug("Repair attempt %d for %s | issues=%s", attempts, stem, issues)
                    rep_messages = build_repair_messages(clean, theme_pack, args.max_words)
                    clean2, _ = call_ollama_chat(
                        base_url=args.ollama_base_url, model=args.model,
                        messages=rep_messages, options=options, timeout_sec=args.timeout_sec,
                        retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                    )
                    clean2 = " ".join(clean2.replace("\n", " ").replace("\"", "").replace("```", "").split())
                    issues2 = detect_contradictions(theme_id, clean2, theme_pack.get("negative_markers", []))
                    if not issues2:
                        clean = clean2
                        repaired = True
                        break
                    else:
                        issues = issues2  # 継続

                image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)

                meta = {
                    "split": split, "camera": args.camera,
                    "npy_path": npy_path, "image_path": image_path,
                    "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                    "scene_hint": scene_hint,
                    "objects_phrase": objects_phrase,
                    "theme": theme_pack,
                    "ollama": {"base_url": args.ollama_base_url, "model": args.model, "options": options},
                    "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "timestamp": int(time.time()),
                    "repaired": bool(repaired),
                }

                save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, clean, meta)
                total_ok += 1
                pbar.set_postfix_str("ok")

            except Exception as e:
                total_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), npy_path)
                dbg = {
                    "npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e),
                    "traceback": tb, "split": split
                }
                try:
                    safe_write_json(out_dbg, dbg)
                except Exception:
                    pass

        logger.info("[%s] done: OK=%d NG=%d -> %s", split, total_ok, total_ng, args.output_root)

    if total_ng == 0:
        print(f"✅ ALL DONE: OK={total_ok}, NG=0, out={args.output_root}")
    else:
        print(f"⚠️ DONE WITH ERRORS: OK={total_ok}, NG={total_ng}, out={args.output_root}")

if __name__ == "__main__":
    main()
