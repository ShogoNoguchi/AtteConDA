# -*- coding: utf-8 -*-
"""
File: generate_waymo_prompts_gptoss_v2_2.py

WaymoV2(Front) セマンティックラベル(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール（自由度重視・一貫性検査・再ランキング・決定論的多様性 / thinkingは保存しない）

v2.2 変更点（安定化フォールバック）:
- /api/chat で空レス時のフォールバックを強化
  1) 通常送信
  2) think フィールドを落として再送
  3) /api/generate への最終フォールバック（System/Userを単一プロンプトに合成）
- Warmup も think あり→なし の二段でチェック
- 候補生成失敗ログに cand index を明記

設計要点（v2系共通）:
- few-shot 例示を撤廃。LLMに時間帯/天候/照明の一貫性選択を委ねる。
- atmosphere/lighting の強制注入なし。出力にメタ指示を書かせない。
- n候補生成 → 矛盾/長さ/禁句/オブジェクト整合のスコアで再ランキング。
- 最終ワンパス校正（coherence向上・軽微修正のみ）。thinkingは保存せず。
- 乱数は run_seed とファイル固有キーから派生。候補ごとに seed オフセットで決定的多様性。
- 画像パス推定バグを修正（*_first_predTrainId.npy → *_first.jpg 等）。
- splitごとのOK/NG集計を明確化。
- 既存CUDA/PyTorch環境に非干渉（requestsのみ任意）。

Author: Shogo支援用
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

# requests があれば使用、無ければ urllib へフォールバック
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

# ===== オブジェクト同義語（整合スコア用 / 単純語彙） =====
OBJ_SYNONYMS: Dict[str, List[str]] = {
    "road": ["road", "street", "asphalt", "lane", "carriageway"],
    "sidewalk": ["sidewalk", "pavement", "footpath"],
    "building": ["building", "buildings"],
    "wall": ["wall", "walls"],
    "fence": ["fence", "fencing"],
    "pole": ["pole", "utility pole", "lamp post", "lamppost"],
    "traffic light": ["traffic light", "traffic lights", "signal", "stoplight"],
    "traffic sign": ["traffic sign", "road sign", "street sign"],
    "vegetation": ["vegetation", "trees", "greenery", "foliage", "bushes"],
    "terrain": ["terrain", "grass", "verge", "embankment", "dirt"],
    "sky": ["sky", "open sky"],
    "person": ["person", "people", "pedestrian", "pedestrians"],
    "rider": ["rider", "cyclist", "motorcyclist"],
    "car": ["car", "cars"],
    "truck": ["truck", "lorry", "trucks", "lorries"],
    "bus": ["bus", "buses", "coach", "coaches"],
    "train": ["train", "trains"],
    "motorcycle": ["motorcycle", "motorbike", "motorbikes"],
    "bicycle": ["bicycle", "bike", "bicycles", "bikes"],
}

# ===== スタイル語彙（矛盾検出のための簡易辞書） =====
DAY_TERMS = [
    "day", "daytime", "sunny", "clear day", "bright daylight",
    "afternoon", "morning", "noon", "midday", "broad daylight"
]
NIGHT_TERMS = [
    "night", "nighttime", "moonlit", "streetlights", "sodium", "neon",
    "headlights", "taillights", "starry", "dark"
]
AMBIGUOUS_TERMS = [
    "dusk", "dawn", "twilight", "sunset", "sunrise", "golden hour", "evening"
]
WEATHER_TERMS = [
    "rain", "rainy", "drizzle", "light rain", "wet", "puddle", "showers",
    "snow", "snowy", "snowfall", "slush", "flurries",
    "fog", "foggy", "mist", "haze", "hazy",
    "overcast", "cloudy", "clear"
]

# ===== 禁句（指示臭・メタ情報の漏出を検出） =====
INSTRUCTIONAL_LEAK_TERMS = [
    "objects:", "scene:", "atmosphere:", "lighting:", "instruction:",
    "Return only", "Output only", "Do not", "should", "must", "bullet points",
    "1–2 sentences", "1-2 sentences", "words", "no labels", "no JSON",
    "camera angle", "composition"
]

# ===== ユーティリティ =====
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run.log")
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
    logger.info("num-predict: %d", args.num_predict)
    logger.info("run-seed : %d", args.run_seed)
    logger.info("overwrite : %s", args.overwrite)
    logger.info("limit : %d", args.limit)
    logger.info("requests : %s", "available" if _HAS_REQUESTS else "fallback to urllib")
    logger.info("api-mode : %s", "chat (forced)")
    logger.info("retries : %d", args.retries)
    logger.info("warmup : %s", args.warmup_check)
    logger.info("think : %s", args.think)
    logger.info("num-candidates : %d", args.num_candidates)
    logger.info("word-range : %d..%d", args.min_words, args.max_words)
    logger.info("polish-final : %s", args.polish_final)

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
    """
    修正: 1507678826876435_first_predTrainId.npy → 1507678826876435_first.jpg 等を探索
    """
    d = os.path.dirname(npy_path)
    try:
        rel_dir = os.path.relpath(d, semseg_split_root)  # front/{segment}
    except Exception:
        return None
    fname = Path(npy_path).name  # e.g., 1507678826876435_first_predTrainId.npy
    stem = None
    if fname.endswith("_predTrainId.npy"):
        stem = fname[:-len("_predTrainId.npy")]  # -> 1507678826876435_first
    elif fname.endswith("_semantic.npy"):
        stem = fname[:-len("_semantic.npy")]
    else:
        stem = fname[:-4]
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
    rng = random.Random()
    rng.seed(seed)
    return rng

# ===== ラベル抽出 =====
def extract_present_labels(seg: np.ndarray, min_area_ratio: float, min_pixels: int) -> List[Tuple[int, float]]:
    if seg.ndim != 2:
        raise ValueError(f"seg ndim expected 2, got {seg.ndim}")
    h, w = seg.shape
    total = float(h * w)
    cnt = np.bincount(seg.flatten(), minlength=19).astype(np.int64)
    present: List[Tuple[int, float]] = []
    for cls_id in range(19):
        px = int(cnt[cls_id])
        ratio = px / total if total > 0 else 0.0
        if px >= min_pixels or ratio >= min_area_ratio:
            present.append((cls_id, ratio))
    present.sort(key=lambda x: x[1], reverse=True)
    return present

def _top_label_names(label_pairs: List[Tuple[int, float]], k: int = 8) -> List[str]:
    return [TRAINID_TO_NAME[i] for (i, _) in label_pairs][:k]

def _scene_hint_from_labels(label_names: List[str]) -> str:
    """
    偏りの少ないシーンヒント（短くジェネリック）
    """
    s = set(label_names)
    if "building" in s and "road" in s and ("sidewalk" in s or "traffic sign" in s):
        return "a city street scene captured from a front-facing dashcam"
    if "terrain" in s and "road" in s and "vegetation" in s and "building" not in s:
        return "a suburban or rural road viewed from a moving vehicle"
    if "road" in s and ("truck" in s or "bus" in s) and "building" not in s:
        return "a highway scene viewed from a vehicle"
    return "a driving scene from a vehicle's front camera"

# ===== LLMメッセージ（few-shot撤廃 / 自由度重視 / メタ露出禁止） =====
def build_chat_messages_v2(label_names: List[str], scene_hint: str, min_words: int, max_words: int) -> List[Dict[str, str]]:
    """
    出力はプロンプト本文のみ。LLMに時間帯/天候/照明を一貫して選ばせる。
    """
    # 2–4個程度の主要要素に自然に触れるよう促すが、列挙はさせない（プロンプトの自由度確保）
    objects_hint = ", ".join(label_names[:6])  # context notes only
    system = (
        "You are an expert prompt writer for photorealistic text-to-image diffusion models used in autonomous driving datasets. "
        f"Write exactly ONE coherent English prompt in {min_words}-{max_words} words, as 1–2 sentences of natural prose. "
        "Implicitly keep the original camera angle and composition; DO NOT mention instructions, lists, or labels. "
        "Choose a single, self-consistent time-of-day, weather, and lighting appropriate for a realistic driving scene. "
        "Avoid contradictions (e.g., 'clear day' with 'streetlights' unless dusk/twilight). Output ONLY the prompt text."
    )
    user = (
        "Context notes (do not quote verbatim; use only to ensure plausibility):\n"
        f"- Scene hint: {scene_hint}\n"
        f"- Present elements (not exhaustive): {objects_hint}\n"
        "- Write naturally and concisely without bullets or meta-instructions."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ===== LLM オプション =====
def build_llm_options(rng: random.Random, base_temperature: float, base_top_p: float,
                      base_repeat_penalty: float, num_predict: int, seed: int) -> Dict[str, Any]:
    def jitter(x: float, low: float, high: float, scale: float = 0.1) -> float:
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

# ===== 例外 =====
class OllamaBadResponse(RuntimeError):
    def __init__(self, message: str, obj: Any = None, which: str = "", payload_hint: Dict[str, Any] = None):
        super().__init__(message)
        self.obj = obj
        self.which = which
        self.payload_hint = payload_hint or {}

# ===== HTTP =====
def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if _HAS_REQUESTS:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout_sec)  # type: ignore
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, data=data, headers=headers)  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:  # type: ignore
            return json.loads(r.read().decode("utf-8"))

# ===== /api/generate 用シンプル変換 =====
def _messages_to_simple_prompt(messages: List[Dict[str, str]], min_words: int, max_words: int) -> str:
    """
    chatメッセージを単一文字列に変換（最終フォールバック用）
    """
    sys_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    usr_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    sys_text = "\n".join([s for s in sys_parts if isinstance(s, str)]).strip()
    usr_text = "\n\n".join([s for s in usr_parts if isinstance(s, str)]).strip()
    prompt = (
        f"{sys_text}\n\nUser request:\n{usr_text}\n\n"
        f"Return exactly ONE prompt ({min_words}-{max_words} words), plain text only."
    ).strip()
    return prompt

# ===== Ollama 呼び出し（/api/chat 固定・フォールバック強化） =====
def call_ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                     timeout_sec: int = 120, retries: int = 2, retry_backoff_ms: int = 800,
                     think: Optional[str] = "low") -> Tuple[str, Dict[str, Any]]:
    """
    フロー:
      A) /api/chat with think(指定どおり)
      B) /api/chat without think
      C) /api/generate (最終フォールバック)
    いずれかで content を得たら即返す。
    """
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    gen_url = base + "/api/generate"
    last_obj: Dict[str, Any] = {}

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
                        # Harmony 由来の断片では 'text' キーに本文が入ることが多い
                        for k in ("text", "content", "value"):
                            v = part.get(k)
                            if isinstance(v, str):
                                parts.append(v)
                text = " ".join(parts).strip()
                if text:
                    return text
        return ""

    # --- A) /api/chat with think ---
    for i in range(max(1, retries + 1)):
        try:
            payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": options}
            if think and think != "none":
                payload["think"] = str(think)
            last_obj = _http_post_json(chat_url, payload, timeout_sec)
            if isinstance(last_obj, dict) and "message" in last_obj:
                txt = _extract_text_from_message_obj(last_obj.get("message"))
                if isinstance(txt, str) and txt.strip():
                    return txt.strip(), last_obj
            raise OllamaBadResponse("Ollama /api/chat returned no usable text (with think)", obj=last_obj, which="chat")
        except Exception:
            if i < retries:
                time.sleep(max(0, retry_backoff_ms) / 1000.0)
                continue
            # 続けて B) へ
            break

    # --- B) /api/chat without think ---
    for i in range(1):  # 1回だけ
        try:
            payload2: Dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": options}
            # think を入れない
            last_obj = _http_post_json(chat_url, payload2, timeout_sec)
            if isinstance(last_obj, dict) and "message" in last_obj:
                txt = _extract_text_from_message_obj(last_obj.get("message"))
                if isinstance(txt, str) and txt.strip():
                    return txt.strip(), last_obj
            raise OllamaBadResponse("Ollama /api/chat returned no usable text (no think)", obj=last_obj, which="chat_nothink")
        except Exception:
            # 続けて C) へ
            pass

    # --- C) /api/generate (最終フォールバック) ---
    try:
        simple_prompt = _messages_to_simple_prompt(messages, options.get("min_words", 28), options.get("max_words", 55))
        gen_opts = dict(options)
        # generate では stop を指定しない（gpt-ossの思考出力に干渉しうるため）
        payload3: Dict[str, Any] = {"model": model, "prompt": simple_prompt, "stream": False, "options": gen_opts}
        last_obj = _http_post_json(gen_url, payload3, timeout_sec)
        resp = last_obj.get("response")
        if isinstance(resp, str) and resp.strip():
            return resp.strip(), last_obj
        # まれに 'message' 形で返ることがある
        if isinstance(last_obj, dict) and "message" in last_obj:
            txt = _extract_text_from_message_obj(last_obj.get("message"))
            if isinstance(txt, str) and txt.strip():
                return txt.strip(), last_obj
        raise OllamaBadResponse("Ollama /api/generate returned no usable text", obj=last_obj, which="generate")
    except Exception as e:
        raise OllamaBadResponse("All fallbacks failed", obj=last_obj, which="fallbacks") from e

# ===== テキスト整形/検査 =====
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def word_count(s: str) -> int:
    return len(_WORD_RE.findall(s))

def _contains_any(s: str, terms: List[str]) -> bool:
    t = s.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", t) for term in terms)

def _count_any(s: str, terms: List[str]) -> int:
    t = s.lower()
    return sum(1 for term in terms if re.search(r"\b" + re.escape(term) + r"\b", t))

def classify_day_night(s: str) -> Tuple[float, float, bool]:
    """
    戻り値: (day_score, night_score, has_ambiguous)
    """
    t = s.lower()
    day_score = 0.0
    night_score = 0.0
    has_amb = False
    for term in DAY_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", t):
            day_score += 1.0
    for term in NIGHT_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", t):
            # 'streetlights' 'headlights' は弱め重み
            if term in ("headlights", "taillights", "streetlights", "sodium", "neon"):
                night_score += 0.6
            else:
                night_score += 1.0
    for term in AMBIGUOUS_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", t):
            has_amb = True
    return day_score, night_score, has_amb

def object_coverage(s: str, label_names: List[str]) -> int:
    """
    上位ラベルのうち、テキストに現れる語彙（同義語含む）の個数
    """
    t = s.lower()
    covered = 0
    for name in label_names[:6]:
        syns = OBJ_SYNONYMS.get(name, [name])
        if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in syns):
            covered += 1
    return covered

def instructional_leak_count(s: str) -> int:
    return _count_any(s, INSTRUCTIONAL_LEAK_TERMS)

def coherence_score(s: str, label_names: List[str], min_words: int, max_words: int) -> Dict[str, Any]:
    """
    スコア詳細を返す（デバッグ保存用）
    """
    wc = word_count(s)
    day_score, night_score, has_amb = classify_day_night(s)
    cov = object_coverage(s, label_names)
    leaks = instructional_leak_count(s)
    # 矛盾ペナルティ
    conflict = 0.0
    if day_score > 0 and night_score > 0:
        # 曖昧語（dusk/dawn等）があるなら軽減
        conflict = 1.0 if not has_amb else 0.5
    # 長さペナルティ
    if wc < min_words:
        len_pen = (min_words - wc) * 0.25
    elif wc > max_words:
        len_pen = (wc - max_words) * 0.10
    else:
        len_pen = 0.0
    # 漏出ペナルティ
    leak_pen = leaks * 1.0
    # 総合: 高いほど良い
    score = (cov * 1.5) - (conflict * 2.0) - len_pen - leak_pen
    return {
        "score": float(score),
        "wc": int(wc),
        "day_score": float(day_score),
        "night_score": float(night_score),
        "has_ambiguous": bool(has_amb),
        "object_cov": int(cov),
        "leaks": int(leaks),
        "len_pen": float(len_pen),
        "conflict_pen": float(conflict * 2.0),
        "leak_pen": float(leak_pen),
    }

def clean_output_text(s: str) -> str:
    # コードフェンスや引用破片除去
    t = s.replace("```", " ").replace("\n", " ").strip()
    # 連続空白
    t = " ".join(t.split())
    return t

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

# ===== ウォームアップ =====
def _ollama_warmup(logger: logging.Logger, base_url: str, model: str, timeout_sec: int, think: Optional[str]) -> None:
    base = base_url.rstrip("/")
    url = base + "/api/chat"

    def _once(payload: Dict[str, Any], tag: str) -> None:
        try:
            obj = _http_post_json(url, payload, timeout_sec)
            msg = (obj.get("message") or {}).get("content", "")
            if isinstance(msg, str) and msg.strip():
                return
            logger.warning("Warmup(chat-%s): returned empty text.", tag)
        except Exception as e:
            logger.warning("Warmup(chat-%s) failed: %s", tag, repr(e))

    # 1) think あり
    payload1: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "stream": False,
        "options": {"num_predict": 8},
    }
    if think and think != "none":
        payload1["think"] = str(think)
    _once(payload1, "think")

    # 2) think なし
    payload2: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "OK?"}],
        "stream": False,
        "options": {"num_predict": 8},
    }
    _once(payload2, "nothink")

# ===== メイン生成 (候補生成→再ランキング→最終校正) =====
def generate_prompt_for_item(
    logger: logging.Logger,
    base_url: str,
    model: str,
    label_pairs: List[Tuple[int, float]],
    image_path: Optional[str],
    min_words: int,
    max_words: int,
    base_options: Dict[str, Any],
    frame_seed: int,
    num_candidates: int,
    timeout_sec: int,
    retries: int,
    think: Optional[str],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    label_names = _top_label_names(label_pairs, k=8)
    scene_hint = _scene_hint_from_labels(label_names)
    # メッセージ生成（自由度重視）
    messages = build_chat_messages_v2(label_names, scene_hint, min_words=min_words, max_words=max_words)
    # 候補生成
    candidates: List[Dict[str, Any]] = []
    for ci in range(num_candidates):
        cand_seed = (frame_seed + 101 * ci) & 0x7FFFFFFF
        rng = _rng_from_seed(cand_seed)
        options = dict(base_options)
        options["seed"] = cand_seed
        # ほんの少しだけ温度をゆらす（決定的・再現可能）
        options["temperature"] = min(1.50, max(0.05, options.get("temperature", 0.6) * (0.95 + 0.1 * rng.random())))
        # 後段の /api/generate フォールバック用に語数範囲も持たせておく
        options["min_words"] = min_words
        options["max_words"] = max_words
        try:
            raw_text, last_obj = call_ollama_chat(
                base_url=base_url, model=model, messages=messages, options=options,
                timeout_sec=timeout_sec, retries=retries, retry_backoff_ms=800, think=think
            )
            text = clean_output_text(raw_text)
            score_info = coherence_score(text, label_names, min_words=min_words, max_words=max_words)
            candidates.append({
                "text": text,
                "score_info": score_info,
                "seed": int(cand_seed),
            })
            logger.debug("[cand %d] wc=%d score=%.3f cov=%d conflict=%.2f leaks=%d text=%s",
                         ci, score_info["wc"], score_info["score"], score_info["object_cov"],
                         score_info["conflict_pen"], score_info["leaks"], text)
        except Exception as e:
            logger.error("[cand %d] Candidate generation failed: %s", ci, repr(e))
            continue

    if not candidates:
        raise RuntimeError("No candidates produced")

    # 再ランキング（最高スコアを選択）
    best = max(candidates, key=lambda c: c["score_info"]["score"])

    # 最終軽微校正（文の一貫性/流暢さのみ、内容大幅変更は不可）
    final_text = best["text"]
    # polish は main() 側で制御
    return final_text, {"scene_hint": scene_hint, "label_names": _top_label_names(label_pairs, k=8)}, candidates

def polish_prompt_if_enabled(
    enabled: bool,
    base_url: str,
    model: str,
    prompt_text: str,
    label_names: List[str],
    min_words: int,
    max_words: int,
    options: Dict[str, Any],
    timeout_sec: int,
    retries: int,
    think: Optional[str],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not enabled:
        return prompt_text, None
    system = (
        "You act as a careful editor. Improve coherence, grammar, and flow of the given driving-scene prompt. "
        f"Keep the meaning and keep it in {min_words}-{max_words} words. "
        "Do NOT add or remove major scene elements, do NOT list labels, and avoid contradictions. "
        "Output ONLY the revised prompt text."
    )
    user = (
        "Context notes (do not quote): "
        + ", ".join(label_names[:6])
        + "\nOriginal prompt:\n"
        + prompt_text
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        raw, last_obj = call_ollama_chat(
            base_url=base_url, model=model, messages=messages, options=options,
            timeout_sec=timeout_sec, retries=retries, retry_backoff_ms=800, think=think
        )
        revised = clean_output_text(raw)
        return revised, {"edited": True}
    except Exception:
        # 校正に失敗しても元を返す（安全側）
        return prompt_text, {"edited": False}

# ===== CLI / main =====
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator (free-form, rerank, deterministic, fallback-hardened)")
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
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    ap.add_argument("--repeat-penalty", type=float, default=1.12, dest="repeat_penalty")
    ap.add_argument("--num-predict", type=int, default=120, dest="num_predict")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")
    # v2追加
    ap.add_argument("--num-candidates", type=int, default=3, help="候補生成数（決定的多様性）。")
    ap.add_argument("--min-words", type=int, default=28)
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--polish-final", action="store_true", help="最終ワンパス校正を有効化")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)
    logger = _setup_logger(args.output_root, verbose=args.verbose)
    _log_env(logger, args)

    think_level = None if (args.think == "none") else args.think
    if args.warmup_check:
        _ollama_warmup(logger, args.ollama_base_url, args.model, args.timeout_sec, think=think_level)

    # ベースオプション（候補ごとにseed/temperatureを微ゆらし）
    base_rng = _rng_from_seed(args.run_seed)
    base_options = build_llm_options(base_rng, args.temperature, args.top_p, args.repeat_penalty, args.num_predict, args.run_seed)

    grand_ok = 0
    grand_ng = 0

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
        split_ok = 0
        split_ng = 0

        pbar = tqdm(files, desc=f"{split}")
        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            fname = Path(npy_path).name
            out_dir = os.path.join(args.output_root, split, rel_dir)
            # 入力stem（first/mid10s/lastを含む）
            if fname.endswith("_predTrainId.npy"):
                stem = fname[:-len("_predTrainId.npy")]  # 150..._first
            elif fname.endswith("_semantic.npy"):
                stem = fname[:-len("_semantic.npy")]
            else:
                stem = fname[:-4]
            out_txt = os.path.join(out_dir, f"{stem}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
            out_dbg = os.path.join(out_dir, f"{stem}_prompt.debug.json")

            try:
                if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                    pbar.set_postfix_str("skip")
                    split_ok += 1
                    continue

                seg = np.load(npy_path)
                if seg.dtype != np.uint8:
                    seg = seg.astype(np.uint8)

                label_pairs = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
                # 空のときはスキップ
                if not label_pairs:
                    raise RuntimeError("No salient labels")

                frame_seed = _derive_seed_for_item(args.run_seed, npy_path)
                image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)

                # 候補生成→再ランキング
                draft_text, scene_meta, candidates = generate_prompt_for_item(
                    logger=logger,
                    base_url=args.ollama_base_url,
                    model=args.model,
                    label_pairs=label_pairs,
                    image_path=image_path,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    base_options=base_options,
                    frame_seed=frame_seed,
                    num_candidates=max(1, args.num_candidates),
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                    think=think_level,
                )

                # 最終校正（任意）
                final_options = dict(base_options)
                final_options["seed"] = (frame_seed + 991) & 0x7FFFFFFF
                final_text, polish_info = polish_prompt_if_enabled(
                    enabled=args.polish_final,
                    base_url=args.ollama_base_url,
                    model=args.model,
                    prompt_text=draft_text,
                    label_names=scene_meta["label_names"],
                    min_words=args.min_words,
                    max_words=args.max_words,
                    options=final_options,
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                    think=think_level,
                )

                # スコア最終計算（記録用）
                final_score = coherence_score(final_text, scene_meta["label_names"], args.min_words, args.max_words)
                meta = {
                    "split": split,
                    "camera": args.camera,
                    "npy_path": npy_path,
                    "image_path": image_path,
                    "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                    "scene_hint": scene_meta["scene_hint"],
                    "ollama": {"base_url": args.ollama_base_url, "model": args.model, "options": base_options, "api_mode": "chat"},
                    "run_seed": int(args.run_seed),
                    "frame_seed": int(frame_seed),
                    "timestamp": int(time.time()),
                    "final_eval": final_score,
                    "polish": polish_info or {"edited": False},
                }
                dbg = {
                    "candidates": candidates,
                    "chosen_text": draft_text,
                    "after_polish_text": final_text,
                }

                save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, final_text, meta)
                safe_write_json(out_dbg, dbg)
                split_ok += 1
                pbar.set_postfix_str("ok")
            except Exception as e:
                split_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), npy_path)
                dbg = {
                    "npy_path": npy_path,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": tb,
                    "split": split,
                }
                try:
                    safe_write_json(out_dbg, dbg)
                except Exception:
                    pass

        logger.info("[%s] done: OK=%d NG=%d -> %s", split, split_ok, split_ng, args.output_root)
        grand_ok += split_ok
        grand_ng += split_ng

    if grand_ng == 0:
        print(f"✅ ALL DONE: OK={grand_ok}, NG=0, out={args.output_root}")
    else:
        print(f"⚠️ DONE WITH ERRORS: OK={grand_ok}, NG={grand_ng}, out={args.output_root}")

if __name__ == "__main__":
    main()
