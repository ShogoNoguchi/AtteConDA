# -*- coding: utf-8 -*-
"""
WaymoV2(Front) セマンティックラベル(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール（構造保持・多様性・再現性対応 / thinkingは保存しない / Harmony対応）

【v2 重要変更点（2025-11-04）】
- 「Coherent Style Bank」を導入：天候/時間帯と照明ヒントをペア設計し、矛盾（昼×街灯支配 等）を根絶。
- 例示（Example）を削除：語彙の写り込み（streetlights固定化）を防止。
- LLMへの指示を「自由度高め＋列挙抑制」に再設計（“自然な2文/<=max_words/列挙しすぎるな”）。
- 出力品質ゲートを追加：ローカル検査（矛盾/列挙過多）→ 必要時に自動Refine（1回）。
- 画像パス推定のバグ修正："..._first_predTrainId.npy" → "..._first.jpg" を正しく復元。
- ログ/メタ/デバッグ拡充：選択スタイル/ゲート判定/Refine履歴を保存。
- そのほか細部の堅牢化（例外ハンドリング、ラベル抽出のソート位置など）。

参考（gpt-oss / Ollama）:
- gpt-oss は Harmony 形式前提。/api/chat を推奨（/api/generateはHarmony不足になりやすい）。
- Ollama "think"（推論痕跡）はトップレベル "think" フィールドで制御。ここでは保存しない設計。
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
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss"
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

# ===== Coherent Style Bank（矛盾しないペア設計） =====
#   - time/weather と lighting_hint を整合させる
#   - avoid_terms はそのスタイルで"出してはならない/主光源になりえない"語の一例
STYLE_BANK: List[Dict[str, Any]] = [
    {
        "id": "day_clear",
        "atmosphere": "on a clear day",
        "lighting_hint": "bright natural daylight with crisp, short shadows",
        "avoid_terms": ["streetlight", "sodium", "neon", "moonlit", "starry", "night"],
    },
    {
        "id": "day_overcast",
        "atmosphere": "in overcast weather",
        "lighting_hint": "soft diffuse light with muted contrast",
        "avoid_terms": ["harsh midday sun", "streetlight", "sodium", "neon", "night"],
    },
    {
        "id": "golden_hour",
        "atmosphere": "at golden hour",
        "lighting_hint": "low warm sun with long gentle shadows and warm tones",
        "avoid_terms": ["streetlight", "neon", "harsh midday sun", "moonlit", "night"],
    },
    {
        "id": "dawn",
        "atmosphere": "at dawn",
        "lighting_hint": "cool early light with long soft shadows",
        "avoid_terms": ["streetlight dominating", "sodium lamps", "harsh midday sun", "neon", "night"],
    },
    {
        "id": "dusk",
        "atmosphere": "at dusk",
        "lighting_hint": "transition light where ambient twilight starts to mix with early headlights",
        "avoid_terms": ["harsh midday sun", "clear bright noon", "full daylight"],
    },
    {
        "id": "night",
        "atmosphere": "at night",
        "lighting_hint": "streetlights and vehicle headlights providing the main illumination",
        "avoid_terms": ["bright sun", "midday sun", "blue sky with crisp shadows", "daylight"],
    },
    {
        "id": "rain",
        "atmosphere": "in rainy weather",
        "lighting_hint": "overcast sky with wet asphalt showing natural reflections",
        "avoid_terms": ["powdery snow", "dusty dry road"],
    },
    {
        "id": "fog",
        "atmosphere": "in foggy weather",
        "lighting_hint": "reduced visibility with light gently scattered by mist",
        "avoid_terms": ["harsh midday sun", "crisp sharp shadows"],
    },
    {
        "id": "snow",
        "atmosphere": "in snowy weather",
        "lighting_hint": "cold diffuse light with snow-muted contrast",
        "avoid_terms": ["wet asphalt reflections", "harsh midday sun"],
    },
]

# ===== ロガー =====
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run.log")
    logger = logging.getLogger("promptgen_gptoss")
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
    logger.info("=== PromptGen (Waymo -> gpt-oss:20b / v2) ===")
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
    logger.info("api-mode : %s", args.api_mode)
    logger.info("retries : %d", args.retries)
    logger.info("warmup : %s", args.warmup_check)
    logger.info("think : %s", args.think)
    logger.info("max-words : %d", args.max_words)
    logger.info("quality-gate : %s", args.quality_gate)
    logger.info("refine : %s", args.refine)

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

def _infer_image_path_for_npy_v2(npy_path: str, semseg_split_root: str, image_split_root: str) -> Optional[str]:
    """
    修正版： "1507..._first_predTrainId.npy" → "1507..._first.jpg" を探索
    """
    d = os.path.dirname(npy_path)
    try:
        rel_dir = os.path.relpath(d, semseg_split_root)  # front/{segment}
    except Exception:
        return None
    name = Path(npy_path).name  # e.g., 1507678826876435_first_predTrainId.npy
    # _predTrainId / _semantic を外してベース名を作る
    if name.endswith("_predTrainId.npy"):
        stem = name[:-len("_predTrainId.npy")]
    elif name.endswith("_semantic.npy"):
        stem = name[:-len("_semantic.npy")]
    else:
        stem = Path(name).stem
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
    # ループ外で一度だけソート（大きい順）
    present.sort(key=lambda x: x[1], reverse=True)
    return present

def _to_oxford_comma(words: List[str]) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"

def _scene_hint_from_labels(label_names: List[str]) -> str:
    s = set(label_names)
    if "building" in s and "road" in s and ("sidewalk" in s or "traffic sign" in s):
        return "a city street scene captured from a front-facing dashcam"
    if "terrain" in s and "road" in s and "vegetation" in s and "building" not in s:
        return "a suburban or rural road scene seen from a moving vehicle"
    if "road" in s and ("truck" in s or "bus" in s) and "building" not in s:
        return "a highway scene viewed from a vehicle"
    return "a driving scene viewed from a vehicle's front camera"

# ===== スタイル選択（矛盾なき一貫性） =====
def _weighted_choice(rng: random.Random, items: List[Any], weights: Optional[List[float]] = None) -> Any:
    if not items:
        raise ValueError("Empty items for weighted choice")
    if weights is None:
        return rng.choice(items)
    if len(weights) != len(items):
        raise ValueError("weights and items length mismatch")
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]

def choose_style_coherent(rng: random.Random, label_names: List[str]) -> Dict[str, Any]:
    """
    sky や traffic light の有無などで僅かに重みを傾ける（過度にはしない）
    """
    s = set(label_names)
    ids = [x["id"] for x in STYLE_BANK]
    # 基本フラット。軽いバイアスのみ。
    w = [1.0 for _ in ids]
    # sky があれば昼系/夕夜境目に少し傾ける
    if "sky" in s:
        for i, sid in enumerate(ids):
            if sid in ("day_clear", "day_overcast", "golden_hour", "dawn", "dusk"):
                w[i] *= 1.15
    # traffic light が多い＝都市夜景の可能性少し（やりすぎない）
    if "traffic light" in s and "building" in s:
        for i, sid in enumerate(ids):
            if sid in ("night", "dusk"):
                w[i] *= 1.10
    # vegetationが支配的なら雨/霧/雪を少し減らす（恣意性は小）
    if "vegetation" in s and "building" not in s:
        for i, sid in enumerate(ids):
            if sid in ("snow", "fog"):
                w[i] *= 0.9

    chosen_id = _weighted_choice(rng, ids, w)
    style = next(x for x in STYLE_BANK if x["id"] == chosen_id)
    return style

def build_style_brief(style: Dict[str, Any]) -> str:
    """
    LLM には 'style brief' として渡す。lighting は「ヒント」であり、LLMが自由に微調整可能。
    """
    atm = style["atmosphere"]
    light = style["lighting_hint"]
    avoid = ", ".join(style.get("avoid_terms", []))
    brief = (
        f"style:\n"
        f"- time_and_weather: {atm}\n"
        f"- lighting_hint: {light}\n"
        f"- avoid_terms: {avoid}\n"
        f"notes:\n"
        f"- Keep physical consistency between time/weather and lighting.\n"
        f"- Do not explicitly contradict the avoid_terms.\n"
    )
    return brief

# ===== メッセージ（v2：自由度高め、列挙抑制、Example削除） =====
def build_chat_messages_v2(label_pairs: List[Tuple[int, float]],
                           scene_hint: str,
                           style_brief: str,
                           max_words: int = 55) -> List[Dict[str, str]]:
    # 上位12件くらいを「重要物体」として提示するが、列挙は強要しない
    names_sorted = [TRAINID_TO_NAME[i] for (i, _) in label_pairs][:12]
    objects_phrase = _to_oxford_comma(names_sorted)

    system = (
        "You are an expert prompt writer for text-to-image diffusion models used in autonomous driving datasets. "
        f"Write one photorealistic English prompt in 1–2 sentences (<= {max_words} words). "
        "Be natural and coherent; avoid bullet points or label-like lists. "
        "You may paraphrase object categories concisely, but do not invent new major object categories not present. "
        "Keep the same camera angle and composition as the original dashcam view. "
        "Ensure time/weather and lighting are physically consistent."
    )

    user = (
        "CONTEXT\n"
        f"- scene_hint: {scene_hint}\n"
        f"- present_categories: {objects_phrase}\n"
        f"{style_brief}\n"
        "TASK\n"
        f"- Return ONLY the final prompt text (1–2 sentences, <= {max_words} words). "
        "Avoid long enumerations; prefer a flowing description mentioning only salient elements and atmosphere."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ===== 生成オプション =====
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

# ===== 抽出（thinking は採用しない） =====
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

def _looks_like_instruction(s: str) -> bool:
    t = s.lower()
    bad_subs = [
        "we need to", "return only", "do not include", "bullet points", "no brand names",
        "avoid", "instruction:", "objects:", "scene:", "atmosphere:", "lighting:", "should", "must",
    ]
    return any(k in t for k in bad_subs)

# ===== ユーティリティ =====
def _is_gptoss_model(model: str) -> bool:
    return "gpt-oss" in (model or "").lower()

# ===== Ollama 呼び出し（Harmony対応: 常に chat を優先 / gpt-ossは chat 強制） =====
def call_ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                     timeout_sec: int = 120, prefer_api: str = "auto", retries: int = 2,
                     retry_backoff_ms: int = 800, think: Optional[str] = "low") -> Tuple[str, Dict[str, Any]]:
    """
    戻り値: (final_text, last_obj)
    - gpt-oss の場合は /api/chat を強制。Harmony テンプレートは chat で自動適用される。
    - think: "low"|"medium"|"high"|None を chat/generate トップレベルに渡す（thinking分離）。
    """
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    gen_url = base + "/api/generate"
    last_obj: Dict[str, Any] = {}

    # gpt-oss は Harmony 必須 → chat 強制
    force_chat = _is_gptoss_model(model)

    # 呼び出し順序
    if force_chat:
        order = ["chat"]
    else:
        order_auto = ["chat", "generate"]
        order = (["chat"] if prefer_api == "chat" else ["generate"] if prefer_api == "generate" else order_auto)

    def _chat_attempt() -> str:
        nonlocal last_obj
        payload = {"model": model, "messages": messages, "stream": False, "options": options}
        if think and think != "none":
            payload["think"] = str(think)  # gpt-oss は "low"/"medium"/"high"
        last_obj = _http_post_json(chat_url, payload, timeout_sec)
        if isinstance(last_obj, dict) and "message" in last_obj:
            txt = _extract_text_from_message_obj(last_obj.get("message"))
            if isinstance(txt, str) and txt.strip() and not _looks_like_instruction(txt):
                return txt.strip()
        raise OllamaBadResponse("Ollama /api/chat returned no usable text", obj=last_obj, which="chat")

    def _gen_once(prompt: str, ctx: Optional[List[int]] = None, npredict: Optional[int] = None, add_stops: bool = False) -> Dict[str, Any]:
        # gpt-oss では generate + 生文字列は Harmony 不足のため原則使わない。
        gen_opts = dict(options)
        if npredict is not None:
            gen_opts["num_predict"] = int(npredict)
        if add_stops and (not _is_gptoss_model(model)):
            stops = gen_opts.get("stop", [])
            if not isinstance(stops, list):
                stops = [stops]
            stops.extend(["\nobjects:", "\nscene:", "\natmosphere:", "\nlighting:", "Instruction:", "instruction:", "We need to"])
            # 重複排除
            seen = set()
            stops_unique = []
            for s in stops:
                if s not in seen:
                    stops_unique.append(s)
                    seen.add(s)
            gen_opts["stop"] = stops_unique
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "options": gen_opts}
        if think and think != "none":
            payload["think"] = str(think)
        if ctx is not None:
            payload["context"] = ctx
        return _http_post_json(gen_url, payload, timeout_sec)

    def _messages_to_single_prompt(messages_: List[Dict[str, str]]) -> str:
        # generate 用の単一文字列（非推奨 / 後方互換）。Harmony 未適用。
        system_msgs = [m["content"] for m in messages_ if m.get("role") == "system"]
        user_msgs = [m["content"] for m in messages_ if m.get("role") == "user"]
        assistant = [m["content"] for m in messages_ if m.get("role") == "assistant"]
        sys_text = ("\n".join(system_msgs)).strip()
        usr_last = (user_msgs[-1] if user_msgs else "").strip()
        ex_ass = ("Example:\n" + assistant[0].strip()) if assistant else ""
        prompt = (
            f"{sys_text}\n\n{ex_ass}\n\nINPUT:\n{usr_last}\n\n"
            "Return ONLY the final prompt in 1–2 sentences (<=55 words). "
            "No thinking / no preface / no labels / no bullet points / no JSON."
        ).strip()
        return prompt

    def _generate_attempt() -> str:
        nonlocal last_obj
        full_prompt = _messages_to_single_prompt(messages)
        last_obj = _gen_once(full_prompt, add_stops=False)  # gpt-oss: stop無効
        resp = last_obj.get("response")
        if isinstance(resp, str) and resp.strip() and not _looks_like_instruction(resp):
            return resp.strip()

        # 継続生成（万一 length 停止した場合の保険）
        done_reason = last_obj.get("done_reason")
        ctx = last_obj.get("context")
        if (not resp or not str(resp).strip()) and str(done_reason) == "length" and isinstance(ctx, list) and len(ctx) > 0:
            cont_prompt = "Continue with ONLY the final prompt (1–2 sentences, <=55 words). No thinking or preface."
            last_obj = _gen_once(cont_prompt, ctx=ctx, npredict=max(40, int(options.get("num_predict", 120) // 2)), add_stops=False)
            resp2 = last_obj.get("response")
            if isinstance(resp2, str) and resp2.strip() and not _looks_like_instruction(resp2):
                return resp2.strip()

        # 最終の保険：message 形が返ってきたとき
        if "message" in last_obj:
            txt2 = _extract_text_from_message_obj(last_obj.get("message"))
            if isinstance(txt2, str) and txt2.strip() and not _looks_like_instruction(txt2):
                return txt2.strip()

        raise OllamaBadResponse("Ollama /api/generate returned no usable text",
                                obj=last_obj, which="generate",
                                payload_hint={"done_reason": last_obj.get("done_reason"),
                                              "has_context": isinstance(last_obj.get("context"), list)})

    for i in range(max(1, retries + 1)):
        for api in order:
            try:
                return (_chat_attempt() if api == "chat" else _generate_attempt()), last_obj
            except Exception:
                if i < retries or api != order[-1]:
                    time.sleep(max(0, retry_backoff_ms) / 1000.0)
                    continue
                raise

# ===== 品質ゲート（ローカル） =====
_DAY_TERMS = {"clear day", "daylight", "sunny", "bright sun", "midday", "blue sky"}
_NIGHT_TERMS = {"night", "streetlight", "streetlights", "sodium", "neon", "moonlit", "starry"}
_RAIN_TERMS = {"rain", "rainy", "shower", "wet", "drizzle"}
_DRY_TERMS = {"dry asphalt", "dry road", "dusty"}
_SNOW_TERMS = {"snow", "snowy"}
_FOG_TERMS = {"fog", "foggy", "mist", "haze", "hazy"}
_OVERCAST_TERMS = {"overcast", "cloudy"}
_SUN_TERMS = {"harsh midday sun", "bright sun", "direct sun", "crisp shadows"}

def _contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)

def quality_gate_check(prompt: str, style: Dict[str, Any], max_commas: int = 7) -> Dict[str, Any]:
    """
    矛盾と列挙過多を検査。必要なら "needs_refine": True を返す。
    """
    p = prompt.strip()
    pl = p.lower()

    issues: List[str] = []

    # スタイルの avoid_terms に触れていないか
    for term in style.get("avoid_terms", []):
        if term.lower() in pl:
            issues.append(f"avoid_term_detected:{term}")

    sid = style["id"]
    # 昼 vs 夜矛盾
    has_day = _contains_any(pl, list(_DAY_TERMS | _SUN_TERMS | {"clear", "day"}))
    has_night = _contains_any(pl, list(_NIGHT_TERMS | {"nighttime"}))
    if sid in ("day_clear", "day_overcast", "golden_hour", "dawn", "dusk"):
        # 昼・薄明系で「街灯支配・ネオン・月光主照明」などはNG
        if has_night and ("streetlight" in pl or "sodium" in pl or "neon" in pl or "moon" in pl or "night" in pl):
            issues.append("contradiction:daystyle_with_night_lighting")
    if sid == "night":
        if has_day and ("sun" in pl or "day" in pl or "blue sky" in pl):
            issues.append("contradiction:nightstyle_with_day_terms")

    # 雨 vs 乾燥
    if sid == "rain":
        if _contains_any(pl, list(_DRY_TERMS)):
            issues.append("contradiction:rain_with_dry_asphalt")
    # 雪 vs 濡れ強調
    if sid == "snow":
        if _contains_any(pl, ["wet asphalt", "rain"]):
            issues.append("contradiction:snow_with_wet_asphalt")
    # 霧 vs 鋭い直射
    if sid == "fog":
        if _contains_any(pl, list(_SUN_TERMS | {"crisp shadows", "sharp shadows"})):
            issues.append("contradiction:fog_with_harsh_sun")

    # 過度な列挙（カンマの数で簡易判定）
    comma_count = p.count(",")
    if comma_count > max_commas:
        issues.append(f"over_enumeration:commas={comma_count}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "needs_refine": len(issues) > 0,
    }

# ===== Refine（必要時のみ1回） =====
def refine_with_llm(base_url: str, model: str, prompt_text: str, style_brief: str,
                    max_words: int, options: Dict[str, Any],
                    timeout_sec: int, retries: int, retry_backoff_ms: int,
                    think: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    system = (
        "You are a careful editor for text-to-image prompts. "
        f"Rewrite the prompt into 1–2 sentences (<= {max_words} words) "
        "ensuring physical consistency between time/weather and lighting, "
        "avoiding long enumerations, and keeping the dashcam composition intact. "
        "Return ONLY the final prompt."
    )
    user = (
        "STYLE BRIEF\n"
        f"{style_brief}\n"
        "ORIGINAL PROMPT\n"
        f"{prompt_text}\n"
        "TASK\n"
        "Fix any contradictions or over-enumeration while preserving meaning and scene."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    fixed, last_obj = call_ollama_chat(
        base_url=base_url, model=model, messages=messages, options=options,
        timeout_sec=timeout_sec, prefer_api="chat", retries=retries,
        retry_backoff_ms=retry_backoff_ms, think=think
    )
    return fixed, last_obj

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
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator (Harmony/chat, v2 coherent style)")
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
    ap.add_argument("--api-mode", type=str, choices=["auto", "chat", "generate"], default="chat")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")
    # v2 追加
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--quality-gate", action="store_true", help="enable local contradiction/enumeration checks")
    ap.add_argument("--refine", action="store_true", help="run one-shot LLM refine if quality gate fails")
    return ap.parse_args()

def _ollama_warmup(logger: logging.Logger, base_url: str, model: str, timeout_sec: int, think: Optional[str]) -> None:
    """
    Harmony対応: /api/chat で短いやり取りを行い、空レス/起動遅延を検知。
    """
    base = base_url.rstrip("/")
    url = base + "/api/chat"
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
            return
        logger.warning("Warmup(chat): returned empty text. Model may be mid-load or using incompatible mode.")
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

    # gpt-oss の場合は chat 強制
    prefer_api = args.api_mode
    if _is_gptoss_model(args.model) and args.api_mode != "chat":
        logger.warning("gpt-oss detected: forcing /api/chat regardless of --api-mode=%s", args.api_mode)
        prefer_api = "chat"

    total_ok = 0
    total_ng = 0

    for split in args.splits:
        split_ok = 0
        split_ng = 0
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
            # 例: 1507678826876435_first_predTrainId.npy → stem_base = 1507678826876435_first
            name = Path(npy_path).name
            if name.endswith("_predTrainId.npy"):
                stem_base = name[:-len("_predTrainId.npy")]
            elif name.endswith("_semantic.npy"):
                stem_base = name[:-len("_semantic.npy")]
            else:
                stem_base = Path(name).stem

            out_dir = os.path.join(args.output_root, split, rel_dir)
            out_txt = os.path.join(out_dir, f"{stem_base}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem_base}_prompt.meta.json")
            out_dbg = os.path.join(out_dir, f"{stem_base}_prompt.debug.json")

            try:
                if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                    pbar.set_postfix_str("skip")
                    continue

                seg = np.load(npy_path)
                if seg.dtype != np.uint8:
                    seg = seg.astype(np.uint8)

                label_pairs = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
                label_names = [TRAINID_TO_NAME[i] for (i, _) in label_pairs]
                scene_hint = _scene_hint_from_labels(label_names)

                frame_seed = _derive_seed_for_item(args.run_seed, npy_path)
                rng = _rng_from_seed(frame_seed)

                # === スタイル一貫性（coherent）選択 ===
                style = choose_style_coherent(rng, label_names)
                style_brief = build_style_brief(style)

                # === メッセージ生成（v2） ===
                messages = build_chat_messages_v2(label_pairs, scene_hint, style_brief, max_words=args.max_words)

                # === オプション ===
                options = build_llm_options(rng, args.temperature, args.top_p, args.repeat_penalty,
                                            args.num_predict, frame_seed)

                image_path = _infer_image_path_for_npy_v2(npy_path, semseg_split_root, image_split_root)

                # === 生成 ===
                prompt_text, last_obj = call_ollama_chat(
                    base_url=args.ollama_base_url, model=args.model, messages=messages, options=options,
                    timeout_sec=args.timeout_sec, prefer_api=prefer_api,
                    retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                )

                clean = " ".join(prompt_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
                if not clean or _looks_like_instruction(clean):
                    raise OllamaBadResponse("Invalid-looking text (instructional)", obj=last_obj)

                # === 品質ゲート（必要ならRefine） ===
                gate_info = {"skipped": True, "result": None}
                refine_applied = False
                refine_obj_preview = None
                if args.quality_gate:
                    gate = quality_gate_check(clean, style)
                    gate_info = {"skipped": False, "result": gate}
                    if gate.get("needs_refine") and args.refine:
                        fixed_text, refine_obj = refine_with_llm(
                            base_url=args.ollama_base_url, model=args.model, prompt_text=clean, style_brief=style_brief,
                            max_words=args.max_words, options=options, timeout_sec=args.timeout_sec,
                            retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                        )
                        fixed_clean = " ".join(fixed_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
                        # 再チェック（再度NGでも保存はするが、デバッグに履歴を残す）
                        gate2 = quality_gate_check(fixed_clean, style)
                        refine_applied = True
                        clean = fixed_clean
                        refine_obj_preview = refine_obj

                        # gate2結果を gate_info に追記
                        gate_info["refine_applied"] = True
                        gate_info["after_refine"] = gate2

                meta = {
                    "split": split, "camera": args.camera,
                    "npy_path": npy_path, "image_path": image_path,
                    "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                    "scene_hint": scene_hint,
                    "style": style,
                    "ollama": {"base_url": args.ollama_base_url, "model": args.model, "options": options, "api_mode": prefer_api},
                    "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "timestamp": int(time.time()),
                    "quality_gate": gate_info,
                }

                save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, clean, meta)
                split_ok += 1
                total_ok += 1
                pbar.set_postfix_str("ok")

                # デバッグjson（Refine時はレスポンスヘッドもスナップショット）
                if refine_applied:
                    dbg = {
                        "npy_path": npy_path,
                        "split": split,
                        "refine_applied": True,
                        "ollama_first_response_head": (json.dumps(last_obj, ensure_ascii=False)[:8000] if last_obj else None),
                        "ollama_refine_response_head": (json.dumps(refine_obj_preview, ensure_ascii=False)[:8000] if refine_obj_preview else None),
                    }
                    try:
                        safe_write_json(out_dbg, dbg)
                    except Exception:
                        pass

            except OllamaBadResponse as e:
                split_ng += 1
                total_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] OllamaBadResponse: %s | %s", str(e), npy_path)
                dbg = {
                    "npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e),
                    "traceback": tb, "split": split, "which": getattr(e, "which", ""),
                    "payload_hint": getattr(e, "payload_hint", {}),
                }
                try:
                    obj = getattr(e, "obj", None)
                    if obj is not None:
                        s = json.dumps(obj, ensure_ascii=False)
                        dbg["ollama_response_preview_head"] = s[:10000]
                except Exception:
                    pass
                try:
                    safe_write_json(out_dbg, dbg)
                except Exception:
                    pass

            except Exception as e:
                split_ng += 1
                total_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), npy_path)
                dbg = {"npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e), "traceback": tb, "split": split}
                try:
                    safe_write_json(out_dbg, dbg)
                except Exception:
                    pass

        logger.info("[%s] done: OK=%d NG=%d -> %s", split, split_ok, split_ng, args.output_root)

    if total_ng == 0:
        print(f"✅ ALL DONE: OK={total_ok}, NG=0, out={args.output_root}")
    else:
        print(f"⚠️ DONE WITH ERRORS: OK={total_ok}, NG={total_ng}, out={args.output_root}")

if __name__ == "__main__":
    main()

"""

# 学習機（Ubuntu / CUDA12.8環境）での例
python /home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v4.py \
  --semseg-root /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes \
  --image-root  /home/shogo/coding/datasets/WaymoV2/extracted \
  --output-root /home/shogo/coding/datasets/WaymoV2/Prompts_gptoss_V4 \
  --splits training validation testing \
  --camera front \
  --naming predTrainId \
  --run-seed 20251029 \
  --overwrite \
  --warmup-check \
  --api-mode chat \
  --retries 3 \
  --num-predict 120 \
  --max-words 55 \
  --quality-gate \
  --refine \
  --verbose

"""