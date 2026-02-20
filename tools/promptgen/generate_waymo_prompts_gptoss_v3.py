# -*- coding: utf-8 -*-
"""
WaymoV2(Front) セマンティックラベル(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール（構造保持・多様性・再現性 / Harmony-chat / 矛盾自動修復）

【このv2のねらい】
- "自由度の高い自然文" を gpt-oss に書かせつつ、セマセグ由来のオブジェクト整合性を担保。
- スタイルは "一貫セット(time×weather×lighting)" だけを軽く提案。合わなければ無視してOK。
- 生成後に軽量な "整合性検査(heuristics)" を実施し、衝突があれば "自動リライト修復" を1回だけ行う。
- 実験再現性(run_seed/frame_seed)は維持。ログ/デバッグは詳細保存。

【前提】
- gpt-oss は Harmony形式の /api/chat を推奨。本ツールは gpt-oss 検出時に /api/chat に強制。
- thinkingは chat/generateトップレベルの "think" を通じて分離（保存はしない）。
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
DEFAULT_IMAGE_ROOT  = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_gptoss_v2"
DEFAULT_CAMERA = "front"
DEFAULT_SPLITS = ["training", "validation", "testing"]
DEFAULT_MODEL  = "gpt-oss:20b"
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

# ===== "一貫セット"のテーマ（time×weather×lighting の整合済プリセット） =====
# LLMには「合わなければ無視しても良い」という方針で軽く示すのみ（自由度を確保）
THEME_COMBOS = [
    # 昼
    {"theme": "clear_day", "hint": "on a clear day with crisp midday sunlight and defined shadows"},
    {"theme": "overcast_day", "hint": "under overcast sky with soft, diffuse lighting"},
    {"theme": "post_rain_day", "hint": "after light rain in daylight, with subtle reflections on asphalt"},
    # 夕方・朝
    {"theme": "golden_hour", "hint": "at golden hour with warm low-angle sunlight and long shadows"},
    {"theme": "dawn", "hint": "at dawn with gentle, cool light and a calm atmosphere"},
    {"theme": "dusk", "hint": "at dusk with low sun and gradually deepening contrast"},
    # 夜
    {"theme": "night_streetlights", "hint": "at night illuminated by streetlights and car headlights"},
    {"theme": "night_moonlit", "hint": "at night with faint moonlit ambience and subtle contrast"},
    {"theme": "night_city_neon", "hint": "at night with occasional neon and headlight highlights"},
    # 天候
    {"theme": "foggy", "hint": "in light fog that gently scatters the light and softens distance"},
    {"theme": "rainy_night", "hint": "at night after rain, streetlights reflecting on wet asphalt"},
    {"theme": "rainy_day", "hint": "in rainy daylight with soft contrast and wet surfaces"},
    {"theme": "snowy_day", "hint": "in light snow with muted colors and softened textures"},
]

# ===== ロガー =====
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
    logger.info("max-words : %d", args.max_words)
    logger.info("top-k-objects : %d", args.top_k_objects)
    logger.info("run-seed : %d", args.run_seed)
    logger.info("overwrite : %s", args.overwrite)
    logger.info("limit : %d", args.limit)
    logger.info("requests : %s", "available" if _HAS_REQUESTS else "fallback to urllib")
    logger.info("api-mode : %s", args.api_mode)
    logger.info("retries : %d", args.retries)
    logger.info("warmup : %s", args.warmup_check)
    logger.info("think : %s", args.think)
    logger.info("repair-pass : %s", args.repair_pass)

# ===== 走査 =====
def _list_semseg_files(semseg_split_root: str, camera: str, naming: str) -> List[str]:
    suffix = "predTrainId" if naming == "predTrainId" else "semantic"
    base = os.path.join(semseg_split_root, camera)
    out: List[str] = []
    if not os.path.isdir(base): return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(f"_{suffix}.npy"):
                out.append(os.path.join(r, f))
    return sorted(out)

def _infer_image_path_for_npy(npy_path: str, semseg_split_root: str, image_split_root: str) -> Optional[str]:
    """
    修正点：拡張子変換は「末尾の _{suffix}.npy を厳密に取り除く」ことで、"first/mid10s/last" を落とさない。
    例: 1507678826876435_first_predTrainId.npy -> 1507678826876435_first.jpg
    """
    d = os.path.dirname(npy_path)
    try:
        rel_dir = os.path.relpath(d, semseg_split_root)  # front/{segment}
    except Exception:
        return None
    fname = Path(npy_path).name
    if fname.endswith("_predTrainId.npy"):
        stem = fname[:-len("_predTrainId.npy")]
    elif fname.endswith("_semantic.npy"):
        stem = fname[:-len("_semantic.npy")]
    else:
        stem = Path(fname).stem  # フォールバック
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
    if seg.ndim != 2: raise ValueError(f"seg ndim expected 2, got {seg.ndim}")
    h, w = seg.shape; total = float(h*w)
    cnt = np.bincount(seg.flatten(), minlength=19).astype(np.int64)
    present: List[Tuple[int, float]] = []
    for cls_id in range(19):
        px = int(cnt[cls_id]); ratio = px/total if total > 0 else 0.0
        if px >= min_pixels or ratio >= min_area_ratio:
            present.append((cls_id, ratio))
    present.sort(key=lambda x: x[1], reverse=True)
    return present

def _to_oxford_comma(words: List[str]) -> str:
    if not words: return ""
    if len(words) == 1: return words[0]
    if len(words) == 2: return f"{words[0]} and {words[1]}"
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

# ===== テーマ（整合済）をランダム選択（ただし LLM は無視可） =====
def choose_theme_hint(rng: random.Random) -> Dict[str, str]:
    return dict(random.choice(THEME_COMBOS))

# ===== メッセージ（v2: 自由度重視＋矛盾しないテーマは軽く提案、合わなければ無視可） =====
def build_chat_messages_v2(
    label_pairs: List[Tuple[int, float]],
    scene_hint: str,
    theme_hint: Optional[str],
    max_words: int = 55,
    top_k_objects: int = 12
) -> List[Dict[str, str]]:
    # トップK（面積ソート済）だけ自然文に混ぜ込む
    names_sorted = [TRAINID_TO_NAME[i] for (i, _) in label_pairs][:top_k_objects]
    objects_phrase = _to_oxford_comma(names_sorted)

    # 例：自然文（1–2文、ラベルを自然に織り込む、カテゴリ追加禁止、カメラ保持）
    example_input = (
        "objects: road, sidewalk, buildings, cars, traffic signs, vegetation, people\n"
        "scene: a city street scene captured from a front-facing dashcam\n"
        "style_suggestion: at night illuminated by streetlights and car headlights (ignore if inconsistent)\n"
        "instruction: one or two sentences, <= 55 words, photorealistic natural English; "
        "do not introduce new object categories; keep camera angle/composition"
    )
    example_output = (
        "A realistic nighttime street scene from a front-facing dashcam, with buildings, road, sidewalk, cars, "
        "traffic signs, vegetation, and people. Streetlights and headlights shape the contrast; keep the original "
        "camera angle and composition without adding new categories."
    )

    # ユーザ入力（最小限ガイド＋styleはoptional）
    user_lines = [
        f"objects: {objects_phrase}",
        f"scene: {scene_hint}",
        "instruction: Write one photorealistic prompt in natural English (one or two sentences, <= {max_words} words). "
        "Do not introduce new object categories. Keep the original camera angle and composition. "
        "Prioritize internal coherence (time-of-day, weather, and lighting must agree)."
    ]
    if theme_hint:
        user_lines.append(f"style_suggestion: {theme_hint} (use only if it naturally fits; otherwise ignore)")

    user_input = "\n".join(user_lines)
    system = (
        "You are an expert prompt writer for text-to-image diffusion models used for autonomous driving datasets. "
        f"Return ONLY the final prompt text in one or two sentences (<= {max_words} words). "
        "No preface, no labels, no bullet points, no JSON."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Here is one example input/output pair."},
        {"role": "user", "content": f"INPUT:\n{example_input}"},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": "Now generate a prompt for the following context."},
        {"role": "user", "content": f"INPUT:\n{user_input}"},
    ]

# ===== 生成オプション =====
def build_llm_options(rng: random.Random, base_temperature: float, base_top_p: float,
                      base_repeat_penalty: float, num_predict: int, seed: int) -> Dict[str, Any]:
    def jitter(x: float, low: float, high: float, scale: float = 0.1) -> float:
        d = (rng.random() * 2 - 1) * scale
        v = x * (1.0 + d)
        return float(max(low, min(high, v)))
    return {
        "temperature": float(jitter(base_temperature, 0.05, 1.30, 0.10)),  # 1.5 -> 1.3に若干保守化（整合性重視）
        "top_p": float(jitter(base_top_p, 0.50, 1.00, 0.10)),
        "repeat_penalty": float(jitter(base_repeat_penalty, 1.00, 1.50, 0.08)),  # 2.0上限→1.5に保守
        "num_predict": int(num_predict),
        "seed": int(seed),
    }

# ===== 例外 =====
class OllamaBadResponse(RuntimeError):
    def __init__(self, message: str, obj: Any = None, which: str = "", payload_hint: Dict[str, Any] = None):
        super().__init__(message); self.obj = obj; self.which = which; self.payload_hint = payload_hint or {}

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

# ===== 抽出（thinking は採用しない） =====
def _extract_text_from_message_obj(m: Any) -> str:
    if isinstance(m, dict):
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts: List[str] = []
            for part in c:
                if isinstance(part, str): parts.append(part)
                elif isinstance(part, dict):
                    for k in ("text", "content", "value"):
                        v = part.get(k)
                        if isinstance(v, str): parts.append(v)
            text = " ".join(parts).strip()
            if text: return text
    return ""

def _looks_like_instruction(s: str) -> bool:
    t = s.lower()
    bad_subs = [
        "we need to", "return only", "do not include", "bullet points", "no brand names",
        "avoid", "instruction:", "objects:", "scene:", "atmosphere:", "lighting:", "should", "must",
    ]
    return any(k in t for k in bad_subs)

# ===== 単語ベースの軽量整合性検査 =====
_DAY_TERMS = {
    "day", "daylight", "clear day", "on a clear day", "midday", "afternoon", "sunny", "blue sky", "bright day"
}
_NIGHT_TERMS = {
    "night", "at night", "moonlit", "starry", "streetlights", "streetlight", "headlights", "neon", "sodium"
}
_TWILIGHT_TERMS = {"dawn", "dusk", "twilight", "sunset", "sunrise", "golden hour", "blue hour"}
_SUN_TERMS = {"sunlit", "sunlight", "midday sun", "bright sun", "warm sunlight", "low-angle sunlight"}
_ARTIF_LIGHT_TERMS = {"streetlight", "streetlights", "headlight", "headlights", "neon", "sodium", "led streetlights"}
_OVERCAST_TERMS = {"overcast", "cloudy", "soft diffuse light", "diffuse"}
_SHADOW_TERMS = {"crisp shadows", "defined shadows", "long shadows"}
_FOG_TERMS = {"fog", "foggy", "mist", "haze", "hazy"}
_CLEAR_VIS_TERMS = {"clear visibility", "crystal clear visibility"}
_RAIN_TERMS = {"rain", "rainy", "after rain", "after light rain", "wet asphalt"}
_DRY_TERMS = {"dry asphalt", "dry road"}

def _norm_text(s: str) -> str:
    # 記号・連続空白をならす
    s2 = re.sub(r"[-–—‑]", " ", s)  # 異体ハイフン→空白
    s2 = re.sub(r"\s+", " ", s2)
    return s2.strip().lower()

def detect_inconsistencies(prompt: str) -> List[str]:
    """
    軽量ヒューリスティック。false positive を避けるために最小限のルール。
    見つけた問題の textual list を返す（空なら整合OK）。
    """
    t = _norm_text(prompt)

    def has_any(terms: set) -> bool:
        return any(term in t for term in terms)

    issues: List[str] = []
    has_day = has_any(_DAY_TERMS)
    has_night = has_any(_NIGHT_TERMS)
    has_twilight = has_any(_TWILIGHT_TERMS)
    has_sun = has_any(_SUN_TERMS)
    has_art = has_any(_ARTIF_LIGHT_TERMS)
    is_overcast = has_any(_OVERCAST_TERMS)
    has_shadow = has_any(_SHADOW_TERMS)
    has_fog = has_any(_FOG_TERMS)
    has_clear_vis = has_any(_CLEAR_VIS_TERMS)
    has_rain = has_any(_RAIN_TERMS)
    has_dry = has_any(_DRY_TERMS)

    # 1) day と night の同居（薄暮を除外）
    if has_day and has_night and not has_twilight:
        issues.append("contains both day and night cues without twilight context")

    # 2) clear/day × streetlights（薄暮を許容）
    if has_day and has_art and not has_twilight:
        issues.append("mentions artificial street lighting alongside daytime without twilight")

    # 3) night × sunlight
    if has_night and has_sun:
        issues.append("mentions strong sunlight while stating night")

    # 4) overcast × crisp shadows
    if is_overcast and has_shadow:
        issues.append("mentions crisp/defined shadows under overcast conditions")

    # 5) fog × crystal clear visibility
    if has_fog and has_clear_vis:
        issues.append("mentions crystal clear visibility together with fog/mist/haze")

    # 6) rain/wet × dry asphalt 同居
    if has_rain and has_dry:
        issues.append("mentions both wet and dry asphalt")

    return issues

# ===== リライト（矛盾修復） =====
def build_repair_messages(prompt_text: str, objects_phrase: str, scene_hint: str,
                          max_words: int = 55) -> List[Dict[str, str]]:
    system = (
        "You fix prompts for diffusion models. Return only the final repaired prompt (one or two sentences, "
        f"<= {max_words} words), no preface."
    )
    user = (
        "The following prompt has minor internal inconsistencies (time-of-day, weather, lighting). "
        "Rewrite it to be coherent while keeping: (1) the same camera angle/composition, "
        "and (2) the same object categories (no new categories). "
        "Make it photorealistic natural English.\n\n"
        f"OBJECTS (must not introduce new categories beyond these): {objects_phrase}\n"
        f"SCENE: {scene_hint}\n"
        f"PROMPT:\n{prompt_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ===== Ollama 呼び出し（Harmony対応: 常に chat を優先 / gpt-ossは chat 強制） =====
def _is_gptoss_model(model: str) -> bool:
    return "gpt-oss" in (model or "").lower()

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
    gen_url  = base + "/api/generate"
    last_obj: Dict[str, Any] = {}

    force_chat = _is_gptoss_model(model)

    # 呼び出し順序
    if force_chat:
        order = ["chat"]
    else:
        order = (["chat"] if prefer_api == "chat" else ["generate"] if prefer_api == "generate" else ["chat", "generate"])

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

    def _gen_once(prompt: str, ctx: Optional[List[int]] = None, npredict: Optional[int] = None) -> Dict[str, Any]:
        gen_opts = dict(options)
        if npredict is not None: gen_opts["num_predict"] = int(npredict)
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "options": gen_opts}
        if think and think != "none":
            payload["think"] = str(think)
        if ctx is not None: payload["context"] = ctx
        return _http_post_json(gen_url, payload, timeout_sec)

    def _messages_to_single_prompt(messages_: List[Dict[str, str]]) -> str:
        system_msgs = [m["content"] for m in messages_ if m.get("role") == "system"]
        user_msgs   = [m["content"] for m in messages_ if m.get("role") == "user"]
        assistant   = [m["content"] for m in messages_ if m.get("role") == "assistant"]
        sys_text = ("\n".join(system_msgs)).strip()
        usr_last = (user_msgs[-1] if user_msgs else "").strip()
        ex_ass   = ("Example:\n" + assistant[0].strip()) if assistant else ""
        prompt = (
            f"{sys_text}\n\n{ex_ass}\n\nINPUT:\n{usr_last}\n\n"
            "Return ONLY the final prompt in one or two sentences (<=55 words). "
            "No thinking / no preface / no labels / no bullet points / no JSON."
        ).strip()
        return prompt

    def _generate_attempt() -> str:
        nonlocal last_obj
        full_prompt = _messages_to_single_prompt(messages)
        last_obj = _gen_once(full_prompt)
        resp = last_obj.get("response")
        if isinstance(resp, str) and resp.strip() and not _looks_like_instruction(resp):
            return resp.strip()
        # 継続生成の保険
        done_reason = last_obj.get("done_reason")
        ctx = last_obj.get("context")
        if (not resp or not str(resp).strip()) and str(done_reason) == "length" and isinstance(ctx, list) and len(ctx) > 0:
            cont_prompt = "Continue with ONLY the final prompt (one or two sentences, <=55 words). No preface."
            last_obj = _gen_once(cont_prompt, ctx=ctx, npredict=max(40, int(options.get("num_predict", 120)//2)))
            resp2 = last_obj.get("response")
            if isinstance(resp2, str) and resp2.strip() and not _looks_like_instruction(resp2):
                return resp2.strip()
        # 最後の保険：message 形
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
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator v2 (Harmony/chat, repair)")
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
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    ap.add_argument("--repeat-penalty", type=float, default=1.10, dest="repeat_penalty")
    ap.add_argument("--num-predict", type=int, default=120, dest="num_predict")
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--top-k-objects", type=int, default=12)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    # gpt-oss は chat 推奨。指定が generate でも gpt-oss 検出時は chat 強制。
    ap.add_argument("--api-mode", type=str, choices=["auto", "chat", "generate"], default="chat")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")
    ap.add_argument("--repair-pass", type=str, choices=["auto", "never", "always"], default="auto",
                    help="auto: run repair only if inconsistencies detected; always: always repair once; never: skip repair.")
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

    total_ok = 0; total_ng = 0

    for split in args.splits:
        semseg_split_root = os.path.join(args.semseg_root, split)
        image_split_root  = os.path.join(args.image_root,  split)
        jsonl_path = os.path.join(args.output_root, f"prompts_{split}.jsonl")

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0: files = files[:args.limit]
        if not files:
            logger.warning("[%s] no targets under: %s", split, semseg_split_root); continue

        logger.info("[%s] target files: %d", split, len(files))
        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            # stem は _predTrainId.npy / _semantic.npy を厳密除去（first/mid10s/last を保持）
            fname = Path(npy_path).name
            if fname.endswith("_predTrainId.npy"):
                stem = fname[:-len("_predTrainId.npy")]
            elif fname.endswith("_semantic.npy"):
                stem = fname[:-len("_semantic.npy")]
            else:
                stem = Path(fname).stem
            out_dir = os.path.join(args.output_root, split, rel_dir)
            out_txt = os.path.join(out_dir, f"{stem}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
            out_dbg = os.path.join(out_dir, f"{stem}_prompt.debug.json")

            try:
                if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                    pbar.set_postfix_str("skip"); continue

                seg = np.load(npy_path)
                if seg.dtype != np.uint8: seg = seg.astype(np.uint8)

                label_pairs = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
                label_names = [TRAINID_TO_NAME[i] for (i, _) in label_pairs]
                scene_hint  = _scene_hint_from_labels(label_names)

                frame_seed = _derive_seed_for_item(args.run_seed, npy_path)
                rng = _rng_from_seed(frame_seed)

                # 整合済テーマ（LLMは無視可）
                theme = choose_theme_hint(rng)
                theme_hint = theme.get("hint", "") if theme else ""

                # メッセージ（v2）
                messages = build_chat_messages_v2(
                    label_pairs=label_pairs,
                    scene_hint=scene_hint,
                    theme_hint=theme_hint,
                    max_words=args.max_words,
                    top_k_objects=args.top_k_objects
                )

                # LLM options
                options = build_llm_options(
                    rng, args.temperature, args.top_p, args.repeat_penalty, args.num_predict, frame_seed
                )

                image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)

                # === 1st 生成 ===
                prompt_text, last_obj = call_ollama_chat(
                    base_url=args.ollama_base_url, model=args.model, messages=messages, options=options,
                    timeout_sec=args.timeout_sec, prefer_api=prefer_api,
                    retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                )

                clean = " ".join(prompt_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
                if not clean or _looks_like_instruction(clean):
                    raise OllamaBadResponse("Invalid-looking text (instructional)", obj=last_obj)

                # === 整合性検査 ===
                issues = detect_inconsistencies(clean)
                repaired = False
                repair_obj: Dict[str, Any] = {}
                objects_phrase = _to_oxford_comma([TRAINID_TO_NAME[i] for (i, _) in label_pairs])

                need_repair = (
                    (args.repair_pass == "always") or
                    (args.repair_pass == "auto" and len(issues) > 0)
                )

                if need_repair:
                    repair_msgs = build_repair_messages(clean, objects_phrase, scene_hint, max_words=args.max_words)
                    repaired_text, repair_obj = call_ollama_chat(
                        base_url=args.ollama_base_url, model=args.model, messages=repair_msgs, options=options,
                        timeout_sec=args.timeout_sec, prefer_api=prefer_api,
                        retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                    )
                    repaired_text = " ".join(repaired_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
                    if repaired_text and not _looks_like_instruction(repaired_text):
                        # 修復案でもう一度チェック
                        issues2 = detect_inconsistencies(repaired_text)
                        if len(issues2) <= len(issues):
                            clean = repaired_text
                            issues = issues2
                            repaired = True

                meta = {
                    "split": split, "camera": args.camera,
                    "npy_path": npy_path, "image_path": image_path,
                    "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                    "scene_hint": scene_hint,
                    "theme_hint": theme_hint,
                    "ollama": {"base_url": args.ollama_base_url, "model": args.model,
                               "options": options, "api_mode": prefer_api},
                    "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "timestamp": int(time.time()),
                    "consistency_issues": issues,
                    "repaired": repaired,
                }

                save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, clean, meta)

                # デバッグ保存（元レスの頭部だけ）
                dbg = {
                    "npy_path": npy_path,
                    "initial_response_head": json.dumps(last_obj, ensure_ascii=False)[:8000],
                    "repair_response_head": json.dumps(repair_obj, ensure_ascii=False)[:8000] if repaired else "",
                    "initial_prompt": prompt_text,
                    "final_prompt": clean,
                    "issues": issues,
                    "repaired": repaired,
                }
                try:
                    safe_write_json(out_dbg, dbg)
                except Exception:
                    pass

                total_ok += 1; pbar.set_postfix_str("ok")

            except OllamaBadResponse as e:
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
                        s = json.dumps(obj, ensure_ascii=False); dbg["ollama_response_preview_head"] = s[:10000]
                except Exception:
                    pass
                try: safe_write_json(out_dbg, dbg)
                except Exception: pass

            except Exception as e:
                total_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s: %s | %s", type(e).__name__, str(e), npy_path)
                dbg = {"npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e), "traceback": tb, "split": split}
                try: safe_write_json(out_dbg, dbg)
                except Exception: pass

        logger.info("[%s] done: OK=%d NG=%d -> %s", split, total_ok, total_ng, args.output_root)

    if total_ng == 0:
        print(f"✅ ALL DONE: OK={total_ok}, NG=0, out={args.output_root}")
    else:
        print(f"⚠️ DONE WITH ERRORS: OK={total_ok}, NG={total_ng}, out={args.output_root}")

if __name__ == "__main__":
    main()
