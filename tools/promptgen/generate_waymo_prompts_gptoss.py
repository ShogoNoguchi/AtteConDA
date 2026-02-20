# -*- coding: utf-8 -*-
"""
WaymoV2(Front) セマンティックラベル(OneFormer Cityscapes trainId) -> gpt-oss:20b (Ollama)
英語プロンプト自動生成ツール（構造保持・多様性・再現性対応 / thinkingは保存しない / Harmony対応）

重要変更点（2025-10-29）:
- gpt-oss は Harmony response format 前提。/api/generate の素プロンプトは非対応になりやすく、空レスの原因。
  -> gpt-oss 検出時は /api/chat を強制使用（ユーザが --api-mode generate を指定しても警告の上で chat に切替）。
- warmup も /api/chat で実施。
- gpt-oss では stop トークンを付与しない（Harmony/思考出力と衝突しやすく空レスの温床）。
- think レベルを chat/generate のトップレベル "think" フィールドで制御（"low"|"medium"|"high"）。既定は "low"。
- 出力抽出は「/api/chat: message.content」のみを採用。message.thinking は保存しない（ファイルに出さない）。

参考:
- gpt-oss モデルカード: Harmony 形式必須。Transformers の chat テンプレート、または手動で Harmony を適用。 
  https://huggingface.co/openai/gpt-oss-20b
- Ollama の gpt-oss テンプレート（Harmony 形式のプロンプトレンダリング）:
  https://ollama.com/library/gpt-oss:20b (template)
- Ollama の thinking / reasoning 設定（chat/generate 最上位の "think" フィールド）:
  https://docs.ollama.com/capabilities/thinking

"""
# =============================================================================
# ⚠️ 重要：GPU対応版Ollama環境でのみ実行すること（RTX5090 + CUDA13.0）
#
# 【起動前チェックリスト】
# 1. NVIDIA Container Toolkit がインストール済みか確認：
#       dpkg -l | grep nvidia-container-toolkit
#
# 2. Ollama コンテナを GPU + ローカルモデル共有で起動：
#       docker stop ollama && docker rm ollama
#       docker run -d --gpus all \
#         -v /home/shogo/.ollama:/root/.ollama \
#         -p 11434:11434 \
#         --name ollama \
#         ollama/ollama:latest
#
# 3. GPU稼働確認：
#       curl -s http://localhost:11434/api/version | jq
#       → {"version":"0.12.x","gpu":true,"device":"NVIDIA RTX 5090","cuda":"13.0"} が出ればOK
#       または watch -n 1 nvidia-smi で `ollama` プロセスが見えること。
#
# 4. モデルは再ダウンロード不要：
#       /home/shogo/.ollama に gpt-oss:20b が既に存在する。
#       Docker側にも -v でマウントして再利用する。
#
# 【よくある失敗】
#   × GPU指定(--gpus all)を忘れる → CPUモードで空レス連発
#   × /home/shogo/.ollama をマウントしない → 毎回40GB再DL
#   × warmup中に実行 → "Warmup(chat): returned empty text" が出る
#
# 【デバッグ用】
#   docker logs -f ollama
#   docker exec -it ollama bash
#   curl -s http://localhost:11434/api/chat ... （手動呼び出しテスト）
#
# =============================================================================

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

import numpy as np
from tqdm import tqdm

# requests があれば使用、無ければ urllib へフォールバック
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request
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

# ===== Atmosphere / Lighting 候補 =====
ATMOSPHERE_CHOICES = [
    "in foggy weather", "in rainy weather", "in snowy weather", "in overcast weather", "at night",
    "at golden hour", "at dawn", "at dusk", "on a clear day", "after light rain",
    "in hazy conditions", "on a humid summer afternoon",
]
LIGHTING_TEXTURE_CHOICES = [
    "soft diffuse light", "harsh midday sun with crisp shadows", "warm sodium streetlights",
    "cool LED streetlights", "headlights and taillights providing rim light",
    "wet asphalt with mild reflections", "backlit with low sun and long shadows",
    "overcast softbox-like illumination", "mist scattering the light slightly",
    "moonlit ambience with subtle contrast", "dry asphalt and clear visibility",
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
    logger.info("=== PromptGen (Waymo -> gpt-oss:20b) ===")
    logger.info("semseg-root: %s", args.semseg_root)
    logger.info("image-root : %s", args.image_root)
    logger.info("output-root: %s", args.output_root)
    logger.info("splits     : %s", " ".join(args.splits))
    logger.info("camera     : %s", args.camera)
    logger.info("naming     : %s", args.naming)
    logger.info("ollama-url : %s", args.ollama_base_url)
    logger.info("model      : %s", args.model)
    logger.info("min-area-ratio: %.6f, min-pixels: %d", args.min_area_ratio, args.min_pixels)
    logger.info("num-predict: %d", args.num_predict)
    logger.info("run-seed   : %d", args.run_seed)
    logger.info("overwrite  : %s", args.overwrite)
    logger.info("limit      : %d", args.limit)
    logger.info("requests   : %s", "available" if _HAS_REQUESTS else "fallback to urllib")
    logger.info("api-mode   : %s", args.api_mode)
    logger.info("retries    : %d", args.retries)
    logger.info("warmup     : %s", args.warmup_check)
    logger.info("think      : %s", args.think)

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
    d = os.path.dirname(npy_path)
    try:
        rel_dir = os.path.relpath(d, semseg_split_root)  # front/{segment}
    except Exception:
        return None
    stem = Path(npy_path).name.split("_")[0]  # first / mid10s / last
    for ext in ALLOWED_IMG_EXTS:
        cand = os.path.join(image_split_root, rel_dir, f"{stem}{ext}")
        if os.path.exists(cand): return cand
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
    present = []
    for cls_id in range(19):
        px = int(cnt[cls_id]); ratio = px/total if total>0 else 0.0
        if px >= min_pixels or ratio >= min_area_ratio:
            present.append((cls_id, ratio))
    present.sort(key=lambda x: x[1], reverse=True)
    return present

def _to_oxford_comma(words: List[str]) -> str:
    if not words: return ""
    if len(words)==1: return words[0]
    if len(words)==2: return f"{words[0]} and {words[1]}"
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

def choose_style_pack(rng: random.Random) -> Dict[str, str]:
    return {"atmosphere": rng.choice(ATMOSPHERE_CHOICES),
            "lighting": rng.choice(LIGHTING_TEXTURE_CHOICES)}

# ===== メッセージ =====
def build_chat_messages(label_pairs: List[Tuple[int, float]], scene_hint: str,
                        style_pack: Dict[str, str], max_words: int=55) -> List[Dict[str, str]]:
    names_sorted = [TRAINID_TO_NAME[i] for (i, _) in label_pairs][:12]
    objects_phrase = _to_oxford_comma(names_sorted)
    example_input = (
        "objects: road, sidewalk, buildings, cars, traffic signs, vegetation, people\n"
        "scene: a city street scene captured from a front-facing dashcam\n"
        "atmosphere: at night\n"
        "lighting: warm sodium streetlights, wet asphalt with mild reflections\n"
        "instruction: keep the same camera angle and composition; do not add objects not listed"
    )
    example_output = (
        "A realistic nighttime city street scene with buildings, road, sidewalk, cars, "
        "traffic signs, vegetation, and people, illuminated by streetlights and car headlights. "
        "The asphalt is slightly wet after light rain, showing subtle reflections. "
        "Keep the same camera angle and composition as the original."
    )
    user_input = (
        f"objects: {objects_phrase}\n"
        f"scene: {scene_hint}\n"
        f"atmosphere: {style_pack.get('atmosphere','')}\n"
        f"lighting: {style_pack.get('lighting','')}\n"
        "instruction: keep the same camera angle and composition; "
        "do not introduce new object categories; describe naturally in 1-2 sentences."
    )
    system = (
        "You are an expert prompt writer for text-to-image diffusion models used for autonomous driving datasets. "
        f"Write one photorealistic English prompt in 1–2 sentences (<= {max_words} words). "
        "Output ONLY the final prompt text and nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Here is one example input/output pair."},
        {"role": "user", "content": f"INPUT:\n{example_input}"},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": "Now generate a prompt for the following context."},
        {"role": "user", "content": f"INPUT:\n{user_input}"},
    ]

# ===== オプション =====
def build_llm_options(rng: random.Random, base_temperature: float, base_top_p: float,
                      base_repeat_penalty: float, num_predict: int, seed: int) -> Dict[str, Any]:
    def jitter(x: float, low: float, high: float, scale: float=0.1) -> float:
        d = (rng.random()*2-1)*scale; v = x*(1.0+d)
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
        super().__init__(message); self.obj = obj; self.which = which; self.payload_hint = payload_hint or {}

# ===== HTTP =====
def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8"); headers = {"Content-Type": "application/json"}
    if _HAS_REQUESTS:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout_sec); resp.raise_for_status(); return resp.json()
    else:
        req = urllib.request.Request(url, data=data, headers=headers)  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:  # type: ignore
            return json.loads(r.read().decode("utf-8"))

# ===== 抽出（thinking は採用しない） =====
def _extract_text_from_message_obj(m: Any) -> str:
    if isinstance(m, dict):
        c = m.get("content")
        if isinstance(c, str) and c.strip(): return c.strip()
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

# ===== ユーティリティ =====
def _is_gptoss_model(model: str) -> bool:
    return "gpt-oss" in (model or "").lower()

# ===== Ollama 呼び出し（Harmony対応: 常に chat を優先 / gpt-ossは chat 強制） =====
def call_ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                     timeout_sec: int=120, prefer_api: str="auto", retries: int=2,
                     retry_backoff_ms: int=800, think: Optional[str]="low") -> Tuple[str, Dict[str, Any]]:
    """
    戻り値: (final_text, last_obj)
    - gpt-oss の場合は /api/chat を強制。Harmony テンプレートは chat で自動適用される。
    - think: "low"|"medium"|"high"|None を chat/generate トップレベルに渡す（thinking分離）。
    """
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    gen_url  = base + "/api/generate"
    last_obj: Dict[str, Any] = {}

    # gpt-oss は Harmony 必須 → chat 強制
    force_chat = _is_gptoss_model(model)
    if force_chat and prefer_api != "chat":
        # ロガーが未確定のため print は避ける。呼び出し側が logger を持っているので例外化はしない。
        pass

    # 呼び出し順序
    if force_chat:
        order = ["chat"]
    else:
        order_auto = ["chat", "generate"]
        order = (["chat"] if prefer_api=="chat" else ["generate"] if prefer_api=="generate" else order_auto)

    def _chat_attempt() -> str:
        nonlocal last_obj
        payload = {"model": model, "messages": messages, "stream": False, "options": options}
        if think:
            payload["think"] = str(think)  # gpt-oss は "low"/"medium"/"high"
        last_obj = _http_post_json(chat_url, payload, timeout_sec)
        if isinstance(last_obj, dict) and "message" in last_obj:
            txt = _extract_text_from_message_obj(last_obj.get("message"))
            if isinstance(txt, str) and txt.strip() and not _looks_like_instruction(txt):
                return txt.strip()
        raise OllamaBadResponse("Ollama /api/chat returned no usable text", obj=last_obj, which="chat")

    def _gen_once(prompt: str, ctx: Optional[List[int]]=None, npredict: Optional[int]=None, add_stops: bool = False) -> Dict[str, Any]:
        # gpt-oss では generate + 生文字列は Harmony 不足のため原則使わない。
        gen_opts = dict(options)
        if npredict is not None: gen_opts["num_predict"] = int(npredict)
        # gpt-oss では stop は付けない（Harmony/思考出力と衝突回避）
        if add_stops and (not _is_gptoss_model(model)):
            stops = gen_opts.get("stop", [])
            if not isinstance(stops, list): stops = [stops]
            stops.extend(["\nobjects:", "\nscene:", "\natmosphere:", "\nlighting:", "Instruction:", "instruction:", "We need to"])
            # 重複排除
            seen = set(); stops_unique = []
            for s in stops:
                if s not in seen:
                    stops_unique.append(s); seen.add(s)
            gen_opts["stop"] = stops_unique
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "options": gen_opts}
        if think:
            payload["think"] = str(think)
        if ctx is not None: payload["context"] = ctx
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
            last_obj = _gen_once(cont_prompt, ctx=ctx, npredict=max(40, int(options.get("num_predict", 120)//2)), add_stops=False)
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

    for i in range(max(1, retries+1)):
        for api in order:
            try:
                return (_chat_attempt() if api=="chat" else _generate_attempt()), last_obj
            except Exception:
                if i < retries or api != order[-1]:
                    time.sleep(max(0, retry_backoff_ms)/1000.0)
                    continue
                raise

# ===== 保存 =====
def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def save_txt_json_and_jsonl(out_txt: str, out_meta_json: str, jsonl_path: str,
                            prompt: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(out_txt))
    with open(out_txt, "w", encoding="utf-8") as f: f.write(prompt.strip() + "\n")
    safe_write_json(out_meta_json, meta)
    _ensure_dir(os.path.dirname(jsonl_path))
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps({"prompt": prompt, **meta}, ensure_ascii=False) + "\n")

# ===== CLI / ウォームアップ / main =====
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WaymoV2 OneFormer(trainId) -> gpt-oss:20b Prompt Generator (Harmony/chat)")
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
    # 重要: 既定は chat。gpt-oss では chat 強制。generate 指定時は警告の上で chat に切替。
    ap.add_argument("--api-mode", type=str, choices=["auto", "chat", "generate"], default="chat")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    # thinking/推論長: gpt-oss は "low"/"medium"/"high"
    ap.add_argument("--think", type=str, choices=["low", "medium", "high", "none"], default="low",
                    help="gpt-oss reasoning trace level. 'none' to omit the field.")
    ap.add_argument("--warmup-check", action="store_true")
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
        image_split_root = os.path.join(args.image_root, split)
        jsonl_path = os.path.join(args.output_root, f"prompts_{split}.jsonl")

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0: files = files[:args.limit]
        if not files:
            logger.warning("[%s] no targets under: %s", split, semseg_split_root); continue

        logger.info("[%s] target files: %d", split, len(files))
        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            stem = Path(npy_path).name.split("_")[0]
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
                scene_hint = _scene_hint_from_labels(label_names)

                frame_seed = _derive_seed_for_item(args.run_seed, npy_path)
                rng = _rng_from_seed(frame_seed)

                style_pack = choose_style_pack(rng)
                messages = build_chat_messages(label_pairs, scene_hint, style_pack, max_words=55)
                options = build_llm_options(rng, args.temperature, args.top_p, args.repeat_penalty,
                                            args.num_predict, frame_seed)

                image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)

                # === LLM 呼び出し（Harmony対応: chat優先、thinkingは分離されるが保存しない） ===
                prompt_text, last_obj = call_ollama_chat(
                    base_url=args.ollama_base_url, model=args.model, messages=messages, options=options,
                    timeout_sec=args.timeout_sec, prefer_api=prefer_api,
                    retries=args.retries, retry_backoff_ms=args.retry_backoff_ms, think=think_level
                )

                clean = " ".join(prompt_text.replace("\n", " ").replace("\"", "").replace("```", "").split())
                if not clean or _looks_like_instruction(clean):
                    raise OllamaBadResponse("Invalid-looking text (instructional)", obj=last_obj)

                meta = {
                    "split": split, "camera": args.camera,
                    "npy_path": npy_path, "image_path": image_path,
                    "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                    "style_pack": style_pack, "scene_hint": scene_hint,
                    "ollama": {"base_url": args.ollama_base_url, "model": args.model, "options": options, "api_mode": prefer_api},
                    "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "timestamp": int(time.time()),
                }

                save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, clean, meta)
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
                except Exception: pass
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
