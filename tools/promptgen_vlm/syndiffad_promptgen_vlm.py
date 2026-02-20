#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynDiff-AD 準拠 VLMプロンプト生成器（CLIP z特定 → VLM caption(サブグループ語禁止) → z*付与 → 矛盾修復）
- RTX5090(32GB)前提 / Torch 2.7.0+cu128 はベースを汚染せず、追加依存は pip --target の overlay に導入して使用
- 失敗しても該当フレームを飛ばさない：「回復可能エラー」は指数バックオフで成功まで無限リトライ
- 再現性：run_seed とフレーム固有 seed から温度/Top-p/RepeatPenalty に微弱ジッタ
- 出力：prompt.txt + prompt.meta.json + prompts_{split}.jsonl
- 根拠：SynDiff-AD Sec.3.2（LLaVAで“サブグループ語を禁止”したcaption→z*一文を付加）、CLIPでz同定

Paper:
- SynDiff-AD (Sec.3.2 / Algorithm 1) https://arxiv.org/abs/2411.16776
- LLaVA (Visual Instruction Tuning) https://arxiv.org/abs/2304.08485
- CLIP (image-text similarity) https://arxiv.org/abs/2103.00020
- Ollama Vision API (images base64) https://docs.ollama.com/capabilities/vision
"""

import os
import sys
import io
import re
import json
import time
import base64
import argparse
import logging
import hashlib
import random
import traceback
from logging import handlers
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch  # ★必須：関数内で使用するためトップレベル import
from PIL import Image
from tqdm import tqdm

# -------- HTTP（requests 優先 / fallback urllib） --------
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request  # type: ignore
    _HAS_REQUESTS = False

# -------- CLIP（open_clip 優先、無ければ openai-clip） --------
_OPENCLIP = None
_CLIP = None
try:
    import open_clip  # type: ignore
    _OPENCLIP = open_clip
except Exception:
    try:
        import clip  # type: ignore
        _CLIP = clip
    except Exception:
        pass

# ===== 既定パス =====
DEFAULT_SEMSEG_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_IMAGE_ROOT  = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_vlm"
DEFAULT_CAMERA = "front"
DEFAULT_SPLITS = ["training","validation","testing"]

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
# 既定 VLM：Qwen2.5‑VL 7B（32GBで余裕、キャプション強い）
DEFAULT_VLM    = "qwen2.5vl:7b"

NAMING_CHOICES = ["predTrainId","semantic"]
ALLOWED_IMG_EXTS = [".jpg",".jpeg",".png",".bmp"]

# ===== Cityscapes trainId -> 英語名 =====
TRAINID_TO_NAME = {
    0:"road",1:"sidewalk",2:"building",3:"wall",4:"fence",5:"pole",
    6:"traffic light",7:"traffic sign",8:"vegetation",9:"terrain",
    10:"sky",11:"person",12:"rider",13:"car",14:"truck",15:"bus",
    16:"train",17:"motorcycle",18:"bicycle"
}

# ===== サブグループ（Z） =====
SUBGROUPS = [
    "Clear, Day", "Clear, Twilight", "Clear, Night",
    "Cloudy, Day", "Cloudy, Twilight", "Cloudy, Night",
    "Rain, Day", "Rain, Twilight", "Rain, Night",
]

# ===== ロガー =====
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    path = os.path.join(out_root, "run.log")
    lg = logging.getLogger("syndiffad_vlm")
    lg.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = handlers.RotatingFileHandler(path, maxBytes=16*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
    if not lg.handlers:
        lg.addHandler(ch); lg.addHandler(fh)
    return lg

def _log_env(lg: logging.Logger, args: argparse.Namespace) -> None:
    lg.info("=== SynDiff-AD VLM PromptGen (CLIP→VLM→z* line) ===")
    lg.info("semseg_root : %s", args.semseg_root)
    lg.info("image_root  : %s", args.image_root)
    lg.info("output_root : %s", args.output_root)
    lg.info("splits      : %s", " ".join(args.splits))
    lg.info("camera      : %s", args.camera)
    lg.info("naming      : %s", args.naming)
    lg.info("ollama_url  : %s", args.ollama_base_url)
    lg.info("vlm_model   : %s", args.vlm_model)
    lg.info("num_predict : %d", args.num_predict)
    lg.info("run_seed    : %d", args.run_seed)
    lg.info("overwrite   : %s", args.overwrite)
    lg.info("limit       : %d", args.limit)
    lg.info("think       : %s", args.think)
    lg.info("vlm_temp/top_p/repeat : %.3f / %.3f / %.3f",
            args.vlm_temperature, args.vlm_top_p, args.vlm_repeat_penalty)

# ===== 走査 =====
def _list_semseg_files(semseg_split_root: str, camera: str, naming: str) -> List[str]:
    suffix = "predTrainId" if naming=="predTrainId" else "semantic"
    base = os.path.join(semseg_split_root, camera)
    out: List[str] = []
    if not os.path.isdir(base): return out
    for r, _d, fs in os.walk(base):
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

# ===== seed & rng =====
def _derive_seed(run_seed: int, key: str) -> int:
    h = hashlib.sha256((str(run_seed)+"@"+key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF

def _rng(seed: int) -> random.Random:
    r = random.Random(); r.seed(seed); return r

# ===== CLIP =====
def _load_clip(device: str="cuda"):
    if _OPENCLIP is not None:
        model, _, preprocess = _OPENCLIP.create_model_and_transforms("ViT-L-14", pretrained="openai")
        tokenizer = _OPENCLIP.get_tokenizer("ViT-L-14")
        model = model.to(device).eval()
        return ("open_clip", model, preprocess, tokenizer)
    if _CLIP is not None:
        model, preprocess = _CLIP.load("ViT-L/14", device=device, jit=False)
        return ("clip", model, preprocess, _CLIP.tokenize)
    raise RuntimeError("CLIP backend not available. Please install open_clip_torch (and timm) or clip.")

@torch.inference_mode()
def _clip_image_text_match(im: Image.Image, candidates: List[str], device: str="cuda") -> Tuple[str, List[float]]:
    kind, model, preprocess, tokenizer = _load_clip(device)
    img = preprocess(im).unsqueeze(0).to(device)
    if kind == "open_clip":
        toks = tokenizer(candidates).to(device)
        img_f = model.encode_image(img)
        txt_f = model.encode_text(toks)
    else:
        toks = tokenizer(candidates, truncate=True).to(device)  # type: ignore
        img_f = model.encode_image(img)
        txt_f = model.encode_text(toks)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    sim = (img_f @ txt_f.t()).squeeze(0).tolist()
    best = int(np.argmax(sim))
    return candidates[best], sim

# ===== 画像→base64 =====
def _image_to_base64(p: str) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ===== Ollama /api/chat（Vision） =====
def _ollama_chat_with_image(base_url: str, model: str, prompt: str, image_path: str,
                            num_predict: int = 256, temperature: float = 0.2,
                            top_p: float = 0.9, repeat_penalty: float = 1.05,
                            think: Optional[str] = "low", timeout_sec: int = 180) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    img_b64 = _image_to_base64(image_path)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [img_b64]
        }],
        "stream": False,
        "options": {
            "num_predict": int(num_predict),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repeat_penalty": float(repeat_penalty)
        }
    }
    if think and think != "none":
        payload["think"] = str(think)

    if _HAS_REQUESTS:
        resp = requests.post(url, data=json.dumps(payload), headers={"Content-Type":"application/json"},
                             timeout=timeout_sec)
        resp.raise_for_status()
        obj = resp.json()
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type":"application/json"})  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:  # type: ignore
            obj = json.loads(r.read().decode("utf-8"))

    msg = obj.get("message", {}).get("content", "")
    if not isinstance(msg, str) or not msg.strip():
        raise RuntimeError("VLM returned empty text")
    return msg.strip()

# ===== VLM 指示（サブグループ語を“明示的に禁止”） =====
SUBGROUP_KEYWORDS = [
    "clear","cloudy","overcast","rain","rainy","snow","snowy","fog","mist","haze","hazy",
    "night","day","daytime","twilight","dusk","dawn","golden hour"
]

def build_vlm_caption_prompt(label_names: List[str]) -> str:
    objects = ", ".join(label_names[:12]) if label_names else "road, sidewalk, building, car, traffic sign, vegetation, sky"
    ban = ", ".join(SUBGROUP_KEYWORDS)
    return (
        "You are a professional captioner for autonomous-driving dashcam images.\n"
        f"Objects present (from semantic segmentation): {objects}\n"
        "- Task: Write a concise, semantically dense description in 2–3 sentences that covers:\n"
        "  (a) scene layout & road topology/markings, (b) relative positions of vehicles/pedestrians,\n"
        "  (c) camera viewpoint and object interactions if visible.\n"
        f"- IMPORTANT: Do NOT use these words or their inflections: {ban}.\n"
        "- Maintain realism suitable for training autonomous driving perception (no artistic/fantasy wording).\n"
        "Return only the description text."
    )

# ===== サブグループ一文（z* → style line） =====
def subgroup_to_style_line(z: str) -> str:
    w, t = z.split(",")
    w = w.strip().lower()
    t = t.strip().lower()
    tod = ("twilight" if "twilight" in t else "night time" if "night" in t else "daytime")
    return f"Image taken in {w} weather at {tod}."

# ===== 矛盾検出＆修復 =====
# 兆候語（caption 側）
KW_NIGHT = ["night", "moonlit", "streetlight", "headlight", "taillight", "sodium"]
KW_TWIL  = ["twilight", "dusk", "dawn", "golden hour"]
KW_RAIN  = ["rain", "rainy", "drizzle", "downpour", "wet", "puddle", "raindrop"]
KW_CLEAR = ["clear", "sunny"]
KW_CLOUD = ["cloudy", "overcast", "hazy", "mist", "fog"]

def _lower(s: str) -> str:
    return " ".join(s.lower().split())

def _caption_hints(caption: str) -> Dict[str, bool]:
    t = _lower(caption)
    def any_in(keys): return any(k in t for k in keys)
    return {
        "night": any_in(KW_NIGHT),
        "twilight": any_in(KW_TWIL),
        "rain": any_in(KW_RAIN),
        "clear": any_in(KW_CLEAR),
        "cloudy": any_in(KW_CLOUD),
        # “daytime”明示はほぼ出ないが、明暗の否定情報として扱う
    }

def has_contradiction(final_prompt: str) -> bool:
    t = _lower(final_prompt)
    day = (" daytime" in t) or (" day " in t)
    night = any(k in t for k in KW_NIGHT) or (" night" in t)
    twi = any(k in t for k in KW_TWIL)
    clear = any(k in t for k in KW_CLEAR)
    rain = any(k in t for k in KW_RAIN)
    # 代表的な矛盾：daytime ∧ streetlight/headlight、clear ∧ rain など
    if day and (night or twi): return True
    if clear and rain: return True
    if day and ("streetlight" in t or "headlight" in t): return True
    return False

def repair_z_star_by_caption(z_star: str, caption: str, rng: random.Random) -> Tuple[str, str]:
    """caption の兆候語に基づき z* を修復する。戻り値: (new_z_star, reason)"""
    hint = _caption_hints(caption)
    # 候補集合の構築
    weather_opts = []
    if hint["rain"]: weather_opts = ["Rain"]
    elif hint["clear"]: weather_opts = ["Clear"]
    elif hint["cloudy"]: weather_opts = ["Cloudy"]
    # time-of-day
    tod_opts = []
    if hint["night"]: tod_opts = ["Night"]
    elif hint["twilight"]: tod_opts = ["Twilight"]
    # 条件が出なければ “どちらでも良い”
    weathers = (weather_opts if weather_opts else ["Clear","Cloudy","Rain"])
    tods = (tod_opts if tod_opts else ["Day","Twilight","Night"])
    # もとの z* は避ける
    candidates = [f"{w}, {t}" for w in weathers for t in tods if f"{w}, {t}" != z_star]
    if not candidates:  # 念のため
        candidates = [s for s in SUBGROUPS if s != z_star]
    new_z = rng.choice(candidates)
    reason = f"repair_by_caption_hints(weather={weather_opts or 'any'}, tod={tod_opts or 'any'})"
    return new_z, reason

# ===== 保存 =====
def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)

def save_txt_json_and_jsonl(out_txt: str, out_meta: str, jsonl: str, prompt: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(out_txt))
    with open(out_txt,"w",encoding="utf-8") as f: f.write(prompt.strip()+"\n")
    safe_write_json(out_meta, meta)
    _ensure_dir(os.path.dirname(jsonl))
    with open(jsonl,"a",encoding="utf-8") as jf:
        jf.write(json.dumps({"prompt":prompt,**meta},ensure_ascii=False)+"\n")

# ===== CLI =====
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SynDiff-AD VLM Prompt Generator (CLIP→VLM→z* line, with infinite retry)")
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--image-root",  type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--naming", type=str, choices=NAMING_CHOICES, default="predTrainId")
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument("--min-pixels", type=int, default=4000)
    ap.add_argument("--run-seed", type=int, default=20251029)
    ap.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA)
    ap.add_argument("--vlm-model", type=str, default=DEFAULT_VLM)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--vlm-temperature", type=float, default=0.25)
    ap.add_argument("--vlm-top-p", type=float, default=0.90)
    ap.add_argument("--vlm-repeat-penalty", type=float, default=1.05)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--think", type=str, choices=["none","low","medium","high"], default="low")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

# ===== ユーティリティ =====
def _jitter(rng: random.Random, x: float, lo: float, hi: float, scale: float=0.10) -> float:
    d = (rng.random()*2-1)*scale
    v = x*(1.0+d)
    return float(max(lo, min(hi, v)))

def _extract_labels(seg: np.ndarray, min_area_ratio: float, min_pixels: int) -> List[Tuple[int,float]]:
    h, w = seg.shape
    total = float(h*w)
    cnt = np.bincount(seg.flatten(), minlength=19).astype(np.int64)
    present=[]
    for cls_id in range(19):
        px = int(cnt[cls_id]); ratio = (px/total if total>0 else 0.0)
        if px >= min_pixels or ratio >= min_area_ratio:
            present.append((cls_id, ratio))
    present.sort(key=lambda x:x[1], reverse=True)
    return present

# ===== main =====
def main() -> None:
    args = parse_args()
    lg = _setup_logger(args.output_root, args.verbose)
    _log_env(lg, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _ = _load_clip(device=device)
        lg.info("CLIP backend loaded on %s.", device)
    except Exception as e:
        lg.error("CLIP load failed: %s", repr(e))
        raise

    for split in args.splits:
        semseg_split_root = os.path.join(args.semseg_root, split)
        image_split_root  = os.path.join(args.image_root, split)
        jsonl_path = os.path.join(args.output_root, f"prompts_{split}.jsonl")

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0: files = files[:args.limit]
        if not files:
            lg.warning("[%s] no targets under: %s", split, semseg_split_root)
            continue

        lg.info("[%s] target files: %d", split, len(files))
        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            stem = Path(npy_path).name.split("_")[0]
            out_dir  = os.path.join(args.output_root, split, rel_dir)
            out_txt  = os.path.join(out_dir, f"{stem}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
            out_dbg  = os.path.join(out_dir, f"{stem}_prompt.debug.json")

            # 既存があればスキップ（--overwrite がなければ）
            if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                pbar.set_postfix_str("skip")
                continue

            # 非回復エラーを先に弾く（ここで失敗したら無限再試行は無意味）
            if not os.path.exists(npy_path):
                lg.error("Missing seg file: %s", npy_path)
                continue
            image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)
            if (not image_path) or (not os.path.exists(image_path)):
                lg.error("Missing image for %s", npy_path)
                continue

            # ここからが “成功まで無限再試行” 範囲
            global_attempt = 0
            while True:
                global_attempt += 1
                try:
                    seg = np.load(npy_path)
                    if seg.dtype != np.uint8: seg = seg.astype(np.uint8)
                    label_pairs = _extract_labels(seg, args.min_area_ratio, args.min_pixels)
                    label_names = [TRAINID_TO_NAME[i] for (i,_) in label_pairs] if label_pairs else []

                    # CLIP による z 同定
                    im = Image.open(image_path).convert("RGB")
                    z, sims = _clip_image_text_match(im, SUBGROUPS, device=device)

                    # フレーム固有 seed と RNG
                    frame_seed = _derive_seed(args.run_seed, npy_path)
                    rng = _rng(frame_seed)

                    # z* は Z\{z} から一様
                    candidates = [s for s in SUBGROUPS if s != z]
                    z_star = rng.choice(candidates)

                    # VLM キャプション（サブグループ語は禁止）
                    caption_prompt = build_vlm_caption_prompt(label_names)

                    # VLM 内部リトライ（空返答/禁止語混入）
                    vlm_attempt = 0
                    while True:
                        vlm_attempt += 1
                        try:
                            temp = _jitter(rng, args.vlm_temperature, 0.05, 1.2, 0.12)
                            top_p = _jitter(rng, args.vlm_top_p, 0.50, 1.00, 0.08)
                            rep   = _jitter(rng, args.vlm_repeat_penalty, 1.00, 1.80, 0.08)
                            cap = _ollama_chat_with_image(
                                args.ollama_base_url, args.vlm_model, caption_prompt, image_path,
                                num_predict=args.num_predict, temperature=temp, top_p=top_p,
                                repeat_penalty=rep, think=args.think, timeout_sec=args.timeout_sec
                            )
                            cap_clean = " ".join(cap.replace("\n"," ").split())
                            low = cap_clean.lower()
                            if any(k in low for k in SUBGROUP_KEYWORDS):
                                raise RuntimeError("caption contains subgroup words")
                            break  # VLM 成功
                        except Exception as e:
                            wait = min(60.0, 1.5 ** min(12, vlm_attempt))
                            lg.warning("[VLM retry %d] %s -> sleep %.1fs", vlm_attempt, repr(e), wait)
                            time.sleep(wait)
                            continue

                    # 最終プロンプト = caption + z* のスタイル一文
                    style_line = subgroup_to_style_line(z_star)
                    final_prompt = f"{cap_clean} {style_line}"

                    # 矛盾検出→修復→再検査（最大数回）
                    repair_reason = ""
                    for _ in range(3):
                        if not has_contradiction(final_prompt):
                            break
                        z_star, repair_reason = repair_z_star_by_caption(z_star, cap_clean, rng)
                        style_line = subgroup_to_style_line(z_star)
                        final_prompt = f"{cap_clean} {style_line}"
                    # それでも矛盾なら VLM からやり直し
                    if has_contradiction(final_prompt):
                        raise RuntimeError("final prompt contradiction after repairs")

                    meta = {
                        "split": split,
                        "camera": args.camera,
                        "npy_path": npy_path,
                        "image_path": image_path,
                        "labels": [{"trainId":int(i),"name":TRAINID_TO_NAME[i],"ratio":float(r)} for (i,r) in label_pairs],
                        "subgroup_clip": {"z": z, "z_star": z_star, "similarities": sims},
                        "vlm": {
                            "base_url": args.ollama_base_url, "model": args.vlm_model,
                            "num_predict": args.num_predict,
                            "temperature": float(temp), "top_p": float(top_p), "repeat_penalty": float(rep)
                        },
                        "run_seed": int(args.run_seed),
                        "attempts": {"global": global_attempt, "vlm": vlm_attempt},
                        "repair_reason": repair_reason,
                        "timestamp": int(time.time())
                    }

                    save_txt_json_and_jsonl(out_txt, out_meta, jsonl_path, final_prompt, meta)
                    pbar.set_postfix_str("ok")
                    break  # ← 成功で無限ループ脱出

                except Exception as e:
                    tb = traceback.format_exc()
                    lg.error("[Global retry %d] %s | %s", global_attempt, repr(e), npy_path)
                    # デバッグ JSON を都度更新（上書き）
                    dbg = {
                        "npy_path": npy_path,
                        "image_path": image_path,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "traceback": tb,
                        "split": split,
                        "attempt": global_attempt
                    }
                    try: safe_write_json(out_dbg, dbg)
                    except Exception: pass
                    # 回復可能扱い：指数バックオフして続行
                    wait = min(90.0, 1.6 ** min(12, global_attempt))
                    time.sleep(wait)
                    continue

    lg.info("ALL DONE -> %s", args.output_root)

if __name__ == "__main__":
    main()
