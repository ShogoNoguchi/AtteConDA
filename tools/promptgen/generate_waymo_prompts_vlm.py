# -*- coding: utf-8 -*-
"""
generate_waymo_prompts_vlm.py
VLM(LLaVA/Qwen2-VL 等, Ollama) を用いた SynDiff-AD 方式の Caption→Style後付け(CaG) プロンプト生成。
- 画像 + 物体リスト→VLMに「天候/時間帯を言うな」で**中立キャプション c**を取得
- CLIP で現サブグループ z、別サブグループ z* を一様サンプル
- c に z* を**後付け**して最終プロンプト c*（1–2文, 実写・運転AI向け）
- 失敗時は**無限再試行**オプション

Ollama 側モデル例:
  - llava:13b           （画像入力可; 32GBで安定運用）
  - qwen2-vl:7b-instruct（画像入力可; 余裕）
  ※“最新最強”の選定はWeb不可のため、まずは上記を**選択式**で対応。

実行例（ucn-eval + pip overlay）:
  docker run --rm --gpus all --network host \
    -e PIP_OVERLAY_DIR=/mnt/hdd/ucn_eval_cache/pip-overlay \
    -e PIP_INSTALL="open_clip_torch pillow numpy requests tqdm" \
    -v /mnt/hdd/ucn_eval_cache:/mnt/hdd/ucn_eval_cache \
    -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
    -v /home/shogo/coding/datasets/WaymoV2/Prompts_vlm:/out \
    -v /home/shogo/coding/tools/promptgen/generate_waymo_prompts_vlm.py:/app/generate_waymo_prompts_vlm.py:ro \
    -v /home/shogo/coding/tools/promptgen/subgroups.py:/app/subgroups.py:ro \
    --entrypoint /usr/bin/python3 ucn-eval /app/generate_waymo_prompts_vlm.py \
    --semseg-root /home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes \
    --image-root  /home/shogo/coding/datasets/WaymoV2/extracted \
    --output-root /out \
    --splits training validation testing \
    --camera front --naming predTrainId \
    --ollama-base-url http://127.0.0.1:11434 \
    --vlm-model llava:13b \
    --clip-arch ViT-L-14 --clip-pretrained openai \
    --run-seed 20251029 --timeout-sec 180 --infinite-retry --verbose
"""

import os, sys, json, time, argparse, logging
from logging import handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib, random, traceback

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request  # type: ignore
    _HAS_REQUESTS = False

import torch
import open_clip

from subgroups import (
    list_all_subgroups, choose_subgroup, render_style_sentence
)

DEFAULT_SEMSEG_ROOT = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"
DEFAULT_IMAGE_ROOT  = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_OUTPUT_ROOT = "/home/shogo/coding/datasets/WaymoV2/Prompts_vlm"
DEFAULT_CAMERA = "front"
DEFAULT_SPLITS = ["training", "validation", "testing"]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VLM_MODEL  = "llava:13b"  # or "qwen2-vl:7b-instruct"

ALLOWED_IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

TRAINID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _setup_logger(out_root: str, verbose: bool) -> logging.Logger:
    _ensure_dir(out_root)
    log_path = os.path.join(out_root, "run.log")
    logger = logging.getLogger("promptgen_vlm")
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

def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if _HAS_REQUESTS:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout_sec)  # type: ignore
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, data=data, headers=headers)  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:    # type: ignore
            return json.loads(r.read().decode("utf-8"))

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

def _derive_seed(run_seed: int, key: str) -> int:
    h = hashlib.sha256((str(run_seed) + "@" + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF

def _rng(seed: int) -> random.Random:
    r = random.Random(); r.seed(seed); return r

# ===== CLIP によるサブグループ推定 =====
def _build_clip_texts() -> List[str]:
    texts = []
    for w,t in list_all_subgroups():
        if t == "Twilight":
            s = f"an outdoor driving scene in {w.lower()} weather at twilight"
        elif t == "Day":
            s = f"an outdoor driving scene in {w.lower()} weather during the day"
        else:
            s = f"an outdoor driving scene in {w.lower()} weather at night"
        texts.append(s)
    return texts

def _clip_argmax_subgroup(device: torch.device,
                          clip_arch: str, clip_pretrained: str,
                          image_pil: Image.Image) -> Tuple[str, str]:
    model, _, preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(clip_arch)
    with torch.no_grad():
        image = preprocess(image_pil).unsqueeze(0).to(device)
        texts = _build_clip_texts()
        text_tokens = tokenizer(texts).to(device)
        image_features = model.encode_image(image)
        text_features  = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(0)  # (9,)
        idx = int(torch.argmax(sims).item())
    return list_all_subgroups()[idx]

# ===== VLM（Ollama /api/chat 画像付き） =====
def _ollama_chat_vision(base_url: str, model: str, image_path: str, user_text: str,
                        timeout_sec: int, num_predict: int = 128, think: Optional[str]="low") -> Dict[str, Any]:
    """
    Ollama vision chat: メッセージに images=[<path>] を付ける形式
    """
    base = base_url.rstrip("/")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": user_text,
            "images": [image_path]  # file path をそのまま渡せる Ollama の標準仕様
        }],
        "stream": False,
        "options": {"num_predict": num_predict}
    }
    if think and think != "none": payload["think"] = str(think)
    url = base + "/api/chat"
    if _HAS_REQUESTS:
        resp = requests.post(url, data=json.dumps(payload).encode("utf-8"),
                             headers={"Content-Type":"application/json"}, timeout=timeout_sec)  # type: ignore
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type":"application/json"})  # type: ignore
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:  # type: ignore
            return json.loads(r.read().decode("utf-8"))

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

def _looks_instructional(s: str) -> bool:
    t = (s or "").lower()
    bad = ["do not", "bullet points", "avoid", "you should", "step", "objects:", "weather", "time:"]
    return any(b in t for b in bad)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Waymo -> VLM (Ollama vision) prompt generator (SynDiff-AD CaG)")
    ap.add_argument("--semseg-root", type=str, default=DEFAULT_SEMSEG_ROOT)
    ap.add_argument("--image-root",  type=str, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    ap.add_argument("--naming", type=str, choices=["predTrainId","semantic"], default="predTrainId")
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument("--min-pixels", type=int, default=4000)
    ap.add_argument("--run-seed", type=int, default=42)
    ap.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    ap.add_argument("--think", type=str, choices=["low","medium","high","none"], default="low")
    ap.add_argument("--num-predict", type=int, default=160)
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clip-arch", type=str, default="ViT-L-14")
    ap.add_argument("--clip-pretrained", type=str, default="openai")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--retry-backoff-ms", type=int, default=800)
    ap.add_argument("--infinite-retry", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def _objects_phrase(label_pairs: List[Tuple[int, float]]) -> str:
    names_sorted = [TRAINID_TO_NAME[i] for (i, _) in label_pairs][:12]
    return ", ".join(names_sorted)

def _build_vlm_user_prompt(objects_phrase: str) -> str:
    """
    SynDiff-AD の LLaVA 指示の本質：
      「物体と関係・背景を述べよ。ただし天候や時間帯は述べるな」
    リポジトリの設定例（LLAVACAPTION.prompt）でも同趣旨の文面（後掲の引用を参照）。
    """
    return (
        "Provide a concise, photorealistic description of the scene for autonomous driving datasets.\n"
        f"Objects (from semantic map): {objects_phrase}\n"
        "Describe object relations, layout, background, and image quality.\n"
        "Do NOT mention weather or time-of-day.\n"
        "Write in 1–2 sentences. Avoid lists or bullet points."
    )

def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)
    logger = _setup_logger(args.output_root, verbose=args.verbose)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    logger.info("device: %s", device)

    texts_for_clip = _build_clip_texts()  # 初回に構築

    total_ok = 0; total_ng = 0

    for split in args.splits:
        semseg_split_root = os.path.join(args.semseg_root, split)
        image_split_root  = os.path.join(args.image_root,  split)
        jsonl_path = os.path.join(args.output_root, f"prompts_{split}.jsonl")

        files = _list_semseg_files(semseg_split_root, args.camera, args.naming)
        if args.limit > 0: files = files[:args.limit]
        if not files:
            logger.warning("[%s] no targets: %s", split, semseg_split_root)
            continue

        logger.info("[%s] targets: %d", split, len(files))
        pbar = tqdm(files, desc=f"{split}")

        for npy_path in pbar:
            rel_dir = os.path.relpath(os.path.dirname(npy_path), semseg_split_root)  # front/{segment}
            stem = Path(npy_path).name.split("_")[0]
            out_dir  = os.path.join(args.output_root, split, rel_dir)
            out_txt  = os.path.join(out_dir, f"{stem}_prompt.txt")
            out_meta = os.path.join(out_dir, f"{stem}_prompt.meta.json")
            out_dbg  = os.path.join(out_dir, f"{stem}_prompt.debug.json")

            try:
                if (not args.overwrite) and os.path.exists(out_txt) and os.path.exists(out_meta):
                    pbar.set_postfix_str("skip"); continue

                seg = np.load(npy_path)
                if seg.dtype != np.uint8: seg = seg.astype(np.uint8)

                label_pairs = extract_present_labels(seg, args.min_area_ratio, args.min_pixels)
                if not label_pairs:
                    raise RuntimeError("no labels over thresholds")

                image_path = _infer_image_path_for_npy(npy_path, semseg_split_root, image_split_root)
                if not image_path:
                    raise RuntimeError("image not found for " + npy_path)

                # 画像読み込み
                img_pil = Image.open(image_path).convert("RGB")

                # フレーム固有 seed
                frame_seed = _derive_seed(args.run_seed, npy_path)
                rng = random.Random(); rng.seed(frame_seed)

                # 1) CLIP で現サブグループ z を推定
                z_weather, z_time = _clip_argmax_subgroup(device, args.clip_arch, args.clip_pretrained, img_pil)

                # 2) 目標サブグループ z* を一様抽出（z != z*）
                weather_tgt, time_tgt = choose_subgroup(rng, exclude=(z_weather, z_time))

                # 3) VLM に**中立キャプション c**を要求（天候/時間帯は言うな）
                objects_phrase = _objects_phrase(label_pairs)
                user_text = _build_vlm_user_prompt(objects_phrase)

                attempt = 0
                while True:
                    attempt += 1
                    try:
                        obj = _ollama_chat_vision(
                            base_url=args.ollama_base_url, model=args.vlm_model,
                            image_path=image_path, user_text=user_text,
                            timeout_sec=args.timeout_sec, num_predict=args.num_predict,
                            think=None if args.think == "none" else args.think
                        )
                        txt = ""
                        if isinstance(obj, dict) and "message" in obj:
                            txt = _extract_text_from_message_obj(obj.get("message"))
                        txt = (txt or "").strip()
                        if (not txt) or _looks_instructional(txt):
                            raise RuntimeError("invalid or instructional caption")

                        # 4) サブグループ z* を**後付け**（短い自然文）
                        style_sentence = render_style_sentence(weather_tgt, time_tgt)
                        final = f"{txt} {style_sentence}".strip()
                        if _looks_instructional(final):
                            raise RuntimeError("became instructional after style append")

                        # 保存
                        _ensure_dir(out_dir)
                        with open(out_txt, "w", encoding="utf-8") as f:
                            f.write(final + "\n")
                        meta = {
                            "split": split, "camera": args.camera,
                            "npy_path": npy_path, "image_path": image_path,
                            "labels": [{"trainId": int(i), "name": TRAINID_TO_NAME[i], "ratio": float(r)} for (i, r) in label_pairs],
                            "clip_detected_subgroup": {"weather": z_weather, "time": z_time},
                            "target_subgroup": {"weather": weather_tgt, "time": time_tgt},
                            "ollama": {"base_url": args.ollama_base_url, "model": args.vlm_model, "num_predict": args.num_predict},
                            "clip": {"arch": args.clip_arch, "pretrained": args.clip_pretrained},
                            "run_seed": int(args.run_seed), "frame_seed": int(frame_seed), "timestamp": int(time.time())
                        }
                        with open(out_meta, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                        with open(jsonl_path, "a", encoding="utf-8") as jf:
                            jf.write(json.dumps({"prompt": final, **meta}, ensure_ascii=False) + "\n")

                        total_ok += 1
                        pbar.set_postfix_str(f"ok({attempt})")
                        break

                    except Exception as e:
                        if args.infinite-retry or (attempt <= max(1, args.retries)):
                            time.sleep(max(0, args.retry_backoff_ms)/1000.0)
                            continue
                        raise

            except Exception as e:
                total_ng += 1
                tb = traceback.format_exc()
                logger.error("[ERR] %s | %s", str(e), npy_path)
                _ensure_dir(out_dir)
                dbg = {"npy_path": npy_path, "error_type": type(e).__name__, "error_msg": str(e), "traceback": tb, "split": split}
                with open(out_dbg, "w", encoding="utf-8") as f:
                    json.dump(dbg, f, ensure_ascii=False, indent=2)

    print(f"DONE: OK={total_ok}, NG={total_ng}, out={args.output_root}")

if __name__ == "__main__":
    main()
