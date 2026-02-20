#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3_vl_client.py  (LLaVA/その他 Ollama VLM 用クライアント)
- /api/chat は使用しない。/api/generate のみを使用（フォールバック禁止）。
- 画像は Base64 JPEG を "images": ["..."] で送信。
- 既定モデルは "llava:13b"（--ollama-model で上書き可）。
- 安定化策:
  1) 画像の長辺を 1536px にリサイズ
  2) JPEG 品質 = 85
  3) keep_alive = "4h" で常駐（連続推論の再ロードを抑止）
  4) do load request: ... "EOF" を 1 回だけ自動リトライ
  5) <think>…</think> および "Thinking..." の除去オプション
  6) 起動中モデルは /api/ps で検出し、/api/generate(keep_alive=0) でアンロード（VRAM解放）
  7) モデル未導入時は /api/pull を自動実行
"""

from __future__ import annotations
import io
import re
import time
import base64
from typing import Any, Union, Optional, List

import numpy as np
import requests
from PIL import Image


class OllamaBackend:
    def __init__(self,
                 host: str = "http://127.0.0.1:11434",
                 model: str = "llava:13b",
                 hidethinking: bool = True,
                 timeout: int = 300,
                 max_image_side: int = 1536,
                 jpeg_quality: int = 85,
                 keep_alive: str = "4h") -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.hidethinking = bool(hidethinking)
        self.timeout = int(timeout)
        self.max_image_side = int(max_image_side)
        self.jpeg_quality = int(jpeg_quality)
        self.keep_alive = str(keep_alive)
        self.session = requests.Session()

    # ---------- ユーティリティ ----------
    def _get(self, path: str) -> requests.Response:
        return self.session.get(f"{self.host}{path}", timeout=self.timeout)

    def _post(self, path: str, json: dict) -> requests.Response:
        return self.session.post(f"{self.host}{path}", json=json, timeout=self.timeout)

    # ---------- モデル存在確認＆自動pull ----------
    def ensure_model_present(self) -> None:
        try:
            r = self._get("/api/tags")
            r.raise_for_status()
            data = r.json() or {}
            names = [m.get("name") for m in data.get("models", [])]
            if self.model not in names:
                # /api/pull はストリーム JSON を返すが、完了まで待機させればOK
                pr = self._post("/api/pull", {"name": self.model})
                pr.raise_for_status()
        except Exception as e:
            raise requests.HTTPError(f"ensure_model_present failed: {e}")

    # ---------- 起動中モデル一覧（VRAM占有モデル） ----------
    def list_running_models(self) -> List[str]:
        try:
            r = self._get("/api/ps")  # 起動中モデル（VRAM上）を列挙
            r.raise_for_status()
            data = r.json() or {}
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception:
            # 旧バージョン等で /api/ps が未実装の環境は空扱い
            return []

    # ---------- モデルを VRAM からアンロード（keep_alive=0） ----------
    def unload_model(self, name: str) -> None:
        # 画像や長いプロンプトは不要。keep_alive=0 を渡すことが重要。
        payload = {
            "model": name,
            "prompt": "",          # 空でよい
            "stream": False,
            "keep_alive": 0
        }
        r = self._post("/api/generate", payload)
        # 一部の実装では 200 以外で返すことがあるため、200 以外なら警告相当扱い（例外にはしない）
        if r.status_code not in (200, 400, 404, 500):
            r.raise_for_status()

    def ensure_unloaded_all(self) -> None:
        names = self.list_running_models()
        for nm in names:
            try:
                self.unload_model(nm)
            except Exception:
                # 失敗しても他を続行
                pass
        # 念のため小休止（ドライバ側の解放反映）
        time.sleep(0.5)

    # ---------- 画像 → Base64 JPEG ----------
    def _encode_rgb_to_b64jpeg(self, img: Union[np.ndarray, Image.Image]) -> str:
        if isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB ndarray, got shape={img.shape}")
            pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
        elif isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            raise TypeError("img must be numpy.ndarray or PIL.Image.Image")

        w, h = pil.size
        m = max(w, h)
        if m > self.max_image_side:
            scale = self.max_image_side / float(m)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            pil = pil.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ---------- “Thinking…”/ <think>…</think> 除去 ----------
    def _strip_thinking(self, s: str) -> str:
        if not s:
            return ""
        out = s
        if self.hidethinking:
            out = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", out, flags=re.IGNORECASE | re.DOTALL)
            out = re.sub(r"^\s*Thinking\.\.\..*\n?", "", out, flags=re.IGNORECASE | re.MULTILINE)
        return out.strip()

    # ---------- /api/generate 呼び出し ----------
    def _call_generate_once(self, payload: dict) -> requests.Response:
        return self._post("/api/generate", payload)

    def _call_generate_stable(self, payload: dict) -> requests.Response:
        r = self._call_generate_once(payload)
        # 初回ロード直後に起き得る EOF を 1 回だけ待って再試行
        if r.status_code == 500 and ("do load request" in r.text and "\"/load\": EOF" in r.text):
            time.sleep(1.0)
            r = self._call_generate_once(payload)
        return r

    # ---------- ウォームアップ（テキスト1発） ----------
    def warmup(self) -> None:
        payload = {
            "model": self.model,
            "prompt": "ok",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": 2
            }
        }
        r = self._call_generate_stable(payload)
        if r.status_code != 200:
            raise requests.HTTPError(f"Ollama warmup failed {r.status_code}: {r.text[:1000]}")

    # ---------- 画像キャプション（本体） ----------
    def caption(self, rgb_image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        img_b64 = self._encode_rgb_to_b64jpeg(rgb_image)

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "keep_alive": self.keep_alive,  # 連続推論中は常駐
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 128
            }
        }

        r = self._call_generate_stable(payload)
        if r.status_code != 200:
            raise requests.HTTPError(f"Ollama /api/generate {r.status_code}: {r.text[:2000]}")
        data = r.json()
        text = data.get("response", "")
        return self._strip_thinking(text)


def build_vlm_backend(kind: str,
                      hidethinking: bool = True,
                      **kwargs) -> OllamaBackend:
    """
    kind == 'ollama' のみサポート（フォールバックなし）。
    モデルは既定 'llava:13b'。--ollama-model で上書き可。
    """
    kind = (kind or "ollama").lower()
    if kind != "ollama":
        raise ValueError("Only 'ollama' backend is supported in this environment (no fallback).")
    host = kwargs.get("host", "http://127.0.0.1:11434")
    model = kwargs.get("model", "llava:13b")
    timeout = int(kwargs.get("timeout", 300))
    return OllamaBackend(host=host, model=model, hidethinking=hidethinking, timeout=timeout)
