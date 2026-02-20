#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subgroups_synad.py
SynDiff-AD 方式のサブグループ定義と、Waymo用の形容詞・文飾テンプレ群。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ===== サブグループ定義（SynDiff-AD + 追加: 雨/雪/霧 を含める） =====
WEATHERS: List[str] = ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy"]  # 追加あり
TIMES: List[str]    = ["Day", "Dawn/Dusk", "Night"]

def all_subgroups() -> List[Tuple[str, str]]:
    return [(w, t) for w in WEATHERS for t in TIMES]

# ===== Cityscapes互換クラス名 =====
CITYSCAPES_TRAINID_TO_NAME = {
    0:"road", 1:"sidewalk", 2:"building", 3:"wall", 4:"fence", 5:"pole",
    6:"traffic light", 7:"traffic sign", 8:"vegetation", 9:"terrain", 10:"sky",
    11:"person", 12:"rider", 13:"car", 14:"truck", 15:"bus", 16:"train",
    17:"motorcycle", 18:"bicycle",
}
CITYSCAPES_NAME_LIST = list(CITYSCAPES_TRAINID_TO_NAME.values())

# ===== Waymo改良プロンプト用：サブグループ→形容詞（一語） =====
ADJECTIVE_BY_WEATHER = {
    "Clear":  "sunlit",
    "Cloudy": "overcast",
    "Rainy":  "rain-soaked",
    "Snowy":  "snow-covered",
    "Foggy":  "misty",
}
ADJECTIVE_BY_TIME = {
    "Day":       "daytime",
    "Dawn/Dusk": "twilight",
    "Night":     "nighttime",
}

# ===== Waymo改良プロンプト用：装飾文（各サブグループで3文以上） =====
DECORATIONS_WEATHER: Dict[str, List[str]] = {
    "Clear": [
        "Crisp sunlight with sharp, well-defined shadows.",
        "The sky appears vividly blue with high visibility.",
        "Surface textures are clear and contrast is strong."
    ],
    "Cloudy": [
        "Soft, diffuse lighting with muted contrast.",
        "Shadows are minimal due to overcast skies.",
        "Colors appear slightly desaturated and cool."
    ],
    "Rainy": [
        "Wet asphalt reflects streetlights and tail-lights.",
        "Raindrops streak across windshields and nearby surfaces.",
        "Headlights are diffused by water spray from passing cars."
    ],
    "Snowy": [
        "Fresh snow accumulates on rooftops and signage.",
        "Road edges and curbs show patches of snow and slush.",
        "Exhaust plumes and tire tracks are faintly visible in the cold air."
    ],
    "Foggy": [
        "Low-visibility haze softens edges and reduces contrast.",
        "Headlamps bloom in the mist, forming light halos.",
        "Distant objects are partially veiled by uniform fog."
    ],
}

DECORATIONS_TIME: Dict[str, List[str]] = {
    "Day": [
        "Color balance is neutral without artificial lighting dominance.",
        "Shadows follow the sun’s direction and are moderately long.",
        "Sky details remain visible throughout the scene."
    ],
    "Dawn/Dusk": [
        "Warm, golden-hour glow with long, soft shadows.",
        "The sky gradients from orange to blue near the horizon.",
        "Specular highlights are gentle and cinematic."
    ],
    "Night": [
        "Streetlights and neon signs dominate the illumination.",
        "Reflections from headlights animate the road surface.",
        "Darker backgrounds bring out high-contrast highlights."
    ],
}

def pick_adjective(weather: str, time: str) -> str:
    # 形容詞は（天候→一語）＋（時刻→一語）を組にして自然に接続
    return f"{ADJECTIVE_BY_TIME.get(time,'')} {ADJECTIVE_BY_WEATHER.get(weather,'')}".strip()

def decorations_for(weather: str, time: str) -> List[str]:
    return DECORATIONS_WEATHER.get(weather, []) + DECORATIONS_TIME.get(time, [])
