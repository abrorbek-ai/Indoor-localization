"""Reference nomlari va logical corridor layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import REF_IMAGE_DIR


def natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def extract_sequence_id(name: str) -> int:
    match = re.search(r"img_(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def extract_view_type(name: str) -> str:
    lowered = name.lower()
    if "_left" in lowered:
        return "left"
    if "_right" in lowered:
        return "right"
    return "front"


def logical_layout_position(name: str) -> np.ndarray:
    seq = extract_sequence_id(name)
    view = extract_view_type(name)
    x_map = {"left": -1.0, "front": 0.0, "right": 1.0}
    return np.array([x_map.get(view, 0.0), -float(seq)], dtype=np.float64)


def build_logical_layout_items() -> List[Tuple[str, np.ndarray]]:
    items = []
    for path in sorted(REF_IMAGE_DIR.glob("*"), key=lambda p: natural_sort_key(p.name)):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        items.append((path.name, logical_layout_position(path.name)))
    return items
