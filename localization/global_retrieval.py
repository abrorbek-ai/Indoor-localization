"""Global deskriptor va reference kesh — o'xshash rasmlarni tez tartiblash."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .colmap_io import ImageRecord
from .config import (
    CACHE_DIR,
    QUERY_QUAD_MIN_SIDE_PX,
    QUERY_RETRIEVAL_GRID_COLS,
    QUERY_RETRIEVAL_GRID_ROWS,
    REF_GLOBAL_CACHE,
    REF_IMAGE_DIR,
)
from .layout_utils import natural_sort_key


def _build_global_descriptor_bgr(bgr: np.ndarray) -> np.ndarray:
    """Ko'p masshtabli intensivlik+gradient, HSV gistogramma, 4x4 grid statistikasi."""
    if bgr is None or bgr.size == 0:
        return np.zeros((1,), dtype=np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    parts: List[np.ndarray] = []

    for size in ((128, 128), (64, 64)):
        r = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
        parts.extend([r.flatten(), gx.flatten(), gy.flatten()])

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    parts.append(hist.astype(np.float32).flatten())

    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    grid_stats = []
    for i in range(4):
        for j in range(4):
            patch = small[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16]
            grid_stats.extend([float(patch.mean()), float(patch.std())])
    parts.append(np.array(grid_stats, dtype=np.float32))

    vec = np.concatenate(parts).astype(np.float32)
    n = float(np.linalg.norm(vec)) + 1e-12
    return vec / n


def build_global_descriptor(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((1,), dtype=np.float32)
    return _build_global_descriptor_bgr(bgr)


def iter_query_bgr_quads(bgr: np.ndarray) -> List[np.ndarray]:
    """
    Query (BGR) ni GRID_ROWS x GRID_COLS qismga bo'lish.
    Kichik rasm bo'lsa bitta butun rasm qaytariladi.
    """
    if bgr is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    rows, cols = QUERY_RETRIEVAL_GRID_ROWS, QUERY_RETRIEVAL_GRID_COLS
    if h < QUERY_QUAD_MIN_SIDE_PX * rows or w < QUERY_QUAD_MIN_SIDE_PX * cols:
        return [bgr]

    cell_h = h // rows
    cell_w = w // cols
    out: List[np.ndarray] = []
    for ri in range(rows):
        for ci in range(cols):
            y0 = ri * cell_h
            x0 = ci * cell_w
            y1 = h if ri == rows - 1 else (ri + 1) * cell_h
            x1 = w if ci == cols - 1 else (ci + 1) * cell_w
            out.append(bgr[y0:y1, x0:x1].copy())
    return out


class ReferenceGlobalCache:
    """ref_images uchun global vektorlarni diskda saqlaydi (mtime bo'yicha)."""

    def __init__(self, cache_path: Path | None = None, ref_dir: Path | None = None):
        self.cache_path = cache_path or REF_GLOBAL_CACHE
        self.ref_dir = ref_dir or REF_IMAGE_DIR
        self._name_to_vec: Dict[str, np.ndarray] = {}
        self._name_to_mtime: Dict[str, float] = {}

    def _load_disk(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("rb") as f:
                data = pickle.load(f)
            self._name_to_vec = data.get("vectors", {})
            self._name_to_mtime = data.get("mtimes", {})
        except (OSError, pickle.PickleError, EOFError):
            self._name_to_vec = {}
            self._name_to_mtime = {}

    def _save_disk(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump({"vectors": self._name_to_vec, "mtimes": self._name_to_mtime}, f)

    def ensure_fresh(self) -> None:
        self._load_disk()
        changed = False
        current_names: set[str] = set()
        for path in sorted(self.ref_dir.glob("*"), key=lambda p: natural_sort_key(p.name)):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            name = path.name
            current_names.add(name)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if name not in self._name_to_vec or self._name_to_mtime.get(name) != mtime:
                vec = build_global_descriptor(path)
                self._name_to_vec[name] = vec
                self._name_to_mtime[name] = mtime
                changed = True
        stale_names = set(self._name_to_vec) - current_names
        for name in stale_names:
            self._name_to_vec.pop(name, None)
            self._name_to_mtime.pop(name, None)
            changed = True
        if changed:
            self._save_disk()

    def dot_with_query(self, query_path: Path) -> Dict[str, float]:
        """Har bir query kvadrati uchun to'liq reference vektor bilan dot; ballar yig'indisi."""
        self.ensure_fresh()
        bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return {}
        quad_vecs = [_build_global_descriptor_bgr(q) for q in iter_query_bgr_quads(bgr)]
        if not quad_vecs:
            return {}

        def _dot(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape != b.shape:
                m = min(a.size, b.size)
                return float(np.dot(a[:m], b[:m]))
            return float(np.dot(a, b))

        scores: Dict[str, float] = {}
        for name, vec in self._name_to_vec.items():
            scores[name] = sum(_dot(qv, vec) for qv in quad_vecs)
        return scores


def rank_images_by_global_similarity(
    query_path: Path,
    images: Dict[int, ImageRecord],
    cache: ReferenceGlobalCache | None = None,
) -> List[Tuple[str, float]]:
    cache = cache or ReferenceGlobalCache()
    scores = cache.dot_with_query(query_path)
    ranked: List[Tuple[str, float]] = []
    for image in images.values():
        if image.name in scores:
            ranked.append((image.name, scores[image.name]))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def order_pnp_candidates(
    images: Dict[int, ImageRecord],
    query_path: Path,
    cache: ReferenceGlobalCache | None = None,
) -> List[ImageRecord]:
    """Avvalo global o'xshashlik bo'yicha, keyin qolgan reference nomlari tartib bilan."""
    cache = cache or ReferenceGlobalCache()
    ranked_names = [n for n, _ in rank_images_by_global_similarity(query_path, images, cache)]
    by_name = {img.name: img for img in images.values()}
    seen: set[str] = set()
    ordered: List[ImageRecord] = []
    for name in ranked_names:
        if name in by_name:
            ordered.append(by_name[name])
            seen.add(name)
    for image in sorted(images.values(), key=lambda i: natural_sort_key(i.name)):
        if image.name not in seen:
            ordered.append(image)
    return ordered
