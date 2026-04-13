"""Image features, matching, geometric verification, and coverage metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from io_colmap import ReferenceFeatures, rootsift


@dataclass
class ImageFeatures:
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    scale_to_original: float = 1.0


@dataclass
class QueryRegionFeatures:
    region_id: int
    row: int
    col: int
    bbox_xyxy: Tuple[int, int, int, int]
    image_shape: Tuple[int, int]
    keypoints_xy: np.ndarray
    descriptors: np.ndarray

    @property
    def feature_count(self) -> int:
        return int(len(self.descriptors))


@dataclass
class MatchResult:
    raw_matches: int
    good_matches: List[cv2.DMatch]
    verified_matches: List[cv2.DMatch]
    fundamental_matrix: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    geometry_method: str
    inlier_ratio: float
    invalid_geometry: bool


@dataclass
class CoverageMetrics:
    occupied_cells: int
    min_cell_inliers: int
    max_cell_inliers: int
    balance: float
    dispersion: float
    coverage_ratio: float
    cell_counts: List[int]


def load_image_gray(path: Path, max_side: int = 0) -> Tuple[np.ndarray, Tuple[int, int], float]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    original_shape = gray.shape[:2]
    if max_side and max(original_shape) > max_side:
        h, w = original_shape
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray, original_shape, 1.0 / scale
    return gray, original_shape, 1.0


def _extract_sift_from_gray(gray: np.ndarray, nfeatures: int, offset_xy=(0.0, 0.0), scale_to_original: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if gray is None or gray.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    xy[:, 0] += float(offset_xy[0])
    xy[:, 1] += float(offset_xy[1])
    xy *= float(scale_to_original)
    desc = (rootsift(descriptors) * 255.0).astype(np.float32)
    return xy, desc


def extract_sift_features(path: Path, nfeatures: int = 6000, max_side: int = 0) -> ImageFeatures:
    """Extract SIFT on the whole image and convert descriptors to RootSIFT*255."""
    gray, original_shape, scale_to_original = load_image_gray(path, max_side=max_side)
    xy, desc = _extract_sift_from_gray(gray, nfeatures=nfeatures, offset_xy=(0.0, 0.0), scale_to_original=scale_to_original)
    return ImageFeatures(xy, desc, original_shape, scale_to_original)


def extract_query_region_features(
    path: Path,
    *,
    grid: Tuple[int, int] = (3, 3),
    nfeatures_per_region: int = 1200,
    max_side: int = 0,
) -> List[QueryRegionFeatures]:
    gray, original_shape, scale_to_original = load_image_gray(path, max_side=max_side)
    h, w = gray.shape[:2]
    rows, cols = grid
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)

    regions: List[QueryRegionFeatures] = []
    region_id = 0
    for row in range(rows):
        for col in range(cols):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = h if row == rows - 1 else (row + 1) * cell_h
            x1 = w if col == cols - 1 else (col + 1) * cell_w
            patch = gray[y0:y1, x0:x1]
            xy, desc = _extract_sift_from_gray(
                patch,
                nfeatures=nfeatures_per_region,
                offset_xy=(float(x0), float(y0)),
                scale_to_original=scale_to_original,
            )
            bbox = (
                int(round(x0 * scale_to_original)),
                int(round(y0 * scale_to_original)),
                int(round(x1 * scale_to_original)),
                int(round(y1 * scale_to_original)),
            )
            regions.append(
                QueryRegionFeatures(
                    region_id=region_id,
                    row=row,
                    col=col,
                    bbox_xyxy=bbox,
                    image_shape=(bbox[3] - bbox[1], bbox[2] - bbox[0]),
                    keypoints_xy=xy,
                    descriptors=desc,
                )
            )
            region_id += 1
    return regions


def cv_keypoints_from_xy(xy: np.ndarray) -> List[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), 4.0) for x, y in xy]


def region_bbox_from_id(image_shape: Tuple[int, int], region_id: int, grid: Tuple[int, int] = (3, 3)) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    rows, cols = grid
    row = region_id // cols
    col = region_id % cols
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)
    y0 = row * cell_h
    x0 = col * cell_w
    y1 = h if row == rows - 1 else (row + 1) * cell_h
    x1 = w if col == cols - 1 else (col + 1) * cell_w
    return (x0, y0, x1, y1)


def match_descriptors(
    query_descriptors: np.ndarray,
    ref_descriptors: np.ndarray,
    lowe_ratio: float,
    mutual_check: bool = False,
) -> Tuple[int, List[cv2.DMatch]]:
    if len(query_descriptors) < 2 or len(ref_descriptors) < 2:
        return 0, []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    q_to_r = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
    raw_matches = len(q_to_r)

    ref_to_query = {}
    if mutual_check:
        r_to_q = matcher.knnMatch(ref_descriptors, query_descriptors, k=2)
        for pair in r_to_q:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < lowe_ratio * second.distance:
                ref_to_query[first.queryIdx] = first.trainIdx

    good: List[cv2.DMatch] = []
    for pair in q_to_r:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= lowe_ratio * second.distance:
            continue
        if mutual_check and ref_to_query.get(first.trainIdx) != first.queryIdx:
            continue
        good.append(first)
    return raw_matches, good


def verify_matches_fundamental(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    good_matches: Sequence[cv2.DMatch],
    ransac_threshold: float = 3.0,
    confidence: float = 0.995,
    compute_homography: bool = True,
) -> MatchResult:
    if len(good_matches) < 8:
        return MatchResult(
            raw_matches=0,
            good_matches=list(good_matches),
            verified_matches=[],
            fundamental_matrix=None,
            homography=None,
            geometry_method="too_few_matches_for_fundamental",
            inlier_ratio=0.0,
            invalid_geometry=True,
        )

    q_pts = np.float32([query_xy[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    r_pts = np.float32([ref_xy[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    F, f_mask = cv2.findFundamentalMat(
        q_pts,
        r_pts,
        cv2.FM_RANSAC,
        ransac_threshold,
        confidence,
    )

    homography = None
    if compute_homography and len(good_matches) >= 4:
        homography, _ = cv2.findHomography(q_pts, r_pts, cv2.RANSAC, ransac_threshold * 1.5)

    if F is None or f_mask is None:
        return MatchResult(
            raw_matches=0,
            good_matches=list(good_matches),
            verified_matches=[],
            fundamental_matrix=F,
            homography=homography,
            geometry_method="fundamental_failed",
            inlier_ratio=0.0,
            invalid_geometry=True,
        )

    mask = f_mask.ravel().astype(bool)
    verified = [m for m, keep in zip(good_matches, mask.tolist()) if keep]
    inlier_ratio = float(len(verified)) / max(1.0, float(len(good_matches)))
    return MatchResult(
        raw_matches=0,
        good_matches=list(good_matches),
        verified_matches=verified,
        fundamental_matrix=F,
        homography=homography,
        geometry_method="fundamental",
        inlier_ratio=inlier_ratio,
        invalid_geometry=False,
    )


def match_and_verify(
    query_features: ImageFeatures,
    ref_features: ReferenceFeatures,
    lowe_ratio: float,
    mutual_check: bool,
    ransac_threshold: float,
    ransac_confidence: float,
) -> MatchResult:
    raw_matches, good = match_descriptors(
        query_features.descriptors,
        ref_features.descriptors,
        lowe_ratio=lowe_ratio,
        mutual_check=mutual_check,
    )
    result = verify_matches_fundamental(
        query_features.keypoints_xy,
        ref_features.keypoints_xy,
        good,
        ransac_threshold=ransac_threshold,
        confidence=ransac_confidence,
        compute_homography=True,
    )
    result.raw_matches = raw_matches
    return result


def compute_spatial_coverage(
    query_xy: np.ndarray,
    image_shape: Tuple[int, int],
    grid: Tuple[int, int] = (3, 3),
) -> CoverageMetrics:
    rows, cols = grid
    counts = np.zeros((rows, cols), dtype=np.int32)
    if len(query_xy) == 0:
        return CoverageMetrics(0, 0, 0, 0.0, 0.0, 0.0, counts.ravel().tolist())

    h, w = image_shape[:2]
    xs = np.clip(query_xy[:, 0] / max(1.0, float(w)), 0.0, 0.999999)
    ys = np.clip(query_xy[:, 1] / max(1.0, float(h)), 0.0, 0.999999)
    col_idx = np.floor(xs * cols).astype(np.int32)
    row_idx = np.floor(ys * rows).astype(np.int32)
    for r, c in zip(row_idx, col_idx):
        counts[int(r), int(c)] += 1

    nonzero = counts[counts > 0]
    occupied = int(len(nonzero))
    max_cell = int(nonzero.max()) if occupied else 0
    min_cell = int(nonzero.min()) if occupied else 0
    balance = float(min_cell) / float(max_cell) if max_cell > 0 else 0.0

    spread_x = float(np.std(query_xy[:, 0])) / max(1.0, float(w))
    spread_y = float(np.std(query_xy[:, 1])) / max(1.0, float(h))
    dispersion = float(np.clip(np.sqrt(spread_x * spread_x + spread_y * spread_y), 0.0, 1.0))

    return CoverageMetrics(
        occupied_cells=occupied,
        min_cell_inliers=min_cell,
        max_cell_inliers=max_cell,
        balance=balance,
        dispersion=dispersion,
        coverage_ratio=float(occupied) / float(rows * cols),
        cell_counts=counts.ravel().tolist(),
    )


def global_descriptor_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Simple deterministic descriptor for large reference sets."""
    if bgr is None or bgr.size == 0:
        return np.zeros((1,), dtype=np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    grad_hist = np.histogram(angle, bins=16, range=(0, 360), weights=mag)[0].astype(np.float32)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hsv_hist = hsv_hist.astype(np.float32)

    grid_stats = []
    for row in range(4):
        for col in range(4):
            patch = small[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16]
            grid_stats.extend([float(patch.mean()), float(patch.std())])

    vec = np.concatenate([small.flatten(), grad_hist, hsv_hist, np.array(grid_stats, dtype=np.float32)])
    norm = float(np.linalg.norm(vec)) + 1e-12
    return (vec / norm).astype(np.float32)


def global_descriptor_from_path(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((1,), dtype=np.float32)
    return global_descriptor_from_bgr(bgr)


def draw_verified_matches(
    query_path: Path,
    ref_path: Path,
    query_features: ImageFeatures,
    ref_features: ReferenceFeatures,
    verified_matches: Sequence[cv2.DMatch],
    output_path: Path,
    max_matches: int = 120,
    allowed_query_bboxes: Optional[Iterable[Tuple[int, int, int, int]]] = None,
) -> None:
    query_img = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if query_img is None or ref_img is None:
        raise FileNotFoundError("Could not read query/reference image for match visualization")

    matches = list(verified_matches)
    if allowed_query_bboxes:
        filtered = []
        bboxes = list(allowed_query_bboxes)
        for match in matches:
            qx, qy = query_features.keypoints_xy[match.queryIdx]
            keep = False
            for x0, y0, x1, y1 in bboxes:
                if x0 <= qx < x1 and y0 <= qy < y1:
                    keep = True
                    break
            if keep:
                filtered.append(match)
        if filtered:
            matches = filtered

    matches = _select_display_matches(
        matches,
        query_features.keypoints_xy,
        ref_features.keypoints_xy,
        query_features.image_shape,
        max_matches=max_matches,
    )
    q_kps = cv_keypoints_from_xy(query_features.keypoints_xy)
    r_kps = cv_keypoints_from_xy(ref_features.keypoints_xy)
    vis = cv2.drawMatches(
        query_img,
        q_kps,
        ref_img,
        r_kps,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    max_width = 2200
    if vis.shape[1] > max_width:
        scale = float(max_width) / float(vis.shape[1])
        vis = cv2.resize(
            vis,
            (max_width, max(1, int(round(vis.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def _select_display_matches(
    matches: Sequence[cv2.DMatch],
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    query_shape: Tuple[int, int],
    *,
    max_matches: int,
    query_grid: Tuple[int, int] = (4, 4),
    ref_grid: Tuple[int, int] = (4, 4),
    max_per_query_cell: int = 2,
    max_per_ref_cell: int = 2,
) -> List[cv2.DMatch]:
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: m.distance)
    if len(sorted_matches) > 12:
        cutoff = float(np.percentile([m.distance for m in sorted_matches], 70))
        trimmed = [m for m in sorted_matches if m.distance <= cutoff]
        if len(trimmed) >= 8:
            sorted_matches = trimmed

    qh, qw = query_shape[:2]
    rh = float(np.max(ref_xy[:, 1])) + 1.0 if len(ref_xy) else 1.0
    rw = float(np.max(ref_xy[:, 0])) + 1.0 if len(ref_xy) else 1.0
    q_rows, q_cols = query_grid
    r_rows, r_cols = ref_grid

    def _cell(x: float, y: float, w: float, h: float, rows: int, cols: int) -> tuple[int, int]:
        col = min(cols - 1, max(0, int(np.floor((x / max(1.0, w)) * cols))))
        row = min(rows - 1, max(0, int(np.floor((y / max(1.0, h)) * rows))))
        return row, col

    query_counts = {}
    ref_counts = {}
    chosen: List[cv2.DMatch] = []
    for match in sorted_matches:
        qx, qy = query_xy[match.queryIdx]
        rx, ry = ref_xy[match.trainIdx]
        q_cell = _cell(float(qx), float(qy), float(qw), float(qh), q_rows, q_cols)
        r_cell = _cell(float(rx), float(ry), float(rw), float(rh), r_rows, r_cols)
        if query_counts.get(q_cell, 0) >= max_per_query_cell:
            continue
        if ref_counts.get(r_cell, 0) >= max_per_ref_cell:
            continue
        chosen.append(match)
        query_counts[q_cell] = query_counts.get(q_cell, 0) + 1
        ref_counts[r_cell] = ref_counts.get(r_cell, 0) + 1
        if len(chosen) >= max_matches:
            break

    if len(chosen) >= 6:
        return chosen
    return sorted_matches[:max_matches]
