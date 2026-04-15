"""Image features, matching, geometric verification, and coverage metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def photometric_normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Make local contrast more stable under slightly darker/brighter captures."""
    if gray is None or gray.size == 0:
        return gray
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    mean_intensity = float(np.mean(norm))
    if mean_intensity > 1e-6:
        gamma = np.clip(np.log(140.0 / 255.0) / np.log(mean_intensity / 255.0), 0.75, 1.35)
        table = np.array(
            [((idx / 255.0) ** gamma) * 255.0 for idx in range(256)],
            dtype=np.float32,
        )
        norm = cv2.LUT(norm, np.clip(table, 0, 255).astype(np.uint8))

    return norm


def load_image_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    return bgr


def photometric_normalize_bgr(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_norm = photometric_normalize_gray(l_chan)
    return cv2.cvtColor(cv2.merge([l_norm, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


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


def extract_sift_features(
    path: Path,
    nfeatures: int = 6000,
    max_side: int = 0,
    photometric_normalization: bool = False,
) -> ImageFeatures:
    """Extract SIFT on the whole image and convert descriptors to RootSIFT*255."""
    gray, original_shape, scale_to_original = load_image_gray(path, max_side=max_side)
    if photometric_normalization:
        gray = photometric_normalize_gray(gray)
    xy, desc = _extract_sift_from_gray(gray, nfeatures=nfeatures, offset_xy=(0.0, 0.0), scale_to_original=scale_to_original)
    return ImageFeatures(xy, desc, original_shape, scale_to_original)


def extract_query_region_features(
    path: Path,
    *,
    grid: Tuple[int, int] = (3, 3),
    nfeatures_per_region: int = 1200,
    max_side: int = 0,
    photometric_normalization: bool = False,
) -> List[QueryRegionFeatures]:
    gray, original_shape, scale_to_original = load_image_gray(path, max_side=max_side)
    if photometric_normalization:
        gray = photometric_normalize_gray(gray)
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


def global_descriptor_from_bgr(bgr: np.ndarray, photometric_normalization: bool = False) -> np.ndarray:
    """Simple deterministic descriptor for large reference sets."""
    if bgr is None or bgr.size == 0:
        return np.zeros((1,), dtype=np.float32)

    if photometric_normalization:
        bgr = photometric_normalize_bgr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    grad_hist = np.histogram(angle, bins=16, range=(0, 360), weights=mag)[0].astype(np.float32)

    hsv_hist = np.zeros((0,), dtype=np.float32)
    if not photometric_normalization:
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


def global_descriptor_from_path(path: Path, photometric_normalization: bool = False) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((1,), dtype=np.float32)
    return global_descriptor_from_bgr(bgr, photometric_normalization=photometric_normalization)


# ---------------------------------------------------------------------------
# Line-based structural features (primary localization signal)
# ---------------------------------------------------------------------------

@dataclass
class LineFeatures:
    """Structural line features extracted via Canny + HoughLinesP."""
    lines: np.ndarray           # (N, 4): x1, y1, x2, y2 in original pixel coords
    orientation_hist: np.ndarray  # 16 bins [0°,90°], length-weighted, L1-normalized
    spatial_hist: np.ndarray      # 16 cells (4x4 grid), length-weighted, L1-normalized
    vertical_count: int           # lines with angle > 70°
    horizontal_count: int         # lines with angle < 20°
    diagonal_count: int           # lines with 20° ≤ angle ≤ 70°
    total_length: float
    image_shape: Tuple[int, int]  # (h, w) original
    zone_hists: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 16), dtype=np.float32)
    )  # Per-zone orientation histograms within ROI. Shape (3, 16).
       # Zone 0 = ceiling (top 0–33% of ROI), Zone 1 = boundary (33–67%), Zone 2 = upper wall (67–100%)
    vertical_density: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )  # 8-bin horizontal distribution of near-vertical lines (angle > 65°) = pillar positions
    parallel_profile: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )  # concentration of dominant orientation families, independent of line count
    perspective_density: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )  # where corridor-direction lines reach the ROI base
    roi_bottom_y: int = 0
    zone_bounds_y: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int32)
    )


def extract_line_features(
    path: Path,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 30,
    hough_min_line_len: int = 30,
    hough_max_gap: int = 10,
    max_side: int = 800,
    normalize: bool = True,
    roi_top_fraction: float = 0.62,
) -> LineFeatures:
    """Extract structural line features from an image.

    Returns orientation histogram, spatial distribution, and orientation
    counts — all invariant to color and robust to lighting changes.
    """
    gray, original_shape, scale_to_original = load_image_gray(path, max_side=max_side)
    h_orig, w_orig = original_shape

    if normalize:
        gray = photometric_normalize_gray(gray)

    # Apply ROI mask: blank out bottom portion (floor, bicycles, clutter)
    h_gray, w_gray = gray.shape[:2]
    roi_h = max(1, int(round(h_gray * roi_top_fraction)))
    masked = gray.copy()
    masked[roi_h:, :] = 0  # zero below ROI; Canny finds nothing there
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_len,
        maxLineGap=hough_max_gap,
    )

    _empty = LineFeatures(
        lines=np.zeros((0, 4), dtype=np.float32),
        orientation_hist=np.zeros(16, dtype=np.float32),
        spatial_hist=np.zeros(16, dtype=np.float32),
        vertical_count=0,
        horizontal_count=0,
        diagonal_count=0,
        total_length=0.0,
        image_shape=original_shape,
        zone_hists=np.zeros((3, 16), dtype=np.float32),
        vertical_density=np.zeros(8, dtype=np.float32),
        parallel_profile=np.zeros(4, dtype=np.float32),
        perspective_density=np.zeros(8, dtype=np.float32),
        roi_bottom_y=0,
        zone_bounds_y=np.zeros(4, dtype=np.int32),
    )
    if raw_lines is None:
        return _empty

    lines = raw_lines.reshape(-1, 4).astype(np.float32) * float(scale_to_original)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    # angle=0 → horizontal, angle=90 → vertical
    angles = np.degrees(np.arctan2(np.abs(dy), np.abs(dx) + 1e-9))
    lengths = np.sqrt(dx**2 + dy**2)
    total_length = float(np.sum(lengths))
    if total_length < 1e-6:
        return _empty

    # Orientation histogram: 16 bins over [0°, 90°], weighted by line length
    orientation_hist, _ = np.histogram(angles, bins=16, range=(0.0, 90.0), weights=lengths)
    orientation_hist = orientation_hist.astype(np.float32)
    orientation_hist /= float(np.sum(orientation_hist)) + 1e-12

    vertical_count = int(np.sum(angles > 70.0))
    horizontal_count = int(np.sum(angles < 20.0))
    diagonal_count = int(np.sum((angles >= 20.0) & (angles <= 70.0)))

    # Spatial histogram: 4×4 grid of image, weighted by line length
    midx = (lines[:, 0] + lines[:, 2]) * 0.5
    midy = (lines[:, 1] + lines[:, 3]) * 0.5
    spatial_hist = np.zeros(16, dtype=np.float32)
    for mx, my, ln in zip(midx.tolist(), midy.tolist(), lengths.tolist()):
        c = min(3, max(0, int(float(mx) / float(w_orig) * 4)))
        r = min(3, max(0, int(float(my) / float(h_orig) * 4)))
        spatial_hist[r * 4 + c] += float(ln)
    spatial_hist /= float(np.sum(spatial_hist)) + 1e-12

    # Zone histograms: 3 horizontal bands within the ROI (original pixel space)
    roi_h_orig = int(round(h_orig * roi_top_fraction))
    zone_bounds = [0, roi_h_orig // 3, 2 * roi_h_orig // 3, roi_h_orig]
    zone_hists = np.zeros((3, 16), dtype=np.float32)
    for z in range(3):
        zmask = (midy >= zone_bounds[z]) & (midy < zone_bounds[z + 1])
        if np.any(zmask):
            zh, _ = np.histogram(angles[zmask], bins=16, range=(0.0, 90.0), weights=lengths[zmask])
            zh = zh.astype(np.float32)
            denom = float(np.sum(zh))
            if denom > 1e-12:
                zh /= denom
            zone_hists[z] = zh

    # Vertical density profile: 8-bin horizontal distribution of near-vertical lines (pillars)
    vert_mask = angles > 65.0
    vertical_density = np.zeros(8, dtype=np.float32)
    if np.any(vert_mask):
        for mx, ln in zip(midx[vert_mask].tolist(), lengths[vert_mask].tolist()):
            b = min(7, max(0, int(float(mx) / float(w_orig) * 8)))
            vertical_density[b] += float(ln)
        denom = float(np.sum(vertical_density))
        if denom > 1e-12:
            vertical_density /= denom

    parallel_profile = np.sort(orientation_hist.astype(np.float32))[-4:][::-1]
    parallel_profile /= float(np.sum(parallel_profile)) + 1e-12

    perspective_density = np.zeros(8, dtype=np.float32)
    diag_mask = (angles >= 20.0) & (angles <= 80.0) & (np.abs(dy) > 1e-6)
    if np.any(diag_mask):
        target_y = float(roi_h_orig)
        x1 = lines[diag_mask, 0]
        y1 = lines[diag_mask, 1]
        x2 = lines[diag_mask, 2]
        y2 = lines[diag_mask, 3]
        inter_x = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
        valid = np.isfinite(inter_x)
        for ix, ln in zip(inter_x[valid].tolist(), lengths[diag_mask][valid].tolist()):
            ix = float(np.clip(ix, 0.0, float(w_orig) - 1.0))
            b = min(7, max(0, int(ix / float(w_orig) * 8)))
            perspective_density[b] += float(ln)
        denom = float(np.sum(perspective_density))
        if denom > 1e-12:
            perspective_density /= denom

    return LineFeatures(
        lines=lines,
        orientation_hist=orientation_hist,
        spatial_hist=spatial_hist,
        vertical_count=vertical_count,
        horizontal_count=horizontal_count,
        diagonal_count=diagonal_count,
        total_length=total_length,
        image_shape=original_shape,
        zone_hists=zone_hists,
        vertical_density=vertical_density,
        parallel_profile=parallel_profile,
        perspective_density=perspective_density,
        roi_bottom_y=int(roi_h_orig),
        zone_bounds_y=np.asarray(zone_bounds, dtype=np.int32),
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b)) / (
        (float(np.linalg.norm(a)) + 1e-12) * (float(np.linalg.norm(b)) + 1e-12)
    )


def compute_line_similarity_breakdown(query: LineFeatures, ref: LineFeatures) -> dict:
    """Count-robust first-stage line shortlist score breakdown."""
    orient_sim = _cos(query.orientation_hist, ref.orientation_hist)
    spatial_sim = _cos(query.spatial_hist, ref.spatial_hist)
    zone_sims = [_cos(query.zone_hists[z], ref.zone_hists[z]) for z in range(3)]
    zone_avg = float(np.mean(zone_sims))
    parallel_sim = _cos(query.parallel_profile, ref.parallel_profile)
    perspective_sim = _cos(query.perspective_density, ref.perspective_density)
    q_total = max(1, query.vertical_count + query.horizontal_count + query.diagonal_count)
    r_total = max(1, ref.vertical_count + ref.horizontal_count + ref.diagonal_count)
    q_v = float(query.vertical_count) / q_total
    r_v = float(ref.vertical_count) / r_total
    q_h = float(query.horizontal_count) / q_total
    r_h = float(ref.horizontal_count) / r_total
    layout_sim = 1.0 - 0.5 * (abs(q_v - r_v) + abs(q_h - r_h))
    final = float(np.clip(
        0.24 * orient_sim
        + 0.18 * parallel_sim
        + 0.20 * spatial_sim
        + 0.18 * zone_avg
        + 0.20 * perspective_sim,
        0.0,
        1.0,
    ))
    return {
        "direction_similarity": float(orient_sim),
        "parallel_similarity": float(parallel_sim),
        "spatial_similarity": float(spatial_sim),
        "zone_similarity": float(zone_avg),
        "perspective_similarity": float(perspective_sim),
        "layout_similarity": float(layout_sim),
        "line_similarity": final,
    }


def compute_line_similarity(query: LineFeatures, ref: LineFeatures) -> float:
    return float(compute_line_similarity_breakdown(query, ref)["line_similarity"])


def compute_structural_similarity(query: LineFeatures, ref: LineFeatures) -> float:
    """Second-stage discriminative score to defeat corridor perceptual aliasing.

    Uses zone-specific orientation histograms and vertical pillar density to
    distinguish locations that share similar global line statistics.

    Returns a score in [0, 1].
    """
    zone_a = _cos(query.zone_hists[0], ref.zone_hists[0])   # ceiling — most stable
    zone_b = _cos(query.zone_hists[1], ref.zone_hists[1])   # ceiling-wall boundary + pillar tops
    vdensity = _cos(query.vertical_density, ref.vertical_density)  # pillar horizontal positions
    perspective = _cos(query.perspective_density, ref.perspective_density)
    return float(np.clip(0.32 * zone_a + 0.24 * zone_b + 0.24 * vdensity + 0.20 * perspective, 0.0, 1.0))


def draw_line_features(
    bgr: np.ndarray,
    lf: LineFeatures,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw detected lines onto a BGR image copy."""
    vis = bgr.copy()
    for x1, y1, x2, y2 in lf.lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return vis


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
