"""SIFT, RootSIFT va geometrik tekshiruv (F + H)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .config import (
    HOMOGRAPHY_RANSAC_THRESH,
    LOWE_RATIO_DEFAULT,
    FUNDAMENTAL_RANSAC_THRESH,
    QUERY_PNP_SIFT_PER_QUAD,
    QUERY_QUAD_MIN_SIDE_PX,
    QUERY_RETRIEVAL_GRID_COLS,
    QUERY_RETRIEVAL_GRID_ROWS,
    MATCH_VIZ_F_THRESH,
    MATCH_VIZ_F_THRESH_RELAXED,
    MATCH_VIZ_H_THRESH,
    MATCH_VIZ_H_THRESH_RELAXED,
    MATCH_VIZ_LOWE_RATIO,
    MATCH_VIZ_LOWE_RATIO_RELAXED,
    MATCH_VIZ_RANSAC_CONF,
    MATCH_VIZ_USE_MUTUAL_NN,
    MATCH_VIS_MAX_LINES,
    SIFT_MAX_FEATURES_PAIR,
    SIFT_MAX_FEATURES_QUERY,
    SIFT_MAX_FEATURES_VIZ,
)
from .colmap_io import to_rootsift


def extract_query_features(query_path: Path, nfeatures: int | None = None):
    nfeatures = nfeatures if nfeatures is not None else SIFT_MAX_FEATURES_QUERY
    image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Query image topilmadi: {query_path}")

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) == 0:
        raise ValueError("Query rasm uchun feature topilmadi.")

    query_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    query_descriptors = to_rootsift(descriptors) * 255.0
    return query_xy, query_descriptors.astype(np.float32)


def extract_sift_from_gray(gray: np.ndarray, nfeatures: int | None = None):
    """Kesim (query kvadrati) uchun SIFT; bo'sh bo'lsa nol."""
    nfeatures = nfeatures if nfeatures is not None else SIFT_MAX_FEATURES_PAIR
    if gray is None or gray.size == 0 or gray.shape[0] < 8 or gray.shape[1] < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    descriptors = to_rootsift(descriptors) * 255.0
    return xy, descriptors.astype(np.float32)


def extract_sift_image_features(image_path: Path, nfeatures: int | None = None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Rasm topilmadi: {image_path}")
    return extract_sift_from_gray(image, nfeatures=nfeatures)


def extract_query_features_spatial_grid(query_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Query ni 2x2 qismga bo'lib, har qismdan SIFT; nuqtalar to'liq query koordinatasiga siljitiladi.
    Chap/o'ng va yuqori/past muvozanatli moslik uchun.
    """
    image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Query image topilmadi: {query_path}")

    h, w = image.shape[:2]
    rows, cols = QUERY_RETRIEVAL_GRID_ROWS, QUERY_RETRIEVAL_GRID_COLS
    if h < QUERY_QUAD_MIN_SIDE_PX * rows or w < QUERY_QUAD_MIN_SIDE_PX * cols:
        return extract_query_features(query_path)

    cell_h = h // rows
    cell_w = w // cols
    per = max(400, QUERY_PNP_SIFT_PER_QUAD)
    all_xy: List[List[float]] = []
    all_desc: List[np.ndarray] = []

    for ri in range(rows):
        for ci in range(cols):
            y0 = ri * cell_h
            x0 = ci * cell_w
            y1 = h if ri == rows - 1 else (ri + 1) * cell_h
            x1 = w if ci == cols - 1 else (ci + 1) * cell_w
            patch = image[y0:y1, x0:x1]
            xy, desc = extract_sift_from_gray(patch, nfeatures=per)
            if len(desc) == 0:
                continue
            xy_global = xy + np.array([float(x0), float(y0)], dtype=np.float32)
            all_xy.extend(xy_global.tolist())
            all_desc.append(desc)

    if not all_desc:
        return extract_query_features(query_path)

    query_xy = np.asarray(all_xy, dtype=np.float32)
    query_descriptors = np.vstack(all_desc).astype(np.float32)
    return query_xy, query_descriptors


def compute_good_matches(
    query_descriptors: np.ndarray,
    ref_descriptors: np.ndarray,
    ratio: float | None = None,
):
    ratio = LOWE_RATIO_DEFAULT if ratio is None else ratio
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)
    return good_matches


def compute_good_matches_mutual(
    query_descriptors: np.ndarray,
    ref_descriptors: np.ndarray,
    ratio: float,
) -> List[cv2.DMatch]:
    """Ikki yo'nalishda Lowe ratio: ref->query eng yaxshi juft query->ref bilan mos bo'lishi kerak."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_q2r = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
    knn_r2q = matcher.knnMatch(ref_descriptors, query_descriptors, k=2)

    ref_to_q: dict[int, int] = {}
    for pair in knn_r2q:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= ratio * second.distance:
            continue
        ref_to_q[first.queryIdx] = first.trainIdx

    good_matches: List[cv2.DMatch] = []
    for pair in knn_q2r:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= ratio * second.distance:
            continue
        q_i, r_i = first.queryIdx, first.trainIdx
        if ref_to_q.get(r_i) == q_i:
            good_matches.append(first)
    return good_matches


def compute_geometric_match_score(
    query_xy: np.ndarray,
    query_descriptors: np.ndarray,
    ref_xy: np.ndarray,
    ref_descriptors: np.ndarray,
    *,
    ratio: float | None = None,
    f_thresh: float | None = None,
    h_thresh: float | None = None,
    ransac_confidence: float = 0.995,
    use_mutual_nn: bool = False,
) -> Dict[str, Any]:
    ratio = LOWE_RATIO_DEFAULT if ratio is None else ratio
    f_thresh = FUNDAMENTAL_RANSAC_THRESH if f_thresh is None else f_thresh
    h_thresh = HOMOGRAPHY_RANSAC_THRESH if h_thresh is None else h_thresh

    if use_mutual_nn:
        good_matches = compute_good_matches_mutual(query_descriptors, ref_descriptors, ratio)
    else:
        good_matches = compute_good_matches(query_descriptors, ref_descriptors, ratio=ratio)
    if len(good_matches) < 4:
        return {
            "good_matches": good_matches,
            "inlier_matches": [],
            "num_good_matches": len(good_matches),
            "num_inliers": 0,
            "geom_method": "none",
        }

    src_pts = np.float32([query_xy[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_xy[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    nf, nh = 0, 0
    mask_f = None
    mask_h = None

    if len(good_matches) >= 8:
        _F, mask_f = cv2.findFundamentalMat(
            src_pts,
            dst_pts,
            cv2.FM_RANSAC,
            f_thresh,
            ransac_confidence,
        )
        if mask_f is not None:
            nf = int(mask_f.ravel().sum())

    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, h_thresh, confidence=ransac_confidence)
    if mask_h is not None:
        nh = int(mask_h.ravel().sum())

    use_f = nf >= 8 and (nf >= nh or nh < 8)
    if use_f and mask_f is not None:
        final_mask = mask_f.ravel().astype(bool)
        method = "fundamental"
    elif mask_h is not None and nh >= 4:
        final_mask = mask_h.ravel().astype(bool)
        method = "homography"
    else:
        return {
            "good_matches": good_matches,
            "inlier_matches": [],
            "num_good_matches": len(good_matches),
            "num_inliers": 0,
            "geom_method": "weak",
            "homography": H,
        }

    inlier_matches = [m for m, keep in zip(good_matches, final_mask.tolist()) if keep]
    inlier_query_xy = np.float32([query_xy[m.queryIdx] for m in inlier_matches]).reshape(-1, 2)
    return {
        "good_matches": good_matches,
        "inlier_matches": inlier_matches,
        "num_good_matches": len(good_matches),
        "num_inliers": len(inlier_matches),
        "geom_method": method,
        "homography": H,
        "inlier_query_xy": inlier_query_xy,
    }


def draw_match_visualization(
    query_path: Path,
    reference_name: str,
    output_path: Path,
    ref_image_dir: Path,
    max_lines: int | None = None,
) -> None:
    """Faqat geometrik inlier chiziqlari (good_matches ga tushmaymiz — noto'g'ri juftlar kamayadi)."""
    max_lines = MATCH_VIS_MAX_LINES if max_lines is None else max_lines
    query_image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    ref_path = ref_image_dir / reference_name
    ref_image = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

    if query_image is None or ref_image is None:
        return

    sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES_VIZ)
    q_kp, q_desc = sift.detectAndCompute(query_image, None)
    r_kp, r_desc = sift.detectAndCompute(ref_image, None)
    if q_desc is None or r_desc is None:
        return

    q_desc = (to_rootsift(q_desc) * 255.0).astype(np.float32)
    r_desc = (to_rootsift(r_desc) * 255.0).astype(np.float32)
    q_xy = np.array([kp.pt for kp in q_kp], dtype=np.float32)
    r_xy = np.array([kp.pt for kp in r_kp], dtype=np.float32)

    attempts: list[tuple[float, float, float, float, bool]] = [
        (
            MATCH_VIZ_LOWE_RATIO,
            MATCH_VIZ_F_THRESH,
            MATCH_VIZ_H_THRESH,
            MATCH_VIZ_RANSAC_CONF,
            MATCH_VIZ_USE_MUTUAL_NN,
        ),
        (
            MATCH_VIZ_LOWE_RATIO_RELAXED,
            MATCH_VIZ_F_THRESH_RELAXED,
            MATCH_VIZ_H_THRESH_RELAXED,
            0.995,
            MATCH_VIZ_USE_MUTUAL_NN,
        ),
        (LOWE_RATIO_DEFAULT, FUNDAMENTAL_RANSAC_THRESH, HOMOGRAPHY_RANSAC_THRESH, 0.995, False),
    ]

    selected: List[cv2.DMatch] = []
    for ratio, f_t, h_t, conf, use_mutual in attempts:
        score = compute_geometric_match_score(
            q_xy,
            q_desc,
            r_xy,
            r_desc,
            ratio=ratio,
            f_thresh=f_t,
            h_thresh=h_t,
            ransac_confidence=conf,
            use_mutual_nn=use_mutual,
        )
        selected = score["inlier_matches"]
        if len(selected) >= 4:
            break

    selected = sorted(selected, key=lambda m: m.distance)[:max_lines]

    vis = cv2.drawMatches(
        query_image,
        q_kp,
        ref_image,
        r_kp,
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), vis)
