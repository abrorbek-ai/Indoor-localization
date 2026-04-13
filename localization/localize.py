"""Query uchun PnP yoki fallback — o'xshash reference tanlash."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .colmap_io import (
    Camera,
    ImageRecord,
    Point3D,
    ReferenceFeatures,
    camera_center_from_pose,
    camera_to_intrinsics,
)
from .config import (
    AMBIGUOUS_RELATIVE_MARGIN,
    FALLBACK_MIN_BALANCE,
    FALLBACK_MIN_QUAD_INLIERS,
    FALLBACK_GLOBAL_RANK_WEIGHT,
    FALLBACK_GOOD_MATCH_SCALE,
    FALLBACK_SCORE_AREA_RATIO_WEIGHT,
    FALLBACK_SCORE_BALANCE_WEIGHT,
    FALLBACK_SCORE_DISPERSION_WEIGHT,
    FALLBACK_SCORE_FUSION_WEIGHT,
    FALLBACK_SCORE_GLOBAL_WEIGHT,
    FALLBACK_SCORE_INLIER_WEIGHT,
    FALLBACK_SCORE_MATCH_WEIGHT,
    FALLBACK_MAX_AREA_RATIO,
    FALLBACK_MIN_AREA_RATIO,
    LOWE_RATIO_DEFAULT,
    MEDIUM_CONFIDENCE_RELATIVE_MARGIN,
    PNP_GLOBAL_RANK_WEIGHT,
    PNP_GLOBAL_SHORTLIST,
    PNP_MIN_ACCEPTED_INLIERS,
    PNP_REF_DISTANCE_MEDIAN_MULTIPLIER,
    PNP_REF_DISTANCE_MIN,
    REF_IMAGE_DIR,
    TOP_REFERENCE_RESULTS,
)
from .feature_matching import (
    compute_geometric_match_score,
    extract_query_features_spatial_grid,
    extract_sift_from_gray,
    extract_sift_image_features,
)
from .global_retrieval import ReferenceGlobalCache, iter_query_bgr_quads, order_pnp_candidates


def _geometric_score_query_quads_vs_ref(query_path: Path, ref_path: Path) -> tuple[int, int, list[int]]:
    """
    Query ni 4 kvadratga bo'lib, har birini to'liq reference bilan solishtiradi.
    Qaytaradi: (inlierlar yig'indisi, yaxshi mosliklar yig'indisi, har qism inlierlari).
    """
    bgr_q = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if bgr_q is None:
        return 0, 0, []

    quads_bgr = iter_query_bgr_quads(bgr_q)
    ref_xy, ref_descriptors = extract_sift_image_features(ref_path)
    if len(ref_descriptors) < 2:
        return 0, 0, []

    per_inl: list[int] = []
    sum_inl = 0
    sum_good = 0
    for qb in quads_bgr:
        gray = cv2.cvtColor(qb, cv2.COLOR_BGR2GRAY)
        q_xy, q_desc = extract_sift_from_gray(gray)
        if len(q_desc) < 2:
            per_inl.append(0)
            continue
        score = compute_geometric_match_score(q_xy, q_desc, ref_xy, ref_descriptors)
        ni = int(score["num_inliers"])
        ng = int(score["num_good_matches"])
        per_inl.append(ni)
        sum_inl += ni
        sum_good += ng

    return sum_inl, sum_good, per_inl


def _query_num_quads(query_path: Path) -> int:
    bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return 1
    return max(1, len(iter_query_bgr_quads(bgr)))


def _confidence_from_rank_margin(best_score: float, second_score: float | None) -> dict:
    if second_score is None:
        return {
            "confidence": "high",
            "is_ambiguous": False,
            "score_margin": None,
            "relative_margin": None,
            "ambiguity_reason": "only one candidate",
        }

    margin = float(best_score - second_score)
    relative = margin / (abs(float(best_score)) + 1e-9)
    if relative < AMBIGUOUS_RELATIVE_MARGIN:
        confidence = "low"
        ambiguous = True
        reason = "top-1 and top-2 scores are very close"
    elif relative < MEDIUM_CONFIDENCE_RELATIVE_MARGIN:
        confidence = "medium"
        ambiguous = True
        reason = "top-1 is only moderately ahead of top-2"
    else:
        confidence = "high"
        ambiguous = False
        reason = "top-1 is clearly ahead"

    return {
        "confidence": confidence,
        "is_ambiguous": ambiguous,
        "score_margin": margin,
        "relative_margin": relative,
        "ambiguity_reason": reason,
    }


def _pnp_reference_distance_threshold(images: Dict[int, ImageRecord]) -> float:
    centers = np.array([camera_center_from_pose(img.qvec, img.tvec) for img in images.values()], dtype=np.float64)
    if len(centers) < 2:
        return PNP_REF_DISTANCE_MIN

    nearest_distances = []
    for idx, center in enumerate(centers):
        distances = np.linalg.norm(centers - center, axis=1)
        distances = distances[distances > 1e-9]
        if len(distances) > 0:
            nearest_distances.append(float(np.min(distances)))

    if not nearest_distances:
        return PNP_REF_DISTANCE_MIN

    median_nn = float(np.median(nearest_distances))
    return max(PNP_REF_DISTANCE_MIN, PNP_REF_DISTANCE_MEDIAN_MULTIPLIER * median_nn)


def _compute_balance(quad_inl: list[int]) -> float:
    if not quad_inl:
        return 0.0
    arr = np.asarray(quad_inl, dtype=np.float64) + 1e-6
    return float(np.min(arr) / np.max(arr))


def _compute_dispersion(kpts: np.ndarray | None) -> float:
    if kpts is None or len(kpts) < 5:
        return 0.0
    return float(np.std(kpts[:, 0]) + np.std(kpts[:, 1]))


def _compute_area_ratio(H: np.ndarray | None, img_w: int, img_h: int) -> float:
    if H is None:
        return 0.0
    pts = np.array(
        [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(pts, H)
        area = float(cv2.contourArea(projected))
        return area / float(img_w * img_h + 1e-6)
    except cv2.error:
        return 0.0


def _compute_fallback_final_score(candidate: dict) -> float:
    quad_inl = candidate["quad_inl"]
    balance = _compute_balance(quad_inl)
    candidate["balance"] = balance

    if not quad_inl or min(quad_inl) < FALLBACK_MIN_QUAD_INLIERS:
        candidate["reject_reason"] = f"min_quad_inliers<{FALLBACK_MIN_QUAD_INLIERS}"
        return -1e9

    if balance < FALLBACK_MIN_BALANCE:
        candidate["reject_reason"] = f"balance<{FALLBACK_MIN_BALANCE}"
        return -1e9

    dispersion = _compute_dispersion(candidate.get("kpts"))
    area_ratio = _compute_area_ratio(candidate.get("H"), candidate.get("img_w", 640), candidate.get("img_h", 480))
    candidate["dispersion"] = dispersion
    candidate["area_ratio"] = area_ratio
    candidate["reject_reason"] = ""

    if area_ratio < FALLBACK_MIN_AREA_RATIO or area_ratio > FALLBACK_MAX_AREA_RATIO:
        candidate["reject_reason"] = f"area_ratio_not_in_{FALLBACK_MIN_AREA_RATIO}_{FALLBACK_MAX_AREA_RATIO}"
        return -1e9

    return float(
        candidate["fusion"] * FALLBACK_SCORE_FUSION_WEIGHT
        + candidate["inliers"] * FALLBACK_SCORE_INLIER_WEIGHT
        + candidate["matches"] * FALLBACK_SCORE_MATCH_WEIGHT
        + candidate["global"] * FALLBACK_SCORE_GLOBAL_WEIGHT
        + balance * FALLBACK_SCORE_BALANCE_WEIGHT
        + dispersion * FALLBACK_SCORE_DISPERSION_WEIGHT
        + area_ratio * FALLBACK_SCORE_AREA_RATIO_WEIGHT
    )


def localize_query_image(
    query_path: Path,
    cameras: Dict[int, Camera],
    images: Dict[int, ImageRecord],
    points3D: Dict[int, Point3D],
    reference_features: Dict[str, ReferenceFeatures],
    global_cache: ReferenceGlobalCache | None = None,
) -> dict:
    global_cache = global_cache or ReferenceGlobalCache()
    global_scores = global_cache.dot_with_query(query_path)
    n_quads = float(_query_num_quads(query_path))

    query_xy, query_descriptors = extract_query_features_spatial_grid(query_path)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pnp_candidates: list[dict] = []
    max_ref_pose_distance = _pnp_reference_distance_threshold(images)

    candidate_images = order_pnp_candidates(images, query_path, global_cache)[:PNP_GLOBAL_SHORTLIST]
    ratio = LOWE_RATIO_DEFAULT

    print(f"Candidate reference rasmlar (4 qism global ball yig'indisi tartibi): {len(candidate_images)} ta")
    print(f"Birinchi {PNP_GLOBAL_SHORTLIST} ta — eng yaqin global matchlar bo'yicha tekshiriladi avval.")

    for idx, image in enumerate(candidate_images, start=1):
        if image.name not in reference_features:
            continue

        print(f"PnP candidate {idx}/{len(candidate_images)}: {image.name}")

        ref_features = reference_features[image.name]
        ref_descriptors = ref_features.descriptors
        if len(ref_descriptors) < 2:
            continue

        knn_matches = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
        object_points = []
        image_points = []
        used_point_ids = set()

        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance >= ratio * second.distance:
                continue

            ref_idx = first.trainIdx
            query_idx = first.queryIdx
            if ref_idx >= len(image.point3D_ids):
                continue

            point3D_id = int(image.point3D_ids[ref_idx])
            if point3D_id == -1 or point3D_id not in points3D or point3D_id in used_point_ids:
                continue

            used_point_ids.add(point3D_id)
            object_points.append(points3D[point3D_id].xyz)
            image_points.append(query_xy[query_idx])

        if len(object_points) < 6:
            continue

        object_points = np.asarray(object_points, dtype=np.float64)
        image_points = np.asarray(image_points, dtype=np.float64)
        camera = cameras[image.camera_id]
        K, dist = camera_to_intrinsics(camera)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=2000,
        )

        if not success or inliers is None or len(inliers) < PNP_MIN_ACCEPTED_INLIERS:
            continue

        inlier_idx = inliers.ravel()
        success, rvec, tvec = cv2.solvePnP(
            object_points[inlier_idx],
            image_points[inlier_idx],
            K,
            dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            continue

        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = (-rotation.T @ tvec).reshape(3)
        reference_center = camera_center_from_pose(image.qvec, image.tvec)
        ref_pose_distance = float(np.linalg.norm(camera_center - reference_center))
        if ref_pose_distance > max_ref_pose_distance:
            print(
                f"  PnP reject: {image.name} pose sakradi "
                f"distance={ref_pose_distance:.2f}, limit={max_ref_pose_distance:.2f}"
            )
            continue

        g = float(global_scores.get(image.name, 0.0))
        g_norm = g / n_quads
        num_inl = int(len(inlier_idx))
        if num_inl < PNP_MIN_ACCEPTED_INLIERS:
            continue
        fusion_rank = num_inl + PNP_GLOBAL_RANK_WEIGHT * g_norm

        result = {
            "best_reference_name": image.name,
            "camera_center": camera_center,
            "rvec": rvec,
            "tvec": tvec,
            "num_correspondences": len(object_points),
            "num_inliers": num_inl,
            "camera_id": image.camera_id,
            "global_similarity_sum": g,
            "pnp_fusion_rank": fusion_rank,
            "reference_pose_distance": ref_pose_distance,
        }
        pnp_candidates.append(result)

    if not pnp_candidates:
        return fallback_reference_pose(query_path, images, global_cache)

    pnp_candidates.sort(
        key=lambda r: (r["pnp_fusion_rank"], r["num_inliers"], r["global_similarity_sum"]),
        reverse=True,
    )
    best_result = pnp_candidates[0]
    second_result = pnp_candidates[1] if len(pnp_candidates) > 1 else None
    confidence = _confidence_from_rank_margin(
        float(best_result["pnp_fusion_rank"]),
        float(second_result["pnp_fusion_rank"]) if second_result else None,
    )
    print("\nPnP muvaffaqiyatli kandidatlar (top-3, fusion: inlier + global o'xshashlik):")
    for r in pnp_candidates[:TOP_REFERENCE_RESULTS]:
        print(
            f"  {r['best_reference_name']}: inliers={r['num_inliers']}, "
            f"global_sum={r['global_similarity_sum']:.3f}, fusion_rank={r['pnp_fusion_rank']:.2f}"
        )

    top_reference_scores = [
        (
            r["best_reference_name"],
            (
                f"fusion={r['pnp_fusion_rank']:.1f}, inliers={r['num_inliers']}, "
                f"matches={r['num_correspondences']}, global={r['global_similarity_sum']:.3f}, "
                f"ref_dist={r.get('reference_pose_distance', 0.0):.2f}"
            ),
        )
        for r in pnp_candidates[:TOP_REFERENCE_RESULTS]
    ]
    best_result = {k: v for k, v in best_result.items() if k != "pnp_fusion_rank"}
    best_result["mode"] = "pnp_ransac"
    best_result["top_reference_scores"] = top_reference_scores
    best_result.update(confidence)
    if second_result:
        best_result["second_reference_name"] = second_result["best_reference_name"]
    return best_result


def fallback_reference_pose(
    query_path: Path,
    images: Dict[int, ImageRecord],
    global_cache: ReferenceGlobalCache | None = None,
) -> dict:
    """2D-2D geometriya + global o'xshashlik — eng yaxshi reference nomi."""
    global_cache = global_cache or ReferenceGlobalCache()
    global_scores = global_cache.dot_with_query(query_path)
    n_quads = float(_query_num_quads(query_path))
    query_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(f"Query image topilmadi: {query_path}")
    img_h, img_w = query_bgr.shape[:2]
    query_xy_full, query_desc_full = extract_sift_image_features(query_path)

    scored = []
    rejected = []

    for image in order_pnp_candidates(images, query_path, global_cache)[:PNP_GLOBAL_SHORTLIST]:
        ref_path = REF_IMAGE_DIR / image.name
        if not ref_path.exists():
            continue

        sum_inl, sum_good, per_quad = _geometric_score_query_quads_vs_ref(query_path, ref_path)
        if sum_good < 2 and sum_inl < 1:
            continue

        g = float(global_scores.get(image.name, 0.0))
        g_norm = g / n_quads
        fusion = (
            float(sum_inl)
            + FALLBACK_GLOBAL_RANK_WEIGHT * g_norm
            + FALLBACK_GOOD_MATCH_SCALE * float(sum_good)
        )
        ref_xy, ref_desc = extract_sift_image_features(ref_path)
        full_score = compute_geometric_match_score(query_xy_full, query_desc_full, ref_xy, ref_desc)
        candidate = {
            "name": image.name,
            "inliers": sum_inl,
            "matches": sum_good,
            "global": g,
            "fusion": fusion,
            "quad_inl": per_quad,
            "kpts": full_score.get("inlier_query_xy"),
            "H": full_score.get("homography"),
            "img_w": img_w,
            "img_h": img_h,
        }
        final_score = _compute_fallback_final_score(candidate)
        candidate["final_score"] = final_score
        if final_score <= -1e8:
            rejected.append(candidate)
        else:
            scored.append(candidate)

    if not scored:
        if not rejected:
            raise RuntimeError("Fallback localization uchun ham mos reference image topilmadi.")
        print("Fallback hard filter hamma candidateni reject qildi; eng yaxshi reject qilingan kandidat ishlatiladi.")
        scored = sorted(rejected, key=lambda item: (item["fusion"], item["inliers"], item["matches"]), reverse=True)[:1]

    scored.sort(key=lambda item: (item["final_score"], item["fusion"], item["inliers"], item["matches"]), reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    best_name = best["name"]
    confidence = _confidence_from_rank_margin(best["final_score"], second["final_score"] if second else None)
    best_image = next(img for img in images.values() if img.name == best_name)
    camera_center = camera_center_from_pose(best_image.qvec, best_image.tvec)
    display_candidates = scored[:TOP_REFERENCE_RESULTS]
    if len(display_candidates) < TOP_REFERENCE_RESULTS:
        rejected_for_display = sorted(
            rejected,
            key=lambda item: (item["fusion"], item["inliers"], item["matches"]),
            reverse=True,
        )
        display_candidates.extend(rejected_for_display[: TOP_REFERENCE_RESULTS - len(display_candidates)])

    def _format_fallback_score(item: dict) -> str:
        reject = item.get("reject_reason")
        prefix = "REJECTED, " if reject else ""
        suffix = f", reason={reject}" if reject else ""
        return (
            f"{prefix}score={item['final_score']:.1f}, fusion={item['fusion']:.1f}, "
            f"inliers={item['inliers']}, matches={item['matches']}, global={item['global']:.3f}, "
            f"balance={item.get('balance', 0.0):.3f}, dispersion={item.get('dispersion', 0.0):.1f}, "
            f"area={item.get('area_ratio', 0.0):.3f}, quad_inl={item['quad_inl']}{suffix}"
        )

    result = {
        "best_reference_name": best_name,
        "camera_center": camera_center,
        "rvec": None,
        "tvec": None,
        "num_correspondences": best["matches"],
        "num_inliers": best["inliers"],
        "camera_id": best_image.camera_id,
        "mode": "fallback_reference_pose",
        "quad_inliers": best["quad_inl"],
        "fallback_fusion": best["fusion"],
        "fallback_final_score": best["final_score"],
        "balance": best.get("balance", 0.0),
        "dispersion": best.get("dispersion", 0.0),
        "area_ratio": best.get("area_ratio", 0.0),
        "top_reference_scores": [
            (
                item["name"],
                _format_fallback_score(item),
            )
            for item in display_candidates
        ],
    }
    result.update(confidence)
    if second:
        result["second_reference_name"] = second["name"]
    return result
