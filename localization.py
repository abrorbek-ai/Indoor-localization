"""Main localization pipeline: primary reference-level localization plus optional PnP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from features import (
    ImageFeatures,
    LineFeatures,
    compute_line_similarity,
    compute_line_similarity_breakdown,
    compute_structural_similarity,
    extract_line_features,
    extract_query_region_features,
    extract_sift_features,
)
from io_colmap import (
    Camera,
    ImageRecord,
    Point3D,
    ReferenceFeatures,
    ReferenceMeta,
    build_reference_metadata,
    camera_center_from_pose,
    camera_to_intrinsics,
    find_coords_path,
    load_colmap_model,
    load_reference_features,
    natural_sort_key,
    resolve_reference_dir,
)
from retrieval import (
    CandidateResult,
    RankingConfig,
    apply_region_voting,
    evaluate_candidate,
    rank_candidates,
    select_candidate_names,
)


@dataclass
class PipelineConfig:
    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    ref_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "ref_images")
    query_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "query" / "test.jpg")
    coords_path: Optional[Path] = None
    floorplan_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "floorplan.png")
    model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "colmap_workspace" / "sparse" / "0")
    database_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "colmap_workspace" / "database.db")
    output_summary: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "localization_summary.txt")
    output_match_png: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "query_best_match.png")
    output_pose_png: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "pose_plot.png")
    output_pnp_inliers_png: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "pnp_inliers.png")
    output_line_viz_png: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "query_line_viz.png")
    output_html: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "colmap_pose_view.html")
    sift_nfeatures: int = 6000
    max_query_side: int = 0
    use_photometric_normalization: bool = True
    use_hybrid_query_variants: bool = True
    use_neighborhood_scoring: bool = True
    neighborhood_radius: int = 1
    allow_primary_feature_fallback: bool = True
    ranking: RankingConfig = field(default_factory=RankingConfig)
    pnp_enabled: bool = True
    pnp_top_k: int = 3
    pnp_min_correspondences: int = 12
    pnp_min_inliers: int = 8
    pnp_reprojection_error: float = 8.0
    pnp_max_mean_reprojection_error: float = 8.0
    pnp_neighbor_distance_min: float = 2.0
    pnp_neighbor_distance_multiplier: float = 4.0

    def resolved(self) -> "PipelineConfig":
        self.ref_dir = resolve_reference_dir(self.ref_dir)
        if self.coords_path is None:
            self.coords_path = find_coords_path(self.project_dir, self.ref_dir)
        return self


@dataclass
class ReferenceEntry:
    meta: ReferenceMeta
    features: ReferenceFeatures
    image_record: Optional[ImageRecord] = None
    line_features: Optional[LineFeatures] = None


@dataclass
class LocalizationContext:
    config: PipelineConfig
    references: Dict[str, ReferenceEntry]
    cameras: Dict[int, Camera]
    images: Dict[int, ImageRecord]
    points3D: Dict[int, Point3D]
    warnings: List[str] = field(default_factory=list)
    reference_neighbor_distance: float = 2.0


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(arr))
    min_value = float(np.min(arr))
    if max_value - min_value <= 1e-12:
        return [1.0 for _ in values]
    return [float((value - min_value) / (max_value - min_value)) for value in arr]


def _image_records_by_name(images: Dict[int, ImageRecord]) -> Dict[str, ImageRecord]:
    by_name: Dict[str, ImageRecord] = {}
    for image in images.values():
        by_name[image.name] = image
        by_name[Path(image.name).name] = image
    return by_name


def _compute_reference_neighbor_distance(images: Dict[int, ImageRecord], config: PipelineConfig) -> float:
    centers = [camera_center_from_pose(img.qvec, img.tvec) for img in images.values()]
    if len(centers) < 2:
        return config.pnp_neighbor_distance_min
    arr = np.asarray(centers, dtype=np.float64)
    nearest = []
    for idx, center in enumerate(arr):
        distances = np.linalg.norm(arr - center, axis=1)
        distances = distances[distances > 1e-9]
        if len(distances):
            nearest.append(float(np.min(distances)))
    if not nearest:
        return config.pnp_neighbor_distance_min
    return max(config.pnp_neighbor_distance_min, config.pnp_neighbor_distance_multiplier * float(np.median(nearest)))


def load_localization_context(config: Optional[PipelineConfig] = None, verbose: bool = True) -> LocalizationContext:
    config = (config or PipelineConfig()).resolved()
    warnings: List[str] = []

    metadata = build_reference_metadata(config.ref_dir, config.coords_path)
    if not metadata:
        raise FileNotFoundError(f"No reference images found in {config.ref_dir}")
    coord_sources = sorted({meta.coord_source for meta in metadata.values()})

    cameras: Dict[int, Camera] = {}
    images: Dict[int, ImageRecord] = {}
    points3D: Dict[int, Point3D] = {}
    if config.model_dir.exists():
        cameras, images, points3D = load_colmap_model(config.model_dir)
    else:
        warnings.append(f"COLMAP model not found: {config.model_dir}; PnP will be disabled.")

    db_features: Dict[str, ReferenceFeatures] = {}
    if config.database_path.exists():
        db_features = load_reference_features(config.database_path)
    else:
        warnings.append(f"COLMAP database not found: {config.database_path}; reference features will be computed for primary mode.")

    image_by_name = _image_records_by_name(images)
    references: Dict[str, ReferenceEntry] = {}
    for name, meta in sorted(metadata.items(), key=lambda item: natural_sort_key(item[0])):
        features = db_features.get(name)
        if features is None and config.allow_primary_feature_fallback:
            computed = extract_sift_features(
                meta.image_path,
                nfeatures=config.sift_nfeatures,
                photometric_normalization=config.use_photometric_normalization,
            )
            features = ReferenceFeatures(computed.keypoints_xy, computed.descriptors, "computed_sift_primary_only")
            warnings.append(f"{name}: using computed SIFT for primary mode; PnP mapping unavailable for this image.")
        if features is None:
            warnings.append(f"{name}: skipped because no descriptors are available.")
            continue
        references[name] = ReferenceEntry(meta=meta, features=features, image_record=image_by_name.get(name))

    if not references:
        raise RuntimeError("No usable references after loading features.")

    # Pre-compute line features for all reference images (color-independent)
    if verbose:
        print(f"  Pre-computing line features for {len(references)} references...")
    line_fail_count = 0
    for entry in references.values():
        try:
            entry.line_features = extract_line_features(entry.meta.image_path)
        except Exception:
            line_fail_count += 1
    if verbose and line_fail_count:
        print(f"  Warning: line feature extraction failed for {line_fail_count} references.")

    neighbor_distance = _compute_reference_neighbor_distance(images, config)

    if verbose:
        print("Context loaded")
        print(f"  ref_dir: {config.ref_dir}")
        print(f"  references: {len(references)}")
        if config.coords_path:
            print(f"  coords: {config.coords_path}")
        else:
            print(f"  coords: auto ({', '.join(coord_sources)})")
        print(f"  COLMAP cameras/images/points3D: {len(cameras)}/{len(images)}/{len(points3D)}")
        print(f"  database features: {len(db_features)} images")
        if warnings:
            print("  warnings:")
            for warning in warnings[:8]:
                print(f"    - {warning}")
            if len(warnings) > 8:
                print(f"    - ... {len(warnings) - 8} more")

    return LocalizationContext(
        config=config,
        references=references,
        cameras=cameras,
        images=images,
        points3D=points3D,
        warnings=warnings,
        reference_neighbor_distance=neighbor_distance,
    )


def _compute_line_and_structural_scores(
    context: LocalizationContext,
    query_path: Path,
) -> tuple[LineFeatures, Dict[str, float], Dict[str, float], Dict[str, dict]]:
    """Compute first-stage line shortlist scores and second-stage structural scores.

    `line_scores` is the retrieval signal used for top-K selection.
    `structural_scores` is a more discriminative layout signal used during ranking.
    """
    query_lf = extract_line_features(query_path)
    line_scores: Dict[str, float] = {}
    structural_scores: Dict[str, float] = {}
    line_breakdowns: Dict[str, dict] = {}
    for name, entry in context.references.items():
        if entry.line_features is not None:
            breakdown = compute_line_similarity_breakdown(query_lf, entry.line_features)
            line_scores[name] = breakdown["line_similarity"]
            structural_scores[name] = compute_structural_similarity(query_lf, entry.line_features)
            line_breakdowns[name] = breakdown
        else:
            line_scores[name] = 0.0
            structural_scores[name] = 0.0
            line_breakdowns[name] = {}
    return query_lf, line_scores, structural_scores, line_breakdowns


def run_primary_localization(
    context: LocalizationContext,
    query_path: Path,
    query_features: ImageFeatures,
    *,
    photometric_normalization: bool,
    variant_label: str,
    verbose: bool = True,
) -> dict:
    query_line_features, global_scores, structural_scores, line_breakdowns = _compute_line_and_structural_scores(context, query_path)
    shortlist_used = True
    candidate_names, _ = select_candidate_names(
        list(context.references.keys()),
        global_scores,
        context.config.ranking,
    )
    query_name = query_path.name
    candidate_names = [name for name in candidate_names if name != query_name]

    candidates: List[CandidateResult] = []
    for idx, name in enumerate(candidate_names, start=1):
        entry = context.references[name]
        candidate = evaluate_candidate(
            name=name,
            meta=entry.meta,
            ref_features=entry.features,
            query_features=query_features,
            global_similarity=global_scores.get(name) if global_scores else None,
            config=context.config.ranking,
            structural_similarity=structural_scores.get(name, 0.0),
        )
        candidate.line_breakdown = dict(line_breakdowns.get(name, {}))
        candidates.append(candidate)
        if verbose:
            status = "ok" if candidate.accepted else ",".join(candidate.reject_reasons)
            print(
                f"  [{idx:03d}/{len(candidate_names):03d}] {name}: "
                f"line={candidate.global_similarity or 0.0:.3f}, structural={candidate.structural_similarity:.3f}, "
                f"inliers={candidate.verified_inliers}, coverage={candidate.coverage_ratio:.2f}, "
                f"balance={candidate.balance:.2f}, status={status}"
            )

    region_voting = {"invalid_regions": [], "region_debug": [], "valid_regions": []}
    if context.config.ranking.use_region_voting:
        query_regions = extract_query_region_features(
            query_path,
            grid=context.config.ranking.region_grid,
            nfeatures_per_region=context.config.ranking.region_nfeatures_per_region,
            max_side=context.config.max_query_side,
            photometric_normalization=photometric_normalization,
        )
        region_voting = apply_region_voting(
            candidates,
            query_regions,
            {name: context.references[name].features for name in candidate_names},
            context.config.ranking,
        )
        if verbose:
            print(
                f"Region voting ({variant_label}): "
                f"valid_regions={len(region_voting['valid_regions'])}, "
                f"invalid_regions={len(region_voting['invalid_regions'])}"
            )

    ranked = rank_candidates(candidates, use_global=bool(global_scores), config=context.config.ranking)
    line_ranked = sorted(
        candidates,
        key=lambda c: (c.global_similarity or 0.0, c.structural_similarity, c.score, c.verified_inliers),
        reverse=True,
    )
    return {
        "variant_label": variant_label,
        "photometric_normalization": photometric_normalization,
        "use_global": bool(global_scores),
        "used_global_shortlist": shortlist_used,
        "query_line_features": query_line_features,
        "candidates": candidates,
        "best": ranked["best"],
        "accepted": ranked["accepted"],
        "rejected": ranked["rejected"],
        "top3": ranked["top3"],
        "line_topk": line_ranked[: min(10, len(line_ranked))],
        "used_relaxed_fallback": ranked["used_relaxed_fallback"],
        "invalid_regions": region_voting["invalid_regions"],
        "region_debug": region_voting["region_debug"],
        "valid_regions": region_voting["valid_regions"],
    }


def _variant_selection_tuple(primary: dict) -> tuple:
    best: CandidateResult = primary["best"]
    top3 = primary["top3"]
    second_score = top3[1].score if len(top3) > 1 else -1.0
    margin = float(best.score - second_score)
    return (
        int(best.accepted),
        int(not primary["used_relaxed_fallback"]),
        len(primary["valid_regions"]),
        best.region_top_hits,
        float(best.score),
        margin,
        best.verified_inliers,
        best.primary_score,
    )


def _variant_summary(primary: dict) -> dict:
    best: CandidateResult = primary["best"]
    top3 = primary["top3"]
    second_score = top3[1].score if len(top3) > 1 else None
    return {
        "variant_label": primary["variant_label"],
        "photometric_normalization": primary["photometric_normalization"],
        "best_reference_name": best.name,
        "accepted": best.accepted,
        "used_relaxed_fallback": primary["used_relaxed_fallback"],
        "score": float(best.score),
        "primary_score": float(best.primary_score),
        "region_vote_score": float(best.region_vote_score),
        "score_margin": None if second_score is None else float(best.score - second_score),
        "valid_regions": len(primary["valid_regions"]),
        "region_top_hits": int(best.region_top_hits),
        "verified_inliers": int(best.verified_inliers),
    }


def compute_neighborhood_ranking(context: LocalizationContext, primary: dict) -> dict:
    candidates: List[CandidateResult] = list(primary["candidates"])
    if not context.config.use_neighborhood_scoring:
        best = primary["best"]
        station_ids = [best.station_id] if best.station_id is not None else []
        return {
            "enabled": False,
            "groups": [],
            "best_group": None,
            "best_reference": best,
            "station_ids": station_ids,
        }

    station_ids = sorted({c.station_id for c in candidates if c.station_id is not None})
    if not station_ids:
        best = primary["best"]
        return {
            "enabled": True,
            "groups": [],
            "best_group": None,
            "best_reference": best,
            "station_ids": [],
        }

    radius = max(0, int(context.config.neighborhood_radius))
    groups = []
    for center_station in station_ids:
        window = set(range(center_station - radius, center_station + radius + 1))
        members = [c for c in candidates if c.station_id in window]
        if not members:
            continue
        accepted_members = [c for c in members if c.accepted]
        score_sum = float(sum(c.score for c in members))
        accepted_score_sum = float(sum(c.score for c in accepted_members))
        total_inliers = int(sum(c.verified_inliers for c in members))
        total_region_hits = int(sum(c.region_top_hits for c in members))
        supporting_refs = int(sum(1 for c in members if c.region_top_hits > 0))
        accepted_count = int(len(accepted_members))
        groups.append(
            {
                "center_station": center_station,
                "station_ids": sorted({c.station_id for c in members if c.station_id is not None}),
                "member_names": [c.name for c in members],
                "members": members,
                "accepted_members": accepted_members,
                "score_sum": score_sum,
                "accepted_score_sum": accepted_score_sum,
                "total_inliers": total_inliers,
                "total_region_hits": total_region_hits,
                "supporting_refs": supporting_refs,
                "accepted_count": accepted_count,
                "group_score": 0.0,
            }
        )

    if not groups:
        best = primary["best"]
        return {
            "enabled": True,
            "groups": [],
            "best_group": None,
            "best_reference": best,
            "station_ids": [best.station_id] if best.station_id is not None else [],
        }

    norm_accepted_scores = _normalize_scores([g["accepted_score_sum"] for g in groups])
    norm_total_scores = _normalize_scores([g["score_sum"] for g in groups])
    norm_inliers = _normalize_scores([float(g["total_inliers"]) for g in groups])
    norm_region_hits = _normalize_scores([float(g["total_region_hits"]) for g in groups])
    norm_accepted_count = _normalize_scores([float(g["accepted_count"]) for g in groups])

    for idx, group in enumerate(groups):
        group["group_score"] = (
            0.35 * norm_accepted_scores[idx]
            + 0.20 * norm_total_scores[idx]
            + 0.20 * norm_inliers[idx]
            + 0.15 * norm_region_hits[idx]
            + 0.10 * norm_accepted_count[idx]
        )

    groups.sort(
        key=lambda g: (
            g["group_score"],
            g["accepted_count"],
            g["total_region_hits"],
            g["total_inliers"],
            g["accepted_score_sum"],
        ),
        reverse=True,
    )
    best_group = groups[0]
    group_members = best_group["accepted_members"] if best_group["accepted_members"] else best_group["members"]
    group_members = sorted(
        group_members,
        key=lambda c: (c.accepted, c.score, c.region_top_hits, c.verified_inliers, c.primary_score),
        reverse=True,
    )
    best_reference = group_members[0]

    serializable_groups = []
    for group in groups:
        serializable_groups.append(
            {
                "center_station": group["center_station"],
                "station_ids": group["station_ids"],
                "member_names": group["member_names"],
                "score_sum": group["score_sum"],
                "accepted_score_sum": group["accepted_score_sum"],
                "total_inliers": group["total_inliers"],
                "total_region_hits": group["total_region_hits"],
                "supporting_refs": group["supporting_refs"],
                "accepted_count": group["accepted_count"],
                "group_score": group["group_score"],
                "best_reference_name": group_members[0].name if group is best_group else None,
            }
        )

    return {
        "enabled": True,
        "groups": serializable_groups,
        "best_group": serializable_groups[0],
        "best_reference": best_reference,
        "station_ids": best_group["station_ids"],
    }


def _pnp_correspondences(
    candidate: CandidateResult,
    query_features: ImageFeatures,
    entry: ReferenceEntry,
    points3D: Dict[int, Point3D],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    image = entry.image_record
    if image is None or entry.features.source != "colmap_database":
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            0,
        )

    object_points = []
    image_points = []
    query_points = []
    reference_points = []
    used_point_ids = set()
    for match in candidate.verified_match_objects:
        ref_idx = int(match.trainIdx)
        query_idx = int(match.queryIdx)
        if ref_idx >= len(image.point3D_ids) or query_idx >= len(query_features.keypoints_xy):
            continue
        point_id = int(image.point3D_ids[ref_idx])
        if point_id == -1 or point_id in used_point_ids or point_id not in points3D:
            continue
        used_point_ids.add(point_id)
        object_points.append(points3D[point_id].xyz)
        image_points.append(query_features.keypoints_xy[query_idx])
        query_points.append(query_features.keypoints_xy[query_idx])
        reference_points.append(entry.features.keypoints_xy[ref_idx])

    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        np.asarray(query_points, dtype=np.float32),
        np.asarray(reference_points, dtype=np.float32),
        len(candidate.verified_match_objects),
    )


def _reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    return np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)


def try_optional_pnp(
    context: LocalizationContext,
    query_features: ImageFeatures,
    primary: dict,
) -> dict:
    config = context.config
    attempts = []
    if not config.pnp_enabled:
        return {"accepted": False, "attempts": attempts, "reject_reason": "pnp_disabled"}
    if not context.cameras or not context.points3D:
        return {"accepted": False, "attempts": attempts, "reject_reason": "missing_colmap_model"}

    best_primary: CandidateResult = primary["best"]
    best_entry = context.references[best_primary.name]
    anchor_center = (
        camera_center_from_pose(best_entry.image_record.qvec, best_entry.image_record.tvec)
        if best_entry.image_record is not None
        else None
    )

    pnp_candidates: Sequence[CandidateResult] = primary["accepted"][: config.pnp_top_k]
    if not pnp_candidates:
        pnp_candidates = primary["top3"][: config.pnp_top_k]

    accepted_attempts = []
    for candidate in pnp_candidates:
        entry = context.references[candidate.name]
        attempt = {
            "reference": candidate.name,
            "correspondences": 0,
            "inliers": 0,
            "mean_reprojection_error": None,
            "median_reprojection_error": None,
            "reference_distance": None,
            "accepted": False,
            "reject_reason": "",
        }

        if entry.image_record is None:
            attempt["reject_reason"] = "reference_not_registered_in_colmap"
            attempts.append(attempt)
            continue
        if entry.features.source != "colmap_database":
            attempt["reject_reason"] = "reference_features_not_colmap_database"
            attempts.append(attempt)
            continue
        if entry.image_record.camera_id not in context.cameras:
            attempt["reject_reason"] = "camera_intrinsics_missing"
            attempts.append(attempt)
            continue

        object_points, image_points, query_points, reference_points, matched_verified = _pnp_correspondences(
            candidate,
            query_features,
            entry,
            context.points3D,
        )
        attempt["verified_2d2d_matches"] = int(matched_verified)
        attempt["correspondences"] = int(len(object_points))
        if len(object_points) < config.pnp_min_correspondences:
            attempt["reject_reason"] = "too_few_2d3d_correspondences"
            attempts.append(attempt)
            continue

        camera = context.cameras[entry.image_record.camera_id]
        K, dist = camera_to_intrinsics(camera, image_shape=query_features.image_shape)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            dist,
            iterationsCount=2000,
            reprojectionError=config.pnp_reprojection_error,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            attempt["reject_reason"] = "solvepnp_failed"
            attempts.append(attempt)
            continue

        inlier_idx = inliers.ravel().astype(np.int32)
        attempt["inliers"] = int(len(inlier_idx))
        if len(inlier_idx) < config.pnp_min_inliers:
            attempt["reject_reason"] = "too_few_pnp_inliers"
            attempts.append(attempt)
            continue

        refine_success, rvec, tvec = cv2.solvePnP(
            object_points[inlier_idx],
            image_points[inlier_idx],
            K,
            dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not refine_success:
            attempt["reject_reason"] = "pnp_refine_failed"
            attempts.append(attempt)
            continue

        errors = _reprojection_errors(object_points[inlier_idx], image_points[inlier_idx], rvec, tvec, K, dist)
        mean_error = float(np.mean(errors)) if len(errors) else float("inf")
        median_error = float(np.median(errors)) if len(errors) else float("inf")
        attempt["mean_reprojection_error"] = mean_error
        attempt["median_reprojection_error"] = median_error
        if mean_error > config.pnp_max_mean_reprojection_error:
            attempt["reject_reason"] = "high_reprojection_error"
            attempts.append(attempt)
            continue

        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = (-rotation.T @ tvec).reshape(3)
        if anchor_center is not None:
            ref_distance = float(np.linalg.norm(camera_center - anchor_center))
            attempt["reference_distance"] = ref_distance
            if ref_distance > context.reference_neighbor_distance:
                attempt["reject_reason"] = "pose_jumped_from_best_reference_neighborhood"
                attempts.append(attempt)
                continue

        attempt.update(
            {
                "accepted": True,
                "rvec": rvec,
                "tvec": tvec,
                "camera_center": camera_center,
                "camera_id": entry.image_record.camera_id,
                "candidate_score": candidate.score,
                "inlier_query_points": query_points[inlier_idx].copy(),
                "inlier_reference_points": reference_points[inlier_idx].copy(),
                "inlier_object_points": object_points[inlier_idx].copy(),
            }
        )
        attempts.append(attempt)
        accepted_attempts.append(attempt)

    if not accepted_attempts:
        return {"accepted": False, "attempts": attempts, "reject_reason": "no_pnp_candidate_accepted"}

    accepted_attempts.sort(
        key=lambda item: (
            item["inliers"],
            -float(item["mean_reprojection_error"] or 1e9),
            item["candidate_score"],
        ),
        reverse=True,
    )
    best = accepted_attempts[0]
    return {"accepted": True, "best": best, "attempts": attempts}


def localize_query_image(
    context: LocalizationContext,
    query_path: Path,
    verbose: bool = True,
) -> dict:
    query_path = Path(query_path)
    variant_specs = [("raw", False)]
    if context.config.use_photometric_normalization and context.config.use_hybrid_query_variants:
        variant_specs.append(("photo_norm", True))
    elif context.config.use_photometric_normalization:
        variant_specs = [("photo_norm", True)]

    variant_runs = []
    for variant_label, photometric_normalization in variant_specs:
        query_features = extract_sift_features(
            query_path,
            nfeatures=context.config.sift_nfeatures,
            max_side=context.config.max_query_side,
            photometric_normalization=photometric_normalization,
        )
        if len(query_features.descriptors) == 0:
            continue
        if verbose:
            print(f"Query features ({variant_label}): {len(query_features.descriptors)}")
            print(f"Primary reference localization ({variant_label}):")
        primary = run_primary_localization(
            context,
            query_path,
            query_features,
            photometric_normalization=photometric_normalization,
            variant_label=variant_label,
            verbose=verbose,
        )
        variant_runs.append(
            {
                "label": variant_label,
                "photometric_normalization": photometric_normalization,
                "query_features": query_features,
                "primary": primary,
            }
        )

    if not variant_runs:
        raise RuntimeError(f"No SIFT features found in query image: {query_path}")

    variant_runs.sort(key=lambda item: _variant_selection_tuple(item["primary"]), reverse=True)
    chosen = variant_runs[0]
    query_features = chosen["query_features"]
    primary = chosen["primary"]
    neighborhood = compute_neighborhood_ranking(context, primary)
    best: CandidateResult = neighborhood["best_reference"]
    primary_for_pnp = dict(primary)
    primary_for_pnp["best"] = best
    pnp = try_optional_pnp(context, query_features, primary_for_pnp)
    pnp_accepted = bool(pnp.get("accepted"))

    final_mode = "pnp_pose" if pnp_accepted else "reference_fallback"
    final_camera_center = pnp["best"]["camera_center"] if pnp_accepted else None

    query_line_features = chosen["primary"].get("query_line_features") or extract_line_features(query_path)
    result = {
        "mode": final_mode,
        "query_path": query_path,
        "query_features": query_features,
        "query_line_features": query_line_features,
        "query_variant": chosen["label"],
        "best_candidate": best,
        "best_reference_name": best.name,
        "best_station_id": best.station_id,
        "best_logical_xy": best.logical_xy,
        "top3_candidates": [c.metrics_dict() for c in primary["top3"]],
        "line_top_candidates": [c.metrics_dict() for c in primary.get("line_topk", [])],
        "best_neighborhood": neighborhood["best_group"],
        "neighborhood_groups": neighborhood["groups"],
        "accepted_candidates": [c.metrics_dict() for c in primary["accepted"]],
        "rejected_candidates": [c.metrics_dict() for c in primary["rejected"]],
        "primary_candidates": [c.metrics_dict() for c in primary["candidates"]],
        "used_global_retrieval": primary["use_global"],
        "used_global_shortlist": primary["used_global_shortlist"],
        "used_relaxed_fallback": primary["used_relaxed_fallback"],
        "used_photometric_normalization": chosen["photometric_normalization"],
        "variant_comparison": [_variant_summary(item["primary"]) for item in variant_runs],
        "invalid_regions": primary["invalid_regions"],
        "valid_regions": primary["valid_regions"],
        "region_debug": primary["region_debug"],
        "pnp": pnp,
        "pnp_camera_center": final_camera_center,
    }

    if verbose:
        print("\nFinal result:")
        print(f"  mode: {result['mode']}")
        print(f"  query variant: {result['query_variant']}")
        if result["best_neighborhood"]:
            print(
                "  best neighborhood: "
                f"center={result['best_neighborhood']['center_station']} "
                f"stations={result['best_neighborhood']['station_ids']}"
            )
        print(f"  best reference: {result['best_reference_name']}")
        print(f"  station: {result['best_station_id']}")
        print(f"  logical xy: ({best.logical_xy[0]:.3f}, {best.logical_xy[1]:.3f})")
        if pnp_accepted:
            p = pnp["best"]
            print(
                "  PnP accepted: "
                f"inliers={p['inliers']}, mean_reproj={p['mean_reprojection_error']:.2f}, "
                f"ref_distance={p.get('reference_distance')}"
            )
        else:
            print(f"  PnP fallback reason: {pnp.get('reject_reason')}")

    return result
