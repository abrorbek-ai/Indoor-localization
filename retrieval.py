"""Candidate selection and normalized primary ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from features import ImageFeatures, QueryRegionFeatures, compute_spatial_coverage, match_and_verify
from io_colmap import ReferenceFeatures, ReferenceMeta, natural_sort_key


@dataclass
class RankingConfig:
    # Line-based shortlist: always applied (threshold=0 means always use shortlist)
    large_dataset_threshold: int = 0
    # How many candidates from line similarity shortlist go through SIFT
    global_top_k: int = 30
    lowe_ratio: float = 0.78
    mutual_check: bool = False
    fundamental_ransac_threshold: float = 3.0
    ransac_confidence: float = 0.995
    min_good_matches: int = 8
    min_verified_inliers: int = 8
    min_coverage_ratio: float = 2.0 / 9.0
    min_balance: float = 0.10
    balance_override_min_coverage: float = 7.0 / 9.0
    balance_override_min_occupied_cells: int = 5
    balance_override_inlier_multiplier: float = 3.0
    min_global_similarity: Optional[float] = None
    use_region_voting: bool = True
    region_grid: tuple[int, int] = (3, 3)
    region_nfeatures_per_region: int = 1200
    region_min_query_features: int = 25
    region_min_good_matches: int = 8
    region_min_verified_inliers: int = 6
    region_min_inlier_ratio: float = 0.20
    region_min_supporting_regions: int = 1
    region_primary_blend_weight: float = 0.55
    region_vote_blend_weight: float = 0.45
    region_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "supporting_regions": 0.35,
            "total_verified_inliers": 0.25,
            "coverage": 0.20,
            "balance": 0.20,
        }
    )
    # Line similarity (global_similarity) is the primary signal.
    # SIFT-based metrics (verified_inliers, inlier_ratio, coverage, balance)
    # serve as secondary verification only.
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "verified_inliers":      0.24,
            "inlier_ratio":          0.16,
            "coverage":              0.16,
            "balance":               0.05,
            "global_similarity":     0.10,
            "structural_similarity": 0.29,
        }
    )


@dataclass
class CandidateResult:
    name: str
    station_id: Optional[int]
    view_label: str
    logical_xy: np.ndarray
    raw_matches: int = 0
    good_matches: int = 0
    verified_inliers: int = 0
    inlier_ratio: float = 0.0
    coverage_ratio: float = 0.0
    occupied_cells: int = 0
    min_cell_inliers: int = 0
    max_cell_inliers: int = 0
    balance: float = 0.0
    dispersion: float = 0.0
    global_similarity: Optional[float] = None
    structural_similarity: float = 0.0
    score: float = 0.0
    primary_score: float = 0.0
    region_vote_score: float = 0.0
    line_breakdown: Dict[str, float] = field(default_factory=dict)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reject_reasons: List[str] = field(default_factory=list)
    geometry_method: str = "none"
    invalid_geometry: bool = False
    supporting_regions: List[int] = field(default_factory=list)
    region_top_hits: int = 0
    region_verified_regions: int = 0
    region_total_inliers: int = 0
    region_coverage_ratio: float = 0.0
    region_balance: float = 0.0
    region_support_inliers: List[int] = field(default_factory=list)
    region_match_debug: List[dict] = field(default_factory=list, repr=False)
    good_match_objects: list = field(default_factory=list, repr=False)
    verified_match_objects: list = field(default_factory=list, repr=False)

    def metrics_dict(self) -> dict:
        return {
            "name": self.name,
            "station_id": self.station_id,
            "view_label": self.view_label,
            "logical_xy": [float(self.logical_xy[0]), float(self.logical_xy[1])],
            "raw_matches": self.raw_matches,
            "good_matches": self.good_matches,
            "verified_inliers": self.verified_inliers,
            "inlier_ratio": self.inlier_ratio,
            "coverage_ratio": self.coverage_ratio,
            "occupied_cells": self.occupied_cells,
            "min_cell_inliers": self.min_cell_inliers,
            "max_cell_inliers": self.max_cell_inliers,
            "balance": self.balance,
            "dispersion": self.dispersion,
            "global_similarity": self.global_similarity,
            "structural_similarity": self.structural_similarity,
            "score": self.score,
            "primary_score": self.primary_score,
            "region_vote_score": self.region_vote_score,
            "line_breakdown": dict(self.line_breakdown),
            "score_breakdown": dict(self.score_breakdown),
            "accepted": self.accepted,
            "reject_reasons": list(self.reject_reasons),
            "geometry_method": self.geometry_method,
            "supporting_regions": list(self.supporting_regions),
            "region_top_hits": self.region_top_hits,
            "region_verified_regions": self.region_verified_regions,
            "region_total_inliers": self.region_total_inliers,
            "region_coverage_ratio": self.region_coverage_ratio,
            "region_balance": self.region_balance,
            "region_support_inliers": list(self.region_support_inliers),
        }


def select_candidate_names(
    reference_names: Sequence[str],
    global_scores: Optional[Dict[str, float]],
    config: RankingConfig,
) -> tuple[List[str], bool]:
    """Return candidate names sorted by line similarity, limited to global_top_k.

    With line-based retrieval, the shortlist is always applied when scores are
    available (large_dataset_threshold=0), so SIFT is only run on the top-K
    most structurally similar references.
    """
    names = sorted(reference_names, key=natural_sort_key)
    use_global = bool(global_scores) and len(names) > config.large_dataset_threshold
    if not use_global:
        return names, False

    ranked = sorted(names, key=lambda name: global_scores.get(name, -1.0), reverse=True)
    return ranked[: min(config.global_top_k, len(ranked))], True


def evaluate_candidate(
    name: str,
    meta: ReferenceMeta,
    ref_features: ReferenceFeatures,
    query_features: ImageFeatures,
    global_similarity: Optional[float],
    config: RankingConfig,
    structural_similarity: float = 0.0,
) -> CandidateResult:
    match = match_and_verify(
        query_features,
        ref_features,
        lowe_ratio=config.lowe_ratio,
        mutual_check=config.mutual_check,
        ransac_threshold=config.fundamental_ransac_threshold,
        ransac_confidence=config.ransac_confidence,
    )
    inlier_xy = (
        np.asarray([query_features.keypoints_xy[m.queryIdx] for m in match.verified_matches], dtype=np.float32)
        if match.verified_matches
        else np.zeros((0, 2), dtype=np.float32)
    )
    coverage = compute_spatial_coverage(inlier_xy, query_features.image_shape, grid=(3, 3))

    candidate = CandidateResult(
        name=name,
        station_id=meta.station_id,
        view_label=meta.view_label,
        logical_xy=meta.logical_xy,
        raw_matches=match.raw_matches,
        good_matches=len(match.good_matches),
        verified_inliers=len(match.verified_matches),
        inlier_ratio=match.inlier_ratio,
        coverage_ratio=coverage.coverage_ratio,
        occupied_cells=coverage.occupied_cells,
        min_cell_inliers=coverage.min_cell_inliers,
        max_cell_inliers=coverage.max_cell_inliers,
        balance=coverage.balance,
        dispersion=coverage.dispersion,
        global_similarity=global_similarity,
        structural_similarity=float(structural_similarity),
        geometry_method=match.geometry_method,
        invalid_geometry=match.invalid_geometry,
        good_match_objects=list(match.good_matches),
        verified_match_objects=list(match.verified_matches),
    )

    if candidate.good_matches < config.min_good_matches:
        candidate.reject_reasons.append("too_few_matches")
    if candidate.invalid_geometry:
        candidate.reject_reasons.append("invalid_geometry")
    if candidate.verified_inliers < config.min_verified_inliers:
        candidate.reject_reasons.append("too_few_geometric_inliers")
    if candidate.coverage_ratio < config.min_coverage_ratio:
        candidate.reject_reasons.append("poor_coverage")
    allow_balance_override = (
        candidate.coverage_ratio >= config.balance_override_min_coverage
        and candidate.occupied_cells >= config.balance_override_min_occupied_cells
        and candidate.verified_inliers >= int(round(config.min_verified_inliers * config.balance_override_inlier_multiplier))
    )
    if candidate.balance < config.min_balance and not allow_balance_override:
        candidate.reject_reasons.append("poor_balance")
    if (
        config.min_global_similarity is not None
        and global_similarity is not None
        and global_similarity < config.min_global_similarity
    ):
        candidate.reject_reasons.append("poor_global_similarity")

    candidate.accepted = len(candidate.reject_reasons) == 0
    return candidate


def _normalize(values: Iterable[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return []
    max_value = float(np.max(arr))
    min_value = float(np.min(arr))
    if max_value <= 1e-12:
        return [0.0 for _ in arr]
    if min_value < 0.0:
        denom = max_value - min_value
        if denom <= 1e-12:
            return [1.0 for _ in arr]
        return [float((v - min_value) / denom) for v in arr]
    return [float(v / max_value) for v in arr]


def score_candidates(candidates: List[CandidateResult], use_global: bool, config: RankingConfig) -> None:
    if not candidates:
        return

    norm_verified = _normalize(c.verified_inliers for c in candidates)
    # Line similarity (global_similarity) is always used when available — it is the primary signal.
    has_line_scores = any(c.global_similarity is not None for c in candidates)
    norm_global = _normalize((c.global_similarity or 0.0) for c in candidates) if has_line_scores else [0.0] * len(candidates)
    # Structural similarity (zone + pillar) — second-stage discrimination signal.
    has_structural = any(c.structural_similarity != 0.0 for c in candidates)
    norm_structural = (
        _normalize(c.structural_similarity for c in candidates)
        if has_structural else [0.0] * len(candidates)
    )

    weights = dict(config.score_weights)
    if not has_line_scores:
        weights.pop("global_similarity", None)
    if not has_structural:
        weights.pop("structural_similarity", None)
    total_weight = sum(weights.values()) or 1.0
    weights = {key: value / total_weight for key, value in weights.items()}

    for idx, candidate in enumerate(candidates):
        candidate.primary_score = (
            weights.get("verified_inliers", 0.0) * norm_verified[idx]
            + weights.get("inlier_ratio", 0.0) * float(np.clip(candidate.inlier_ratio, 0.0, 1.0))
            + weights.get("coverage", 0.0) * float(np.clip(candidate.coverage_ratio, 0.0, 1.0))
            + weights.get("balance", 0.0) * float(np.clip(candidate.balance, 0.0, 1.0))
            + weights.get("global_similarity", 0.0) * norm_global[idx]
            + weights.get("structural_similarity", 0.0) * norm_structural[idx]
        )
        candidate.score_breakdown = {
            "verified_inliers_norm": float(norm_verified[idx]),
            "inlier_ratio": float(np.clip(candidate.inlier_ratio, 0.0, 1.0)),
            "coverage": float(np.clip(candidate.coverage_ratio, 0.0, 1.0)),
            "balance": float(np.clip(candidate.balance, 0.0, 1.0)),
            "line_similarity_norm": float(norm_global[idx]),
            "structural_similarity_norm": float(norm_structural[idx]),
            "weighted_primary_score": float(candidate.primary_score),
        }
        candidate.score = candidate.primary_score


def _region_passes(candidate_metrics: dict, config: RankingConfig) -> bool:
    return (
        not candidate_metrics["invalid_geometry"]
        and candidate_metrics["good_matches"] >= config.region_min_good_matches
        and candidate_metrics["verified_inliers"] >= config.region_min_verified_inliers
        and candidate_metrics["inlier_ratio"] >= config.region_min_inlier_ratio
    )


def _region_balance(support_inliers: Sequence[int]) -> float:
    arr = np.asarray([value for value in support_inliers if value > 0], dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(arr.min() / arr.max())


def apply_region_voting(
    candidates: List[CandidateResult],
    query_regions: List[QueryRegionFeatures],
    reference_features_by_name: Dict[str, ReferenceFeatures],
    config: RankingConfig,
) -> dict:
    if not config.use_region_voting:
        return {"invalid_regions": [], "region_debug": [], "valid_regions": []}

    for candidate in candidates:
        candidate.supporting_regions = []
        candidate.region_top_hits = 0
        candidate.region_verified_regions = 0
        candidate.region_total_inliers = 0
        candidate.region_coverage_ratio = 0.0
        candidate.region_balance = 0.0
        candidate.region_vote_score = 0.0
        candidate.region_support_inliers = [0 for _ in range(len(query_regions))]
        candidate.region_match_debug = []

    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    invalid_regions: List[dict] = []
    region_debug: List[dict] = []
    valid_regions: List[int] = []

    for region in query_regions:
        if region.feature_count < config.region_min_query_features:
            invalid_regions.append(
                {
                    "region_id": region.region_id,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "reason": "too_few_region_features",
                    "feature_count": region.feature_count,
                }
            )
            continue

        region_query_features = ImageFeatures(
            keypoints_xy=region.keypoints_xy,
            descriptors=region.descriptors,
            image_shape=region.image_shape,
            scale_to_original=1.0,
        )
        metrics_per_candidate = []
        passed = []
        for candidate in candidates:
            ref_features = reference_features_by_name[candidate.name]
            match = match_and_verify(
                region_query_features,
                ref_features,
                lowe_ratio=config.lowe_ratio,
                mutual_check=config.mutual_check,
                ransac_threshold=config.fundamental_ransac_threshold,
                ransac_confidence=config.ransac_confidence,
            )
            item = {
                "name": candidate.name,
                "good_matches": int(len(match.good_matches)),
                "verified_inliers": int(len(match.verified_matches)),
                "inlier_ratio": float(match.inlier_ratio),
                "invalid_geometry": bool(match.invalid_geometry),
                "passed": False,
            }
            item["passed"] = _region_passes(item, config)
            metrics_per_candidate.append(item)

            if item["passed"]:
                candidate.region_verified_regions += 1
                candidate.region_total_inliers += item["verified_inliers"]
                passed.append(item)

        metrics_per_candidate.sort(
            key=lambda item: (item["passed"], item["verified_inliers"], item["inlier_ratio"], item["good_matches"]),
            reverse=True,
        )
        top_candidates = metrics_per_candidate[:3]

        if not passed:
            invalid_regions.append(
                {
                    "region_id": region.region_id,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "reason": "no_reliable_region_match",
                    "feature_count": region.feature_count,
                    "top_candidates": top_candidates,
                }
            )
            region_debug.append(
                {
                    "region_id": region.region_id,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "feature_count": region.feature_count,
                    "valid": False,
                    "top_candidates": top_candidates,
                }
            )
            continue

        passed.sort(
            key=lambda item: (item["verified_inliers"], item["inlier_ratio"], item["good_matches"]),
            reverse=True,
        )
        valid_regions.append(region.region_id)
        winner = passed[0]
        winner_candidate = candidate_by_name[winner["name"]]
        winner_candidate.region_top_hits += 1
        winner_candidate.supporting_regions.append(region.region_id)
        winner_candidate.region_support_inliers[region.region_id] = winner["verified_inliers"]

        region_debug.append(
            {
                "region_id": region.region_id,
                "bbox_xyxy": list(region.bbox_xyxy),
                "feature_count": region.feature_count,
                "valid": True,
                "winner": winner,
                "top_candidates": top_candidates,
            }
        )

    total_regions = max(1, len(query_regions))
    for candidate in candidates:
        candidate.region_coverage_ratio = float(candidate.region_top_hits) / float(total_regions)
        candidate.region_balance = _region_balance(candidate.region_support_inliers)
        candidate.region_match_debug = [
            item
            for item in region_debug
            if item.get("valid") and item.get("winner", {}).get("name") == candidate.name
        ]

    norm_top_hits = _normalize(candidate.region_top_hits for candidate in candidates)
    norm_region_inliers = _normalize(candidate.region_total_inliers for candidate in candidates)
    weights = dict(config.region_score_weights)
    total_weight = sum(weights.values()) or 1.0
    weights = {key: value / total_weight for key, value in weights.items()}

    for idx, candidate in enumerate(candidates):
        candidate.region_vote_score = (
            weights.get("supporting_regions", 0.0) * norm_top_hits[idx]
            + weights.get("total_verified_inliers", 0.0) * norm_region_inliers[idx]
            + weights.get("coverage", 0.0) * float(np.clip(candidate.region_coverage_ratio, 0.0, 1.0))
            + weights.get("balance", 0.0) * float(np.clip(candidate.region_balance, 0.0, 1.0))
        )

    return {
        "invalid_regions": invalid_regions,
        "region_debug": region_debug,
        "valid_regions": valid_regions,
    }


def weak_fallback_score(candidate: CandidateResult) -> tuple:
    """Used only when hard filters reject every candidate."""
    return (
        candidate.region_top_hits,
        candidate.region_vote_score,
        candidate.verified_inliers,
        candidate.coverage_ratio,
        candidate.balance,
        candidate.inlier_ratio,
        candidate.good_matches,
        candidate.global_similarity or 0.0,
    )


def rank_candidates(candidates: List[CandidateResult], use_global: bool, config: RankingConfig) -> dict:
    score_candidates(candidates, use_global, config)
    if config.use_region_voting:
        for candidate in candidates:
            candidate.score = (
                config.region_primary_blend_weight * candidate.primary_score
                + config.region_vote_blend_weight * candidate.region_vote_score
            )
            if candidate.accepted and candidate.region_top_hits < config.region_min_supporting_regions:
                candidate.accepted = False
                if "weak_region_support" not in candidate.reject_reasons:
                    candidate.reject_reasons.append("weak_region_support")
    accepted = sorted(
        [c for c in candidates if c.accepted],
        key=lambda c: (c.score, c.verified_inliers, c.inlier_ratio, c.good_matches),
        reverse=True,
    )
    rejected = sorted(
        [c for c in candidates if not c.accepted],
        key=weak_fallback_score,
        reverse=True,
    )

    used_relaxed_fallback = False
    if accepted:
        best = accepted[0]
    elif rejected:
        best = rejected[0]
        used_relaxed_fallback = True
    else:
        raise RuntimeError("No reference candidates could be evaluated.")

    all_ranked = sorted(
        candidates,
        key=lambda c: (c.accepted, c.score, c.primary_score, c.region_vote_score, c.verified_inliers, c.good_matches),
        reverse=True,
    )

    return {
        "best": best,
        "accepted": accepted,
        "rejected": rejected,
        "top3": all_ranked[:3],
        "used_relaxed_fallback": used_relaxed_fallback,
    }
