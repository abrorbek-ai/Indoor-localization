"""Lokalizatsiya pipeline va summary."""

from __future__ import annotations

from pathlib import Path

from .colmap_io import LocalizationContext
from .config import REF_IMAGE_DIR
from .feature_matching import draw_match_visualization
from .global_retrieval import ReferenceGlobalCache
from .localize import localize_query_image
from .viz_plots import plot_pose_result


def save_localization_summary(result: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        center = result["camera_center"]
        f.write(f"mode {result['mode']}\n")
        f.write(f"best_reference {result['best_reference_name']}\n")
        f.write(f"x {center[0]:.6f}\n")
        f.write(f"y {center[1]:.6f}\n")
        f.write(f"z {center[2]:.6f}\n")
        f.write(f"correspondences {result['num_correspondences']}\n")
        f.write(f"inliers {result['num_inliers']}\n")
        f.write(f"confidence {result.get('confidence', 'unknown')}\n")
        f.write(f"ambiguous {result.get('is_ambiguous', False)}\n")
        if result.get("second_reference_name"):
            f.write(f"second_reference {result['second_reference_name']}\n")
        if result.get("score_margin") is not None:
            f.write(f"score_margin {result['score_margin']:.6f}\n")
        if result.get("relative_margin") is not None:
            f.write(f"relative_margin {result['relative_margin']:.6f}\n")
        if result.get("ambiguity_reason"):
            f.write(f"ambiguity_reason {result['ambiguity_reason']}\n")
        if result.get("quad_inliers"):
            f.write(f"quad_inliers {' '.join(str(x) for x in result['quad_inliers'])}\n")
        for idx, item in enumerate(result.get("top_reference_scores", []), start=1):
            f.write(f"top_{idx} {item[0]} {item[1]}\n")


def run_localization_pipeline(
    context: LocalizationContext,
    query_path: Path,
    pose_plot_path: Path,
    match_vis_path: Path,
    summary_path: Path,
    *,
    verbose: bool = True,
    global_cache: ReferenceGlobalCache | None = None,
) -> dict:
    global_cache = global_cache or ReferenceGlobalCache()

    if verbose:
        print("\nQuery rasm lokalizatsiya qilinmoqda...")

    result = localize_query_image(
        query_path=query_path,
        cameras=context.cameras,
        images=context.images,
        points3D=context.points3D,
        reference_features=context.reference_features,
        global_cache=global_cache,
    )

    query_center = result["camera_center"]
    plot_pose_result(
        context.images,
        query_center,
        result["best_reference_name"],
        result["mode"],
        pose_plot_path,
    )
    draw_match_visualization(
        query_path,
        result["best_reference_name"],
        match_vis_path,
        REF_IMAGE_DIR,
    )
    save_localization_summary(result, summary_path)

    if verbose:
        print("\nLocalization natijasi:")
        print(f"Localization mode: {result['mode']}")
        print(f"Query world position: x={query_center[0]:.4f}, y={query_center[1]:.4f}, z={query_center[2]:.4f}")
        print(f"Eng yaqin reference rasm: {result['best_reference_name']}")
        if result.get("mode") == "pnp_ransac":
            print(f"2D-3D correspondences: {result['num_correspondences']}")
            print(f"PnP inliers: {result['num_inliers']}")
        else:
            print(f"2D-2D yaxshi mosliklar: {result['num_correspondences']}")
            print(f"Geometrik inlierlar (F/H): {result['num_inliers']}")
        if result.get("top_reference_scores"):
            print("Top reference poses:")
            for rank, (name, score) in enumerate(result["top_reference_scores"], start=1):
                print(f"{rank}. {name} | {score}")
        print(f"Confidence: {result.get('confidence', 'unknown')}")
        if result.get("is_ambiguous"):
            print(
                "Ogohlantirish: natija noaniq. "
                f"Top-2 yaqin: {result.get('second_reference_name', 'unknown')} "
                f"({result.get('ambiguity_reason', 'reason unavailable')})."
            )
        print(f"Pose plot saqlandi: {pose_plot_path}")
        print(f"Query-best-match visual saqlandi: {match_vis_path}")
        print(f"Summary saqlandi: {summary_path}")

    return result
