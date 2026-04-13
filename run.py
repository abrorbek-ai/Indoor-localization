"""CLI entry point for the clean indoor localization MVP.

Default:
    python run.py

Reads query/test.jpg, writes:
    localization_summary.txt
    query_best_match.png
    pose_plot.png
    colmap_pose_view.html
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

from localization import PipelineConfig, load_localization_context, localize_query_image
from visualize import write_all_outputs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pick_query_image(project_dir: Path, requested: str | None = None) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()

    query_dir = project_dir / "query"
    preferred = query_dir / "test.jpg"
    if preferred.exists():
        return preferred

    if not query_dir.exists():
        raise FileNotFoundError(f"Query folder not found: {query_dir}")
    images = [p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No query image found in {query_dir}")
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor reference-level localization with optional COLMAP PnP.")
    parser.add_argument("--query", help="Path to query image. Defaults to query/test.jpg.")
    parser.add_argument("--ref-dir", default="ref_images", help="Reference image directory.")
    parser.add_argument("--coords", default=None, help="coords.txt path. Defaults to coords.txt or coords.text.")
    parser.add_argument("--model-dir", default="colmap_workspace/sparse/0", help="COLMAP sparse model directory.")
    parser.add_argument("--database", default="colmap_workspace/database.db", help="COLMAP database.db path.")
    parser.add_argument("--no-pnp", action="store_true", help="Disable optional PnP and always return reference fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    query_path = pick_query_image(project_dir, args.query)

    config = PipelineConfig(
        project_dir=project_dir,
        ref_dir=(project_dir / args.ref_dir).resolve(),
        query_path=query_path,
        coords_path=(Path(args.coords).resolve() if args.coords else None),
        model_dir=(project_dir / args.model_dir).resolve(),
        database_path=(project_dir / args.database).resolve(),
        pnp_enabled=not args.no_pnp,
    ).resolved()

    print(f"Query image: {query_path}")
    context = load_localization_context(config, verbose=True)
    result = localize_query_image(context, query_path, verbose=True)
    write_all_outputs(
        context,
        result,
        summary_path=config.output_summary,
        match_png=config.output_match_png,
        pose_png=config.output_pose_png,
        pnp_inliers_png=config.output_pnp_inliers_png,
        html_path=config.output_html,
    )

    print("\nOutputs written:")
    print(f"  summary: {config.output_summary}")
    print(f"  match visualization: {config.output_match_png}")
    print(f"  logical pose plot: {config.output_pose_png}")
    print(f"  pnp inlier visualization: {config.output_pnp_inliers_png}")
    print(f"  html report: {config.output_html}")


if __name__ == "__main__":
    main()
