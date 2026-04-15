"""Prepare a lightweight reference set from an ARKit capture folder.

Keeps the original dataset untouched and creates a clean reference directory with:
- selected JPG frames
- coords.txt derived from ARKit poses
- a short preparation summary

Example:
    python prepare_arkit_refs.py \
        --dataset-dir "ref_images/dataset-20260410-142733 2" \
        --output-dir "ref_images/dataset-20260410-142733-2_prepared" \
        --stride 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from io_colmap import iter_reference_images, natural_sort_key, read_arkit_pose_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean JPG+coords reference set from an ARKit dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Source ARKit dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Prepared reference directory.")
    parser.add_argument("--stride", type=int, default=3, help="Keep every Nth frame. Default: 3.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    return parser.parse_args()


def ensure_image_link(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        shutil.copy2(src, dst)
        return
    dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(iter_reference_images(dataset_dir), key=lambda p: natural_sort_key(p.name))
    if not images:
        raise FileNotFoundError(f"No usable reference images found in {dataset_dir}")

    pose_coords = read_arkit_pose_coords(dataset_dir)
    if not pose_coords:
        raise RuntimeError(f"No ARKit pose coordinates found in {dataset_dir}")

    stride = max(1, int(args.stride))
    selected = [image for idx, image in enumerate(images) if idx % stride == 0]
    if images[-1] not in selected:
        selected.append(images[-1])
    selected = sorted({image.name: image for image in selected}.values(), key=lambda p: natural_sort_key(p.name))

    coords_rows = []
    for image_path in selected:
        if image_path.name not in pose_coords:
            continue
        ensure_image_link(image_path, output_dir / image_path.name, args.copy)
        x, y = pose_coords[image_path.name]
        coords_rows.append(f"{image_path.name} {x:.6f} {y:.6f}")

    coords_path = output_dir / "coords.txt"
    coords_path.write_text("\n".join(coords_rows) + "\n", encoding="utf-8")

    summary_lines = [
        f"dataset_dir: {dataset_dir}",
        f"output_dir: {output_dir}",
        f"total_images: {len(images)}",
        f"selected_images: {len(coords_rows)}",
        f"stride: {stride}",
        f"link_mode: {'copy' if args.copy else 'symlink'}",
        "ignored_auxiliary: *_depth.bin, *_depth.png, *_depth_meta.json, *_confidence.png",
    ]
    (output_dir / "prepared_dataset_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Prepared {len(coords_rows)} references in {output_dir}")
    print(f"coords: {coords_path}")


if __name__ == "__main__":
    main()
