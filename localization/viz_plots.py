"""Matplotlib logical corridor plot."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .colmap_io import ImageRecord
from .layout_utils import (
    build_logical_layout_items,
    extract_sequence_id,
    extract_view_type,
    logical_layout_position,
)


def plot_pose_result(
    images: Dict[int, ImageRecord],
    query_center: np.ndarray,
    best_reference_name: str,
    localization_mode: str,
    output_path: Path,
) -> None:
    layout_items = build_logical_layout_items()
    if not layout_items:
        return

    sorted_names = [name for name, _ in layout_items]
    first_name = sorted_names[0]
    last_name = sorted_names[-1]
    best_pos = logical_layout_position(best_reference_name)
    query_pos = best_pos.copy()
    if localization_mode == "fallback_reference_pose":
        query_pos = query_pos + np.array([0.18, 0.0], dtype=np.float64)

    plt.figure(figsize=(9, 12))

    front_points = np.array([pos for name, pos in layout_items if extract_view_type(name) == "front"], dtype=np.float64)
    if len(front_points) > 0:
        sorted_front = front_points[np.argsort(front_points[:, 1])]
        plt.plot(
            sorted_front[:, 0],
            sorted_front[:, 1],
            color="black",
            linewidth=2.5,
            alpha=0.75,
            label="Front capture path",
        )

    left_added = False
    front_added = False
    right_added = False
    for name, pos in layout_items:
        seq = extract_sequence_id(name)
        view = extract_view_type(name)
        if name == best_reference_name:
            continue

        if view == "left":
            plt.scatter(pos[0], pos[1], c="seagreen", s=90, marker="<", label="Left images" if not left_added else None)
            left_added = True
        elif view == "right":
            plt.scatter(pos[0], pos[1], c="mediumpurple", s=90, marker=">", label="Right images" if not right_added else None)
            right_added = True
        else:
            plt.scatter(pos[0], pos[1], c="steelblue", s=65, label="Front images" if not front_added else None)
            front_added = True

        plt.text(pos[0] + 0.06, pos[1] + 0.06, str(seq), color="#333333", fontsize=9, weight="bold")

    plt.scatter(best_pos[0], best_pos[1], c="orange", s=150, edgecolors="black", linewidths=0.8, label="Best reference")
    plt.text(best_pos[0] + 0.06, best_pos[1] + 0.06, Path(best_reference_name).stem, color="darkorange", fontsize=10, weight="bold")

    plt.scatter(query_pos[0], query_pos[1], c="red", s=160, label="Query image")
    plt.text(query_pos[0] + 0.06, query_pos[1] + 0.06, "query", color="red", fontsize=11, weight="bold")

    plt.xlabel("Left  <---- corridor ---->  Right")
    plt.ylabel("Capture order")
    plt.title("Logical Corridor Layout View")
    plt.figtext(0.02, 0.08, f"Start: {Path(first_name).stem} -> End: {Path(last_name).stem}", fontsize=10, color="black")
    plt.figtext(0.02, 0.05, "Green <: left images | Blue: front images | Purple >: right images", fontsize=10, color="black")
    if localization_mode == "fallback_reference_pose":
        plt.figtext(
            0.02,
            0.02,
            "Note: fallback mode is active, so the query point is placed on the best matched reference image location.",
            fontsize=10,
            color="darkred",
        )
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.xlim(-1.8, 1.8)
    plt.gca().invert_yaxis()
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(output_path, dpi=180)
    plt.close()
