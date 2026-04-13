"""Text, PNG, and HTML outputs for the localization MVP."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import draw_verified_matches, region_bbox_from_id
import cv2


def _fmt_xy(xy) -> str:
    return f"({float(xy[0]):.3f}, {float(xy[1]):.3f})"


def save_localization_summary(result: dict, output_path: Path) -> None:
    best = result["best_candidate"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Indoor image-based localization summary\n")
        f.write("========================================\n")
        f.write(f"mode: {result['mode']}\n")
        f.write(f"best_reference: {result['best_reference_name']}\n")
        f.write(f"best_station: {result['best_station_id']}\n")
        f.write(f"logical_coordinate: {_fmt_xy(result['best_logical_xy'])}\n")
        f.write(f"used_global_similarity: {result['used_global_retrieval']}\n")
        f.write(f"used_global_shortlist: {result.get('used_global_shortlist', False)}\n")
        f.write(f"used_relaxed_fallback: {result['used_relaxed_fallback']}\n")
        f.write("\nPrimary best metrics\n")
        f.write("--------------------\n")
        for key, value in best.metrics_dict().items():
            f.write(f"{key}: {value}\n")

        f.write("\nTop3 candidates\n")
        f.write("---------------\n")
        for idx, item in enumerate(result["top3_candidates"], start=1):
            f.write(
                f"{idx}. {item['name']} station={item['station_id']} "
                f"score={item['score']:.4f} inliers={item['verified_inliers']} "
                f"ratio={item['inlier_ratio']:.3f} coverage={item['coverage_ratio']:.3f} "
                f"balance={item['balance']:.3f} region_support={item['supporting_regions']} "
                f"region_hits={item['region_top_hits']} region_vote={item['region_vote_score']:.3f} "
                f"accepted={item['accepted']} "
                f"reasons={item['reject_reasons']}\n"
            )

        f.write("\nRejected candidates\n")
        f.write("-------------------\n")
        for item in result["rejected_candidates"]:
            f.write(
                f"{item['name']}: reasons={item['reject_reasons']} "
                f"good={item['good_matches']} inliers={item['verified_inliers']} "
                f"coverage={item['coverage_ratio']:.3f} balance={item['balance']:.3f}\n"
            )

        f.write("\nInvalid query regions\n")
        f.write("---------------------\n")
        for item in result.get("invalid_regions", []):
            f.write(
                f"region={item['region_id']} bbox={item['bbox_xyxy']} "
                f"features={item.get('feature_count')} reason={item['reason']}\n"
            )

        f.write("\nRegion debug\n")
        f.write("------------\n")
        for item in result.get("region_debug", []):
            if item.get("valid"):
                winner = item.get("winner", {})
                f.write(
                    f"region={item['region_id']} winner={winner.get('name')} "
                    f"inliers={winner.get('verified_inliers')} ratio={winner.get('inlier_ratio'):.3f} "
                    f"top3={[cand['name'] for cand in item.get('top_candidates', [])]}\n"
                )
            else:
                f.write(
                    f"region={item['region_id']} invalid "
                    f"top3={[cand['name'] for cand in item.get('top_candidates', [])]}\n"
                )

        f.write("\nPnP metrics\n")
        f.write("-----------\n")
        pnp = result.get("pnp", {})
        f.write(f"accepted: {pnp.get('accepted', False)}\n")
        if pnp.get("reject_reason"):
            f.write(f"reject_reason: {pnp.get('reject_reason')}\n")
        if pnp.get("accepted"):
            best_pnp = pnp["best"]
            f.write(f"reference: {best_pnp['reference']}\n")
            f.write(f"correspondences: {best_pnp['correspondences']}\n")
            f.write(f"inliers: {best_pnp['inliers']}\n")
            f.write(f"mean_reprojection_error: {best_pnp['mean_reprojection_error']:.4f}\n")
            f.write(f"median_reprojection_error: {best_pnp['median_reprojection_error']:.4f}\n")
            f.write(f"reference_distance: {best_pnp.get('reference_distance')}\n")
            center = best_pnp["camera_center"]
            f.write(f"camera_center_xyz: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n")
        for idx, attempt in enumerate(pnp.get("attempts", []), start=1):
            f.write(
                f"attempt_{idx}: ref={attempt.get('reference')} accepted={attempt.get('accepted')} "
                f"corr={attempt.get('correspondences')} inliers={attempt.get('inliers')} "
                f"mean_reproj={attempt.get('mean_reprojection_error')} "
                f"reason={attempt.get('reject_reason')}\n"
            )


def save_best_match_visualization(context, result: dict, output_path: Path) -> None:
    best = result["best_candidate"]
    entry = context.references[best.name]
    allowed_query_bboxes = None
    if getattr(best, "supporting_regions", None):
        allowed_query_bboxes = [
            region_bbox_from_id(
                result["query_features"].image_shape,
                region_id,
                grid=context.config.ranking.region_grid,
            )
            for region_id in best.supporting_regions
        ]
    draw_verified_matches(
        result["query_path"],
        entry.meta.image_path,
        result["query_features"],
        entry.features,
        best.verified_match_objects,
        output_path,
        allowed_query_bboxes=allowed_query_bboxes,
    )


def _fit_canvas_width(image: np.ndarray, max_width: int = 2200) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = float(max_width) / float(image.shape[1])
    return cv2.resize(
        image,
        (max_width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def save_pnp_inlier_visualization(context, result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pnp = result.get("pnp", {})
    if not pnp.get("accepted"):
        canvas = np.full((420, 900, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "PnP not accepted", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Reason: {pnp.get('reject_reason', 'unknown')}",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )
        y = 220
        for attempt in pnp.get("attempts", [])[:3]:
            text = (
                f"{attempt.get('reference')}: corr={attempt.get('correspondences')} "
                f"inliers={attempt.get('inliers')} reason={attempt.get('reject_reason')}"
            )
            cv2.putText(canvas, text[:110], (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 90, 90), 1, cv2.LINE_AA)
            y += 48
        cv2.imwrite(str(output_path), canvas)
        return

    best_pnp = pnp["best"]
    ref_name = best_pnp["reference"]
    entry = context.references[ref_name]
    query_img = cv2.imread(str(result["query_path"]), cv2.IMREAD_COLOR)
    ref_img = cv2.imread(str(entry.meta.image_path), cv2.IMREAD_COLOR)
    if query_img is None or ref_img is None:
        raise FileNotFoundError("Could not read images for PnP inlier visualization")

    q_pts = np.asarray(best_pnp.get("inlier_query_points", []), dtype=np.float32)
    r_pts = np.asarray(best_pnp.get("inlier_reference_points", []), dtype=np.float32)
    left_h, left_w = query_img.shape[:2]
    right_h, right_w = ref_img.shape[:2]
    canvas_h = max(left_h, right_h)
    canvas = np.full((canvas_h, left_w + right_w, 3), 255, dtype=np.uint8)
    canvas[:left_h, :left_w] = query_img
    canvas[:right_h, left_w : left_w + right_w] = ref_img

    rng = np.random.default_rng(0)
    for idx, (q_pt, r_pt) in enumerate(zip(q_pts, r_pts, strict=False)):
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        qx, qy = int(round(float(q_pt[0]))), int(round(float(q_pt[1])))
        rx, ry = int(round(float(r_pt[0] + left_w))), int(round(float(r_pt[1])))
        cv2.circle(canvas, (qx, qy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (rx, ry), 6, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (qx, qy), (rx, ry), color, 2, cv2.LINE_AA)

    title = (
        f"PnP inliers: {best_pnp.get('inliers')}  "
        f"corr: {best_pnp.get('correspondences')}  "
        f"mean reproj: {best_pnp.get('mean_reprojection_error', 0.0):.2f}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), _fit_canvas_width(canvas))


def plot_pose_result(context, result: dict, output_path: Path, top_n: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refs = list(context.references.values())
    if not refs:
        return

    xs = np.array([entry.meta.logical_xy[0] for entry in refs], dtype=np.float64)
    ys = np.array([entry.meta.logical_xy[1] for entry in refs], dtype=np.float64)
    best_xy = result["best_logical_xy"]

    plt.figure(figsize=(10, 5.5))
    plt.scatter(xs, ys, c="#4c78a8", s=50, label="references", alpha=0.85)

    seen_labels = set()
    for entry in refs:
        label = str(entry.meta.station_id) if entry.meta.station_id is not None else Path(entry.meta.image_name).stem
        if label in seen_labels:
            continue
        seen_labels.add(label)
        plt.text(
            entry.meta.logical_xy[0] + 0.04,
            entry.meta.logical_xy[1] + 0.04,
            label,
            fontsize=8,
            color="#222222",
        )

    top = result["top3_candidates"][:top_n]
    for rank, item in enumerate(top, start=1):
        xy = item["logical_xy"]
        plt.scatter([xy[0]], [xy[1]], s=140 - rank * 20, marker="o", facecolors="none", edgecolors="#f58518", linewidths=2)
        plt.text(xy[0] + 0.06, xy[1] - 0.08, f"top{rank}", color="#f58518", fontsize=9)

    plt.scatter([best_xy[0]], [best_xy[1]], c="#e45756", s=170, marker="x", linewidths=3, label="prediction")
    plt.title(f"Logical corridor localization: {result['mode']}")
    plt.xlabel("logical x")
    plt.ylabel("logical y")
    plt.grid(True, alpha=0.25)
    if len(xs):
        x_pad = max(1.0, 0.08 * (float(xs.max() - xs.min()) + 1e-6))
        y_pad = max(1.0, 0.20 * (float(ys.max() - ys.min()) + 1e-6))
        plt.xlim(float(xs.min()) - x_pad, float(xs.max()) + x_pad)
        plt.ylim(float(ys.min()) - y_pad, float(ys.max()) + y_pad)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_html_report(
    result: dict,
    output_path: Path,
    *,
    project_dir: Path,
    match_png: Path,
    pose_png: Path,
    pnp_inliers_png: Optional[Path] = None,
    query_path: Optional[Path] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_path = query_path or result["query_path"]
    rows = []
    for item in result["top3_candidates"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['station_id']}</td>"
            f"<td>{item['score']:.4f}</td>"
            f"<td>{item['verified_inliers']}</td>"
            f"<td>{item['coverage_ratio']:.3f}</td>"
            f"<td>{item['balance']:.3f}</td>"
            f"<td>{html.escape(str(item['supporting_regions']))}</td>"
            f"<td>{html.escape(', '.join(item['reject_reasons']) or 'accepted')}</td>"
            "</tr>"
        )

    pnp = result.get("pnp", {})
    pnp_text = "accepted" if pnp.get("accepted") else f"fallback: {pnp.get('reject_reason', 'not accepted')}"
    pnp_section = ""
    if pnp_inliers_png is not None:
        pnp_section = f"""
    <section>
      <h2>PnP Inliers</h2>
      <img src="{html.escape(_rel(pnp_inliers_png, project_dir))}" alt="PnP inlier visualization">
    </section>
"""

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Indoor Localization Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d6dde5; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d6dde5; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f2f5f8; }}
    .pill {{ display: inline-block; padding: 4px 8px; border: 1px solid #b8c4d1; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Indoor Localization Result</h1>
  <p><span class="pill">mode: {html.escape(result['mode'])}</span></p>
  <p>Best reference: <strong>{html.escape(result['best_reference_name'])}</strong></p>
  <p>Station: <strong>{html.escape(str(result['best_station_id']))}</strong></p>
  <p>Logical coordinate: <strong>{html.escape(_fmt_xy(result['best_logical_xy']))}</strong></p>
  <p>PnP: {html.escape(pnp_text)}</p>
  <p>Invalid regions: {len(result.get('invalid_regions', []))} / 9</p>
  <div class="grid">
    <section>
      <h2>Query</h2>
      <img src="{html.escape(_rel(query_path, project_dir))}" alt="query image">
    </section>
    <section>
      <h2>Verified Match</h2>
      <img src="{html.escape(_rel(match_png, project_dir))}" alt="verified matches">
    </section>
    <section>
      <h2>Logical Pose</h2>
      <img src="{html.escape(_rel(pose_png, project_dir))}" alt="logical pose plot">
    </section>
{pnp_section}
  </div>
  <h2>Top Candidates</h2>
  <table>
    <thead><tr><th>reference</th><th>station</th><th>score</th><th>inliers</th><th>coverage</th><th>balance</th><th>regions</th><th>status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")


def write_all_outputs(
    context,
    result: dict,
    summary_path: Path,
    match_png: Path,
    pose_png: Path,
    pnp_inliers_png: Optional[Path] = None,
    html_path: Optional[Path] = None,
) -> None:
    save_localization_summary(result, summary_path)
    save_best_match_visualization(context, result, match_png)
    plot_pose_result(context, result, pose_png)
    if pnp_inliers_png is not None:
        save_pnp_inlier_visualization(context, result, pnp_inliers_png)
    if html_path is not None:
        build_html_report(
            result,
            html_path,
            project_dir=context.config.project_dir,
            match_png=match_png,
            pose_png=pose_png,
            pnp_inliers_png=pnp_inliers_png,
            query_path=result["query_path"],
        )
