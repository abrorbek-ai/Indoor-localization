"""Text, PNG, and HTML outputs for the localization MVP."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import draw_line_features, draw_verified_matches, extract_line_features, region_bbox_from_id
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
        f.write(f"query_variant: {result.get('query_variant', 'unknown')}\n")
        f.write(f"best_reference: {result['best_reference_name']}\n")
        f.write(f"best_station: {result['best_station_id']}\n")
        f.write(f"logical_coordinate: {_fmt_xy(result['best_logical_xy'])}\n")
        if result.get("best_neighborhood"):
            neighborhood = result["best_neighborhood"]
            f.write(f"best_neighborhood_center: {neighborhood['center_station']}\n")
            f.write(f"best_neighborhood_stations: {neighborhood['station_ids']}\n")
        f.write(f"used_global_similarity: {result['used_global_retrieval']}\n")
        f.write(f"used_global_shortlist: {result.get('used_global_shortlist', False)}\n")
        f.write(f"used_photometric_normalization: {result.get('used_photometric_normalization', False)}\n")
        f.write(f"used_relaxed_fallback: {result['used_relaxed_fallback']}\n")
        f.write("\nPrimary best metrics\n")
        f.write("--------------------\n")
        for key, value in best.metrics_dict().items():
            f.write(f"{key}: {value}\n")
        qlf = result.get("query_line_features")
        if qlf is not None:
            floor_ignored = 1.0 - (float(getattr(qlf, "roi_bottom_y", 0)) / max(1.0, float(qlf.image_shape[0])))
            f.write("\nROI debug\n")
            f.write("---------\n")
            f.write(f"selected_roi_bottom_y: {getattr(qlf, 'roi_bottom_y', 0)}\n")
            f.write(f"ignored_floor_fraction: {floor_ignored:.3f}\n")
            f.write(f"zone_bounds_y: {getattr(qlf, 'zone_bounds_y', []).tolist() if hasattr(getattr(qlf, 'zone_bounds_y', []), 'tolist') else getattr(qlf, 'zone_bounds_y', [])}\n")
            f.write(f"filtered_line_count: {len(getattr(qlf, 'lines', []))}\n")

        f.write("\nVariant comparison\n")
        f.write("------------------\n")
        for item in result.get("variant_comparison", []):
            f.write(
                f"{item['variant_label']}: best={item['best_reference_name']} "
                f"accepted={item['accepted']} score={item['score']:.4f} "
                f"margin={item['score_margin']} valid_regions={item['valid_regions']} "
                f"region_hits={item['region_top_hits']} inliers={item['verified_inliers']}\n"
            )

        f.write("\nNeighborhood ranking\n")
        f.write("--------------------\n")
        for item in result.get("neighborhood_groups", [])[:5]:
            f.write(
                f"center={item['center_station']} stations={item['station_ids']} "
                f"group_score={item['group_score']:.4f} accepted_count={item['accepted_count']} "
                f"region_hits={item['total_region_hits']} total_inliers={item['total_inliers']} "
                f"members={item['member_names']}\n"
            )

        f.write("\nLine Shortlist (First Stage)\n")
        f.write("----------------------------\n")
        for idx, item in enumerate(result.get("line_top_candidates", [])[:10], start=1):
            f.write(
                f"{idx}. {item['name']} line={float(item.get('global_similarity') or 0.0):.4f} "
                f"structural={float(item.get('structural_similarity') or 0.0):.4f} "
                f"inliers={item['verified_inliers']} region_hits={item['region_top_hits']} "
                f"accepted={item['accepted']} reasons={item['reject_reasons']}\n"
            )
            if item.get("line_breakdown"):
                f.write(f"   breakdown: {item['line_breakdown']}\n")

        f.write("\nTop3 candidates\n")
        f.write("---------------\n")
        for idx, item in enumerate(result["top3_candidates"], start=1):
            f.write(
                f"{idx}. {item['name']} station={item['station_id']} "
                f"line={float(item.get('global_similarity') or 0.0):.4f} "
                f"structural={float(item.get('structural_similarity') or 0.0):.4f} "
                f"score={item['score']:.4f} inliers={item['verified_inliers']} "
                f"ratio={item['inlier_ratio']:.3f} coverage={item['coverage_ratio']:.3f} "
                f"balance={item['balance']:.3f} region_support={item['supporting_regions']} "
                f"region_hits={item['region_top_hits']} region_vote={item['region_vote_score']:.3f} "
                f"accepted={item['accepted']} "
                f"reasons={item['reject_reasons']}\n"
            )
            if item.get("line_breakdown"):
                f.write(f"   line_breakdown: {item['line_breakdown']}\n")
            if item.get("score_breakdown"):
                f.write(f"   final_score_breakdown: {item['score_breakdown']}\n")

        f.write("\nRejected candidates\n")
        f.write("-------------------\n")
        for item in result["rejected_candidates"]:
            f.write(
                f"{item['name']}: reasons={item['reject_reasons']} "
                f"line={float(item.get('global_similarity') or 0.0):.4f} "
                f"structural={float(item.get('structural_similarity') or 0.0):.4f} "
                f"good={item['good_matches']} inliers={item['verified_inliers']} "
                f"coverage={item['coverage_ratio']:.3f} balance={item['balance']:.3f}\n"
            )

        f.write("\nWhy Similar Wrong Frames Were Rejected\n")
        f.write("--------------------------------------\n")
        best_line = max([float(item.get("global_similarity") or 0.0) for item in result.get("line_top_candidates", [])] or [0.0])
        for item in result.get("line_top_candidates", []):
            if item["name"] == result["best_reference_name"]:
                continue
            if float(item.get("global_similarity") or 0.0) < best_line - 0.08:
                continue
            reasons = item.get("reject_reasons") or []
            explanation = []
            if float(item.get("structural_similarity") or 0.0) < float(best.metrics_dict().get("structural_similarity") or 0.0):
                explanation.append("weaker structural layout")
            if item.get("region_top_hits", 0) < best.region_top_hits:
                explanation.append("weaker region consistency")
            if item.get("verified_inliers", 0) < best.verified_inliers:
                explanation.append("weaker point verification")
            if reasons:
                explanation.append(f"reject={reasons}")
            f.write(
                f"{item['name']}: line={float(item.get('global_similarity') or 0.0):.4f}, "
                f"structural={float(item.get('structural_similarity') or 0.0):.4f}, "
                f"inliers={item.get('verified_inliers', 0)}, "
                f"why={'; '.join(explanation) or 'lower final score'}\n"
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
    line_viz_png: Optional[Path] = None,
    query_path: Optional[Path] = None,
    context=None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_path = query_path or result["query_path"]
    best_name = result["best_reference_name"]
    top3 = result["top3_candidates"]
    pnp = result.get("pnp", {})

    # ── Relative paths ──────────────────────────────────────────────────────
    query_rel = html.escape(_rel(query_path, project_dir))
    match_rel = html.escape(_rel(match_png, project_dir))
    line_viz_rel = html.escape(_rel(line_viz_png, project_dir)) if line_viz_png and line_viz_png.exists() else ""

    # ── Interactive map data (JSON) ──────────────────────────────────────────
    top3_name_to_rank = {item["name"]: i + 1 for i, item in enumerate(top3)}
    cand_lookup = {c["name"]: c for c in result.get("primary_candidates", [])}

    map_refs = []
    if context is not None:
        for name, entry in context.references.items():
            lx = float(entry.meta.logical_xy[0])
            ly = float(entry.meta.logical_xy[1])
            img_rel = _rel(entry.meta.image_path, project_dir)
            cand = cand_lookup.get(name, {})
            line_sim = cand.get("global_similarity")
            structural_sim = cand.get("structural_similarity")
            map_refs.append({
                "name": name,
                "station": entry.meta.station_id,
                "lx": lx,
                "ly": ly,
                "img": img_rel,
                "isBest": name == best_name,
                "rank": top3_name_to_rank.get(name),
                "score": round(float(cand.get("score", 0.0)), 4),
                "line_sim": round(float(line_sim), 4) if line_sim is not None else None,
                "structural_sim": round(float(structural_sim), 4) if structural_sim is not None else None,
                "inliers": int(cand.get("verified_inliers", 0)),
                "accepted": bool(cand.get("accepted", False)),
            })

    map_data_json = json.dumps({
        "refs": map_refs,
        "best": best_name,
        "query_img": _rel(query_path, project_dir),
        "predicted_xy": [float(result["best_logical_xy"][0]), float(result["best_logical_xy"][1])],
    })

    # ── Top-candidates cards ─────────────────────────────────────────────────
    rank_colors = ["#e74c3c", "#f39c12", "#3498db"]
    rank_labels = ["🥇 #1 Best Match", "🥈 #2", "🥉 #3"]
    cand_cards_html = ""
    for rank, item in enumerate(top3):
        ref_img_src = ""
        if context is not None and item["name"] in context.references:
            ref_img_src = _rel(context.references[item["name"]].meta.image_path, project_dir)
        line_sim = item.get("global_similarity")
        structural_sim = item.get("structural_similarity")
        sim_str = f"{line_sim:.3f}" if line_sim is not None else "—"
        structural_str = f"{structural_sim:.3f}" if structural_sim is not None else "—"
        reasons = item.get("reject_reasons", [])
        status_txt = "accepted" if item["accepted"] else (", ".join(reasons) or "rejected")
        status_color = "#27ae60" if item["accepted"] else "#e74c3c"
        border_color = rank_colors[rank] if rank < len(rank_colors) else "#aaa"
        rank_label = rank_labels[rank] if rank < len(rank_labels) else f"#{rank+1}"
        img_block = (
            f'<img src="{html.escape(ref_img_src)}" loading="lazy" class="cand-img">'
            if ref_img_src else '<div class="cand-img no-img">No image</div>'
        )
        cand_cards_html += f"""
      <div class="cand-card" style="border-top:4px solid {border_color}">
        <div class="cand-rank-label" style="color:{border_color}">{rank_label}</div>
        {img_block}
        <div class="cand-name" title="{html.escape(item['name'])}">{html.escape(item['name'])}</div>
        <div class="cand-stats">
          <div class="stat-row"><span class="stat-key">Line sim</span><span class="stat-val accent">{sim_str}</span></div>
          <div class="stat-row"><span class="stat-key">Structural</span><span class="stat-val">{structural_str}</span></div>
          <div class="stat-row"><span class="stat-key">SIFT inliers</span><span class="stat-val">{item['verified_inliers']}</span></div>
          <div class="stat-row"><span class="stat-key">Score</span><span class="stat-val">{item['score']:.4f}</span></div>
          <div class="stat-row"><span class="stat-key">Coverage</span><span class="stat-val">{item['coverage_ratio']:.2f}</span></div>
          <div class="stat-row"><span class="stat-key">Balance</span><span class="stat-val">{item['balance']:.2f}</span></div>
          <div class="stat-row"><span class="stat-key">Station</span><span class="stat-val">{item['station_id']}</span></div>
          <div class="stat-row full"><span class="stat-key">Status</span>
            <span class="stat-val" style="color:{status_color}">{html.escape(status_txt)}</span></div>
        </div>
      </div>"""

    # ── PnP info ─────────────────────────────────────────────────────────────
    pnp_txt = "Accepted" if pnp.get("accepted") else f"Fallback — {pnp.get('reject_reason', 'n/a')}"
    pnp_color = "#27ae60" if pnp.get("accepted") else "#e67e22"

    # ── Full document ────────────────────────────────────────────────────────
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indoor Localization — {html.escape(best_name)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f3f7;color:#1e293b;min-height:100vh}}
/* ── Topbar ── */
.topbar{{background:#1e293b;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:100}}
.topbar h1{{font-size:16px;font-weight:700;letter-spacing:.3px;white-space:nowrap}}
.pill{{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;background:rgba(255,255,255,.15)}}
.meta-row{{font-size:13px;color:#94a3b8;margin-left:auto;display:flex;gap:16px;flex-wrap:wrap;align-items:center}}
.meta-row strong{{color:#e2e8f0}}
/* ── Upload panel ── */
#uploadPanel{{background:#fff;border-bottom:2px solid #e2e8f0;padding:20px 24px;display:flex;justify-content:center}}
.up-inner{{display:flex;align-items:center;gap:16px;max-width:1400px;width:100%;flex-wrap:wrap}}
#dropZone{{flex:1;min-width:300px;border:2px dashed #cbd5e1;border-radius:12px;padding:18px 28px;
  cursor:pointer;display:flex;align-items:center;gap:16px;transition:all .2s;background:#f8fafc}}
#dropZone:hover,#dropZone.drag{{border-color:#3b82f6;background:#eff6ff;border-style:solid}}
#dropZone .dz-icon{{font-size:32px;flex-shrink:0}}
#dropZone .dz-text{{font-size:13px;color:#64748b;line-height:1.5}}
#dropZone .dz-text strong{{color:#1e293b;display:block;font-size:15px;margin-bottom:2px}}
#fileInput{{display:none}}
#previewWrap{{display:none;align-items:center;gap:14px;flex-shrink:0}}
#previewThumb{{width:88px;height:66px;object-fit:cover;border-radius:8px;border:2px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
.prev-meta{{display:flex;flex-direction:column;gap:4px}}
#previewName{{font-size:13px;color:#475569;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}}
#runBtn{{background:#3b82f6;color:#fff;border:none;border-radius:10px;padding:11px 26px;font-size:15px;font-weight:700;cursor:pointer;transition:background .15s;white-space:nowrap;box-shadow:0 2px 8px rgba(59,130,246,.3)}}
#runBtn:hover{{background:#2563eb;box-shadow:0 4px 12px rgba(59,130,246,.4)}}
#runBtn:disabled{{background:#94a3b8;cursor:not-allowed;box-shadow:none}}
#loadingWrap{{display:none;align-items:center;gap:12px;color:#475569;font-size:14px;flex-shrink:0}}
.spinner{{width:22px;height:22px;border:3px solid #e2e8f0;border-top-color:#3b82f6;border-radius:50%;animation:spin .7s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
#errorMsg{{color:#dc2626;font-size:13px;display:none;flex-shrink:0}}
/* ── Image modal ── */
#imgModal{{position:fixed;inset:0;background:rgba(0,0,0,.82);z-index:20000;display:flex;
  align-items:center;justify-content:center;backdrop-filter:blur(6px);padding:24px}}
#imgModal.hidden{{display:none}}
#modalInner{{background:#1e293b;border-radius:16px;max-width:90vw;max-height:92vh;
  display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.7)}}
#modalHeader{{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;
  border-bottom:1px solid rgba(255,255,255,.1);flex-shrink:0}}
#modalTitle{{color:#f1f5f9;font-size:14px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:calc(100% - 48px)}}
#modalClose{{background:rgba(255,255,255,.12);border:none;color:#f1f5f9;width:32px;height:32px;
  border-radius:8px;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:background .15s;flex-shrink:0}}
#modalClose:hover{{background:rgba(255,255,255,.22)}}
#modalImgWrap{{overflow:auto;display:flex;align-items:center;justify-content:center;padding:20px;flex:1}}
#modalImg{{max-width:100%;max-height:72vh;object-fit:contain;border-radius:10px;display:block}}
#modalMeta{{display:flex;gap:16px;flex-wrap:wrap;padding:12px 20px;border-top:1px solid rgba(255,255,255,.1);
  background:rgba(0,0,0,.2);flex-shrink:0}}
.modal-stat{{display:flex;flex-direction:column;gap:2px}}
.modal-stat .ms-key{{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}}
.modal-stat .ms-val{{font-size:13px;font-weight:700;color:#e2e8f0}}
.modal-stat .ms-val.hi{{color:#60a5fa}}
#modalHint{{padding:8px 20px 12px;text-align:center;font-size:11px;color:#475569}}
/* ── Cards ── */
.page{{max-width:1400px;margin:0 auto;padding:20px}}
.card{{background:#fff;border-radius:14px;box-shadow:0 1px 6px rgba(0,0,0,.08);margin-bottom:22px;overflow:hidden}}
.card-header{{padding:14px 20px;border-bottom:1px solid #f1f5f9;display:flex;align-items:center;gap:10px}}
.card-header h2{{font-size:15px;font-weight:700;color:#334155}}
.card-header .badge{{margin-left:auto;font-size:12px;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:2px 8px}}
.card-body{{padding:18px 20px}}
/* Section 1 */
.line-viz-img{{width:100%;border-radius:8px;display:block}}
/* Section 2 — Map */
.map-wrap{{position:relative;user-select:none}}
#poseMap{{display:block;width:100%;border-radius:8px;background:#f8fafc;cursor:crosshair}}
#mapTooltip{{position:fixed;z-index:9999;background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  box-shadow:0 8px 24px rgba(0,0,0,.18);padding:10px;pointer-events:none;min-width:200px;max-width:240px}}
#mapTooltip.hidden{{display:none}}
#mapTooltip img{{width:220px;height:160px;object-fit:cover;border-radius:6px;display:block}}
#tipInfo{{font-size:12px;margin-top:7px;line-height:1.6;color:#334155}}
#tipInfo .tip-name{{font-weight:700;font-size:13px;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
#tipInfo .tip-metric{{display:flex;justify-content:space-between;color:#64748b}}
#tipInfo .tip-metric span{{font-weight:600;color:#1e293b}}
#tipInfo .tip-best{{color:#e74c3c;font-weight:700;font-size:12px;margin-top:4px}}
#tipInfo .tip-rank{{color:#f39c12;font-weight:600;font-size:12px;margin-top:4px}}
.map-legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:#64748b}}
.leg-item{{display:flex;align-items:center;gap:5px}}
.leg-dot{{width:14px;height:14px;border-radius:50%;display:inline-block}}
/* Section 3 */
.query-row{{display:flex;align-items:flex-start;gap:14px;margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid #f1f5f9;flex-wrap:wrap}}
.query-label{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;margin-bottom:6px}}
#queryImg,#matchImg{{width:180px;height:130px;object-fit:cover;border-radius:8px;border:1px solid #e2e8f0;display:block}}
.query-arrow{{font-size:28px;color:#cbd5e1;align-self:center;margin:0 4px}}
.cand-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
.cand-card{{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;transition:box-shadow .2s}}
.cand-card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.12)}}
.cand-rank-label{{padding:7px 12px;font-size:13px;font-weight:700;background:#fafafa;border-bottom:1px solid #f1f5f9}}
.cand-img{{width:100%;height:170px;object-fit:cover;display:block}}
.no-img{{width:100%;height:170px;display:flex;align-items:center;justify-content:center;background:#f1f5f9;color:#94a3b8;font-size:13px}}
.cand-name{{padding:8px 12px 0;font-size:12px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.cand-stats{{padding:8px 12px 12px;display:flex;flex-direction:column;gap:4px}}
.stat-row{{display:flex;justify-content:space-between;font-size:12px;color:#64748b}}
.stat-val{{font-weight:600;color:#1e293b}}
.stat-val.accent{{color:#2563eb;font-size:13px}}
</style>
</head>
<body>

<!-- ── Sticky topbar ── -->
<div class="topbar">
  <h1>Indoor Localization</h1>
  <span class="pill" id="topbarMode">{html.escape(result['mode'])}</span>
  <span class="pill" id="topbarPnp" style="background:{'#16a34a' if result['mode']=='pnp_pose' else '#d97706'}">PnP: {html.escape(pnp_txt)}</span>
  <div class="meta-row">
    <span>Best: <strong id="topbarBest">{html.escape(best_name)}</strong></span>
    <span>Station: <strong id="topbarStation">{result['best_station_id']}</strong></span>
    <span>Coord: <strong id="topbarCoord">{_fmt_xy(result['best_logical_xy'])}</strong></span>
  </div>
</div>

<!-- ── Upload panel ── -->
<div id="uploadPanel">
  <div class="up-inner">
    <div id="dropZone">
      <span class="dz-icon">📷</span>
      <div class="dz-text">
        <strong>Query rasm yuklash</strong>
        Rasm shu yerga tashlang yoki bosib tanlang — dastur lokalizatsiya qiladi
      </div>
      <input type="file" id="fileInput" accept="image/*">
    </div>
    <div id="previewWrap">
      <img id="previewThumb" src="" alt="">
      <div class="prev-meta">
        <span id="previewName"></span>
        <button id="runBtn">▶ Lokalizatsiya qil</button>
      </div>
    </div>
    <div id="loadingWrap">
      <div class="spinner"></div>
      <span id="loadingMsg">Ishlanmoqda...</span>
    </div>
    <div id="errorMsg"></div>
  </div>
</div>

<!-- ── Image modal (double-click on map) ── -->
<div id="imgModal" class="hidden">
  <div id="modalInner">
    <div id="modalHeader">
      <span id="modalTitle"></span>
      <button id="modalClose" title="Yopish (ESC)">×</button>
    </div>
    <div id="modalImgWrap">
      <img id="modalImg" src="" alt="">
    </div>
    <div id="modalMeta"></div>
    <div id="modalHint">ESC yoki × tugmasini bosib yoping &nbsp;·&nbsp; Rasmni scroll qilib kattalashtiring</div>
  </div>
</div>

<div class="page">

<!-- SECTION 1 — Line viz -->
<div class="card" id="sec1"{' style="display:none"' if not line_viz_rel else ''}>
  <div class="card-header"><h2>Line Features Visualization</h2><div class="badge">Canny + HoughLinesP — rang yo'q</div></div>
  <div class="card-body"><img id="lineVizImg" class="line-viz-img" src="{line_viz_rel}" alt="Line features"></div>
</div>

<!-- SECTION 2 — Interactive map -->
<div class="card">
  <div class="card-header"><h2>Interactive Location Map</h2><div class="badge">Hover → ma'lumot &nbsp;|&nbsp; 2× bosing → rasmni katta ochadi</div></div>
  <div class="card-body">
    <div class="map-wrap">
      <canvas id="poseMap" height="480"></canvas>
      <div id="mapTooltip" class="hidden">
        <img id="tipImg" src="" alt="">
        <div id="tipInfo"></div>
      </div>
    </div>
    <div class="map-legend">
      <div class="leg-item"><span class="leg-dot" style="background:#e74c3c"></span>Best match</div>
      <div class="leg-item"><span class="leg-dot" style="background:#f97316"></span>Top-2</div>
      <div class="leg-item"><span class="leg-dot" style="background:#3b82f6"></span>Top-3</div>
      <div class="leg-item"><span class="leg-dot" style="background:#64748b"></span>Boshqa referencelar</div>
    </div>
  </div>
</div>

<!-- SECTION 3 — Top candidates -->
<div class="card">
  <div class="card-header"><h2>Top Candidates</h2><div class="badge" id="candBadge">{len(top3)} ta · line similarity = asosiy signal</div></div>
  <div class="card-body">
    <div class="query-row">
      <div><div class="query-label">Query rasmi</div><img id="queryImg" src="{query_rel}" alt="query"></div>
      <div class="query-arrow">→</div>
      <div><div class="query-label">SIFT match</div><img id="matchImg" src="{match_rel}" alt="match" style="width:auto;height:130px"></div>
    </div>
    <div class="cand-grid" id="candGrid">{cand_cards_html}</div>
  </div>
</div>

</div><!-- /page -->

<script>
// ═══════════════════════════════════════════
// Candidate cards builder (reused by upload)
// ═══════════════════════════════════════════
function buildCandCards(top3){{
  const RC=['#e74c3c','#f97316','#3b82f6'];
  const RL=['🥇 #1 Best Match','🥈 #2','🥉 #3'];
  return top3.map((item,i)=>{{
    const col=RC[i]||'#aaa', lbl=RL[i]||`#${{i+1}}`;
    const sim=item.global_similarity!=null?item.global_similarity.toFixed(3):'—';
    const structural=item.structural_similarity!=null?item.structural_similarity.toFixed(3):'—';
    const reasons=(item.reject_reasons||[]);
    const status=item.accepted?'accepted':(reasons.join(', ')||'rejected');
    const sc=item.accepted?'#27ae60':'#e74c3c';
    const imgH=item.ref_img_b64?`<img src="${{item.ref_img_b64}}" class="cand-img" loading="lazy">`
                               :`<div class="cand-img no-img">No image</div>`;
    const lxy=Array.isArray(item.logical_xy)?item.logical_xy:[0,0];
    return `<div class="cand-card" style="border-top:4px solid ${{col}}">
      <div class="cand-rank-label" style="color:${{col}}">${{lbl}}</div>
      ${{imgH}}
      <div class="cand-name" title="${{item.name}}">${{item.name}}</div>
      <div class="cand-stats">
        <div class="stat-row"><span>Line sim</span><span class="stat-val accent">${{sim}}</span></div>
        <div class="stat-row"><span>Structural</span><span class="stat-val">${{structural}}</span></div>
        <div class="stat-row"><span>SIFT inliers</span><span class="stat-val">${{item.verified_inliers}}</span></div>
        <div class="stat-row"><span>Score</span><span class="stat-val">${{item.score.toFixed(4)}}</span></div>
        <div class="stat-row"><span>Coverage</span><span class="stat-val">${{item.coverage_ratio.toFixed(2)}}</span></div>
        <div class="stat-row"><span>Balance</span><span class="stat-val">${{item.balance.toFixed(2)}}</span></div>
        <div class="stat-row"><span>Station</span><span class="stat-val">${{item.station_id}}</span></div>
        <div class="stat-row"><span>Coord</span><span class="stat-val" style="font-size:11px">(${{lxy[0].toFixed(2)}},${{lxy[1].toFixed(2)}})</span></div>
        <div class="stat-row"><span>Status</span><span class="stat-val" style="color:${{sc}}">${{status}}</span></div>
      </div></div>`;
  }}).join('');
}}

// ═══════════════════════════════════════════
// Map controller
// ═══════════════════════════════════════════
let _mapCtrl=null;
(function(){{
let DATA={map_data_json};
const canvas=document.getElementById('poseMap');
const tooltip=document.getElementById('mapTooltip');
const tipImg=document.getElementById('tipImg');
const tipInfo=document.getElementById('tipInfo');
if(!canvas) return;
const ctx=canvas.getContext('2d');
const PAD=55;
let pts=[],hovIdx=-1,B={{}};

function calcBounds(){{
  if(!DATA.refs.length) return {{xMin:-1,xMax:1,yMin:-1,yMax:1}};
  const lxs=DATA.refs.map(r=>r.lx),lys=DATA.refs.map(r=>r.ly);
  let xMin=Math.min(...lxs),xMax=Math.max(...lxs);
  let yMin=Math.min(...lys),yMax=Math.max(...lys);
  const xR=xMax-xMin||2,yR=yMax-yMin||2;
  return {{xMin:xMin-xR*.12,xMax:xMax+xR*.12,yMin:yMin-yR*.12,yMax:yMax+yR*.12}};
}}

function w2c(lx,ly){{
  const cw=canvas.width-2*PAD,ch=canvas.height-2*PAD;
  return [PAD+((lx-B.xMin)/(B.xMax-B.xMin))*cw,
          PAD+((B.yMax-ly)/(B.yMax-B.yMin))*ch];
}}

function initPts(){{
  B=calcBounds();
  pts=DATA.refs.map(r=>{{const[cx,cy]=w2c(r.lx,r.ly);return{{...r,cx,cy}}}});
}}

function resize(){{
  const rect=canvas.parentElement.getBoundingClientRect();
  canvas.width=Math.max(380,rect.width-40);
  canvas.height=Math.max(280,Math.min(520,canvas.width*0.56));
  initPts();draw();
}}

function draw(){{
  const W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){{const x=PAD+i*(W-2*PAD)/5;ctx.beginPath();ctx.moveTo(x,PAD);ctx.lineTo(x,H-PAD);ctx.stroke();}}
  for(let i=0;i<=4;i++){{const y=PAD+i*(H-2*PAD)/4;ctx.beginPath();ctx.moveTo(PAD,y);ctx.lineTo(W-PAD,y);ctx.stroke();}}
  ctx.fillStyle='#94a3b8';ctx.font='11px sans-serif';
  ctx.textAlign='center';
  for(let i=0;i<=5;i++){{const lx=B.xMin+i*(B.xMax-B.xMin)/5;const[x]=w2c(lx,B.yMin);ctx.fillText(lx.toFixed(1),x,H-PAD+18);}}
  ctx.textAlign='right';
  for(let i=0;i<=4;i++){{const ly=B.yMin+i*(B.yMax-B.yMin)/4;const[,y]=w2c(B.xMin,ly);ctx.fillText(ly.toFixed(1),PAD-6,y+4);}}
  pts.forEach((p,i)=>{{
    const hov=i===hovIdx;
    let r,fill,stroke;
    if(p.isBest)       {{r=hov?18:14;fill='#e74c3c';stroke='#b91c1c';}}
    else if(p.rank===1){{r=hov?16:13;fill='#f97316';stroke='#c2410c';}}
    else if(p.rank===2){{r=hov?15:12;fill='#eab308';stroke='#a16207';}}
    else if(p.rank===3){{r=hov?14:11;fill='#3b82f6';stroke='#1d4ed8';}}
    else               {{r=hov?12: 9;fill='#64748b';stroke='#475569';}}
    if(hov){{ctx.shadowColor='rgba(0,0,0,.25)';ctx.shadowBlur=10;}}
    ctx.beginPath();ctx.arc(p.cx,p.cy,r,0,Math.PI*2);
    ctx.fillStyle=fill;ctx.fill();
    ctx.strokeStyle=stroke;ctx.lineWidth=hov?2:1.5;ctx.stroke();
    ctx.shadowBlur=0;
    ctx.fillStyle='#fff';ctx.font=`bold ${{Math.max(8,r-2)}}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(p.station!=null?String(p.station):'?',p.cx,p.cy);
  }});
  if(DATA.predicted_xy){{
    const[px,py]=w2c(DATA.predicted_xy[0],DATA.predicted_xy[1]);
    ctx.strokeStyle='#e74c3c';ctx.lineWidth=3;ctx.lineCap='round';
    const s=11;
    ctx.beginPath();ctx.moveTo(px-s,py-s);ctx.lineTo(px+s,py+s);
    ctx.moveTo(px+s,py-s);ctx.lineTo(px-s,py+s);ctx.stroke();
  }}
}}

canvas.addEventListener('mousemove',function(e){{
  const rect=canvas.getBoundingClientRect();
  const sx=canvas.width/rect.width,sy=canvas.height/rect.height;
  const mx=(e.clientX-rect.left)*sx,my=(e.clientY-rect.top)*sy;
  let best=-1,bestD=22*sx;
  pts.forEach((p,i)=>{{const d=Math.hypot(p.cx-mx,p.cy-my);if(d<bestD){{bestD=d;best=i;}}}});
  if(best!==hovIdx){{hovIdx=best;draw();}}
  if(best>=0){{
    const p=pts[best];
    tipImg.src=p.img;
    let h=`<div class="tip-name">Station ${{p.station!=null?p.station:'?'}} — ${{p.name}}</div>`;
    if(p.line_sim!=null) h+=`<div class="tip-metric">Line sim <span>${{p.line_sim}}</span></div>`;
    if(p.structural_sim!=null) h+=`<div class="tip-metric">Structural <span>${{p.structural_sim}}</span></div>`;
    if(p.inliers)        h+=`<div class="tip-metric">SIFT inliers <span>${{p.inliers}}</span></div>`;
    if(p.score)          h+=`<div class="tip-metric">Score <span>${{p.score}}</span></div>`;
    if(p.isBest)  h+=`<div class="tip-best">★ Best Match</div>`;
    else if(p.rank) h+=`<div class="tip-rank">Top-${{p.rank}}</div>`;
    tipInfo.innerHTML=h;
    const TW=250,TH=220;
    let L=e.clientX+16,T=e.clientY-60;
    if(L+TW>window.innerWidth-8) L=e.clientX-TW-16;
    if(T+TH>window.innerHeight-8) T=window.innerHeight-TH-8;
    if(T<8) T=8;
    tooltip.style.left=L+'px';tooltip.style.top=T+'px';
    tooltip.classList.remove('hidden');
  }}else{{tooltip.classList.add('hidden');}}
}});
canvas.addEventListener('mouseleave',function(){{hovIdx=-1;draw();tooltip.classList.add('hidden');}});

// Double-click → open modal
canvas.addEventListener('dblclick',function(e){{
  const rect=canvas.getBoundingClientRect();
  const sx=canvas.width/rect.width,sy=canvas.height/rect.height;
  const mx=(e.clientX-rect.left)*sx,my=(e.clientY-rect.top)*sy;
  let best=-1,bestD=28*sx;
  pts.forEach((p,i)=>{{const d=Math.hypot(p.cx-mx,p.cy-my);if(d<bestD){{bestD=d;best=i;}}}});
  if(best>=0) openImgModal(pts[best]);
}});

resize();
window.addEventListener('resize',resize);

// Expose update API
_mapCtrl={{
  update:function(d){{
    DATA=d; hovIdx=-1; resize();
  }}
}};
}})();

// ═══════════════════════════════════════════
// Image modal (double-click on map point)
// ═══════════════════════════════════════════
function openImgModal(p){{
  const modal=document.getElementById('imgModal');
  const img=document.getElementById('modalImg');
  const title=document.getElementById('modalTitle');
  const meta=document.getElementById('modalMeta');
  if(!modal) return;

  img.src=p.img;
  title.textContent=`Station ${{p.station!=null?p.station:'?'}} — ${{p.name}}`;

  const parts=[];
  if(p.line_sim!=null) parts.push([`Line sim`,p.line_sim,'hi']);
  if(p.structural_sim!=null) parts.push([`Structural`,p.structural_sim,'']);
  if(p.inliers)        parts.push([`SIFT inliers`,p.inliers,'']);
  if(p.score)          parts.push([`Score`,p.score,'']);
  if(p.accepted!=null) parts.push([`Status`,p.accepted?'✓ accepted':'✗ rejected',p.accepted?'hi':'']);
  if(p.isBest)         parts.push([`Natija`,'★ Best Match','hi']);
  else if(p.rank)      parts.push([`Natija`,`Top-${{p.rank}}`,'']);
  meta.innerHTML=parts.map(([k,v,cls])=>
    `<div class="modal-stat"><span class="ms-key">${{k}}</span><span class="ms-val ${{cls}}">${{v}}</span></div>`
  ).join('');

  modal.classList.remove('hidden');
  document.body.style.overflow='hidden';
}}

function closeImgModal(){{
  const modal=document.getElementById('imgModal');
  if(modal) modal.classList.add('hidden');
  document.body.style.overflow='';
}}

document.getElementById('modalClose')?.addEventListener('click',closeImgModal);
document.getElementById('imgModal')?.addEventListener('click',function(e){{
  if(e.target===this) closeImgModal(); // click backdrop
}});
document.addEventListener('keydown',function(e){{
  if(e.key==='Escape') closeImgModal();
}});

// ═══════════════════════════════════════════
// Page-level update (called after upload)
// ═══════════════════════════════════════════
function updatePage(d){{
  // Topbar
  document.getElementById('topbarBest').textContent=d.best_name;
  document.getElementById('topbarStation').textContent=d.best_station??'—';
  document.getElementById('topbarCoord').textContent=
    `(${{d.best_xy[0].toFixed(3)}}, ${{d.best_xy[1].toFixed(3)}})`;
  document.getElementById('topbarMode').textContent=d.mode;
  // Section 1
  const lv=document.getElementById('lineVizImg');
  const sec1=document.getElementById('sec1');
  if(lv&&d.line_viz_b64){{lv.src=d.line_viz_b64;if(sec1)sec1.style.display='';}}
  // Section 2
  if(_mapCtrl) _mapCtrl.update(d.map_data);
  // Section 3
  const qi=document.getElementById('queryImg');
  const mi=document.getElementById('matchImg');
  if(qi&&d.query_b64) qi.src=d.query_b64;
  if(mi&&d.match_b64) mi.src=d.match_b64;
  const cg=document.getElementById('candGrid');
  if(cg) cg.innerHTML=buildCandCards(d.top3);
  const cb=document.getElementById('candBadge');
  if(cb) cb.textContent=`${{d.top3.length}} ta · line similarity = asosiy signal`;
}}

// ═══════════════════════════════════════════
// Upload logic
// ═══════════════════════════════════════════
(function(){{
const dropZone=document.getElementById('dropZone');
const fileInput=document.getElementById('fileInput');
const previewWrap=document.getElementById('previewWrap');
const previewThumb=document.getElementById('previewThumb');
const previewName=document.getElementById('previewName');
const runBtn=document.getElementById('runBtn');
const loadingWrap=document.getElementById('loadingWrap');
const loadingMsg=document.getElementById('loadingMsg');
const errorMsg=document.getElementById('errorMsg');

let selectedFile=null;

function showPreview(file){{
  selectedFile=file;
  const url=URL.createObjectURL(file);
  previewThumb.src=url;
  previewName.textContent=file.name;
  previewWrap.style.display='flex';
  runBtn.disabled=false;
  errorMsg.style.display='none';
}}

dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>{{if(e.target.files[0]) showPreview(e.target.files[0]);}});
dropZone.addEventListener('dragover',e=>{{e.preventDefault();dropZone.classList.add('drag');}});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop',e=>{{
  e.preventDefault();dropZone.classList.remove('drag');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')) showPreview(f);
}});

runBtn.addEventListener('click',async()=>{{
  if(!selectedFile) return;
  runBtn.disabled=true;
  previewWrap.style.display='none';
  loadingWrap.style.display='flex';
  loadingMsg.textContent=`"${{selectedFile.name}}" lokalizatsiya qilinmoqda...`;
  errorMsg.style.display='none';

  const fd=new FormData();
  fd.append('image',selectedFile);

  try{{
    const resp=await fetch('/localize',{{method:'POST',body:fd}});
    const data=await resp.json();
    if(!resp.ok||data.error){{
      throw new Error(data.error||`Server xatosi ${{resp.status}}`);
    }}
    updatePage(data);
    // Reset upload UI
    loadingWrap.style.display='none';
    previewThumb.src=data.query_b64||previewThumb.src;
    previewName.textContent=selectedFile.name+' ✓';
    previewWrap.style.display='flex';
    runBtn.textContent='Yana lokalizatsiya qil';
    runBtn.disabled=false;
    // Scroll to map
    document.getElementById('poseMap')?.scrollIntoView({{behavior:'smooth',block:'center'}});
  }}catch(err){{
    loadingWrap.style.display='none';
    previewWrap.style.display='flex';
    runBtn.disabled=false;
    errorMsg.style.display='block';
    errorMsg.textContent='Xato: '+err.message;
  }}
}});
}})();
</script>
</body>
</html>"""
    output_path.write_text(doc, encoding="utf-8")


def save_line_visualization(context, result: dict, output_path: Path) -> None:
    """Side-by-side visualization: query ROI + top line-shortlist references with scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_path = result["query_path"]
    query_lf = result.get("query_line_features")
    top_line = result.get("line_top_candidates") or result["top3_candidates"]

    query_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        return

    if query_lf is None:
        query_lf = extract_line_features(query_path)

    # Resize to a common display height
    display_h = 360
    def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        scale = float(h) / float(img.shape[0])
        w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _add_roi_overlay(img: np.ndarray, lf, color=(255, 255, 255)) -> np.ndarray:
        vis = img.copy()
        h, w = vis.shape[:2]
        if getattr(lf, "roi_bottom_y", 0) > 0:
            y = int(lf.roi_bottom_y)
            shaded = vis.copy()
            cv2.rectangle(shaded, (0, y), (w, h), (40, 40, 40), -1)
            vis = cv2.addWeighted(shaded, 0.30, vis, 0.70, 0.0)
            cv2.line(vis, (0, y), (w, y), color, 2, cv2.LINE_AA)
        bounds = np.asarray(getattr(lf, "zone_bounds_y", []), dtype=np.int32).reshape(-1)
        if len(bounds) == 4:
            for zy in bounds[1:-1]:
                cv2.line(vis, (0, int(zy)), (w, int(zy)), (120, 220, 255), 1, cv2.LINE_AA)
        return vis

    query_vis = draw_line_features(query_bgr, query_lf, color=(0, 220, 0), thickness=2)
    query_vis = _add_roi_overlay(query_vis, query_lf)
    query_vis = _resize_to_h(query_vis, display_h)

    # Header label for query
    header_h = 32
    def _add_header(img: np.ndarray, text: str, bg: tuple = (30, 30, 30)) -> np.ndarray:
        header = np.full((header_h, img.shape[1], 3), bg, dtype=np.uint8)
        cv2.putText(header, text[:60], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        return np.vstack([header, img])

    q_label = (
        f"QUERY  ROI(top+mid)  lines={len(query_lf.lines)}  "
        f"V={query_lf.vertical_count} H={query_lf.horizontal_count}"
    )
    panels = [_add_header(query_vis, q_label, bg=(20, 80, 20))]

    ref_colors = [(0, 180, 255), (255, 160, 0), (180, 0, 255)]
    for rank, item in enumerate(top_line[:3]):
        ref_name = item["name"]
        if ref_name not in context.references:
            continue
        entry = context.references[ref_name]
        ref_bgr = cv2.imread(str(entry.meta.image_path), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            continue
        ref_lf = entry.line_features
        if ref_lf is None:
            ref_lf = extract_line_features(entry.meta.image_path)
        ref_vis = draw_line_features(ref_bgr, ref_lf, color=ref_colors[rank], thickness=2)
        ref_vis = _add_roi_overlay(ref_vis, ref_lf)
        ref_vis = _resize_to_h(ref_vis, display_h)
        line_sim = item.get("global_similarity")
        structural_sim = item.get("structural_similarity")
        sim_str = f"{line_sim:.3f}" if line_sim is not None else "n/a"
        structural_str = f"{structural_sim:.3f}" if structural_sim is not None else "n/a"
        mode_tag = "final" if ref_name == result["best_reference_name"] else "shortlist"
        label = (
            f"#{rank+1} {ref_name}  {mode_tag}  line={sim_str}  structural={structural_str}"
            f"  inliers={item['verified_inliers']}  score={item['score']:.3f}"
        )
        panels.append(_add_header(ref_vis, label))

    if not panels:
        return

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)

    canvas = np.hstack(padded)
    max_width = 2400
    if canvas.shape[1] > max_width:
        scale = float(max_width) / float(canvas.shape[1])
        canvas = cv2.resize(
            canvas,
            (max_width, max(1, int(round(canvas.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(output_path), canvas)


def write_all_outputs(
    context,
    result: dict,
    summary_path: Path,
    match_png: Path,
    pose_png: Path,
    pnp_inliers_png: Optional[Path] = None,
    html_path: Optional[Path] = None,
    line_viz_png: Optional[Path] = None,
) -> None:
    save_localization_summary(result, summary_path)
    save_best_match_visualization(context, result, match_png)
    plot_pose_result(context, result, pose_png)
    if pnp_inliers_png is not None:
        save_pnp_inlier_visualization(context, result, pnp_inliers_png)
    if line_viz_png is not None:
        save_line_visualization(context, result, line_viz_png)
    if html_path is not None:
        build_html_report(
            result,
            html_path,
            project_dir=context.config.project_dir,
            match_png=match_png,
            pose_png=pose_png,
            pnp_inliers_png=pnp_inliers_png,
            line_viz_png=line_viz_png,
            query_path=result["query_path"],
            context=context,
        )
