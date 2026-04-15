#!/usr/bin/env python3
"""
Live indoor localization server.

Loads the reference context once, then accepts image uploads and returns
full localization results as JSON (all images base64-encoded).

Usage:
    python server.py --ref-dir "ref_images/dataset-20260410-142733-2_prepared"
    python server.py --ref-dir ref_images/corridor --port 8080

Then open  http://localhost:5000  in your browser.
Upload any image — the page updates in-place without reloading.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_file

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from localization import PipelineConfig, load_localization_context, localize_query_image
from visualize import save_best_match_visualization, save_line_visualization

app = Flask(__name__, static_folder=None)

# Loaded once at startup
_ctx = None
_project_dir = PROJECT_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _build_map_refs(result: dict) -> list:
    top3_rank = {item["name"]: i + 1 for i, item in enumerate(result["top3_candidates"])}
    cand_lookup = {c["name"]: c for c in result.get("primary_candidates", [])}
    refs = []
    for name, entry in _ctx.references.items():
        cand = cand_lookup.get(name, {})
        line_sim = cand.get("global_similarity")
        structural_sim = cand.get("structural_similarity")
        try:
            img_path = entry.meta.image_path.relative_to(_project_dir).as_posix()
        except ValueError:
            img_path = str(entry.meta.image_path)
        refs.append({
            "name": name,
            "station": entry.meta.station_id,
            "lx": float(entry.meta.logical_xy[0]),
            "ly": float(entry.meta.logical_xy[1]),
            "img": img_path,
            "isBest": name == result["best_reference_name"],
            "rank": top3_rank.get(name),
            "score": round(float(cand.get("score", 0.0)), 4),
            "line_sim": round(float(line_sim), 4) if line_sim is not None else None,
            "structural_sim": round(float(structural_sim), 4) if structural_sim is not None else None,
            "inliers": int(cand.get("verified_inliers", 0)),
            "accepted": bool(cand.get("accepted", False)),
        })
    return refs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = _project_dir / "colmap_pose_view.html"
    if not html_path.exists():
        return (
            "<h2>HTML report not found.</h2>"
            "<p>Run <code>python run.py --ref-dir &lt;your-ref-dir&gt;</code> first.</p>",
            404,
        )
    return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/localize", methods=["POST"])
def localize():
    if _ctx is None:
        return jsonify({"error": "Context not loaded on server"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in multipart form data"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return jsonify({"error": f"Unsupported image format: {suffix}"}), 400

    query_dir = _project_dir / "query"
    query_dir.mkdir(exist_ok=True)

    tmp_query = query_dir / f"_upload{suffix}"
    tmp_line = query_dir / "_upload_line_viz.jpg"
    tmp_match = query_dir / "_upload_match.jpg"

    try:
        file.save(str(tmp_query))

        result = localize_query_image(_ctx, tmp_query, verbose=False)

        save_line_visualization(_ctx, result, tmp_line)
        save_best_match_visualization(_ctx, result, tmp_match)

        # Top-3 with reference images as base64
        top3_out = []
        for item in result["top3_candidates"]:
            ref_b64 = ""
            if item["name"] in _ctx.references:
                ref_b64 = _file_to_b64(_ctx.references[item["name"]].meta.image_path)
            top3_out.append({**item, "ref_img_b64": ref_b64,
                             "logical_xy": [float(item["logical_xy"][0]), float(item["logical_xy"][1])]})

        payload = {
            "status": "ok",
            "best_name": result["best_reference_name"],
            "best_station": result["best_station_id"],
            "best_xy": [float(result["best_logical_xy"][0]), float(result["best_logical_xy"][1])],
            "mode": result["mode"],
            "query_b64": _file_to_b64(tmp_query),
            "line_viz_b64": _file_to_b64(tmp_line),
            "match_b64": _file_to_b64(tmp_match),
            "map_data": {
                "refs": _build_map_refs(result),
                "best": result["best_reference_name"],
                "predicted_xy": [float(result["best_logical_xy"][0]), float(result["best_logical_xy"][1])],
            },
            "top3": top3_out,
            "line_top_candidates": result.get("line_top_candidates", []),
        }
        return jsonify(payload)

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return jsonify({"error": "Localization failed", "detail": tb}), 500

    finally:
        for p in [tmp_query, tmp_line, tmp_match]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# Serve any project file (so image src="ref_images/..." works via http)
@app.route("/<path:filepath>")
def serve_project_file(filepath):
    full = _project_dir / filepath
    if full.exists() and full.is_file():
        return send_file(str(full))
    return "", 404


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _ctx

    parser = argparse.ArgumentParser(description="Live indoor localization server")
    parser.add_argument("--ref-dir", default="ref_images/corridor", help="Reference image directory")
    parser.add_argument("--coords", default=None, help="coords.txt path")
    parser.add_argument("--no-pnp", action="store_true", help="Disable PnP")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    ref_dir = (PROJECT_DIR / args.ref_dir).resolve()

    config = PipelineConfig(
        project_dir=PROJECT_DIR,
        ref_dir=ref_dir,
        coords_path=(Path(args.coords).resolve() if args.coords else None),
        pnp_enabled=not args.no_pnp,
    ).resolved()

    print(f"Loading context from {ref_dir} ...")
    _ctx = load_localization_context(config, verbose=True)
    print(f"\nServer ready →  http://{args.host}:{args.port}")
    print("Upload an image on the page to localize it.\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
