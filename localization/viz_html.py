"""HTML vizualizatsiya."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .colmap_io import ImageRecord
from .layout_utils import (
    build_logical_layout_items,
    extract_sequence_id,
    extract_view_type,
    logical_layout_position,
)


def build_interactive_pose_html(
    _images: Dict[int, ImageRecord],
    query_center: np.ndarray,
    best_reference_name: str,
    localization_mode: str,
    output_path: Path,
    query_image_path: str = "query/test.jpg",
) -> None:
    layout_items = build_logical_layout_items()
    if not layout_items:
        return

    sorted_names = [name for name, _ in layout_items]
    first_name = sorted_names[0]
    last_name = sorted_names[-1]
    centers = np.array([pos for _, pos in layout_items], dtype=np.float64)
    x_vals = centers[:, 0]
    z_vals = centers[:, 1]

    min_x, max_x = float(np.min(x_vals)), float(np.max(x_vals))
    min_z, max_z = float(np.min(z_vals)), float(np.max(z_vals))
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
    if abs(max_z - min_z) < 1e-9:
        max_z += 1.0

    items = []
    for name, pos in layout_items:
        items.append(
            {
                "name": name,
                "x": float(pos[0]),
                "z": float(pos[1]),
                "raw_x": float(pos[0]),
                "raw_z": float(pos[1]),
                "type": "best_reference" if name == best_reference_name else extract_view_type(name),
                "image_path": f"ref_images/{name}",
                "label": str(extract_sequence_id(name)),
            }
        )

    query_pos = logical_layout_position(best_reference_name)
    if localization_mode == "fallback_reference_pose":
        query_pos = query_pos + np.array([0.18, 0.0], dtype=np.float64)
    items.append(
        {
            "name": "query",
            "x": float(query_pos[0]),
            "z": float(query_pos[1]),
            "raw_x": float(query_center[0]),
            "raw_z": float(query_center[2]),
            "type": "query",
            "image_path": query_image_path,
            "label": "query",
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Logical Corridor Layout</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f5;
      color: #1c1c1c;
    }}
    .wrap {{
      display: grid;
      grid-template-columns: minmax(700px, 1.4fr) minmax(320px, 0.8fr);
      gap: 18px;
      padding: 18px;
      box-sizing: border-box;
    }}
    .panel {{
      background: white;
      border: 1px solid #ddd8cf;
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.06);
      overflow: hidden;
    }}
    .plot-header {{
      padding: 16px 18px 8px;
      border-bottom: 1px solid #eee7de;
    }}
    .plot-header h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .plot-header p {{
      margin: 0;
      color: #5c5a56;
      line-height: 1.45;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      padding: 10px 18px 0;
      color: #4d4b47;
      font-size: 14px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .svg-wrap {{
      padding: 10px 16px 16px;
    }}
    svg {{
      width: 100%;
      height: 760px;
      background: linear-gradient(180deg, #fffdf9, #f8f5ef);
      border-radius: 14px;
      border: 1px solid #ece5da;
    }}
    .side {{
      padding: 16px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f2ede5;
      color: #5d564d;
      font-size: 13px;
      font-weight: 600;
    }}
    .meta {{
      line-height: 1.6;
      font-size: 15px;
    }}
    .preview {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid #e8dfd2;
      background: #fbfaf7;
      min-height: 260px;
      object-fit: contain;
    }}
    .hint {{
      color: #6f6a62;
      font-size: 13px;
    }}
    .point {{
      cursor: pointer;
    }}
    .point-hit {{
      cursor: pointer;
    }}
    .axis-label {{
      fill: #5e5b56;
      font-size: 14px;
      font-weight: 600;
    }}
    .grid-line {{
      stroke: rgba(80,80,80,0.12);
      stroke-width: 1;
    }}
    .trend-line {{
      stroke: rgba(0, 0, 0, 0.6);
      stroke-width: 2.5;
      fill: none;
    }}
    .name-label {{
      fill: #3a3a3a;
      font-size: 11px;
      font-weight: 700;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="plot-header">
        <h1>Logical Corridor Layout</h1>
        <p>Bu view siz rasm olgan tartibga mos: front markazda, left chapda, right o‘ngda. Nuqtani bossangiz o‘ng tomonda rasm chiqadi.</p>
      </div>
      <div class="legend">
        <span><i class="dot" style="background: seagreen"></i>Left images</span>
        <span><i class="dot" style="background: steelblue"></i>Front images</span>
        <span><i class="dot" style="background: mediumpurple"></i>Right images</span>
        <span><i class="dot" style="background: red"></i>Query image</span>
        <span><i class="dot" style="background: orange"></i>Best reference</span>
      </div>
      <div class="svg-wrap">
        <svg id="pose-svg" viewBox="0 0 980 760" preserveAspectRatio="xMidYMid meet"></svg>
      </div>
    </div>

    <div class="panel side">
      <div>
        <span class="badge">Mode: {localization_mode}</span>
      </div>
      <div class="meta">
        <div><strong id="point-name">query</strong></div>
        <div id="point-type">Query image</div>
        <div id="point-coord">layout x={query_pos[0]:.2f}, order y={query_pos[1]:.2f}</div>
        <div id="point-raw-coord">raw x={query_center[0]:.4f}, raw z={query_center[2]:.4f}</div>
      </div>
      <div>
        <img id="preview" class="preview" src="{query_image_path}" alt="preview" />
        <p class="hint">Query nuqta best reference bilan bir joyga tushsa, bu fallback mode ekanini bildiradi.</p>
      </div>
    </div>
  </div>

  <script>
    const items = {json.dumps(items)};
    const minX = {min_x};
    const maxX = {max_x};
    const minZ = {min_z};
    const maxZ = {max_z};
    const svg = document.getElementById("pose-svg");
    const preview = document.getElementById("preview");
    const pointName = document.getElementById("point-name");
    const pointType = document.getElementById("point-type");
    const pointCoord = document.getElementById("point-coord");
    const pointRawCoord = document.getElementById("point-raw-coord");

    const margin = {{ left: 90, right: 40, top: 40, bottom: 70 }};
    const width = 980;
    const height = 760;
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    function sx(x) {{
      return margin.left + ((x - minX) / (maxX - minX)) * plotW;
    }}
    function sy(z) {{
      return margin.top + ((z - minZ) / (maxZ - minZ)) * plotH;
    }}
    function add(tag, attrs, parent = svg) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      parent.appendChild(el);
      return el;
    }}

    for (let i = 0; i <= 8; i++) {{
      const x = margin.left + (plotW / 4) * (i % 5);
      const y = margin.top + (plotH / 8) * i;
      add("line", {{ x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }});
      if (i < 5) add("line", {{ x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, class: "grid-line" }});
    }}

    add("text", {{ x: width / 2, y: height - 18, class: "axis-label", "text-anchor": "middle" }}).textContent = "Left  <---- corridor ---->  Right";
    const yLabel = add("text", {{ x: 24, y: height / 2, class: "axis-label", transform: `rotate(-90 24 ${{height / 2}})`, "text-anchor": "middle" }});
    yLabel.textContent = "Capture order";
    add("text", {{ x: margin.left, y: 22, class: "axis-label" }}).textContent = "Start: {Path(first_name).stem} -> End: {Path(last_name).stem}";

    const frontItems = items.filter(item => item.type === "front" || item.type === "best_reference");
    const sortedFront = [...frontItems].sort((a, b) => Number.parseInt(a.label, 10) - Number.parseInt(b.label, 10));
    if (sortedFront.length > 1) {{
      const points = sortedFront.map(item => `${{sx(item.x)}},${{sy(item.z)}}`).join(" ");
      add("polyline", {{ points, class: "trend-line" }});
    }}

    function describeType(type) {{
      if (type === "query") return "Query image";
      if (type === "best_reference") return "Best matched reference";
      if (type === "left") return "Left reference image";
      if (type === "right") return "Right reference image";
      return "Front reference image";
    }}

    function selectItem(item) {{
      preview.src = item.image_path;
      pointName.textContent = item.name;
      pointType.textContent = describeType(item.type);
      pointCoord.textContent = `layout x=${{item.x.toFixed(2)}}, order y=${{item.z.toFixed(2)}}`;
      pointRawCoord.textContent = `raw x=${{item.raw_x.toFixed(4)}}, raw z=${{item.raw_z.toFixed(4)}}`;
    }}

    function attachPointEvents(target, item) {{
      target.addEventListener("mouseenter", () => selectItem(item));
      target.addEventListener("click", () => selectItem(item));
    }}

    items.forEach(item => {{
      const cx = sx(item.x);
      const cy = sy(item.z);
      const hit = add("circle", {{ cx, cy, r: 18, fill: "transparent", class: "point-hit" }});

      if (item.type === "left") {{
        const tri = add("polygon", {{ points: `${{cx-9}},${{cy}} ${{cx+7}},${{cy-7}} ${{cx+7}},${{cy+7}}`, fill: "seagreen", stroke: "white", "stroke-width": 1.2, class: "point" }});
        attachPointEvents(tri, item);
        attachPointEvents(hit, item);
      }} else if (item.type === "right") {{
        const tri = add("polygon", {{ points: `${{cx+9}},${{cy}} ${{cx-7}},${{cy-7}} ${{cx-7}},${{cy+7}}`, fill: "mediumpurple", stroke: "white", "stroke-width": 1.2, class: "point" }});
        attachPointEvents(tri, item);
        attachPointEvents(hit, item);
      }} else {{
        const fill = item.type === "query" ? "red" : (item.type === "best_reference" ? "orange" : "steelblue");
        const stroke = item.type === "best_reference" ? "#4b2500" : "white";
        const radius = item.type === "query" ? 9 : (item.type === "best_reference" ? 8 : 6);
        const circle = add("circle", {{ cx, cy, r: radius, fill, stroke, "stroke-width": 1.2, class: "point" }});
        attachPointEvents(circle, item);
        attachPointEvents(hit, item);
      }}

      const label = add("text", {{ x: cx + 8, y: cy - 6, class: "name-label" }});
      label.textContent = item.label;
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

def render_localizer_web_html(state: dict) -> str:
    result = state["result"]
    center = result["camera_center"]
    top_rows = "\n".join(
        f"<li><strong>{html.escape(name)}</strong> <span>{score} matches</span></li>"
        for name, score in result.get("top_reference_scores", [])
    )
    if not top_rows:
        top_rows = "<li>Top matchlar topilmadi</li>"

    query_name = html.escape(state["query_name"])
    best_reference = html.escape(result["best_reference_name"])
    mode = html.escape(result["mode"])
    confidence = html.escape(str(result.get("confidence", "unknown")))
    second_reference = html.escape(str(result.get("second_reference_name", "-")))
    ambiguity_reason = html.escape(str(result.get("ambiguity_reason", "")))
    ambiguous_note = ""
    if result.get("is_ambiguous"):
        ambiguous_note = (
            f"<p class=\"warning\">Natija noaniq: top-2 juda yaqin. "
            f"Ikkinchi variant: <strong>{second_reference}</strong>. {ambiguity_reason}</p>"
        )
    query_rel_path = html.escape(state["query_rel_path"])
    cache_bust = int(state["updated_at"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Indoor Localizer</title>
  <style>
    :root {{
      --bg: #f3efe8;
      --panel: #fffdf9;
      --line: #ddd3c4;
      --ink: #1f1e1b;
      --soft: #6d675d;
      --accent: #be5d34;
      --accent-2: #d98b00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(190,93,52,0.10), transparent 28%),
        linear-gradient(180deg, #f8f5ef, var(--bg));
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 14px 38px rgba(0,0,0,0.06);
    }}
    .hero {{
      padding: 22px;
      display: grid;
      gap: 14px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--soft);
      line-height: 1.5;
    }}
    .upload-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .file-input {{
      min-width: 260px;
      padding: 12px;
      background: #faf7f2;
      border: 1px dashed #cbbca6;
      border-radius: 14px;
    }}
    .btn {{
      border: 0;
      border-radius: 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      padding: 12px 18px;
      cursor: pointer;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: #f1ece4;
      color: #524c44;
      font-size: 14px;
      width: fit-content;
    }}
    .warning {{
      margin: 0;
      padding: 12px 14px;
      border-radius: 14px;
      background: #fff3cf;
      color: #6b4a00;
      border: 1px solid #efd488;
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    .panel {{
      padding: 18px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .panel img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid #e6dccd;
      background: #faf8f4;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .card {{
      padding: 14px;
      border: 1px solid #ebe1d3;
      border-radius: 16px;
      background: #fcfaf6;
    }}
    .card strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      color: var(--soft);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card span {{
      font-size: 20px;
      font-weight: 700;
    }}
    .dual {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .dual h3, .side h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .side {{
      display: grid;
      gap: 14px;
    }}
    ol {{
      margin: 0;
      padding-left: 22px;
    }}
    li {{
      margin-bottom: 8px;
      color: #403a34;
    }}
    li span {{
      color: var(--soft);
    }}
    .hint {{
      margin: 0;
      color: var(--soft);
      font-size: 14px;
      line-height: 1.5;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .dual {{
        grid-template-columns: 1fr;
      }}
      .meta {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="status">Mode: {mode}</div>
      <div class="status">Confidence: {confidence}</div>
      <h1>Indoor Localization Upload Demo</h1>
      <p>Bu sahifada yangi query rasm yuklaysiz, tizim eng yaqin reference rasmni topadi, feature-match chizadi va logical corridor view ichida joyini ko‘rsatadi.</p>
      <form class="upload-row" action="/upload" method="post" enctype="multipart/form-data">
        <input class="file-input" type="file" name="query_image" accept=".jpg,.jpeg,.png" required />
        <button class="btn" type="submit">Upload And Localize</button>
      </form>
      <p class="hint">Hozirgi query: <strong>{query_name}</strong>. Yangi rasm tanlasangiz shu sahifa yangilanadi.</p>
      {ambiguous_note}
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Localization Result</h2>
        <div class="meta">
          <div class="card"><strong>Best Reference</strong><span>{best_reference}</span></div>
          <div class="card"><strong>Confidence</strong><span>{confidence}</span></div>
          <div class="card"><strong>2D-3D Matches</strong><span>{result['num_correspondences']}</span></div>
          <div class="card"><strong>World Position</strong><span>x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}</span></div>
          <div class="card"><strong>PnP Inliers</strong><span>{result['num_inliers']}</span></div>
        </div>
        <img src="/web_runtime/results/current_match_vis.png?v={cache_bust}" alt="match visualization" />
        <p class="hint">Yuqoridagi rasm query va topilgan reference orasidagi feature match’larni ko‘rsatadi. Agar qizil/yashil/ko‘k chiziqlar mantiqan bir xil strukturalarni bog‘lasa, retrieval to‘g‘ri ishlayapti degani.</p>
      </div>

      <div class="panel side">
        <div class="dual">
          <div>
            <h3>Query Image</h3>
            <img src="/{query_rel_path}?v={cache_bust}" alt="query image" />
          </div>
          <div>
            <h3>Best Reference</h3>
            <img src="/ref_images/{best_reference}?v={cache_bust}" alt="best reference image" />
          </div>
        </div>
        <div>
          <h3>Logical Corridor View</h3>
          <img src="/web_runtime/results/current_pose_plot.png?v={cache_bust}" alt="pose plot" />
        </div>
        <div>
          <h3>Top-3 Reference Matches</h3>
          <ol>{top_rows}</ol>
        </div>
        <p class="hint">Interaktiv nuqtali eski view ham saqlanadi: <a href="/colmap_pose_view.html" target="_blank" rel="noreferrer">open interactive pose view</a>.</p>
      </div>
    </section>
  </div>
</body>
</html>
"""
