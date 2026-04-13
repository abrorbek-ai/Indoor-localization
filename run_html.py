"""Small local web UI for upload-based localization.

Run:
    python run_html.py --serve
"""

from __future__ import annotations

import argparse
import errno
import html
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import cv2
import numpy as np

from localization import PipelineConfig, load_localization_context, localize_query_image
from visualize import write_all_outputs


PROJECT_DIR = Path(__file__).resolve().parent
WEB_RUNTIME_DIR = PROJECT_DIR / "web_runtime"
UPLOAD_DIR = WEB_RUNTIME_DIR / "uploads"
RESULTS_DIR = WEB_RUNTIME_DIR / "results"
UPLOAD_PATH = UPLOAD_DIR / "current_query.jpg"
SUMMARY_PATH = RESULTS_DIR / "current_summary.txt"
MATCH_PNG = RESULTS_DIR / "current_match_vis.png"
POSE_PNG = RESULTS_DIR / "current_pose_plot.png"
PNP_INLIERS_PNG = RESULTS_DIR / "current_pnp_inliers.png"
HTML_REPORT = RESULTS_DIR / "current_report.html"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_within_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def parse_multipart_file(content_type: str, body: bytes, field_name: str) -> tuple[str, bytes]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary")
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    if not boundary:
        raise ValueError("Empty multipart boundary")

    boundary_bytes = ("--" + boundary).encode("utf-8")
    for part in body.split(boundary_bytes):
        if not part or part in {b"--\r\n", b"--"}:
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_blob, payload = part.split(b"\r\n\r\n", 1)
        headers = header_blob.decode("utf-8", errors="ignore")
        if f'name="{field_name}"' not in headers:
            continue
        filename = "upload.jpg"
        if "filename=" in headers:
            filename = headers.split("filename=", 1)[1].split("\r\n", 1)[0].strip().strip('"')
        return Path(filename).name, payload.rstrip(b"\r\n-")
    raise ValueError(f"Upload field not found: {field_name}")


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def render_page(state: dict) -> str:
    result = state.get("result")
    message = html.escape(state.get("message", "Upload a corridor query image."))
    result_block = ""
    if result:
        result_block = f"""
        <section>
          <h2>Result</h2>
          <p><strong>mode:</strong> {html.escape(result['mode'])}</p>
          <p><strong>best reference:</strong> {html.escape(result['best_reference_name'])}</p>
          <p><strong>station:</strong> {html.escape(str(result['best_station_id']))}</p>
          <p><strong>logical xy:</strong> ({float(result['best_logical_xy'][0]):.3f}, {float(result['best_logical_xy'][1]):.3f})</p>
          <div class="grid">
            <a href="/{html.escape(rel(UPLOAD_PATH))}"><img src="/{html.escape(rel(UPLOAD_PATH))}" alt="uploaded query"></a>
            <a href="/{html.escape(rel(MATCH_PNG))}"><img src="/{html.escape(rel(MATCH_PNG))}" alt="verified matches"></a>
            <a href="/{html.escape(rel(POSE_PNG))}"><img src="/{html.escape(rel(POSE_PNG))}" alt="logical pose"></a>
            <a href="/{html.escape(rel(PNP_INLIERS_PNG))}"><img src="/{html.escape(rel(PNP_INLIERS_PNG))}" alt="PnP inliers"></a>
          </div>
          <p><a href="/{html.escape(rel(SUMMARY_PATH))}">summary txt</a> | <a href="/{html.escape(rel(HTML_REPORT))}">html report</a></p>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Indoor Localization MVP</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    form {{ display: flex; gap: 10px; align-items: center; margin: 18px 0; }}
    button {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #52606d; background: #f2f5f8; cursor: pointer; }}
    img {{ width: 100%; border: 1px solid #d6dde5; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .note {{ padding: 10px 12px; background: #f2f5f8; border: 1px solid #d6dde5; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Indoor Localization MVP</h1>
  <p class="note">{message}</p>
  <form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="query_image" accept="image/*" required>
    <button type="submit">Localize</button>
  </form>
  {result_block}
</body>
</html>
"""


class LocalizerServer(ThreadingHTTPServer):
    def __init__(self, address, handler, context):
        super().__init__(address, handler)
        self.context = context
        self.state = {"message": "Upload a query image. Max size: 12 MB."}


def bind_server_with_fallback(host: str, port: int, context, max_tries: int = 10) -> tuple[LocalizerServer, int]:
    last_error = None
    for candidate_port in range(port, port + max_tries):
        try:
            return LocalizerServer((host, candidate_port), Handler, context), candidate_port
        except PermissionError as exc:
            raise SystemExit(
                "Could not start the local web server because this environment does not allow "
                f"binding to {host}:{candidate_port}.\n"
                "If you are running inside a restricted sandbox, start it from your normal terminal instead:\n"
                f"  .venv/bin/python run_html.py --serve --host {host} --port {candidate_port}"
            ) from exc
        except OSError as exc:
            last_error = exc
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
    raise SystemExit(
        f"Could not start the web server. Ports {port}-{port + max_tries - 1} look unavailable.\n"
        "Try another port, for example:\n"
        f"  .venv/bin/python run_html.py --serve --host {host} --port {port + max_tries}"
    ) from last_error


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = render_page(self.server.state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return
        target = (PROJECT_DIR / parsed.path.lstrip("/")).resolve()
        if not is_within_path(target, PROJECT_DIR) or not target.exists() or not target.is_file():
            self.send_error(404, "File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = render_page(self.server.state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self.send_error(404, "Unknown route")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_error(400, "Empty upload")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload too large")
            return

        body = self.rfile.read(length)
        try:
            filename, raw_bytes = parse_multipart_file(self.headers.get("Content-Type", ""), body, "query_image")
        except ValueError as exc:
            self.send_error(400, str(exc))
            return

        decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            self.send_error(400, "Uploaded file is not a readable image")
            return

        ensure_dirs()
        cv2.imwrite(str(UPLOAD_PATH), decoded)

        try:
            result = localize_query_image(self.server.context, UPLOAD_PATH, verbose=False)
            write_all_outputs(
                self.server.context,
                result,
                summary_path=SUMMARY_PATH,
                match_png=MATCH_PNG,
                pose_png=POSE_PNG,
                pnp_inliers_png=PNP_INLIERS_PNG,
                html_path=HTML_REPORT,
            )
            self.server.state = {
                "message": f"Localized {Path(filename).name} at {time.strftime('%H:%M:%S')}.",
                "result": result,
            }
        except Exception as exc:  # noqa: BLE001 - local demo server should show useful errors
            self.server.state = {"message": f"Localization failed: {exc}"}

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def serve_static(self, route: str) -> None:
        target = (PROJECT_DIR / route.lstrip("/")).resolve()
        if not is_within_path(target, PROJECT_DIR) or not target.exists() or not target.is_file():
            self.send_error(404, "File not found")
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the indoor localization upload UI.")
    parser.add_argument("--serve", action="store_true", help="Start the local HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-pnp", action="store_true", help="Disable optional PnP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.serve:
        print("Use: python run_html.py --serve")
        return

    ensure_dirs()
    config = PipelineConfig(
        project_dir=PROJECT_DIR,
        pnp_enabled=not args.no_pnp,
        output_summary=SUMMARY_PATH,
        output_match_png=MATCH_PNG,
        output_pose_png=POSE_PNG,
        output_pnp_inliers_png=PNP_INLIERS_PNG,
        output_html=HTML_REPORT,
    ).resolved()
    context = load_localization_context(config, verbose=True)
    server, actual_port = bind_server_with_fallback(args.host, args.port, context)
    print(f"Serving on http://{args.host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
