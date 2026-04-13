"""Oddiy upload web server."""

from __future__ import annotations

import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .colmap_io import LocalizationContext
from .config import (
    MAX_UPLOAD_BYTES,
    POSE_HTML,
    PROJECT_DIR,
    QUERY_IMAGE,
    WEB_MATCH_VIS,
    WEB_POSE_PLOT,
    WEB_QUERY_IMAGE,
    WEB_RESULTS_DIR,
    WEB_SUMMARY_TXT,
    WEB_UPLOAD_DIR,
)
from .global_retrieval import ReferenceGlobalCache
from .http_utils import ensure_web_runtime_dirs, is_within_path, parse_multipart_file
from .pipeline import run_localization_pipeline
from .viz_html import build_interactive_pose_html, render_localizer_web_html


class LocalizerHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        context: LocalizationContext,
        initial_state: dict,
        global_cache: ReferenceGlobalCache,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.context = context
        self.latest_state = initial_state
        self.global_cache = global_cache


class LocalizerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            html_text = render_localizer_web_html(self.server.latest_state)
            payload = html_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.serve_static_file(route)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self.send_error(404, "Unknown route")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_error(400, "Upload body bo'sh")
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_error(413, f"Fayl juda katta. Limit: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            return
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_length)

        try:
            filename, raw_bytes = parse_multipart_file(content_type, body, "query_image")
        except ValueError as exc:
            self.send_error(400, str(exc))
            return

        file_ext = Path(filename or "upload.jpg").suffix.lower()
        if file_ext not in {".jpg", ".jpeg", ".png"}:
            file_ext = ".jpg"

        ensure_web_runtime_dirs(WEB_UPLOAD_DIR, WEB_RESULTS_DIR)
        image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded is None:
            self.send_error(400, "Yuklangan fayl rasm emas yoki ochilmadi")
            return

        if file_ext == ".png":
            cv2.imwrite(str(WEB_QUERY_IMAGE.with_suffix(".png")), decoded)
            if WEB_QUERY_IMAGE.exists():
                WEB_QUERY_IMAGE.unlink()
            saved_query_path = WEB_QUERY_IMAGE.with_suffix(".png")
        else:
            cv2.imwrite(str(WEB_QUERY_IMAGE), decoded)
            png_path = WEB_QUERY_IMAGE.with_suffix(".png")
            if png_path.exists():
                png_path.unlink()
            saved_query_path = WEB_QUERY_IMAGE

        result = run_localization_pipeline(
            self.server.context,
            saved_query_path,
            WEB_POSE_PLOT,
            WEB_MATCH_VIS,
            WEB_SUMMARY_TXT,
            verbose=False,
            global_cache=self.server.global_cache,
        )
        build_interactive_pose_html(
            self.server.context.images,
            result["camera_center"],
            result["best_reference_name"],
            result["mode"],
            POSE_HTML,
            query_image_path=str(saved_query_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
        )
        self.server.latest_state = {
            "query_name": Path(filename or saved_query_path.name).name,
            "query_rel_path": str(saved_query_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "result": result,
            "updated_at": time.time(),
        }

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def serve_static_file(self, route: str) -> None:
        relative = route.lstrip("/")
        target = (PROJECT_DIR / relative).resolve()
        if not is_within_path(target, PROJECT_DIR) or not target.exists() or not target.is_file():
            self.send_error(404, "Fayl topilmadi")
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return
