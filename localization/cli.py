"""CLI va web server ishga tushirish."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from .colmap_io import load_localization_context
from .config import (
    LOCALIZATION_SUMMARY_TXT,
    MATCH_VIS_IMAGE,
    POSE_HTML,
    POSE_PLOT,
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
from .http_server import LocalizerHTTPServer, LocalizerRequestHandler
from .http_utils import ensure_web_runtime_dirs
from .pipeline import run_localization_pipeline
from .viz_html import build_interactive_pose_html


def run_cli() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
    global_cache = ReferenceGlobalCache()
    context = load_localization_context(verbose=True)
    result = run_localization_pipeline(
        context,
        QUERY_IMAGE,
        POSE_PLOT,
        MATCH_VIS_IMAGE,
        LOCALIZATION_SUMMARY_TXT,
        verbose=True,
        global_cache=global_cache,
    )
    build_interactive_pose_html(
        context.images,
        result["camera_center"],
        result["best_reference_name"],
        result["mode"],
        POSE_HTML,
        query_image_path="query/test.jpg",
    )
    print(f"Interactive pose view saqlandi: {POSE_HTML}")


def run_server(host: str, port: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
    global_cache = ReferenceGlobalCache()
    ensure_web_runtime_dirs(WEB_UPLOAD_DIR, WEB_RESULTS_DIR)
    context = load_localization_context(verbose=True)

    default_query_path = QUERY_IMAGE
    if default_query_path.exists():
        image = cv2.imread(str(default_query_path), cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imwrite(str(WEB_QUERY_IMAGE), image)
            query_path = WEB_QUERY_IMAGE
        else:
            query_path = default_query_path
    else:
        raise FileNotFoundError(f"Default query image topilmadi: {default_query_path}")

    result = run_localization_pipeline(
        context,
        query_path,
        WEB_POSE_PLOT,
        WEB_MATCH_VIS,
        WEB_SUMMARY_TXT,
        verbose=True,
        global_cache=global_cache,
    )
    build_interactive_pose_html(
        context.images,
        result["camera_center"],
        result["best_reference_name"],
        result["mode"],
        POSE_HTML,
        query_image_path=str(query_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
    )

    initial_state = {
        "query_name": query_path.name,
        "query_rel_path": str(query_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "result": result,
        "updated_at": time.time(),
    }

    server = LocalizerHTTPServer((host, port), LocalizerRequestHandler, context, initial_state, global_cache)
    print(f"\nWeb app tayyor: http://{host}:{port}")
    print("Brauzerda ochib query rasm yuklang.")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indoor localization — o'xshash reference retrieval + COLMAP PnP")
    parser.add_argument("--serve", action="store_true", help="Local upload web app ni ishga tushirish")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
