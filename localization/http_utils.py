"""Multipart va fayl xavfsizligi."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


def ensure_web_runtime_dirs(upload_dir: Path, results_dir: Path) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)


def is_within_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def parse_multipart_file(content_type: str, body: bytes, field_name: str) -> Tuple[str, bytes]:
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        raise ValueError("multipart boundary topilmadi")

    boundary = boundary_match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    parts = body.split(delimiter)

    for part in parts:
        part = part.lstrip(b"\r\n")
        if not part or part == b"--\r\n" or part == b"--":
            continue

        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue

        header_text = header_blob.decode("utf-8", errors="replace")
        disposition_line = ""
        for line in header_text.splitlines():
            if line.lower().startswith("content-disposition:"):
                disposition_line = line
                break
        if not disposition_line:
            continue

        field_match = re.search(r'name="([^"]+)"', disposition_line, re.IGNORECASE)
        filename_match = re.search(r'filename="([^"]*)"', disposition_line, re.IGNORECASE)
        if not field_match:
            continue

        current_field = field_match.group(1)
        filename = (filename_match.group(1) if filename_match else "") or "upload.jpg"
        if current_field != field_name:
            continue

        if content.endswith(b"\r\n"):
            content = content[:-2]
        if content.endswith(b"--"):
            content = content[:-2]
        return filename, content

    raise ValueError(f"{field_name} field topilmadi")
