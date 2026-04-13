"""COLMAP model/database readers and reference metadata helpers.

This module intentionally keeps IO boring and explicit:
- coords.txt maps image_name -> logical (x, y)
- COLMAP *.bin files provide camera/image/point3D data
- database.db provides the exact COLMAP SIFT keypoints/descriptors used by
  registered reference images
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Camera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


@dataclass
class ReferenceFeatures:
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    source: str = "colmap_database"


@dataclass
class ReferenceMeta:
    image_name: str
    image_path: Path
    station_id: Optional[int]
    view_label: str
    logical_xy: np.ndarray


def natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_station_and_view(image_name: str) -> Tuple[Optional[int], str]:
    """Parse names like img_09_front.jpg -> (9, "front")."""
    stem = Path(image_name).stem
    match = re.search(r"(?:^|_)img[_-]?(\d+)(?:[_-]([A-Za-z0-9]+))?", stem, re.IGNORECASE)
    if match:
        station_id = int(match.group(1))
        view = (match.group(2) or "front").lower()
        return station_id, view

    generic = re.search(r"(\d+)", stem)
    station_id = int(generic.group(1)) if generic else None
    lowered = stem.lower()
    if "left" in lowered:
        return station_id, "left"
    if "right" in lowered:
        return station_id, "right"
    if "back" in lowered:
        return station_id, "back"
    return station_id, "front"


def find_coords_path(project_dir: Path) -> Optional[Path]:
    for name in ("coords.txt", "coords.text"):
        path = project_dir / name
        if path.exists():
            return path
    return None


def read_coords_file(path: Optional[Path]) -> Dict[str, np.ndarray]:
    """Read whitespace or comma separated rows: image_name x y."""
    if path is None or not path.exists():
        return {}

    coords: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = re.split(r"[\s,]+", line)
            if len(tokens) < 3:
                raise ValueError(f"Bad coords row at {path}:{line_no}: {raw_line.rstrip()}")
            name = Path(tokens[0]).name
            try:
                x = float(tokens[1])
                y = float(tokens[2])
            except ValueError as exc:
                raise ValueError(f"Bad numeric coords at {path}:{line_no}: {raw_line.rstrip()}") from exc
            coords[name] = np.array([x, y], dtype=np.float64)
    return coords


def infer_logical_xy(image_name: str) -> np.ndarray:
    station_id, view = parse_station_and_view(image_name)
    y_offsets = {"left": -0.25, "front": 0.0, "right": 0.25, "back": 0.0}
    x = float(station_id or 0)
    y = y_offsets.get(view, 0.0)
    return np.array([x, y], dtype=np.float64)


def iter_reference_images(ref_dir: Path) -> Iterable[Path]:
    if not ref_dir.exists():
        return []
    return sorted(
        [p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )


def build_reference_metadata(ref_dir: Path, coords_path: Optional[Path]) -> Dict[str, ReferenceMeta]:
    coords = read_coords_file(coords_path)
    metadata: Dict[str, ReferenceMeta] = {}
    for image_path in iter_reference_images(ref_dir):
        station_id, view_label = parse_station_and_view(image_path.name)
        logical_xy = coords.get(image_path.name, infer_logical_xy(image_path.name))
        metadata[image_path.name] = ReferenceMeta(
            image_name=image_path.name,
            image_path=image_path,
            station_id=station_id,
            view_label=view_label,
            logical_xy=np.asarray(logical_xy, dtype=np.float64),
        )
    return metadata


def read_next_bytes(fid, num_bytes: int, format_sequence: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + format_sequence, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec.astype(np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_center_from_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec)
    return -rotation.T @ tvec


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model_name, int(width), int(height), params)
    return cameras


def read_images_binary(path: Path) -> Dict[int, ImageRecord]:
    images: Dict[int, ImageRecord] = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = int(props[0])
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = int(props[8])

            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                if char == b"":
                    raise EOFError("Unexpected EOF while reading image name")
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            raw_points = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack(
                [
                    np.array(raw_points[0::3], dtype=np.float64),
                    np.array(raw_points[1::3], dtype=np.float64),
                ]
            )
            point3D_ids = np.array(raw_points[2::3], dtype=np.int64)
            images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    points3D: Dict[int, Point3D] = {}
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = int(props[0])
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)
            points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error)
    return points3D


def load_colmap_model(model_dir: Path) -> Tuple[Dict[int, Camera], Dict[int, ImageRecord], Dict[int, Point3D]]:
    cameras_path = model_dir / "cameras.bin"
    images_path = model_dir / "images.bin"
    points_path = model_dir / "points3D.bin"
    for path in (cameras_path, images_path, points_path):
        if not path.exists():
            raise FileNotFoundError(f"COLMAP model file missing: {path}")
    return read_cameras_binary(cameras_path), read_images_binary(images_path), read_points3d_binary(points_path)


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    desc = descriptors.astype(np.float32)
    desc /= desc.sum(axis=1, keepdims=True) + 1e-12
    desc = np.sqrt(desc)
    return desc.astype(np.float32)


def _blob_to_array(blob: bytes, dtype, shape: Tuple[int, int]) -> np.ndarray:
    array = np.frombuffer(blob, dtype=dtype)
    return array.reshape(shape)


def load_reference_features(database_path: Path) -> Dict[str, ReferenceFeatures]:
    """Load COLMAP feature rows and convert descriptors to RootSIFT*255."""
    if not database_path.exists():
        raise FileNotFoundError(f"COLMAP database missing: {database_path}")

    conn = sqlite3.connect(str(database_path))
    try:
        cursor = conn.cursor()
        image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
        features: Dict[str, ReferenceFeatures] = {}
        for image_id, name in image_rows:
            kp_row = cursor.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            desc_row = cursor.execute(
                "SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            if kp_row is None or desc_row is None:
                continue

            kp_rows, kp_cols, kp_blob = kp_row
            desc_rows, desc_cols, desc_blob = desc_row
            if kp_rows == 0 or desc_rows == 0:
                continue

            keypoints = _blob_to_array(kp_blob, np.float32, (kp_rows, kp_cols))[:, :2].copy()
            raw_descriptors = _blob_to_array(desc_blob, np.uint8, (desc_rows, desc_cols))
            count = min(len(keypoints), len(raw_descriptors))
            descriptors = (rootsift(raw_descriptors[:count]) * 255.0).astype(np.float32)
            keypoints = keypoints[:count].astype(np.float32)

            clean_name = Path(name).name
            features[clean_name] = ReferenceFeatures(keypoints, descriptors, "colmap_database")
            if clean_name != name:
                features[name] = features[clean_name]
        return features
    finally:
        conn.close()


def camera_to_intrinsics(camera: Camera, image_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    params = camera.params
    model = camera.model_name
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = [float(x) for x in params[:4]]
    else:
        raise ValueError(f"Unsupported camera model for intrinsics: {model}")

    if image_shape is not None:
        h, w = image_shape[:2]
        sx = float(w) / float(camera.width)
        sy = float(h) / float(camera.height)
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        dist = np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL":
        dist = np.array([params[3], params[4], 0.0, 0.0], dtype=np.float64)
    elif model == "OPENCV":
        dist = np.array(params[4:8], dtype=np.float64)
    elif model == "FULL_OPENCV":
        dist = np.array(params[4:12], dtype=np.float64)
    else:
        dist = np.zeros((4,), dtype=np.float64)
    return K, dist
