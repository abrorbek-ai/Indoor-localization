"""COLMAP binary model va SQLite database o'qish."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import DATABASE_PATH_ENV, DEFAULT_WORKSPACE_DIR, MODEL_DIR_ENV, PROJECT_DIR
from .layout_utils import natural_sort_key

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
    keypoints: np.ndarray
    descriptors: np.ndarray


@dataclass
class LocalizationContext:
    model_dir: Path
    database_path: Path
    cameras: Dict[int, Camera]
    images: Dict[int, ImageRecord]
    points3D: Dict[int, Point3D]
    reference_features: Dict[str, ReferenceFeatures]


def find_model_dir() -> Path:
    model_dir = Path(MODEL_DIR_ENV)
    if (model_dir / "cameras.bin").exists():
        return model_dir

    candidates = [
        DEFAULT_WORKSPACE_DIR / "sparse" / "0",
        DEFAULT_WORKSPACE_DIR / "sparse",
        PROJECT_DIR / "sparse" / "0",
        PROJECT_DIR / "sparse",
    ]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists():
            return candidate

    raise FileNotFoundError(
        "COLMAP model topilmadi. "
        "Project ichida `colmap_workspace/sparse/0` yarating yoki "
        "COLMAP_MODEL_DIR env orqali model papkani ko'rsating."
    )


def find_database_path() -> Path:
    db_path = Path(DATABASE_PATH_ENV)
    if db_path.exists():
        return db_path

    candidates = [
        DEFAULT_WORKSPACE_DIR / "database.db",
        PROJECT_DIR / "database.db",
        PROJECT_DIR / "colmap.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "COLMAP database topilmadi. "
        "Project ichida `colmap_workspace/database.db` yarating yoki "
        "COLMAP_DATABASE_PATH env bilan database.db yo'lini bering."
    )


def read_next_bytes(fid, num_bytes: int, format_sequence: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_sequence, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
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
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_images_binary(path: Path) -> Dict[int, ImageRecord]:
    images = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name_chars = []
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_chars.append(char.decode("utf-8"))
            name = "".join(name_chars)

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            raw_points = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack(
                [
                    np.array(raw_points[0::3], dtype=np.float64),
                    np.array(raw_points[1::3], dtype=np.float64),
                ]
            )
            point3D_ids = np.array(raw_points[2::3], dtype=np.int64)

            images[image_id] = ImageRecord(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    points3D = {}
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])

            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)

            points3D[point3D_id] = Point3D(
                point3D_id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
            )
    return points3D


def load_colmap_model(model_dir: Path):
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    points3D = read_points3d_binary(model_dir / "points3D.bin")
    return cameras, images, points3D


def camera_to_intrinsics(camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    params = camera.params
    model = camera.model_name

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Hoziroq qo'llab-quvvatlanmagan camera model: {model}")

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    if model == "SIMPLE_RADIAL":
        dist = np.array([params[3], 0, 0, 0], dtype=np.float64)
    elif model == "RADIAL":
        dist = np.array([params[3], params[4], 0, 0], dtype=np.float64)
    elif model == "OPENCV":
        dist = np.array(params[4:8], dtype=np.float64)
    elif model == "FULL_OPENCV":
        dist = np.array(params[4:12], dtype=np.float64)
    else:
        dist = np.zeros((4,), dtype=np.float64)

    return K, dist


def blob_to_array(blob: bytes, dtype, shape: Tuple[int, int]) -> np.ndarray:
    array = np.frombuffer(blob, dtype=dtype)
    return array.reshape(shape)


def to_rootsift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = descriptors.astype(np.float32)
    row_sums = descriptors.sum(axis=1, keepdims=True) + 1e-12
    descriptors = descriptors / row_sums
    descriptors = np.sqrt(descriptors)
    return descriptors


def load_reference_features(database_path: Path) -> Dict[str, ReferenceFeatures]:
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
    features = {}

    for image_id, name in image_rows:
        keypoints_row = cursor.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        descriptors_row = cursor.execute(
            "SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
            (image_id,),
        ).fetchone()

        if keypoints_row is None or descriptors_row is None:
            continue

        kp_rows, kp_cols, kp_blob = keypoints_row
        desc_rows, desc_cols, desc_blob = descriptors_row

        if kp_rows == 0 or desc_rows == 0:
            continue

        keypoints = blob_to_array(kp_blob, np.float32, (kp_rows, kp_cols))
        raw_descriptors = blob_to_array(desc_blob, np.uint8, (desc_rows, desc_cols))
        # Query descriptorlari RootSIFT formatida. Reference descriptorlarni ham
        # shu formatga keltirmasak, PnP matching raw SIFT vs RootSIFT bo'lib qoladi.
        descriptors = (to_rootsift(raw_descriptors) * 255.0).astype(np.float32)
        features[name] = ReferenceFeatures(keypoints=keypoints, descriptors=descriptors)

    conn.close()
    return features


def save_reference_camera_centers(images: Dict[int, ImageRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for image in sorted(images.values(), key=lambda item: natural_sort_key(item.name)):
            center = camera_center_from_pose(image.qvec, image.tvec)
            f.write(f"{image.name} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")


def print_colmap_file_explanation() -> None:
    print("\nCOLMAP fayllari:")
    print("1. cameras.bin  -> kamera intrinsics: model turi, width/height, fx/fy/cx/cy va distortion")
    print("2. images.bin   -> har rasm uchun pose: quaternion (R), translation (t), camera_id, 2D keypoints")
    print("3. points3D.bin -> sparse point cloud: har 3D nuqta koordinatasi va kuzatilgan track'lar")


def load_localization_context(verbose: bool = True) -> LocalizationContext:
    from .config import CAMERA_CENTERS_TXT

    model_dir = find_model_dir()
    database_path = find_database_path()

    if verbose:
        print(f"COLMAP model: {model_dir}")
        print(f"COLMAP database: {database_path}")
        print_colmap_file_explanation()

    cameras, images, points3D = load_colmap_model(model_dir)
    reference_features = load_reference_features(database_path)

    if verbose:
        save_reference_camera_centers(images, CAMERA_CENTERS_TXT)
        print("\nModel summary:")
        print(f"Cameras: {len(cameras)}")
        print(f"Registered images: {len(images)}")
        print(f"3D points: {len(points3D)}")
        print(f"Reference camera centers saqlandi: {CAMERA_CENTERS_TXT}")

        print("\nReference image pose misollari:")
        for image in list(sorted(images.values(), key=lambda item: natural_sort_key(item.name)))[:5]:
            center = camera_center_from_pose(image.qvec, image.tvec)
            print(f"{image.name}: x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}")

        print("\nCOLMAP database dan keypoint/descriptor o'qilmoqda...")
        print(f"Database features topilgan rasmlar soni: {len(reference_features)}")

    return LocalizationContext(
        model_dir=model_dir,
        database_path=database_path,
        cameras=cameras,
        images=images,
        points3D=points3D,
        reference_features=reference_features,
    )
