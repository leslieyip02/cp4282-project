"""Load NerfBaselines Mip-NeRF 360 scenes from their COLMAP binary reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from shared.gaussian_set import GaussianSet


def _read(stream, pattern):
    fmt = "<" + pattern
    data = stream.read(struct.calcsize(fmt))
    if len(data) != struct.calcsize(fmt):
        raise ValueError("Truncated COLMAP binary reconstruction")
    return struct.unpack(fmt, data)


def _rotation(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ], np.float64)


def _read_cameras(path):
    cameras = {}
    with path.open("rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            camera_id, model_id, width, height = _read(stream, "iiQQ")
            if model_id != 1:
                raise ValueError(
                    f"COLMAP camera {camera_id} uses model {model_id}; only undistorted "
                    "PINHOLE cameras are supported. Undistort the scene first."
                )
            intrinsics = np.asarray(_read(stream, "4d"), np.float64)
            cameras[camera_id] = (width, height, intrinsics)
    return cameras


def _read_images(path):
    images = {}
    with path.open("rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            image_id = _read(stream, "i")[0]
            q = _read(stream, "4d")
            t = _read(stream, "3d")
            camera_id = _read(stream, "i")[0]
            name_bytes = bytearray()
            while (char := stream.read(1)) != b"\0":
                if not char:
                    raise ValueError("Truncated COLMAP image name")
                name_bytes.extend(char)
            name = name_bytes.decode("utf8")
            points2d = _read(stream, "Q")[0]
            stream.seek(points2d * 24, 1)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = _rotation(q)
            pose[:3, 3] = t
            if name in images:
                raise ValueError(f"Duplicate COLMAP image name: {name}")
            images[name] = (image_id, camera_id, pose)
    return images


def _read_points(path):
    xyz, rgb = [], []
    with path.open("rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            _read(stream, "Q")  # point ID
            xyz.append(_read(stream, "3d"))
            rgb.append(_read(stream, "3B"))
            _read(stream, "d")  # reprojection error
            stream.seek(_read(stream, "Q")[0] * 8, 1)  # image ID, 2D point index
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32) / 255.0


class RealScene:
    def __init__(self, root: Path, validation_views: int = 8):
        self.root = Path(root)
        info = json.loads((self.root / "nb-info.json").read_text())
        if info.get("id") != "mipnerf360" or info.get("loader") != "colmap":
            raise ValueError(f"{root} is not a NerfBaselines Mip-NeRF 360 COLMAP scene")
        kwargs = info["loader_kwargs"]
        self.image_root = self.root / kwargs["images_path"]
        sparse = self.root / kwargs["colmap_path"]
        self.cameras = _read_cameras(sparse / "cameras.bin")
        self.images = _read_images(sparse / "images.bin")
        centers = np.stack([-pose[:3, :3].T @ pose[:3, 3]
                            for _, _, pose in self.images.values()])
        self.scene_radius = float(1.1 * np.max(np.linalg.norm(
            centers - centers.mean(axis=0), axis=1,
        )))
        self.points_path = sparse / "points3D.bin"
        official_train = self._names("train_list.txt")
        self.test_names = self._names("test_list.txt")
        if set(official_train) & set(self.test_names):
            raise ValueError("COLMAP train and test lists overlap")
        if len(official_train) <= validation_views:
            raise ValueError("Not enough training images to reserve validation views")
        validation_ids = np.linspace(0, len(official_train) - 1, validation_views, dtype=int)
        self.validation_names = [official_train[i] for i in validation_ids]
        validation_set = set(self.validation_names)
        self.train_names = [name for name in official_train if name not in validation_set]

    def _names(self, filename):
        names = (self.root / filename).read_text().splitlines()
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{filename} is empty or contains duplicate images")
        missing = [name for name in names if name not in self.images or not (self.image_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{filename} has missing COLMAP images: {missing[:3]}")
        return names

    def load_views(self, split: str, width: int, height: int, frame_ids=None):
        names = {"train": self.train_names, "val": self.validation_names,
                 "test": self.test_names}[split]
        if frame_ids is not None:
            names = [names[int(i)] for i in frame_ids]
        targets, poses, intrinsics, distances = [], [], [], []
        for name in names:
            _, camera_id, pose = self.images[name]
            original_width, original_height, intrinsic = self.cameras[camera_id]
            if abs(width / height - original_width / original_height) > 0.01:
                raise ValueError(
                    f"Requested {width}x{height} changes the aspect ratio of {name} "
                    f"({original_width}x{original_height}); use aspect-preserving dimensions."
                )
            with Image.open(self.image_root / name) as image:
                if image.size != (original_width, original_height):
                    raise ValueError(f"{name} size differs from its COLMAP camera")
                target = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), np.float32) / 255.0
            targets.append(target)
            poses.append(pose)
            intrinsic = intrinsic * [width / original_width, height / original_height,
                                     width / original_width, height / original_height]
            intrinsics.append(intrinsic)
            distances.append(np.linalg.norm(-pose[:3, :3].T @ pose[:3, 3]))
        return (np.stack(targets), np.stack(poses), np.asarray(intrinsics, np.float32),
                float(np.mean(distances)), names)

    def initial_gaussians(self, capacity: int) -> GaussianSet:
        xyz, rgb = _read_points(self.points_path)
        if not len(xyz):
            raise ValueError("COLMAP reconstruction has no 3D points")
        rng = np.random.default_rng(0)
        selected = rng.choice(len(xyz), size=min(len(xyz), capacity), replace=False)
        means, colors = xyz[selected], rgb[selected]
        distances = cKDTree(xyz).query(means, k=2, workers=-1)[0][:, 1]
        scales = np.clip(distances, self.scene_radius / 500.0,
                         self.scene_radius / 3.0)
        return GaussianSet(
            means=means, scales=np.repeat(scales[:, None], 3, axis=1).astype(np.float32),
            rotations=np.tile(np.array([1, 0, 0, 0], np.float32), (len(means), 1)),
            opacities=np.full(len(means), 0.1, np.float32), colors=colors,
        )
