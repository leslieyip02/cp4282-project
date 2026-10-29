"""Small COLMAP fixture for scene splits, intrinsics, poses, and initialization."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from real_scene import RealScene
from shared.gaussian_set import GaussianSet


class RealSceneTests(unittest.TestCase):
    def test_render_size_preserves_non_bonsai_aspect_ratio(self):
        scene = object.__new__(RealScene)
        scene.cameras = {1: (1297, 840, np.ones(4))}
        self.assertEqual(scene.render_size(390), (390, 253))

    def test_renderer_uses_each_views_principal_point(self):
        trainer_module = importlib.import_module("3dgs_trainer")
        scene = GaussianSet(
            means=np.array([[0, 0, 2]], np.float32),
            scales=np.array([[0.15, 0.15, 0.15]], np.float32),
            rotations=np.array([[1, 0, 0, 0]], np.float32),
            opacities=np.array([0.9], np.float32),
            colors=np.array([[1, 0, 0]], np.float32),
        )
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        intrinsics = np.array([[12, 10, 5, 6], [12, 10, 11, 6]], np.float32)
        trainer = trainer_module.WarpImageTrainer(
            np.zeros((2, 12, 16, 3), np.float32), cameras, intrinsics,
            1.0, 1, 1, "cpu", 0, init_scene=scene, background=(0, 0, 0),
        )
        first, _ = trainer.render_single_view_image(0)
        first = first.copy()
        second, _ = trainer.render_single_view_image(1)
        first_peak = np.unravel_index(np.argmax(first[..., 0]), first.shape[:2])
        second_peak = np.unravel_index(np.argmax(second[..., 0]), second.shape[:2])
        self.assertEqual(second_peak[0], first_peak[0])
        self.assertEqual(second_peak[1] - first_peak[1], 6)

    def test_colmap_scene_loader_and_initial_gaussians(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images_2"
            sparse = root / "sparse_2/0"
            images.mkdir()
            sparse.mkdir(parents=True)
            (root / "nb-info.json").write_text(json.dumps({
                "id": "mipnerf360", "loader": "colmap", "loader_kwargs": {
                    "images_path": "images_2", "colmap_path": "sparse_2/0",
                },
            }))
            with (sparse / "cameras.bin").open("wb") as stream:
                stream.write(struct.pack("<QiiQQ4d", 1, 1, 1, 16, 12, 12, 10, 7, 5))
            names = [f"view_{i:02d}.JPG" for i in range(11)]
            with (sparse / "images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", len(names)))
                for image_id, name in enumerate(names, 1):
                    stream.write(struct.pack("<i4d3di", image_id, 1, 0, 0, 0,
                                             0, 0, float(image_id), 1))
                    stream.write(name.encode() + b"\0")
                    stream.write(struct.pack("<Q", 0))
                    Image.fromarray(np.full((12, 16, 3), image_id, np.uint8)).save(images / name)
            with (sparse / "points3D.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 2))
                for point_id, x in enumerate((0.0, 0.1), 1):
                    stream.write(struct.pack("<Q3d3BdQ", point_id, x, 0, 2,
                                             100, 120, 140, 0, 0))
            (root / "train_list.txt").write_text("\n".join(names[:10]) + "\n")
            (root / "test_list.txt").write_text(names[10] + "\n")

            scene = RealScene(root, validation_views=2)
            self.assertEqual(scene.render_size(40), (40, 30))
            self.assertEqual((len(scene.train_names), len(scene.validation_names),
                              len(scene.test_names)), (8, 2, 1))
            self.assertFalse(set(scene.train_names) & set(scene.validation_names))
            self.assertFalse(set(scene.test_names) & set(scene.train_names))
            targets, poses, intrinsics, _, loaded = scene.load_views("test", 32, 24)
            self.assertEqual(loaded, names[10:])
            self.assertEqual(targets.shape, (1, 24, 32, 3))
            np.testing.assert_allclose(intrinsics[0], [24, 20, 14, 10])
            np.testing.assert_allclose(poses[0][:3, 3], [0, 0, 11])
            initial = scene.initial_gaussians(2)
            self.assertEqual(initial.means.shape, (2, 3))
            self.assertTrue(np.all(initial.scales > 0))
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                scene.load_views("test", 32, 32)


if __name__ == "__main__":
    unittest.main()
