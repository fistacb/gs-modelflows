import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "gs-modeling" / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    import step2_fit_skeleton as skeleton


class EyeDepthTests(unittest.TestCase):
    def test_cli_seats_globes_independently_without_moving_the_carve(self):
        radii = np.array([0.020, 0.028, 0.016])
        centers = {"eye_L": [0.04, -0.20, 0.77], "eye_R": [-0.04, -0.21, 0.77]}
        u, v = np.meshgrid(np.linspace(-1.6, 1.6, 55), np.linspace(-2, 1.2, 55))
        u, v = u.ravel(), v.ravel()
        cloud = []
        for center in centers.values():
            y = 0.08 + 0.05 * u - 0.12 * v + 0.10 * u**2
            cloud.append(center + np.column_stack((u, y, v)) * radii)
        cloud = np.concatenate(cloud)
        fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
        data = np.zeros(len(cloud), dtype=[(key, "f4") for key in fields])
        data["x"], data["y"], data["z"] = cloud.T
        for index, color in enumerate((0.90, 0.72, 0.63)):
            data[f"f_dc_{index}"] = (color - 0.5) / skeleton.SH_C0
        data["opacity"] = 4

        def detect(_xyz, _rgb, side, *_args):
            x, y, z = centers[side]
            return {"surface_center": [x, y, z], "eyeball_center": [x, y + 0.020, z],
                    "eyeball_radius": 0.020, "exclusion_radius": radii.tolist(),
                    "carve_center": [x, y + 0.002, z]}

        cfg = {"work_dir": ".", "proportions": {"face_z_range": [0.72, 0.82],
                                                "neck_z_range": [0.68, 0.71]}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "skeleton.json"
            with patch.object(skeleton, "read_ply", return_value=(data, None)), \
                 patch.object(skeleton, "load_config", return_value=cfg), \
                 patch.object(skeleton, "estimate_face_cx", return_value=0.0), \
                 patch.object(skeleton, "detect_eye", side_effect=detect), \
                 patch.object(sys, "argv", ["step2_fit_skeleton.py", "--dst", str(output)]):
                skeleton.main()
            bones = json.loads(output.read_text(encoding="utf-8"))["bones"]

        for side, (x, y, z) in centers.items():
            with self.subTest(side=side):
                eye = bones[side]["meta"]
                self.assertAlmostEqual(eye["surface_center"][1], y + 0.08 * radii[1], delta=2e-5)
                self.assertAlmostEqual(bones[side]["head"][1] - eye["eyeball_radius"], eye["surface_center"][1])
                self.assertEqual(eye["carve_center"], [x, y + 0.002, z])

    def test_sparse_support_cannot_anchor_an_eye(self):
        xyz = np.zeros((10, 3))
        self.assertIsNone(skeleton.fit_eye_surface(xyz, np.ones(10, dtype=bool), np.zeros(3), np.ones(3)))


if __name__ == "__main__":
    unittest.main()
