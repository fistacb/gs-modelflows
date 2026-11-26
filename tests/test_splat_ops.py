from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "gs-modeling" / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    import splat_ops
    import check_candidate
    import make_sheet

FIELDS = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def make_body(n=20000, lean_deg=0.0, seed=0):
    """Upright capsule-ish body of height 1, optionally sheared to lean back (+Y)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    r = 0.08 + 0.04 * np.sin(np.pi * z)
    x, y = r * np.cos(ang), r * np.sin(ang)
    y = y + np.tan(np.radians(lean_deg)) * z
    data = np.zeros(n, dtype=[(k, "f4") for k in FIELDS])
    data["x"], data["y"], data["z"] = x, y, z
    data["opacity"] = 5.0
    data["scale_0"] = data["scale_1"] = data["scale_2"] = np.log(0.004)
    data["rot_0"] = 1.0
    return data


class PostureTests(unittest.TestCase):
    def test_lean_sign_and_size(self):
        for lean in (0.0, 15.0, -12.0):
            got = splat_ops.trunk_lean_deg(splat_ops.opaque_xyz(make_body(lean_deg=lean)))
            self.assertAlmostEqual(got, lean, delta=1.5)

    def test_flat_soles(self):
        self.assertLess(abs(splat_ops.sole_tilt_deg(splat_ops.xyz_of(make_body()))), 3.0)


class TransformTests(unittest.TestCase):
    def test_stretch_updates_scale_and_identity_is_noop(self):
        body = make_body(n=3000)
        same, rep = splat_ops.transform_gaussians(body, lambda p: p.copy())
        self.assertEqual(rep["moved"], 0)
        stretched, rep = splat_ops.transform_gaussians(body, lambda p: p * np.array([1.0, 1.0, 2.0]))
        self.assertEqual(rep["moved"], len(body) - int((body["z"] == 0).sum()))
        moved = stretched["z"] != 0
        largest = np.exp(np.max(np.column_stack([stretched[f"scale_{i}"][moved] for i in range(3)]), axis=1))
        self.assertTrue(np.allclose(largest, 0.008, rtol=1e-3))

    def test_fold_is_rejected(self):
        body = make_body(n=3000)
        fold = lambda p: np.column_stack([p[:, 0], p[:, 1], 0.5 - np.abs(p[:, 2] - 0.5)])
        with self.assertRaises(ValueError):
            splat_ops.transform_gaussians(body, fold)


class CandidateTests(unittest.TestCase):
    def test_lean_fails_and_protected_move_is_reported(self):
        ref = make_body()
        _, fails = check_candidate.evaluate(make_body(lean_deg=20.0), ref)
        self.assertTrue(any(f.startswith("lean=") for f in fails))
        cand = ref.copy()
        soles = cand["z"] < 0.03
        cand["x"][soles] += 0.01
        boxes = [{"name": "soles", "min": [-1, -1, -0.01], "max": [1, 1, 0.03]}]
        metrics, fails = check_candidate.evaluate(cand, ref, boxes)
        self.assertAlmostEqual(metrics["protect"]["max"], 0.01, places=4)
        self.assertTrue(any(f.startswith("protected_moved") for f in fails))
        _, fails = check_candidate.evaluate(ref.copy(), ref, boxes)
        self.assertEqual(fails, [])

    def test_oversize_splats_are_counted(self):
        body = make_body()
        body["scale_0"][:100] = np.log(0.2)
        self.assertGreaterEqual(splat_ops.oversize_splats(body)["count"], 100)


class SheetTests(unittest.TestCase):
    def test_sheet_is_downscaled_and_labels_parse(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                p = Path(tmp) / f"v{i}.png"
                Image.new("RGB", (1024, 1024), (40 * i, 80, 120)).save(p)
                paths.append(str(p))
            items = [make_sheet.parse_item(f"{p}:view {i}") for i, p in enumerate(paths)]
            self.assertEqual(items[0], (paths[0], "view 0"))
            self.assertEqual(make_sheet.parse_item(paths[1])[0], paths[1])
            sheet = make_sheet.build_sheet(items, cols=3, long_edge=900)
            self.assertLessEqual(max(sheet.size), 900)


if __name__ == "__main__":
    unittest.main()
