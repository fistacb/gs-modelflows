from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from test_splat_ops import make_body

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "gs-modeling" / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    import splat_ops
    import correct_posture
    import align_rows
    import remove_detached
    import measure_landmarks
    import step0_prep_image
    import step0_matte_comfy


class CorrectPostureTests(unittest.TestCase):
    def test_removes_lean_and_keeps_feet(self):
        out, rep = correct_posture.correct(make_body(lean_deg=15.0))
        self.assertAlmostEqual(rep["core_lean_deg"]["before"], 15.0, delta=2.0)
        self.assertLess(abs(rep["core_lean_deg"]["after"]), 1.0)
        # above the neck the block moves rigidly, so the synthetic body (leaning to the top) keeps a little
        self.assertLess(abs(rep["trunk_lean_deg"]["after"]), 4.0)
        self.assertLess(abs(rep["trunk_lean_deg"]["after"]), abs(rep["trunk_lean_deg"]["before"]) / 4)
        self.assertLess(rep["feet_max_move"], 1e-6)
        self.assertAlmostEqual(rep["jacobian"]["det_min"], 1.0, delta=1e-3)


class AlignRowsTests(unittest.TestCase):
    def test_reaches_targets(self):
        _, rep = align_rows.align(make_body(n=8000), [(0.3, 0.33), (0.55, 0.58)])
        for p in rep["pairs"]:
            self.assertAlmostEqual(p["reached"], p["target"], delta=2e-3)

    def test_out_of_order_targets_rejected(self):
        with self.assertRaises(ValueError):
            align_rows.build_map([(0.3, 0.6), (0.5, 0.4)], 0.0, 1.0)


class RemoveDetachedTests(unittest.TestCase):
    def test_drops_far_blob_and_keeps_body(self):
        body = make_body(n=10000)
        blob = make_body(n=60, seed=3)
        blob["x"] = blob["x"] * 0.1 + 0.5
        blob["y"] = blob["y"] * 0.1
        blob["z"] = blob["z"] * 0.03 + 0.6
        out, rep = remove_detached.remove_detached(np.concatenate([body, blob]))
        self.assertEqual(len(out), len(body))
        self.assertEqual(rep["removed_opaque"], len(blob))
        self.assertGreater(rep["removed_blobs"][0]["centre"][0], 0.4)

    def test_box_limits_removal(self):
        body = make_body(n=10000)
        blob = make_body(n=60, seed=3)
        blob["x"] = blob["x"] * 0.1 + 0.5
        blob["z"] = blob["z"] * 0.03 + 0.6
        boxes = [{"name": "elsewhere", "min": [-1, -1, -1], "max": [-0.5, 1, 1]}]
        out, _ = remove_detached.remove_detached(np.concatenate([body, blob]), boxes=boxes)
        self.assertEqual(len(out), len(body) + len(blob))


class InteriorPocketTests(unittest.TestCase):
    def test_large_and_outlined_pockets_become_transparent(self):
        img = np.full((300, 300, 3), 255, np.uint8)            # white background
        cv2.rectangle(img, (40, 40), (260, 260), (60, 120, 200), -1)  # character
        cv2.rectangle(img, (60, 60), (120, 120), (255, 255, 255), -1)  # large gap (3721 px)
        cv2.rectangle(img, (150, 60), (170, 80), (0, 0, 0), -1)        # dark outline ...
        cv2.rectangle(img, (153, 63), (167, 77), (255, 255, 255), -1)  # ... around a small gap
        cv2.rectangle(img, (200, 200), (212, 212), (255, 255, 255), -1)  # small bright highlight
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        dist = np.linalg.norm(lab.astype(np.float32) - lab[0, 0].astype(np.float32), axis=2)
        outside = np.zeros(dist.shape, bool)
        outside[:40, :] = outside[261:, :] = outside[:, :40] = outside[:, 261:] = True
        mask, kept = step0_prep_image.interior_pockets(dist, outside, lab, 30.0, 12.0)
        self.assertTrue(mask[90, 90])
        self.assertTrue(mask[70, 160])
        self.assertFalse(mask[206, 206])
        self.assertEqual(len(kept), 2)


class MeasureLandmarksTests(unittest.TestCase):
    def test_heights_and_pairs(self):
        img = np.zeros((200, 100, 4), np.uint8)
        img[50:151, 30:71] = (200, 200, 200, 255)   # rows 50..150, height 100
        rep = measure_landmarks.measure(img, [("chin", None, 70.0), ("eye_l", 45.0, 60.0), ("eye_r", 55.0, 60.0)])
        self.assertEqual(rep["height_px"], 100.0)
        self.assertAlmostEqual(rep["marks"]["chin"]["t"], 0.8)
        self.assertAlmostEqual(rep["D_eye"], 0.1)
        design = {"marks": {"chin": {"t": 0.78}, "eye_l": {"t": 0.88}}}
        pr = measure_landmarks.pairs_with(rep, design)
        self.assertEqual(pr["names"], ["chin", "eye_l"])
        self.assertEqual(pr["pairs"], [[0.8, 0.78], [0.9, 0.88]])

    def test_opaque_render_uses_background_colour(self):
        img = np.full((120, 120, 4), (30, 30, 30, 255), np.uint8)
        img[20:101, 40:81, :3] = 220
        rep = measure_landmarks.measure(img, [])
        self.assertEqual(rep["bbox"], [40, 20, 80, 100])


class MatteTests(unittest.TestCase):
    def test_clean_alpha_keeps_largest_part_with_soft_edge(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:60] = 255
        mask[20:80, 19] = 120           # soft edge next to the body
        mask[5:8, 90:93] = 255          # speck
        alpha = step0_matte_comfy.clean_alpha(mask)
        self.assertEqual(alpha[50, 40], 255)
        self.assertEqual(alpha[50, 19], 120)
        self.assertEqual(alpha[6, 91], 0)

    def test_canvas_centres_and_pads(self):
        rgb = np.zeros((200, 100, 3), np.uint8)
        alpha = np.zeros((200, 100), np.uint8)
        alpha[10:190, 20:80] = 255
        canvas = np.array(step0_matte_comfy.to_canvas(rgb, alpha, size=256))
        ys, xs = np.nonzero(canvas[..., 3] > 16)
        self.assertEqual(ys.min(), 20)
        self.assertLessEqual(abs((xs.min() + xs.max()) / 2 - 127.5), 1.0)


if __name__ == "__main__":
    unittest.main()
