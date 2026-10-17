import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wardogs.detector import (  # noqa: E402
    DetectorConfig,
    MarkerDetector,
    shift_detection,
)
from wardogs.auto_build import collect_test_images  # noqa: E402


class DetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        cls.detector = MarkerDetector(DetectorConfig.from_dict(config["detector"]))
        cls.assets_available = (
            (ROOT / "assets" / "reference" / "1.png").is_file()
            and (ROOT / "assets" / "validate").is_dir()
        )

    def require_assets(self):
        if not self.assets_available:
            self.skipTest("local validation screenshots are not present")

    def detect(self, name):
        self.require_assets()
        frame = cv2.imread(str(ROOT / "assets" / "reference" / name))
        self.assertIsNotNone(frame, f"missing image {name}")
        return self.detector.detect(frame)

    def test_yellow_marker_in_three_views(self):
        expected = {
            "1.png": (1071.0, 788.0),
            "3.png": (1034.0, 786.0),
            "4.png": (1515.0, 794.0),
        }
        for name, center in expected.items():
            det = self.detect(name)
            self.assertEqual(len(det.yellow), 1, name)
            self.assertIsNotNone(det.best, name)
            self.assertFalse(det.locked, name)
            self.assertLessEqual(math.dist(det.best.center, center), 12.0, name)
            self.assertGreaterEqual(det.best.area, 600, name)
            self.assertLessEqual(det.best.aspect, 3.0, name)

    def test_4_png_oblique_view_regression(self):
        # The old aspect <= 2.0 filter rejected this real marker (2.13).
        det = self.detect("4.png")
        self.assertEqual(len(det.yellow), 1)
        self.assertGreater(det.best.aspect, 2.0)
        self.assertLess(det.best.aspect, 3.0)

    def test_white_lock(self):
        det = self.detect("2.png")
        self.assertEqual(len(det.yellow), 0)
        self.assertTrue(det.locked)
        self.assertGreaterEqual(det.center_v, 235.0)
        self.assertLessEqual(det.center_s, 35.0)

    def test_no_curb_false_positive(self):
        for name in ("3.png", "4.png"):
            det = self.detect(name)
            self.assertEqual(len(det.yellow), 1, name)

    def test_annotate_shape(self):
        self.require_assets()
        frame = cv2.imread(str(ROOT / "assets" / "reference" / "1.png"))
        det = self.detector.detect(frame)
        from wardogs.detector import annotate
        out = annotate(frame, det)
        self.assertEqual(out.shape, frame.shape)

    def test_validate_wardogs_views(self):
        self.require_assets()
        expected = {
            "WARDOGS Screenshot 2026.10.17 - 17.47.16.09.png": (1127.0, 757.0),
            "WARDOGS Screenshot 2026.10.17 - 17.47.21.44.png": (1414.0, 774.0),
            "WARDOGS Screenshot 2026.10.17 - 17.47.31.43.png": (1172.0, 958.0),
            "WARDOGS Screenshot 2026.10.17 - 17.47.39.09.png": (1004.0, 812.0),
            "WARDOGS Screenshot 2026.10.17 - 17.47.57.30.png": (1026.0, 712.0),
            "WARDOGS Screenshot 2026.10.17 - 17.48.12.88.png": (1376.0, 754.0),
        }
        for name, center in expected.items():
            path = ROOT / "assets" / "validate" / name
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            det = self.detector.detect(frame)
            self.assertEqual(len(det.yellow), 1, name)
            self.assertIsNotNone(det.best, name)
            self.assertFalse(det.locked, name)
            self.assertLessEqual(math.dist(det.best.center, center), 15.0, name)
            self.assertGreaterEqual(det.best.area, 600, name)
            self.assertLessEqual(det.best.aspect, 3.0, name)

    def test_white_lock_when_crosshair_covers_center(self):
        # The game may draw its crosshair over the white X, making the exact
        # centre pixel dark.  The white component's bbox still contains it.
        frame = np.full((1440, 2560, 3), 120, dtype=np.uint8)
        white = (245, 245, 245)
        cv2.line(frame, (1210, 650), (1350, 790), white, 22)
        cv2.line(frame, (1350, 650), (1210, 790), white, 22)
        frame[716:725, 1276:1285] = (15, 15, 15)
        det = self.detector.detect(frame)
        self.assertEqual(len(det.yellow), 0)
        self.assertTrue(det.locked)

    def test_white_center_wins_over_yellow_ghost(self):
        frame = np.full((1440, 2560, 3), 120, dtype=np.uint8)
        white = (245, 245, 245)
        yellow = (0, 255, 255)
        cv2.line(frame, (1210, 650), (1350, 790), white, 22)
        cv2.line(frame, (1350, 650), (1210, 790), white, 22)
        cv2.line(frame, (1000, 600), (1150, 750), yellow, 20)
        cv2.line(frame, (1150, 600), (1000, 750), yellow, 20)
        det = self.detector.detect(frame)
        self.assertIsNotNone(det.best)
        self.assertTrue(det.locked)

    def test_validate_white_wardogs_views(self):
        self.require_assets()
        files = sorted((ROOT / "assets" / "validate" / "white").glob("*.png"))
        if not files:
            self.skipTest("validate/white is empty")
        for path in files:
            frame = cv2.imread(str(path))
            det = self.detector.detect(frame)
            self.assertEqual(len(det.yellow), 0, path.name)
            self.assertTrue(det.locked, path.name)

    def test_fast_mode_matches_full_detector(self):
        self.require_assets()
        fast = MarkerDetector(DetectorConfig.from_dict({"fast_mode": True}))
        full = MarkerDetector(DetectorConfig.from_dict({"fast_mode": False}))
        paths = [
            ROOT / "assets" / "validate" /
            "WARDOGS Screenshot 2026.10.17 - 17.47.16.09.png",
            ROOT / "assets" / "validate" / "white" /
            "WARDOGS Screenshot 2026.10.17 - 18.13.55.27.png",
        ]
        for path in paths:
            frame = cv2.imread(str(path))
            self.assertIsNotNone(frame, path.name)
            a = fast.detect(frame)
            b = full.detect(frame)
            self.assertEqual(a.locked, b.locked, path.name)
            self.assertEqual(len(a.yellow), len(b.yellow), path.name)
            if a.best is not None and b.best is not None:
                self.assertLessEqual(
                    math.dist(a.best.center, b.best.center), 10.0, path.name
                )

    def test_generic_test_dir_discovery(self):
        self.require_assets()
        base = str(ROOT / "assets" / "validate")
        top = collect_test_images([base], recursive=False)
        recursive = collect_test_images([base], recursive=True)
        self.assertTrue(top)
        self.assertGreaterEqual(len(recursive), len(top))
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.assertTrue(all(Path(p).suffix.lower() in allowed for p in recursive))

    def test_roi_detection_shifts_to_full_frame(self):
        self.require_assets()
        frame = cv2.imread(str(ROOT / "assets" / "reference" / "1.png"))
        full = self.detector.detect(frame)
        x0, y0, half = 780, 220, 500
        roi = self.detector.detect(frame[y0:y0 + 2 * half, x0:x0 + 2 * half])
        shifted = shift_detection(roi, x0, y0, frame.shape[1], frame.shape[0])
        self.assertEqual(shifted.best.bbox, full.best.bbox)
        self.assertLessEqual(
            math.dist(shifted.best.center, full.best.center), 1.0
        )

    def test_small_yellow_ui_at_crosshair_is_ignored(self):
        frame = np.full((1440, 2560, 3), 120, dtype=np.uint8)
        yellow = (0, 255, 255)
        cv2.rectangle(frame, (1255, 693), (1305, 746), yellow, -1)
        cv2.rectangle(frame, (1299, 683), (1340, 762), yellow, -1)
        cv2.line(frame, (1050, 650), (1150, 750), yellow, 20)
        cv2.line(frame, (1150, 650), (1050, 750), yellow, 20)
        det = self.detector.detect(frame)
        self.assertIsNotNone(det.best)
        self.assertGreater(math.dist(det.best.center, (1280.0, 720.0)), 60.0)


if __name__ == "__main__":
    unittest.main()
