"""Test sulle euristiche pixel (senza GPU)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.heuristics import detect_visual_heuristics, verify_visual_errors
from core.models import EditError
from core.windows import Window


def _solid(path: Path, color: int) -> None:
    Image.new("L", (160, 90), color=color).save(path, "JPEG")


class HeuristicsTests(unittest.TestCase):
    def test_detects_prolonged_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, times = [], []
            # 3s interval: 0,3,6,9,12 → black for >5s
            for i, t in enumerate([0.0, 3.0, 6.0, 9.0, 12.0]):
                p = root / f"f{i}.jpg"
                _solid(p, 0)
                paths.append(p)
                times.append(t)
            win = Window(0, 0.0, 15.0, frame_paths=paths, frame_times=times)
            errs = detect_visual_heuristics([win], log=lambda *_: None)
            blacks = [e for e in errs if e.type == "black_screen"]
            self.assertTrue(blacks)
            self.assertGreater(blacks[0].end - blacks[0].start, 5.0)

    def test_verify_drops_false_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, times = [], []
            for i, t in enumerate([0.0, 3.0, 6.0]):
                p = root / f"f{i}.jpg"
                _solid(p, 180)  # bright frames
                paths.append(p)
                times.append(t)
            win = Window(0, 0.0, 10.0, frame_paths=paths, frame_times=times)
            fake = [EditError("black_screen", 0, 8, "hallucinated", 0.9)]
            kept = verify_visual_errors(fake, [win], log=lambda *_: None)
            self.assertEqual(kept, [])

    def test_verify_keeps_confirmed_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, times = [], []
            for i, t in enumerate([0.0, 3.0, 6.0]):
                p = root / f"f{i}.jpg"
                _solid(p, 0)
                paths.append(p)
                times.append(t)
            win = Window(0, 0.0, 10.0, frame_paths=paths, frame_times=times)
            real = [EditError("black_screen", 0, 8, "real", 0.9)]
            kept = verify_visual_errors(real, [win], log=lambda *_: None)
            self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
