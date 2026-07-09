"""Test su normalizzazione timestamp e parsing errori LLM."""

from __future__ import annotations

import unittest

from core.llm_client import extract_json
from core.parse_errors import (OMNI_POLICY, VIDEO_POLICY, VISION_POLICY,
                               normalize_times, parse_errors)
from core.windows import Window


def _win(start: float = 40.0, duration: float = 20.0) -> Window:
    return Window(index=0, start=start, duration=duration)


class NormalizeTimesTests(unittest.TestCase):
    def test_absolute_heuristic_shifts_relative(self):
        w = _win()
        self.assertEqual(normalize_times(2.0, 5.0, w, "absolute_heuristic"),
                         (42.0, 45.0))

    def test_absolute_heuristic_keeps_absolute(self):
        w = _win()
        self.assertEqual(normalize_times(42.0, 45.0, w, "absolute_heuristic"),
                         (42.0, 45.0))

    def test_relative_clip_shifts(self):
        w = _win()
        self.assertEqual(normalize_times(2.0, 5.0, w, "relative_clip"),
                         (42.0, 45.0))

    def test_relative_clip_keeps_clearly_absolute(self):
        w = _win()
        # end > duration+1 → già assoluti, poi clamp a win_end=60
        self.assertEqual(normalize_times(42.0, 70.0, w, "relative_clip"),
                         (42.0, 60.0))


class ParseErrorsTests(unittest.TestCase):
    def test_omni_penalizes_unquoted_repeat(self):
        w = _win()
        data = {"errors": [{
            "type": "repeated_phrase",
            "start": 41, "end": 44,
            "description": "frase ripetuta senza quote",
            "confidence": 0.9,
        }]}
        errs = parse_errors(data, w, OMNI_POLICY)
        self.assertEqual(len(errs), 1)
        self.assertLessEqual(errs[0].confidence, 0.4)

    def test_vision_remaps_audio_types(self):
        w = _win()
        data = {"errors": [{
            "type": "audio_glitch",
            "start": 41, "end": 43,
            "description": "x",
            "confidence": 0.9,
        }]}
        errs = parse_errors(data, w, VISION_POLICY)
        self.assertEqual(errs[0].type, "other")

    def test_video_relative_times(self):
        w = _win()
        data = {"errors": [{
            "type": "black_screen",
            "start": 1.0, "end": 3.0,
            "description": "nero",
            "confidence": 0.8,
        }]}
        errs = parse_errors(data, w, VIDEO_POLICY)
        self.assertEqual(errs[0].start, 41.0)
        self.assertEqual(errs[0].end, 43.0)


class ExtractJsonTests(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(extract_json('```json\n{"errors":[]}\n```'),
                         {"errors": []})

    def test_embedded(self):
        text = 'blah {"errors":[{"type":"other","start":1,"end":2,' \
               '"description":"d","confidence":0.5}]}'
        self.assertTrue(extract_json(text)["errors"])

    def test_empty(self):
        self.assertEqual(extract_json(""), {"errors": []})


if __name__ == "__main__":
    unittest.main()
