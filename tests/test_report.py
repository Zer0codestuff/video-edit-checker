"""Test su merge/filter errori e formattazione tempi."""

from __future__ import annotations

import unittest

from core.constants import BLACK_MIN_DURATION_SECONDS
from core.models import EditError
from core.report import filter_errors, fmt_time, merge_errors


class MergeFilterTests(unittest.TestCase):
    def test_merge_overlapping_same_type(self):
        errs = [
            EditError("black_screen", 10, 14, "a", 0.7),
            EditError("black_screen", 13, 18, "piu lunga descrizione", 0.9),
        ]
        merged = merge_errors(errs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 10)
        self.assertEqual(merged[0].end, 18)
        self.assertEqual(merged[0].confidence, 0.9)
        self.assertIn("piu lunga", merged[0].description)

    def test_merge_does_not_mutate_input_identity_for_new(self):
        a = EditError("missed_cut", 1, 2, "x", 0.6)
        b = EditError("other", 10, 11, "y", 0.6)
        merged = merge_errors([a, b])
        self.assertEqual(len(merged), 2)
        self.assertIsNot(merged[0], a)

    def test_filter_confidence_and_short_black(self):
        errs = [
            EditError("other", 0, 1, "low", 0.2),
            EditError("black_screen", 0, BLACK_MIN_DURATION_SECONDS,
                      "short", 0.9),
            EditError("black_screen", 0, BLACK_MIN_DURATION_SECONDS + 1,
                      "long", 0.9),
        ]
        out = filter_errors(errs, 0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].description, "long")


class FmtTimeTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(fmt_time(65), "1:05")

    def test_hours(self):
        self.assertEqual(fmt_time(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()
