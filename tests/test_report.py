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

    def test_merge_does_not_fuse_distinct_speech_quotes(self):
        # «soggetto» e «fornisce» vicini non devono diventare un solo evento.
        errs = [
            EditError("repeated_phrase", 17.0, 20.0,
                      "Ripresa/stutter da tagliare: «a un soggetto» ripetuto (3 parole).", 0.88),
            EditError("repeated_phrase", 22.0, 25.0,
                      "Parola in più da tagliare: «fornisce» ripetuta subito dopo.", 0.86),
        ]
        merged = merge_errors(errs)
        self.assertEqual(len(merged), 2)

    def test_merge_fuses_same_speech_quote(self):
        errs = [
            EditError("repeated_phrase", 10.0, 12.0,
                      "Parola in più da tagliare: «fornisce» ripetuta subito dopo.", 0.8),
            EditError("repeated_phrase", 11.5, 13.0,
                      "Parola in più da tagliare: «fornisce» ripetuta subito dopo.", 0.9),
        ]
        merged = merge_errors(errs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].end, 13.0)
        self.assertEqual(merged[0].confidence, 0.9)

    def test_merge_does_not_fuse_distinct_nearby_audio_glitches(self):
        # Boilerplate quasi identico + gap < 3s non deve fondere due silenzi distinti.
        errs = [
            EditError("audio_glitch", 10.0, 16.0,
                      "Silenzio o vuoto audio anomalo di circa 6.0 secondi.", 0.7),
            EditError("audio_glitch", 17.5, 23.0,
                      "Silenzio o vuoto audio anomalo di circa 5.5 secondi.", 0.75),
        ]
        merged = merge_errors(errs)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].start, 10.0)
        self.assertEqual(merged[1].start, 17.5)

    def test_merge_fuses_overlapping_audio_glitches(self):
        errs = [
            EditError("audio_glitch", 10.0, 16.0,
                      "Silenzio o vuoto audio anomalo di circa 6.0 secondi.", 0.7),
            EditError("audio_glitch", 15.0, 18.0,
                      "Silenzio o vuoto audio anomalo di circa 3.0 secondi.", 0.8),
        ]
        merged = merge_errors(errs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 10.0)
        self.assertEqual(merged[0].end, 18.0)
        self.assertEqual(merged[0].confidence, 0.8)

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
