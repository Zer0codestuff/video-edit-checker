"""Test sull'ensemble multi-temperatura (senza whisper reale)."""

from __future__ import annotations

import unittest

from core.models import EditError
from core.speech_ensemble import detect_ensemble
from core.whisper_cpp import TranscriptSegment, TranscriptWord


def _seg(words):
    return [TranscriptSegment(
        words[0].start, words[-1].end,
        " ".join(w.text for w in words), words=words)]


class EnsembleTests(unittest.TestCase):
    def test_union_recovers_stutter_from_one_pass(self):
        # Pass A collassa; pass B ha la ripetizione.
        a = _seg([
            TranscriptWord(0, 0.3, "la"),
            TranscriptWord(0.3, 0.8, "fornisce"),
            TranscriptWord(0.8, 1.2, "un"),
        ])
        b = _seg([
            TranscriptWord(0, 0.3, "la"),
            TranscriptWord(0.3, 0.7, "fornisce"),
            TranscriptWord(0.8, 1.2, "fornisce"),
            TranscriptWord(1.2, 1.5, "un"),
        ])
        errs = detect_ensemble([a, b], 10.0, "it")
        self.assertTrue(any("fornisce" in e.description for e in errs))

    def test_empty_lists(self):
        self.assertEqual(detect_ensemble([], 10.0, "it"), [])
        self.assertEqual(detect_ensemble([[]], 10.0, "it"), [])


if __name__ == "__main__":
    unittest.main()
