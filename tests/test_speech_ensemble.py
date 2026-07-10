"""Test sull'ensemble multi-temperatura (senza whisper reale)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.speech_ensemble import detect_ensemble, transcribe_multi_temp
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

    def test_baseline_runs_once_word_union_preserved(self):
        # Due passaggi con lo stesso trigger segment-level + stutter solo nel 2°.
        trigger_a = [TranscriptSegment(
            0.0, 2.0, "aspetta lo ripeto dalla cima",
            words=[
                TranscriptWord(0.0, 0.4, "aspetta"),
                TranscriptWord(0.4, 0.7, "lo"),
                TranscriptWord(0.7, 1.2, "ripeto"),
                TranscriptWord(1.2, 1.6, "dalla"),
                TranscriptWord(1.6, 2.0, "cima"),
            ],
        )]
        stutter_b = [TranscriptSegment(
            0.0, 2.0, "la fornisce fornisce un",
            words=[
                TranscriptWord(0.0, 0.3, "la"),
                TranscriptWord(0.3, 0.7, "fornisce"),
                TranscriptWord(0.8, 1.2, "fornisce"),
                TranscriptWord(1.2, 1.5, "un"),
            ],
        )]
        with mock.patch(
            "core.speech_ensemble.detect_transcript_errors",
            wraps=__import__(
                "core.whisper_cpp", fromlist=["detect_transcript_errors"]
            ).detect_transcript_errors,
        ) as baseline:
            errs = detect_ensemble([trigger_a, stutter_b], 10.0, "it")
        self.assertEqual(baseline.call_count, 1)
        self.assertTrue(any(e.type == "missed_cut" for e in errs))
        self.assertTrue(any("fornisce" in e.description for e in errs))


class TranscribeMultiTempTests(unittest.TestCase):
    def test_restores_temperature_after_success(self):
        os.environ["WHISPER_TEMPERATURE"] = "0.3"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch(
                    "core.speech_ensemble.transcribe_video",
                    return_value=[],
                ) as tv:
                    transcribe_multi_temp(
                        Path("dummy.mp4"), Path(tmp),
                        temperatures=(0.0, 0.8), log=lambda *_: None,
                    )
                self.assertEqual(tv.call_count, 2)
            self.assertEqual(os.environ["WHISPER_TEMPERATURE"], "0.3")
        finally:
            os.environ.pop("WHISPER_TEMPERATURE", None)

    def test_restores_temperature_on_error(self):
        os.environ["WHISPER_TEMPERATURE"] = "0.25"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch(
                    "core.speech_ensemble.transcribe_video",
                    side_effect=RuntimeError("boom"),
                ):
                    with self.assertRaises(RuntimeError):
                        transcribe_multi_temp(
                            Path("dummy.mp4"), Path(tmp),
                            temperatures=(0.0, 0.8), log=lambda *_: None,
                        )
            self.assertEqual(os.environ["WHISPER_TEMPERATURE"], "0.25")
        finally:
            os.environ.pop("WHISPER_TEMPERATURE", None)

    def test_unsets_temperature_if_was_absent(self):
        os.environ.pop("WHISPER_TEMPERATURE", None)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "core.speech_ensemble.transcribe_video",
                return_value=[],
            ):
                transcribe_multi_temp(
                    Path("dummy.mp4"), Path(tmp),
                    temperatures=(0.0,), log=lambda *_: None,
                )
        self.assertNotIn("WHISPER_TEMPERATURE", os.environ)


if __name__ == "__main__":
    unittest.main()
