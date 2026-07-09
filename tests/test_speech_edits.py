"""Test sul rilevamento word-level (ripetizioni, n-gram, filler)."""

from __future__ import annotations

import unittest

from core.speech_edits import (
    SpeechEditConfig,
    detect_fillers,
    detect_ngram_repeats,
    detect_speech_edit_errors,
    detect_word_repeats,
)
from core.whisper_cpp import TranscriptSegment, TranscriptWord, _merge_subword_tokens, _parse_words


def _w(start: float, text: str, dur: float = 0.3) -> TranscriptWord:
    return TranscriptWord(start, start + dur, text)


class MergeSubwordTests(unittest.TestCase):
    def test_merges_bpe_pieces(self):
        toks = [
            TranscriptWord(0.0, 0.1, " La"),
            TranscriptWord(0.1, 0.3, " gest"),
            TranscriptWord(0.3, 0.5, "ione"),
            TranscriptWord(0.5, 0.7, " for"),
            TranscriptWord(0.7, 0.9, "nisce"),
            TranscriptWord(0.9, 1.1, " for"),
            TranscriptWord(1.1, 1.3, "nisce"),
        ]
        words = _merge_subword_tokens(toks)
        self.assertEqual([w.text for w in words],
                         ["la", "gestione", "fornisce", "fornisce"])

    def test_parse_words_skips_specials(self):
        raw = [
            {"text": "[_BEG_]", "timestamps": {"from": "00:00:00,000", "to": "00:00:00,000"}},
            {"text": " ehh", "timestamps": {"from": "00:00:01,000", "to": "00:00:01,400"}},
            {"text": ".", "timestamps": {"from": "00:00:01,400", "to": "00:00:01,500"}},
        ]
        words = _parse_words(raw)
        self.assertEqual([w.text for w in words], ["ehh"])


class WordRepeatTests(unittest.TestCase):
    def test_detects_fornisce_fornisce(self):
        words = [
            _w(0.0, "la"), _w(0.3, "procedura"),
            _w(0.8, "fornisce"), _w(1.2, "fornisce"),
            _w(1.6, "un"), _w(1.9, "quadro"),
        ]
        errs = detect_word_repeats(words, 10.0, __import__("core.language", fromlist=["resolve_language"]).resolve_language("it"))
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].type, "repeated_phrase")
        self.assertIn("fornisce", errs[0].description)

    def test_ignores_stopword_unigram(self):
        from core.language import resolve_language
        words = [_w(0.0, "di"), _w(0.2, "di"), _w(0.5, "rischio")]
        errs = detect_word_repeats(words, 5.0, resolve_language("it"))
        self.assertEqual(errs, [])


class NgramRepeatTests(unittest.TestCase):
    def test_detects_a_un_soggetto(self):
        from core.language import resolve_language
        words = [
            _w(0.0, "si"), _w(0.3, "applica"),
            _w(0.7, "a"), _w(0.9, "un"), _w(1.1, "soggetto"),
            _w(1.5, "a"), _w(1.7, "un"), _w(1.9, "soggetto"),
            _w(2.4, "esposto"),
        ]
        errs = detect_ngram_repeats(words, 10.0, resolve_language("it"))
        self.assertTrue(errs)
        self.assertIn("soggetto", errs[0].description)


class FillerTests(unittest.TestCase):
    def test_detects_ehh(self):
        from core.language import resolve_language
        words = [_w(0.0, "in"), _w(0.3, "questa"), _w(0.7, "ehh"), _w(1.1, "fase")]
        errs = detect_fillers(words, 5.0, resolve_language("it"))
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].type, "missed_cut")


class PipelineTests(unittest.TestCase):
    def test_combined_finds_all_gt_patterns(self):
        from core.language import resolve_language
        # Simula i 5 errori GT del video 3.5 in un unico stream di parole.
        words = [
            _w(135.0, "anche"), _w(135.4, "anche"),          # 2:15 extra word
            _w(240.0, "potrebbe"), _w(240.5, "potrebbe"),    # 4:00
            _w(367.0, "a"), _w(367.2, "un"), _w(367.4, "soggetto"),
            _w(367.9, "a"), _w(368.1, "un"), _w(368.3, "soggetto"),  # 6:07
            _w(600.0, "fornisce"), _w(600.4, "fornisce"),    # 10:00
            _w(673.0, "ehh"),                                 # 11:13
        ]
        segs = [TranscriptSegment(0, 700, "x", words=words)]
        cfg = SpeechEditConfig(enable_segment_baseline=False)
        errs = detect_speech_edit_errors(segs, 700.0, "it", cfg=cfg)
        types_desc = " | ".join(e.description for e in errs)
        self.assertGreaterEqual(len(errs), 5, types_desc)
        # Tutti i pattern chiave devono comparire.
        blob = types_desc.lower()
        for needle in ("anche", "potrebbe", "soggetto", "fornisce", "ehh"):
            self.assertIn(needle, blob)

    def test_clean_speech_no_false_positives(self):
        words = [
            _w(0.0, "la"), _w(0.2, "gestione"), _w(0.5, "del"),
            _w(0.7, "rischio"), _w(1.0, "fornisce"), _w(1.4, "un"),
            _w(1.6, "quadro"), _w(1.9, "chiaro"),
        ]
        segs = [TranscriptSegment(0, 3, "x", words=words)]
        cfg = SpeechEditConfig(enable_segment_baseline=False)
        errs = detect_speech_edit_errors(segs, 3.0, "it", cfg=cfg)
        self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()
