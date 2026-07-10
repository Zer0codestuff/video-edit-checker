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

    def test_parse_words_skips_specials_keeps_boundaries(self):
        raw = [
            {"text": "[_BEG_]", "timestamps": {"from": "00:00:00,000", "to": "00:00:00,000"}},
            {"text": " ehh", "timestamps": {"from": "00:00:01,000", "to": "00:00:01,400"}},
            {"text": " for", "timestamps": {"from": "00:00:01,500", "to": "00:00:01,700"}},
            {"text": "nisce", "timestamps": {"from": "00:00:01,700", "to": "00:00:01,900"}},
            {"text": " for", "timestamps": {"from": "00:00:02,000", "to": "00:00:02,200"}},
            {"text": "nisce", "timestamps": {"from": "00:00:02,200", "to": "00:00:02,400"}},
            {"text": ".", "timestamps": {"from": "00:00:02,400", "to": "00:00:02,500"}},
        ]
        words = _parse_words(raw)
        self.assertEqual([w.text for w in words], ["ehh", "fornisce", "fornisce"])

    def test_strip_bug_would_merge_all(self):
        # Regressione: strip() prematuro fondeva tutte le parole in una.
        raw = [
            {"text": " Questo", "timestamps": {"from": "00:00:00,000", "to": "00:00:00,400"}},
            {"text": " pot", "timestamps": {"from": "00:00:00,500", "to": "00:00:00,700"}},
            {"text": "rebbe", "timestamps": {"from": "00:00:00,700", "to": "00:00:01,000"}},
            {"text": " pot", "timestamps": {"from": "00:00:01,000", "to": "00:00:01,200"}},
            {"text": "rebbe", "timestamps": {"from": "00:00:01,200", "to": "00:00:01,500"}},
        ]
        words = _parse_words(raw)
        self.assertEqual([w.text for w in words], ["questo", "potrebbe", "potrebbe"])


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

    def test_detects_asr_variants_emm_ehm(self):
        from core.language import resolve_language
        lang = resolve_language("it")
        for tok in ("emm", "ehm", "ehh", "uhm", "mm", "em"):
            errs = detect_fillers([_w(1.0, tok)], 5.0, lang)
            self.assertEqual(len(errs), 1, tok)

    def test_isolated_m_filler(self):
        from core.language import resolve_language
        words = [
            _w(26.0, "fase", 0.5),
            _w(28.0, "m", 0.3),
            _w(28.8, "valutiamo", 0.5),
        ]
        errs = detect_fillers(words, 40.0, resolve_language("it"))
        self.assertEqual(len(errs), 1)
        self.assertIn("m", errs[0].description)

    def test_conjunction_e_not_filler(self):
        from core.language import resolve_language
        # «e» tra parole senza gap: congiunzione, non filler
        words = [
            _w(1.0, "probabilità", 0.5),
            _w(1.5, "e", 0.1),
            _w(1.6, "impatto", 0.4),
        ]
        errs = detect_fillers(words, 10.0, resolve_language("it"))
        self.assertEqual(errs, [])


class NearRepeatTests(unittest.TestCase):
    def test_detects_potrebbe_with_intervening_words(self):
        from core.language import resolve_language
        from core.speech_edits import detect_near_word_repeats
        words = [
            _w(240.0, "evento"), _w(240.5, "dannoso"),
            _w(241.0, "potrebbe"), _w(241.5, "non"), _w(241.9, "prevenuto"),
            _w(242.5, "potrebbe"), _w(243.0, "avere"),
        ]
        errs = detect_near_word_repeats(words, 300.0, resolve_language("it"))
        self.assertTrue(errs)
        self.assertIn("potrebbe", errs[0].description)

    def test_detects_near_ngram_soggetto(self):
        from core.language import resolve_language
        from core.speech_edits import detect_near_ngram_repeats
        words = [
            _w(367.0, "a"), _w(367.2, "un"), _w(367.4, "soggetto"),
            _w(368.0, "dell'evento"), _w(368.5, "potenzialmente"), _w(369.0, "dannoso"),
            _w(370.0, "a"), _w(370.2, "un"), _w(370.4, "soggetto"),
            _w(371.0, "esterno"),
        ]
        errs = detect_near_ngram_repeats(words, 400.0, resolve_language("it"))
        self.assertTrue(errs)
        self.assertIn("soggetto", errs[0].description)

    def test_ignores_rhetorical_abbiamo(self):
        from core.language import resolve_language
        from core.speech_edits import detect_near_word_repeats
        # Parallelismo retorico con molte parole in mezzo, senza cue di ripresa.
        words = [
            _w(20.0, "abbiamo"), _w(20.5, "identificato"), _w(21.0, "le"),
            _w(21.3, "tipologie"), _w(21.8, "specifiche"),
            _w(22.5, "abbiamo"), _w(23.0, "analizzato"),
        ]
        errs = detect_near_word_repeats(words, 30.0, resolve_language("it"))
        self.assertEqual(errs, [])


class FunctionWordRepeatTests(unittest.TestCase):
    def test_detects_il_il_tight_gap(self):
        from core.language import resolve_language
        from core.speech_edits import detect_function_word_repeats
        words = [_w(137.35, "il", 0.13), _w(137.56, "il", 0.13), _w(137.7, "rischio")]
        # gap = 137.56 - 137.48 = 0.08
        errs = detect_function_word_repeats(words, 200.0, resolve_language("it"))
        self.assertEqual(len(errs), 1)
        self.assertIn("il", errs[0].description)

    def test_ignores_stopword_with_wide_gap(self):
        from core.language import resolve_language
        from core.speech_edits import detect_function_word_repeats
        words = [_w(1.0, "di", 0.2), _w(2.0, "di", 0.2), _w(2.5, "rischio")]
        errs = detect_function_word_repeats(words, 10.0, resolve_language("it"))
        self.assertEqual(errs, [])


class StemRestartTests(unittest.TestCase):
    def test_detects_fornisce_fornire_false_start(self):
        from core.language import resolve_language
        from core.speech_edits import detect_stem_restarts
        words = [
            _w(601.5, "massimo", 0.4),
            _w(602.0, "fornisce", 0.6),
            _w(602.8, "sarebbe", 0.5),
            _w(603.4, "in", 0.2),
            _w(603.7, "grado", 0.3),
            _w(604.1, "di", 0.2),
            _w(604.4, "fornire", 0.6),
        ]
        segs = [TranscriptSegment(
            600, 606,
            "massimo fornisce, sarebbe in grado di fornire",
            words=words,
        )]
        errs = detect_stem_restarts(words, segs, 700.0, resolve_language("it"))
        self.assertTrue(errs)
        self.assertIn("fornisce", errs[0].description.lower())
        self.assertIn("fornire", errs[0].description.lower())

    def test_ignores_unrelated_shared_prefix_without_cue(self):
        from core.language import resolve_language
        from core.speech_edits import detect_stem_restarts
        words = [
            _w(1.0, "limiti", 0.4),
            _w(1.5, "la", 0.2),
            _w(1.8, "tensione", 0.4),
            _w(2.3, "limitando", 0.5),
        ]
        segs = [TranscriptSegment(0, 4, "limiti la tensione limitando", words=words)]
        errs = detect_stem_restarts(words, segs, 10.0, resolve_language("it"))
        self.assertEqual(errs, [])


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

    def test_pipeline_finds_real_asr_restart_shapes(self):
        """Forme tipiche del video reale (non adiacenti / stem), senza GT hardcoded."""
        words = [
            _w(137.35, "il", 0.13), _w(137.56, "il", 0.13), _w(137.7, "rischio"),
            _w(241.0, "potrebbe"), _w(241.5, "non"), _w(241.9, "prevenuto"),
            _w(242.5, "potrebbe"), _w(243.0, "avere"),
            _w(367.0, "a"), _w(367.2, "un"), _w(367.4, "soggetto"),
            _w(368.0, "dell'evento"), _w(368.5, "potenzialmente"), _w(369.0, "dannoso"),
            _w(370.0, "a"), _w(370.2, "un"), _w(370.4, "soggetto"),
            _w(602.0, "fornisce", 0.6), _w(602.8, "sarebbe", 0.5),
            _w(603.4, "in", 0.2), _w(603.7, "grado", 0.3),
            _w(604.1, "di", 0.2), _w(604.4, "fornire", 0.6),
        ]
        segs = [TranscriptSegment(
            0, 700,
            "riducono il il rischio. potrebbe non prevenuto, potrebbe avere. "
            "a un soggetto, dell'evento potenzialmente dannoso, a un soggetto. "
            "fornisce, sarebbe in grado di fornire",
            words=words,
        )]
        cfg = SpeechEditConfig(enable_segment_baseline=False)
        errs = detect_speech_edit_errors(segs, 700.0, "it", cfg=cfg)
        blob = " ".join(e.description.lower() for e in errs)
        self.assertIn("il", blob)
        self.assertIn("potrebbe", blob)
        self.assertIn("soggetto", blob)
        self.assertIn("fornisce", blob)

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
