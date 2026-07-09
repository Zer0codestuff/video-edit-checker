"""Test sul LanguagePack e sui trigger transcript per lingua."""

from __future__ import annotations

import unittest

from core.language import resolve_language
from core.whisper_cpp import TranscriptSegment, detect_transcript_errors


class LanguageResolveTests(unittest.TestCase):
    def test_italian_label(self):
        pack = resolve_language("Italiano")
        self.assertEqual(pack.code, "it")
        self.assertEqual(pack.description_name, "Italian")

    def test_english_label(self):
        pack = resolve_language("English")
        self.assertEqual(pack.code, "en")
        self.assertEqual(pack.description_name, "English")

    def test_code_and_default(self):
        self.assertEqual(resolve_language("en").code, "en")
        self.assertEqual(resolve_language(None).code, "it")
        self.assertEqual(resolve_language("??").code, "it")


class TranscriptLanguageTests(unittest.TestCase):
    def test_italian_trigger(self):
        segs = [TranscriptSegment(1.0, 3.0, "ok lo ripeto da qui")]
        errs = detect_transcript_errors(segs, 60.0, language="it")
        self.assertTrue(any(e.type == "missed_cut" for e in errs))
        self.assertIn("taglio mancato", errs[0].description.lower())

    def test_english_trigger(self):
        segs = [TranscriptSegment(1.0, 3.0, "sorry, let me repeat that")]
        errs = detect_transcript_errors(segs, 60.0, language="en")
        self.assertTrue(any(e.type == "missed_cut" for e in errs))
        self.assertIn("missed cut", errs[0].description.lower())

    def test_english_trigger_ignored_in_italian(self):
        segs = [TranscriptSegment(1.0, 3.0, "sorry, let me repeat that")]
        errs = detect_transcript_errors(segs, 60.0, language="it")
        self.assertFalse(any(e.type == "missed_cut" for e in errs))


class PromptLanguageTests(unittest.TestCase):
    def test_omni_prompt_switches_language(self):
        from core.analyzer import _build_prompt
        from core.windows import Window

        win = Window(0, 0.0, 20.0, frame_paths=[], frame_times=[0.0, 3.0])
        it = _build_prompt(win, resolve_language("it"))
        en = _build_prompt(win, resolve_language("en"))
        self.assertIn("speech in the audio is in Italian", it)
        self.assertIn("MUST be written entirely in Italian", it)
        self.assertIn("in Italian only", it)
        self.assertIn("speech in the audio is in English", en)
        self.assertIn("MUST be written entirely in English", en)
        self.assertIn("lo ripeto", it)
        self.assertIn("let me repeat", en)


if __name__ == "__main__":
    unittest.main()
