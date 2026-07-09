"""Lingua dell'analisi: whisper, prompt LLM e testi degli errori."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Etichette UI -> codice whisper.cpp / ISO 639-1
LANGUAGE_CHOICES: dict[str, str] = {
    "Italiano": "it",
    "English": "en",
}
DEFAULT_LANGUAGE_LABEL = "Italiano"


@dataclass(frozen=True)
class LanguagePack:
    """Stringhe e pattern legati alla lingua del video / del report."""

    code: str  # "it" | "en"
    # Nome della lingua in inglese, per i prompt LLM.
    speech_name: str
    # Lingua in cui scrivere le description nel JSON.
    description_name: str
    # Esempi di frasi di ripresa per missed_cut (testo nel prompt).
    missed_cut_examples: str
    # Regex trigger per whisper transcript.
    trigger_re: re.Pattern[str]
    # Descrizioni euristiche / transcript.
    black_screen_desc: str
    frozen_frame_desc: str
    missed_cut_desc: str  # format con {quote}
    repeated_phrase_desc: str  # format con {a}, {b}
    audio_gap_desc: str  # format con {gap}


_PACKS: dict[str, LanguagePack] = {
    "it": LanguagePack(
        code="it",
        speech_name="Italian",
        description_name="Italian",
        missed_cut_examples=(
            '"lo ripeto", "aspetta", "rifacciamo", "da capo" '
            '("let me repeat", "wait", "let\'s redo it", "from the top")'
        ),
        trigger_re=re.compile(
            r"\b("
            r"lo ripeto|ripeto|aspetta|rifacciamo|da capo|taglia|tagliamo|"
            r"scusa|sbagliato|ho sbagliato|errore|riparto|riprovo|un attimo"
            r")\b",
            re.IGNORECASE,
        ),
        black_screen_desc="Schermo nero prolungato rilevato con analisi luminanza.",
        frozen_frame_desc="Sequenza di frame quasi identici rilevata con confronto pixel.",
        missed_cut_desc="Possibile taglio mancato: frase di ripresa nel parlato («{quote}»).",
        repeated_phrase_desc="Possibile frase ripetuta: «{a}» / «{b}».",
        audio_gap_desc="Silenzio o vuoto audio anomalo di circa {gap:.1f} secondi.",
    ),
    "en": LanguagePack(
        code="en",
        speech_name="English",
        description_name="English",
        missed_cut_examples=(
            '"let me repeat", "wait", "let\'s redo it", "from the top", '
            '"sorry", "I messed up", "take two", "cut"'
        ),
        trigger_re=re.compile(
            r"\b("
            r"let me repeat|let's redo|from the top|take two|take it again|"
            r"hold on|wait a (sec|second|minute)|sorry|my bad|i messed up|"
            r"start over|do that again|cut that|one more time"
            r")\b",
            re.IGNORECASE,
        ),
        black_screen_desc="Prolonged black screen detected via luminance analysis.",
        frozen_frame_desc="Near-identical frame sequence detected via pixel comparison.",
        missed_cut_desc="Possible missed cut: retake phrase in speech («{quote}»).",
        repeated_phrase_desc="Possible repeated phrase: «{a}» / «{b}».",
        audio_gap_desc="Abnormal silence or audio gap of about {gap:.1f} seconds.",
    ),
}


def resolve_language(label_or_code: str | None) -> LanguagePack:
    """Accetta etichetta UI ('Italiano') o codice ('it'/'en'). Default: italiano."""
    raw = (label_or_code or "").strip()
    if raw in _PACKS:
        return _PACKS[raw]
    code = LANGUAGE_CHOICES.get(raw, "it")
    return _PACKS.get(code, _PACKS["it"])
