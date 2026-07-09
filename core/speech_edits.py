"""Rilevamento errori di montaggio a livello di parlato (parole/filler).

Le euristiche su segmenti interi (frasi quasi duplicate, trigger tipo
"aspetta/lo ripeto") non catturano i casi tipici di raw footage:
parola ripetuta subito dopo, n-gram stutter, filler ("ehh").

Questo modulo lavora sui token word-level di whisper.cpp (-ojf) e puo'
essere combinato con `detect_transcript_errors` (segment-level).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.language import LanguagePack, resolve_language
from core.models import EditError
from core.whisper_cpp import TranscriptSegment, TranscriptWord


# Parole funzionali troppo comuni: una doppia "di/di" o "a/a" da sola
# e' spesso rumore ASR, non un errore di montaggio.
_STOP_IT = frozenset({
    "a", "ad", "al", "allo", "ai", "agli", "alla", "alle",
    "di", "da", "dai", "dal", "dalla", "dalle", "dei", "degli", "delle",
    "in", "nel", "nella", "nei", "nelle", "su", "sul", "sulla",
    "e", "ed", "o", "od", "ma", "se", "che", "chi", "cui",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "per", "con", "tra", "fra", "come", "non", "si", "ci", "ne",
    "mi", "ti", "vi", "li", "le",
})
_STOP_EN = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or",
    "but", "if", "as", "by", "with", "from", "is", "are", "was", "were",
    "be", "been", "it", "this", "that", "i", "you", "he", "she", "we",
    "they", "my", "your", "his", "her", "our", "their",
})


@dataclass(frozen=True)
class SpeechEditConfig:
    """Parametri degli esperimenti di detection sul parlato."""

    enable_word_repeat: bool = True
    enable_ngram_repeat: bool = True
    enable_fillers: bool = True
    enable_segment_baseline: bool = True
    ngram_sizes: tuple[int, ...] = (2, 3, 4)
    # Gap massimo (s) tra fine della 1a occorrenza e inizio della 2a.
    max_repeat_gap: float = 1.5
    # Ignora ripetizioni di sole stopword (n=1).
    ignore_stopword_unigrams: bool = True
    # Confidence base.
    word_repeat_confidence: float = 0.86
    ngram_repeat_confidence: float = 0.88
    filler_confidence: float = 0.8
    # Espandi leggermente l'intervallo segnalato per il taglio.
    pad_before: float = 0.15
    pad_after: float = 0.35


DEFAULT_SPEECH_EDIT_CONFIG = SpeechEditConfig()


def _stops(lang: LanguagePack) -> frozenset[str]:
    return _STOP_EN if lang.code == "en" else _STOP_IT


def _is_content(token: str, stops: frozenset[str]) -> bool:
    return bool(token) and token not in stops and len(token) > 1


def _words_from_segments(segments: list[TranscriptSegment]) -> list[TranscriptWord]:
    """Preferisce i word-token whisper; fallback: spezza il testo del segmento."""
    words: list[TranscriptWord] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
            continue
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", seg.text)
        if not tokens:
            continue
        dur = max(0.05, seg.end - seg.start)
        step = dur / len(tokens)
        for i, tok in enumerate(tokens):
            t0 = seg.start + i * step
            words.append(TranscriptWord(t0, t0 + step, tok.lower()))
    return words


def _emit_repeat(
    words: list[TranscriptWord],
    i: int,
    n: int,
    video_duration: float,
    lang: LanguagePack,
    conf: float,
    cfg: SpeechEditConfig,
) -> EditError:
    span = words[i:i + 2 * n]
    quote = " ".join(w.text for w in span[:n])
    start = max(0.0, span[0].start - cfg.pad_before)
    end = min(video_duration, span[-1].end + cfg.pad_after)
    if n == 1:
        desc = lang.word_repeat_desc.format(quote=quote)
    else:
        desc = lang.ngram_repeat_desc.format(quote=quote, n=n)
    return EditError(
        type="repeated_phrase",
        start=start,
        end=end,
        description=desc,
        confidence=conf,
    )


def detect_word_repeats(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Stessa parola (contenuto) ripetuta subito dopo: «fornisce fornisce»."""
    if not cfg.enable_word_repeat or len(words) < 2:
        return []
    stops = _stops(lang)
    errors: list[EditError] = []
    i = 0
    while i < len(words) - 1:
        a, b = words[i], words[i + 1]
        gap = b.start - a.end
        if (
            a.text == b.text
            and gap <= cfg.max_repeat_gap
            and (not cfg.ignore_stopword_unigrams or _is_content(a.text, stops))
        ):
            errors.append(_emit_repeat(
                words, i, 1, video_duration, lang,
                cfg.word_repeat_confidence, cfg,
            ))
            i += 2  # non segnalare tripli come due eventi
        else:
            i += 1
    return errors


def detect_ngram_repeats(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """N-gram stutter: «a un soggetto» «a un soggetto».

    Richiede almeno una content-word nell'n-gram per evitare
    «di di» / «a a» da soli.
    """
    if not cfg.enable_ngram_repeat or len(words) < 4:
        return []
    stops = _stops(lang)
    errors: list[EditError] = []
    covered: set[int] = set()  # indici gia' coperti da un match piu' lungo
    for n in sorted(cfg.ngram_sizes, reverse=True):
        if len(words) < 2 * n:
            continue
        i = 0
        while i <= len(words) - 2 * n:
            if any(j in covered for j in range(i, i + 2 * n)):
                i += 1
                continue
            left = words[i:i + n]
            right = words[i + n:i + 2 * n]
            texts_l = [w.text for w in left]
            texts_r = [w.text for w in right]
            gap = right[0].start - left[-1].end
            if (
                texts_l == texts_r
                and gap <= cfg.max_repeat_gap
                and any(_is_content(t, stops) for t in texts_l)
            ):
                errors.append(_emit_repeat(
                    words, i, n, video_duration, lang,
                    cfg.ngram_repeat_confidence, cfg,
                ))
                covered.update(range(i, i + 2 * n))
                i += 2 * n
            else:
                i += 1
    return errors


def detect_fillers(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Filler evidenti da tagliare: ehh, uhm, mh, ..."""
    if not cfg.enable_fillers:
        return []
    errors: list[EditError] = []
    for w in words:
        if lang.filler_re.fullmatch(w.text):
            errors.append(EditError(
                type="missed_cut",
                start=max(0.0, w.start - cfg.pad_before),
                end=min(video_duration, w.end + cfg.pad_after),
                description=lang.filler_desc.format(quote=w.text),
                confidence=cfg.filler_confidence,
            ))
    return errors


def detect_speech_edit_errors(
    segments: list[TranscriptSegment],
    video_duration: float,
    language: str | LanguagePack = "it",
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
    baseline_fn=None,
) -> list[EditError]:
    """Pipeline parlato: word-level + (opzionale) baseline a segmenti."""
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    errors: list[EditError] = []
    if cfg.enable_segment_baseline and baseline_fn is not None:
        errors.extend(baseline_fn(segments, video_duration, language=lang))

    words = _words_from_segments(segments)
    # Prima n-gram (piu' lunghi), poi unigram: evita doppio conteggio
    # «fornisce fornisce» dentro un n-gram gia' segnalato.
    ngram_errs = detect_ngram_repeats(words, video_duration, lang, cfg)
    errors.extend(ngram_errs)
    covered_spans = [(e.start, e.end) for e in ngram_errs]

    def _overlaps(start: float, end: float) -> bool:
        for a, b in covered_spans:
            if start < b and end > a:
                return True
        return False

    for err in detect_word_repeats(words, video_duration, lang, cfg):
        if not _overlaps(err.start, err.end):
            errors.append(err)
    errors.extend(detect_fillers(words, video_duration, lang, cfg))
    return errors
