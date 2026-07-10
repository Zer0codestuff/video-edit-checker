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
    # Fallback regex sul testo del segmento (utile se i token BPE sono rumorosi).
    enable_text_fallback: bool = True
    # Riprese non adiacenti: stessa parola/n-gram con materiale in mezzo
    # (tipico di false-start ASR: «potrebbe non prevenuto, potrebbe…»).
    enable_near_repeats: bool = True
    # Falso inizio morfologico: «fornisce, … fornire» con cue di ripresa.
    enable_stem_restarts: bool = True
    # Ripetizioni adiacenti di sole function-word («il il») con gap strettissimo.
    enable_function_word_repeats: bool = True
    ngram_sizes: tuple[int, ...] = (2, 3, 4)
    # Gap massimo (s) tra fine della 1a occorrenza e inizio della 2a.
    max_repeat_gap: float = 1.5
    # Near-repeat: finestra piu' ampia, ma con ancore lessicali piu' forti.
    near_repeat_max_gap: float = 3.2
    near_repeat_max_intervening: int = 5
    near_unigram_max_gap: float = 2.5
    near_unigram_max_intervening: int = 4
    max_function_repeat_gap: float = 0.35
    stem_restart_max_gap: float = 2.8
    # Ignora ripetizioni di sole stopword (n=1) nel detector adiacente classico.
    ignore_stopword_unigrams: bool = True
    # Confidence base.
    word_repeat_confidence: float = 0.86
    ngram_repeat_confidence: float = 0.88
    near_repeat_confidence: float = 0.84
    stem_restart_confidence: float = 0.72
    function_word_confidence: float = 0.78
    filler_confidence: float = 0.8
    # Espandi leggermente l'intervallo segnalato per il taglio.
    pad_before: float = 0.15
    pad_after: float = 0.35


# Function-word / deittici spesso ripetuti in retorica: esclusi dai near-unigram.
_NEAR_WEAK_IT = frozenset({
    "del", "della", "dei", "degli", "delle", "nel", "nella", "nei", "nelle",
    "al", "allo", "ai", "agli", "alla", "alle", "sul", "sulla", "sullo",
    "ha", "ho", "hai", "hanno", "è", "sono", "sia", "siamo", "siete",
    "che", "chi", "cui", "come", "quando", "dove", "quanto",
    "questo", "questa", "questi", "queste", "quello", "quella",
    "più", "meno", "molto", "poco", "ogni", "altro", "altra",
    "n", "né", "ne",
})
_NEAR_WEAK_EN = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "this", "that", "these", "those", "it", "we", "they", "you",
    "more", "less", "very", "just", "also",
})

# Cue lessicali di ripresa dopo un falso inizio (modali/ausiliari/congiunzioni).
_RESTART_CUES_IT = frozenset({
    "sarebbe", "potrebbe", "dovrebbe", "potrebbero", "dovrebbero",
    "deve", "devono", "può", "possono", "puo",
    "è", "era", "sarà", "sono", "sia",
    "ha", "hanno", "aveva", "avrebbe",
    "quindi", "però", "pero", "invece", "cioè", "cioe", "ovvero",
    "oppure", "o", "ma", "anzi",
})
_RESTART_CUES_EN = frozenset({
    "would", "could", "should", "might", "must", "can", "will",
    "is", "are", "was", "were", "has", "have", "had",
    "so", "but", "or", "instead", "actually", "rather",
})


DEFAULT_SPEECH_EDIT_CONFIG = SpeechEditConfig()


def _stops(lang: LanguagePack) -> frozenset[str]:
    return _STOP_EN if lang.code == "en" else _STOP_IT


def _near_weak(lang: LanguagePack) -> frozenset[str]:
    return _NEAR_WEAK_EN if lang.code == "en" else _NEAR_WEAK_IT


def _restart_cues(lang: LanguagePack) -> frozenset[str]:
    return _RESTART_CUES_EN if lang.code == "en" else _RESTART_CUES_IT


def _is_content(token: str, stops: frozenset[str]) -> bool:
    return bool(token) and token not in stops and len(token) > 1


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _strong_content(token: str, stops: frozenset[str], weak: frozenset[str]) -> bool:
    return _is_content(token, stops) and token not in weak and len(token) > 2


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


def detect_near_ngram_repeats(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """N-gram ripetuto con parole in mezzo (ripresa/false-start).

    Richiede ancore lessicali solide per limitare ripetizioni retoriche
    tipo «del rischio» … «del rischio».
    """
    if not cfg.enable_near_repeats or not cfg.enable_ngram_repeat or len(words) < 5:
        return []
    stops = _stops(lang)
    weak = _near_weak(lang)
    errors: list[EditError] = []
    covered: set[int] = set()
    for n in sorted(cfg.ngram_sizes, reverse=True):
        if len(words) < 2 * n + 1:
            continue
        i = 0
        while i <= len(words) - 2 * n:
            if any(j in covered for j in range(i, i + n)):
                i += 1
                continue
            left = words[i:i + n]
            texts = [w.text for w in left]
            strong = [t for t in texts if _strong_content(t, stops, weak)]
            if len(strong) >= 2:
                ok = True
            elif n >= 3 and len(strong) == 1 and len(strong[0]) >= 6:
                ok = True
            else:
                ok = False
            if not ok:
                i += 1
                continue
            max_j = min(
                len(words) - n + 1,
                i + n + cfg.near_repeat_max_intervening + 1,
            )
            for j in range(i + n, max_j):
                right = words[j:j + n]
                if [w.text for w in right] != texts:
                    continue
                intervening = j - (i + n)
                gap = right[0].start - left[-1].end
                if intervening < 1 or gap < 0 or gap > cfg.near_repeat_max_gap:
                    continue
                if any(k in covered for k in range(i, j + n)):
                    continue
                errors.append(EditError(
                    type="repeated_phrase",
                    start=max(0.0, left[0].start - cfg.pad_before),
                    end=min(video_duration, right[-1].end + cfg.pad_after),
                    description=lang.ngram_repeat_desc.format(
                        quote=" ".join(texts), n=n,
                    ),
                    confidence=cfg.near_repeat_confidence,
                ))
                covered.update(range(i, j + n))
                break
            i += 1
    return errors


def detect_near_word_repeats(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
    covered: set[int] | None = None,
) -> list[EditError]:
    """Parola contenuto ripetuta con 1+ token in mezzo entro una finestra breve."""
    if not cfg.enable_near_repeats or not cfg.enable_word_repeat or len(words) < 3:
        return []
    stops = _stops(lang)
    weak = _near_weak(lang)
    cues = _restart_cues(lang)
    covered = covered if covered is not None else set()
    errors: list[EditError] = []
    for i, a in enumerate(words):
        if i in covered:
            continue
        if not _strong_content(a.text, stops, weak) or len(a.text) < 5:
            continue
        max_j = min(len(words), i + 1 + cfg.near_unigram_max_intervening + 1)
        for j in range(i + 1, max_j):
            if j in covered:
                continue
            b = words[j]
            if b.text != a.text:
                continue
            intervening = j - i - 1
            gap = b.start - a.end
            if intervening < 1 or gap < 0 or gap > cfg.near_unigram_max_gap:
                continue
            nxt = words[i + 1].text if i + 1 < len(words) else ""
            if (
                intervening > 2
                and nxt not in cues
                and not any(words[k].text in cues for k in range(i + 1, j))
            ):
                continue
            errors.append(EditError(
                type="repeated_phrase",
                start=max(0.0, a.start - cfg.pad_before),
                end=min(video_duration, b.end + cfg.pad_after),
                description=lang.word_repeat_desc.format(quote=a.text),
                confidence=cfg.near_repeat_confidence,
            ))
            covered.update(range(i, j + 1))
            break
    return errors


def detect_function_word_repeats(
    words: list[TranscriptWord],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Function-word adiacenti ripetute con gap molto stretto («il il», «di di»).

    Il detector unigram classico le ignora (troppo rumorose a gap larghi);
    qui il vincolo di gap le rende rare e spesso vere.
    """
    if not cfg.enable_function_word_repeats or len(words) < 2:
        return []
    stops = _stops(lang)
    errors: list[EditError] = []
    i = 0
    while i < len(words) - 1:
        a, b = words[i], words[i + 1]
        gap = b.start - a.end
        if (
            a.text == b.text
            and a.text in stops
            and len(a.text) >= 2
            and 0 <= gap <= cfg.max_function_repeat_gap
        ):
            errors.append(EditError(
                type="repeated_phrase",
                start=max(0.0, a.start - cfg.pad_before),
                end=min(video_duration, b.end + cfg.pad_after),
                description=lang.word_repeat_desc.format(quote=a.text),
                confidence=cfg.function_word_confidence,
            ))
            i += 2
        else:
            i += 1
    return errors


def detect_stem_restarts(
    words: list[TranscriptWord],
    segments: list[TranscriptSegment],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Falso inizio morfologico: stessa radice dopo una cue di ripresa.

    Esempio ASR: «…massimo fornisce, sarebbe in grado di fornire…»
    (la prima forma verbale e' un false-start lasciato nel montaggio).
    """
    if not cfg.enable_stem_restarts or len(words) < 4:
        return []
    stops = _stops(lang)
    weak = _near_weak(lang)
    cues = _restart_cues(lang)
    # Timestamp approssimati di virgole nel testo segmento (cue di false-start).
    comma_ends: list[float] = []
    for seg in segments:
        if "," not in seg.text:
            continue
        for w in seg.words or []:
            idx = seg.text.lower().find(w.text)
            if idx >= 0 and "," in seg.text[idx:idx + len(w.text) + 10]:
                comma_ends.append(w.end)

    errors: list[EditError] = []
    for i, a in enumerate(words):
        if not _strong_content(a.text, stops, weak) or len(a.text) < 6:
            continue
        pause = (words[i + 1].start - a.end) if i + 1 < len(words) else 0.0
        near_comma = any(abs(a.end - ct) <= 0.35 for ct in comma_ends)
        if pause < 0.1 and not near_comma:
            continue
        for j in range(i + 2, min(len(words), i + 6)):
            b = words[j]
            if b.text == a.text or len(b.text) < 5:
                continue
            pref = _shared_prefix(a.text, b.text)
            if pref < 5:
                continue
            if pref / max(len(a.text), len(b.text)) < 0.55 and pref < 6:
                continue
            gap = b.start - a.end
            if gap < 0.1 or gap > cfg.stem_restart_max_gap:
                continue
            intervening = [words[k].text for k in range(i + 1, j)]
            if not intervening or not any(t in cues for t in intervening):
                continue
            quote = f"{a.text} / {b.text}"
            errors.append(EditError(
                type="repeated_phrase",
                start=max(0.0, a.start - cfg.pad_before),
                end=min(video_duration, b.end + cfg.pad_after),
                description=lang.word_repeat_desc.format(quote=quote),
                confidence=cfg.stem_restart_confidence,
            ))
            break
    return errors


def _is_isolated_letter_filler(w: TranscriptWord, prev: TranscriptWord | None,
                               nxt: TranscriptWord | None) -> bool:
    """Singola lettera tipo «m»/«e» solo se isolata (gap intorno, durata breve).

    Whisper a volte riduce «ehh»/«ehm» a «m» o «e». Matchare ogni «e» nel
    discorso italiano sarebbe un disastro di FP: richiediamo isolamento.
    """
    if w.text not in {"m", "e", "eh"}:
        return False
    dur = w.end - w.start
    if dur > 0.55:
        return False
    gap_before = (w.start - prev.end) if prev is not None else 1.0
    gap_after = (nxt.start - w.end) if nxt is not None else 1.0
    return gap_before >= 0.25 and gap_after >= 0.25


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
    for i, w in enumerate(words):
        hit = bool(lang.filler_re.fullmatch(w.text))
        if not hit:
            prev = words[i - 1] if i > 0 else None
            nxt = words[i + 1] if i + 1 < len(words) else None
            hit = _is_isolated_letter_filler(w, prev, nxt)
        if hit:
            errors.append(EditError(
                type="missed_cut",
                start=max(0.0, w.start - cfg.pad_before),
                end=min(video_duration, w.end + cfg.pad_after),
                description=lang.filler_desc.format(quote=w.text),
                confidence=cfg.filler_confidence,
            ))
    return errors


def detect_text_fillers(
    segments: list[TranscriptSegment],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Filler nel testo del segmento (es. «ehh» / «ehm» isolati)."""
    if not cfg.enable_fillers:
        return []
    errors: list[EditError] = []
    token_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", re.UNICODE)
    for seg in segments:
        tokens = token_re.findall(seg.text)
        if not tokens:
            continue
        for i, tok in enumerate(tokens):
            if lang.filler_re.fullmatch(tok):
                frac = i / max(1, len(tokens))
                start = max(0.0, seg.start + frac * (seg.end - seg.start) - cfg.pad_before)
                end = min(video_duration, start + 0.6 + cfg.pad_after)
                errors.append(EditError(
                    type="missed_cut",
                    start=start,
                    end=end,
                    description=lang.filler_desc.format(quote=tok.lower()),
                    confidence=max(0.5, cfg.filler_confidence - 0.05),
                ))
    return errors


def detect_text_repeats(
    segments: list[TranscriptSegment],
    video_duration: float,
    lang: LanguagePack,
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
) -> list[EditError]:
    """Fallback sul testo del segmento: cattura «fornisce fornisce» anche
    quando i word-token BPE sono rumorosi o fusi male.

    Cerca unigram e n-gram (2..4) ripetuti adiacenti nel testo normalizzato.
    I timestamp sono quelli del segmento intero (meno precisi dei word-token).
    """
    if not (cfg.enable_word_repeat or cfg.enable_ngram_repeat):
        return []
    stops = _stops(lang)
    errors: list[EditError] = []
    token_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", re.UNICODE)
    for seg in segments:
        tokens = [t.lower() for t in token_re.findall(seg.text)]
        if len(tokens) < 2:
            continue
        covered: set[int] = set()
        sizes = []
        if cfg.enable_ngram_repeat:
            sizes.extend(sorted(cfg.ngram_sizes, reverse=True))
        if cfg.enable_word_repeat:
            sizes.append(1)
        for n in sizes:
            if len(tokens) < 2 * n:
                continue
            i = 0
            while i <= len(tokens) - 2 * n:
                if any(j in covered for j in range(i, i + 2 * n)):
                    i += 1
                    continue
                left = tokens[i:i + n]
                right = tokens[i + n:i + 2 * n]
                if left != right:
                    i += 1
                    continue
                if n == 1 and cfg.ignore_stopword_unigrams and not _is_content(left[0], stops):
                    i += 1
                    continue
                if n > 1 and not any(_is_content(t, stops) for t in left):
                    i += 1
                    continue
                quote = " ".join(left)
                # Timestamp grezzi: distribuisci sul segmento.
                frac = i / max(1, len(tokens))
                span = max(0.4, (seg.end - seg.start) * (2 * n) / max(1, len(tokens)))
                start = max(0.0, seg.start + frac * (seg.end - seg.start) - cfg.pad_before)
                end = min(video_duration, start + span + cfg.pad_after)
                if n == 1:
                    desc = lang.word_repeat_desc.format(quote=quote)
                    conf = cfg.word_repeat_confidence - 0.05  # testo < word-token
                else:
                    desc = lang.ngram_repeat_desc.format(quote=quote, n=n)
                    conf = cfg.ngram_repeat_confidence - 0.05
                errors.append(EditError(
                    type="repeated_phrase",
                    start=start,
                    end=end,
                    description=desc,
                    confidence=max(0.5, conf),
                ))
                covered.update(range(i, i + 2 * n))
                i += 2 * n
    return errors


def detect_speech_edit_errors(
    segments: list[TranscriptSegment],
    video_duration: float,
    language: str | LanguagePack = "it",
    cfg: SpeechEditConfig = DEFAULT_SPEECH_EDIT_CONFIG,
    baseline_fn=None,
) -> list[EditError]:
    """Pipeline parlato: word-level + near-repeat/restart + fallback + baseline."""
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    errors: list[EditError] = []
    if cfg.enable_segment_baseline and baseline_fn is not None:
        errors.extend(baseline_fn(segments, video_duration, language=lang))

    words = _words_from_segments(segments)
    # Prima n-gram adiacenti (piu' lunghi), poi unigram adiacenti.
    ngram_errs = detect_ngram_repeats(words, video_duration, lang, cfg)
    errors.extend(ngram_errs)
    covered_spans = [(e.start, e.end) for e in ngram_errs]

    def _overlaps(start: float, end: float) -> bool:
        for a, b in covered_spans:
            if start < b and end > a:
                return True
        return False

    def _add(err: EditError) -> None:
        if not _overlaps(err.start, err.end):
            errors.append(err)
            covered_spans.append((err.start, err.end))

    for err in detect_word_repeats(words, video_duration, lang, cfg):
        _add(err)

    # Near-repeat (riprese con materiale in mezzo) e function-word adiacenti.
    near_ng = detect_near_ngram_repeats(words, video_duration, lang, cfg)
    near_covered: set[int] = set()
    # Marca indici approssimati gia' coperti da span temporali near-ngram.
    if near_ng:
        for idx, w in enumerate(words):
            if any(e.start <= w.start and w.end <= e.end for e in near_ng):
                near_covered.add(idx)
    for err in near_ng:
        _add(err)
    for err in detect_near_word_repeats(
        words, video_duration, lang, cfg, covered=near_covered,
    ):
        _add(err)
    for err in detect_function_word_repeats(words, video_duration, lang, cfg):
        _add(err)
    for err in detect_stem_restarts(words, segments, video_duration, lang, cfg):
        _add(err)

    for err in detect_fillers(words, video_duration, lang, cfg):
        errors.append(err)
        covered_spans.append((err.start, err.end))
    if cfg.enable_fillers:
        for err in detect_text_fillers(segments, video_duration, lang, cfg):
            _add(err)

    # Fallback testo: aggiungi solo se non c'e' gia' un match sovrapposto.
    if cfg.enable_text_fallback:
        for err in detect_text_repeats(segments, video_duration, lang, cfg):
            _add(err)
    return errors
