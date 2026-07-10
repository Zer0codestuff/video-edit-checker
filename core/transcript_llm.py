"""Analisi LLM del transcript (senza vision) per errori di montaggio parlato.

Dopo whisper, un modello testo-only (es. Qwen3.5-4B) segnala parole ripetute,
stutter e filler citando le parole esatte. Utile come secondo parere rispetto
alle euristiche deterministiche; puo' inventare, quindi va valutato su GT.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.llm_client import call_chat, extract_json
from core.models import EditError
from core.report import merge_errors
from core.whisper_cpp import TranscriptSegment

# Modelli testo-only / vision usabili senza mmproj per analisi transcript.
TRANSCRIPT_LLM_MODELS: dict[str, tuple[str, bool]] = {
    # (hf_repo:quant, needs_jinja)
    "Qwen3.5 4B UD-Q4_K_XL (thinking)":
        ("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL", True),
    "Qwen3.5 4B MTP UD-Q4_K_XL (thinking + MTP)":
        ("unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL", True),
    "Gemma 4 E2B QAT (testo)":
        ("unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL", False),
}
DEFAULT_TRANSCRIPT_LLM = "Qwen3.5 4B UD-Q4_K_XL (thinking)"

PROMPT_IT = """\
Sei un montatore video. Trova SOLO errori da TAGLIARE nella trascrizione word-level.

Segnala SOLO se vedi chiaramente:
1. Stessa parola ripetuta subito dopo o con 1-4 parole in mezzo (false-start).
2. Stessa sequenza di 2-4 parole ripetuta a distanza breve.
3. Filler isolati: ehh, ehm, uhm, mh, mmm, em.
4. Frasi di ripresa: aspetta, lo ripeto, rifacciamo.

NON segnalare liste, ritornelli intenzionali, pause, contenuto, ripetizioni retoriche distanti.

IMPORTANTE:
- NON elencare tutte le righe.
- NON riscrivere la trascrizione.
- Scansiona in cerca di ripetizioni e rispondi SUBITO con JSON.
- Nella description cita le parole tra «».
- Usa i timestamp assoluti delle righe.

Trascrizione (start-end testo):
{transcript}

Rispondi SOLO con JSON valido, nient'altro:
{{"errors":[{{"type":"repeated_phrase|missed_cut|other","start":0.0,"end":1.0,"description":"...","confidence":0.0}}]}}
Se nessuno: {{"errors":[]}}
"""

PROMPT_EN = """\
You are a video editor. Below is a word-level transcript. Report ONLY speech
editing mistakes: immediate word/n-gram repeats, fillers (uh/um), retake phrases.
Do NOT report lists, intentional refrains, natural pauses, or topic content.
Quote exact words and use their absolute start/end seconds.

Transcript:
{transcript}

Reply with ONLY JSON:
{{"errors":[{{"type":"repeated_phrase|missed_cut|other","start":0.0,"end":1.0,"description":"...","confidence":0.0}}]}}
If none: {{"errors":[]}}
"""


def _format_transcript(
    segments: list[TranscriptSegment],
    *,
    t0: float | None = None,
    t1: float | None = None,
    max_chars: int = 12000,
) -> str:
    lines: list[str] = []
    for seg in segments:
        if t0 is not None and seg.end < t0:
            continue
        if t1 is not None and seg.start > t1:
            continue
        if seg.words:
            for w in seg.words:
                if t0 is not None and w.end < t0:
                    continue
                if t1 is not None and w.start > t1:
                    continue
                lines.append(f"{w.start:.2f}-{w.end:.2f} {w.text}")
        else:
            lines.append(f"{seg.start:.2f}-{seg.end:.2f} {seg.text}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n..."
    return text


def _parse_errors(text: str | None) -> list[EditError]:
    if not text:
        return []
    data = extract_json(text)
    errors: list[EditError] = []
    for item in data.get("errors", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            etype = str(item.get("type", "other")).strip().lower()
            if etype not in {"repeated_phrase", "missed_cut", "audio_glitch", "other"}:
                etype = "other"
            start = float(item.get("start", 0))
            end = float(item.get("end", start))
            conf = float(item.get("confidence", 0.5))
            desc = str(item.get("description", "")).strip()
        except (TypeError, ValueError):
            continue
        # Senza citazione abbassa confidence (stessa policy omni).
        if etype == "repeated_phrase" and not re.search(r"[«»\"']", desc):
            if not any(w in desc.lower() for w in ("ripet", "repeat", "duplic")):
                conf = min(conf, 0.45)
        errors.append(EditError(
            type=etype,
            start=max(0.0, start),
            end=max(start, end),
            description=desc or "Segnalato dal modello sul transcript",
            confidence=max(0.0, min(conf, 1.0)),
        ))
    return errors


def analyze_transcript_chunk(
    segments: list[TranscriptSegment],
    *,
    language: str | LanguagePack = "it",
    t0: float | None = None,
    t1: float | None = None,
    timeout: float = 300.0,
    max_tokens: int = 1600,
    enable_thinking: bool = True,
    log=print,
) -> list[EditError]:
    """Analizza una finestra temporale del transcript."""
    if not segments:
        return []
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    transcript = _format_transcript(segments, t0=t0, t1=t1)
    if not transcript.strip():
        return []
    template = PROMPT_IT if lang.code == "it" else PROMPT_EN
    prompt = template.format(transcript=transcript)
    text = call_chat(
        [{"type": "text", "text": prompt}],
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=0.2,
        log=log,
        error_label="transcript-llm",
        enable_thinking=enable_thinking,
        # Con thinking il json_schema a volte lascia content vuoto; il prompt
        # chiede gia' JSON e extract_json e' tollerante.
        json_schema=not enable_thinking,
    )
    errs = _parse_errors(text)
    if text and not errs:
        data = extract_json(text)
        if "errors" not in data:
            log(f"  transcript-LLM: JSON non parsabile ({len(text)} char).")
    return errs


def analyze_transcript_with_llm(
    segments: list[TranscriptSegment],
    language: str | LanguagePack = "it",
    timeout: float = 300.0,
    log=print,
    *,
    enable_thinking: bool = True,
    chunk_seconds: float = 120.0,
    overlap_seconds: float = 20.0,
    max_tokens: int = 1600,
) -> list[EditError]:
    """Analizza il transcript a chunk sovrapposti e unisce gli errori.

    `enable_thinking=True` richiede llama-server avviato con --jinja e
    reasoning_budget > 0 (o -1).
    """
    if not segments:
        return []
    dur = max((s.end for s in segments), default=0.0)
    if dur <= 0:
        return []

    # Clip corti: una sola chiamata.
    if dur <= chunk_seconds + 1.0:
        return analyze_transcript_chunk(
            segments, language=language, timeout=timeout,
            max_tokens=max_tokens, enable_thinking=enable_thinking, log=log,
        )

    step = max(30.0, chunk_seconds - overlap_seconds)
    errors: list[EditError] = []
    t = 0.0
    idx = 0
    while t < dur:
        t1 = min(dur + 0.1, t + chunk_seconds)
        idx += 1
        log(f"  transcript-LLM chunk {idx}: {t:.0f}s–{t1:.0f}s "
            f"(thinking={'on' if enable_thinking else 'off'})")
        errors.extend(analyze_transcript_chunk(
            segments, language=language, t0=t, t1=t1,
            timeout=timeout, max_tokens=max_tokens,
            enable_thinking=enable_thinking, log=log,
        ))
        if t1 >= dur:
            break
        t += step
    return merge_errors(errors)
