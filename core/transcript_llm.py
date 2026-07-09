"""Analisi LLM del transcript (senza vision) per errori di montaggio parlato.

Esperimento: dopo whisper, chiedere a un modello testo-only di segnalare
parole ripetute, stutter e filler citando le parole esatte. Utile quando
le euristiche deterministiche falliscono su ASR rumoroso, ma puo' inventare.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.llm_client import call_chat, extract_json
from core.models import EditError
from core.whisper_cpp import TranscriptSegment

PROMPT_IT = """\
Sei un montatore video. Analizza la trascrizione word-level e trova errori da TAGLIARE.

Regola principale: se la STESSA parola (o la stessa sequenza di 2-4 parole) appare
due volte di fila nelle righe consecutive, DEVI segnalarla come repeated_phrase.
Esempio obbligatorio: se vedi
  1.40-2.00 fornisce
  2.50-3.20 fornisce
allora output:
{{"errors":[{{"type":"repeated_phrase","start":1.4,"end":3.2,"description":"Parola ripetuta «fornisce»","confidence":0.9}}]}}

Altri errori da segnalare:
- n-gram stutter (es. "a un soggetto" poi di nuovo "a un soggetto")
- filler isolati: ehh, ehm, uhm, mh, mmm
- frasi di ripresa: aspetta, lo ripeto, rifacciamo

NON segnalare: liste (probabilità, impatto, probabilità residua), ritornelli
intenzionali su frasi diverse, pause, contenuto del discorso.

Trascrizione (ogni riga: start-end testo):
{transcript}

Rispondi SOLO con JSON valido, senza altro testo.
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


def _format_transcript(segments: list[TranscriptSegment], max_chars: int = 6000) -> str:
    lines: list[str] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                lines.append(f"{w.start:.2f}-{w.end:.2f} {w.text}")
        else:
            lines.append(f"{seg.start:.2f}-{seg.end:.2f} {seg.text}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n..."
    return text


def analyze_transcript_with_llm(
    segments: list[TranscriptSegment],
    language: str | LanguagePack = "it",
    timeout: float = 120.0,
    log=print,
) -> list[EditError]:
    """Chiama llama-server (modello gia' avviato) sul transcript formattato."""
    if not segments:
        return []
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    template = PROMPT_IT if lang.code == "it" else PROMPT_EN
    prompt = template.format(transcript=_format_transcript(segments))
    content = [{"type": "text", "text": prompt}]
    text = call_chat(
        content,
        timeout=timeout,
        max_tokens=800,
        log=log,
        error_label="transcript-llm",
        enable_thinking=False,
    )
    if text is None:
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
