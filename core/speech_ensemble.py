"""Ensemble: trascrivi a più temperature e unisci gli errori speech.

I modelli medium/large a temp bassa collassano gli stutter; a temp alta
a volte li preservano. Unire i risultati di 2-3 passaggi aumenta il recall
a costo di tempo (N× whisper).
"""

from __future__ import annotations

import os
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.models import EditError
from core.report import merge_errors
from core.speech_edits import SpeechEditConfig, detect_speech_edit_errors
from core.whisper_cpp import detect_transcript_errors, transcribe_video


def transcribe_multi_temp(
    video: Path,
    workdir: Path,
    temperatures: tuple[float, ...] = (0.0, 0.8),
    model_label: str = "",
    language: str = "it",
    log=print,
) -> list:
    """Esegue whisper a più temperature; restituisce lista di liste di segmenti."""
    all_segs = []
    for i, temp in enumerate(temperatures):
        os.environ["WHISPER_TEMPERATURE"] = str(temp)
        sub = workdir / f"temp_{temp:g}"
        segs = transcribe_video(
            video, sub, model_label=model_label, language=language, log=log,
        )
        all_segs.append(segs)
        log(f"  temp={temp:g}: {len(segs)} segmenti, "
            f"{sum(len(s.words) for s in segs)} parole")
    return all_segs


def detect_ensemble(
    segment_lists: list,
    video_duration: float,
    language: str | LanguagePack = "it",
    cfg: SpeechEditConfig | None = None,
) -> list[EditError]:
    """Applica speech-edit a ogni trascrizione e fa merge."""
    cfg = cfg or SpeechEditConfig()
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    errors: list[EditError] = []
    for segs in segment_lists:
        if not segs:
            continue
        errors.extend(detect_speech_edit_errors(
            segs, video_duration, language=lang, cfg=cfg,
            baseline_fn=detect_transcript_errors,
        ))
    return merge_errors(errors)
