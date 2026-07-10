"""Ensemble: trascrivi a più temperature e unisci gli errori speech.

I modelli medium/large a temp bassa collassano gli stutter; a temp alta
a volte li preservano. Unire i risultati di 2-3 passaggi aumenta il recall
a costo di tempo (N× whisper).
"""

from __future__ import annotations

import os
from dataclasses import replace
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
    speech_mode: bool = True,
    log=print,
) -> list:
    """Esegue whisper a più temperature; restituisce lista di liste di segmenti.

    Ripristina sempre WHISPER_TEMPERATURE (anche se una trascrizione fallisce).
    """
    all_segs = []
    prev_temp = os.environ.get("WHISPER_TEMPERATURE")
    try:
        for temp in temperatures:
            os.environ["WHISPER_TEMPERATURE"] = str(temp)
            sub = workdir / f"temp_{temp:g}"
            segs = transcribe_video(
                video, sub, model_label=model_label, language=language,
                speech_mode=speech_mode, log=log,
            )
            all_segs.append(segs)
            log(f"  temp={temp:g}: {len(segs)} segmenti, "
                f"{sum(len(s.words) for s in segs)} parole")
    finally:
        if prev_temp is None:
            os.environ.pop("WHISPER_TEMPERATURE", None)
        else:
            os.environ["WHISPER_TEMPERATURE"] = prev_temp
    return all_segs


def detect_ensemble(
    segment_lists: list,
    video_duration: float,
    language: str | LanguagePack = "it",
    cfg: SpeechEditConfig | None = None,
) -> list[EditError]:
    """Applica speech-edit a ogni trascrizione e fa merge.

    La baseline segment-level gira una sola volta (prima trascrizione non vuota);
    i detector word-level restano in unione su tutte le temperature.
    """
    cfg = cfg or SpeechEditConfig()
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    errors: list[EditError] = []
    word_cfg = replace(cfg, enable_segment_baseline=False)

    baseline_segs = next((s for s in segment_lists if s), None)
    if baseline_segs is not None and cfg.enable_segment_baseline:
        errors.extend(detect_transcript_errors(
            baseline_segs, video_duration, language=lang,
        ))

    for segs in segment_lists:
        if not segs:
            continue
        errors.extend(detect_speech_edit_errors(
            segs, video_duration, language=lang, cfg=word_cfg,
        ))
    return merge_errors(errors)
