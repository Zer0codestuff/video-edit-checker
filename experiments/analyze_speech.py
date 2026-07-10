#!/usr/bin/env python3
"""Analisi solo-parlato da CLI (senza Gradio).

Uso:
  python -m experiments.analyze_speech video.mp4
  python -m experiments.analyze_speech video.mp4 --model "Small Q8 (~250 MB, piu veloce)" --ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.binaries import setup_path
from core.heuristics import detect_visual_heuristics
from core.language import resolve_language
from core.report import export_json, filter_errors, fmt_time, merge_errors
from core.speech_edits import detect_speech_edit_errors
from core.speech_ensemble import detect_ensemble, transcribe_multi_temp
from core.whisper_cpp import detect_transcript_errors, transcribe_video
from core.windows import make_windows, probe_duration

setup_path()


def main() -> None:
    ap = argparse.ArgumentParser(description="Video Edit Checker — solo parlato")
    ap.add_argument("video", type=Path, help="File video o wav")
    ap.add_argument("--model", default="Small Q8 (~250 MB, piu veloce)")
    ap.add_argument("--language", default="it")
    ap.add_argument("--ensemble", action="store_true",
                    help="Whisper a temp 0.0 e 0.8, unisci errori")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        sys.exit(f"File non trovato: {video}")
    outdir = (args.outdir or (ROOT / "runs" / f"speech_{video.stem}")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    lang = resolve_language(args.language)

    print(f"Video: {video}")
    print(f"Lingua: {lang.code}  modello: {args.model}  ensemble={args.ensemble}")

    errors = []
    try:
        wins = make_windows(video, outdir / "windows", log=print,
                            with_audio=False, with_clips=False)
        errors.extend(detect_visual_heuristics(wins, log=print, language=lang))
    except RuntimeError as err:
        print(f"Niente frame (input audio-only?): {err}; salto euristiche pixel.")

    if args.ensemble:
        seg_lists = transcribe_multi_temp(
            video, outdir / "whisper", temperatures=(0.0, 0.8),
            model_label=args.model, language=lang.code,
            speech_mode=True, log=print)
        errors.extend(detect_ensemble(seg_lists, probe_duration(video), language=lang))
    else:
        segs = transcribe_video(
            video, outdir / "whisper", model_label=args.model,
            language=lang.code, speech_mode=True, log=print)
        if segs:
            # Durata: da segmenti se ffprobe fallisce su wav strani
            try:
                dur = probe_duration(video)
            except Exception:
                dur = max((s.end for s in segs), default=1.0) + 1.0
            errors.extend(detect_speech_edit_errors(
                segs, dur, language=lang,
                baseline_fn=detect_transcript_errors))

    errors = filter_errors(merge_errors(errors), args.min_confidence)
    print(f"\n{len(errors)} errori:")
    for e in errors:
        print(f"  {e.label} {fmt_time(e.start)}–{fmt_time(e.end)} "
              f"(conf {e.confidence:.2f}): {e.description}")

    report = export_json(video.name, errors, outdir / f"{video.stem}_report.json")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
