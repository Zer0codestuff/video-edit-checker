"""Aggregazione degli errori, thumbnail ed export JSON/CSV."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from core.analyzer import ERROR_TYPES, EditError

MERGE_GAP_SECONDS = 3.0


def merge_errors(errors: list[EditError]) -> list[EditError]:
    """Unisce errori dello stesso tipo vicini/sovrapposti (da finestre overlappanti)."""
    merged: list[EditError] = []
    for err in sorted(errors, key=lambda e: (e.type, e.start)):
        if merged and merged[-1].type == err.type and err.start <= merged[-1].end + MERGE_GAP_SECONDS:
            prev = merged[-1]
            prev.end = max(prev.end, err.end)
            prev.confidence = max(prev.confidence, err.confidence)
            if len(err.description) > len(prev.description):
                prev.description = err.description
        else:
            merged.append(EditError(err.type, err.start, err.end, err.description, err.confidence))
    return sorted(merged, key=lambda e: e.start)


def filter_errors(errors: list[EditError], min_confidence: float) -> list[EditError]:
    return [e for e in errors if e.confidence >= min_confidence]


def extract_thumbnail(video: Path, t: float, out_path: Path) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "4", str(out_path)],
        capture_output=True, check=False,
    )
    return out_path if out_path.exists() else None


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def export_json(video_name: str, errors: list[EditError], out_path: Path) -> Path:
    data = {
        "video": video_name,
        "errors": [
            {
                "type": e.type,
                "label": ERROR_TYPES.get(e.type, e.type),
                "start_seconds": round(e.start, 2),
                "end_seconds": round(e.end, 2),
                "start": fmt_time(e.start),
                "end": fmt_time(e.end),
                "description": e.description,
                "confidence": round(e.confidence, 2),
            }
            for e in errors
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_csv(video_name: str, errors: list[EditError], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "tipo", "inizio", "fine", "descrizione", "confidence"])
        for e in errors:
            writer.writerow([video_name, e.type, fmt_time(e.start), fmt_time(e.end),
                             e.description, f"{e.confidence:.2f}"])
    return out_path
