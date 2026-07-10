"""Aggregazione degli errori, thumbnail ed export JSON/CSV."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from core.constants import BLACK_MIN_DURATION_SECONDS
from core.models import ERROR_TYPES, EditError

MERGE_GAP_SECONDS = 3.0
# Errori di parlato distinti (parole diverse) non vanno fusi solo perche'
# cadono entro MERGE_GAP: altrimenti «soggetto» + «fornisce» a 2s di distanza
# diventano un solo evento e se ne perde uno nel report.
_SPEECH_TYPES = frozenset({"repeated_phrase", "missed_cut"})
# audio_glitch: descrizioni boilerplate quasi identiche ("Silenzio o vuoto...");
# fondi solo se gli intervalli si sovrappongono/toccano, non per similarita' testuale.
_OVERLAP_ONLY_TYPES = frozenset({"audio_glitch"})


def _speech_same_event(a: EditError, b: EditError) -> bool:
    """True se due errori speech sembrano lo stesso evento (stessa citazione)."""
    import re
    qa = re.findall(r"[«\"]([^»\"]+)[»\"]", a.description)
    qb = re.findall(r"[«\"]([^»\"]+)[»\"]", b.description)
    if qa and qb:
        return qa[0].strip().lower() == qb[0].strip().lower()
    # Senza citazione: fondi solo se le description condividono molte parole.
    wa = set(re.findall(r"[a-zàèéìòù]+", a.description.lower()))
    wb = set(re.findall(r"[a-zàèéìòù]+", b.description.lower()))
    if not wa or not wb:
        return True
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.5


def merge_errors(errors: list[EditError]) -> list[EditError]:
    """Unisce errori dello stesso tipo vicini/sovrapposti (da finestre overlappanti)."""
    merged: list[EditError] = []
    for err in sorted(errors, key=lambda e: (e.type, e.start)):
        gap = 0.0 if err.type in _OVERLAP_ONLY_TYPES else MERGE_GAP_SECONDS
        if (
            merged
            and merged[-1].type == err.type
            and err.start <= merged[-1].end + gap
            and (err.type not in _SPEECH_TYPES or _speech_same_event(merged[-1], err))
        ):
            prev = merged[-1]
            prev.end = max(prev.end, err.end)
            prev.confidence = max(prev.confidence, err.confidence)
            if len(err.description) > len(prev.description):
                prev.description = err.description
        else:
            merged.append(EditError(err.type, err.start, err.end, err.description, err.confidence))
    return sorted(merged, key=lambda e: e.start)


def filter_errors(errors: list[EditError], min_confidence: float) -> list[EditError]:
    out: list[EditError] = []
    for e in errors:
        if e.confidence < min_confidence:
            continue
        if e.type == "black_screen" and (e.end - e.start) <= BLACK_MIN_DURATION_SECONDS:
            continue
        out.append(e)
    return out


def extract_thumbnail(video: Path, t: float, out_path: Path) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "4", str(out_path)],
        capture_output=True, check=False,
    )
    return out_path if out_path.exists() else None


def fmt_time(seconds: float) -> str:
    """Formatta secondi come M:SS o H:MM:SS (arrotondata al secondo)."""
    total = max(0, int(round(seconds)))
    m, s = divmod(total, 60)
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


def export_batch(results: dict[str, list[EditError]], out_json: Path,
                 out_csv: Path) -> tuple[Path, Path]:
    """Esporta il report combinato di tutti i video di una run (playlist)."""
    data = {
        "videos": [
            {
                "video": name,
                "error_count": len(errors),
                "errors": [
                    {
                        "type": e.type,
                        "label": ERROR_TYPES.get(e.type, e.type),
                        "start": fmt_time(e.start),
                        "end": fmt_time(e.end),
                        "description": e.description,
                        "confidence": round(e.confidence, 2),
                    }
                    for e in errors
                ],
            }
            for name, errors in results.items()
        ],
        "total_errors": sum(len(v) for v in results.values()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "tipo", "inizio", "fine", "descrizione", "confidence"])
        for name, errors in results.items():
            for e in errors:
                writer.writerow([name, e.type, fmt_time(e.start), fmt_time(e.end),
                                 e.description, f"{e.confidence:.2f}"])
    return out_json, out_csv


def batch_summary_md(results: dict[str, list[EditError]]) -> str:
    """Riepilogo markdown della run: una riga per video + dettaglio errori."""
    if not results:
        return "Nessun video analizzato."
    total = sum(len(v) for v in results.values())
    clean = sum(1 for v in results.values() if not v)
    lines = [
        f"## 📊 Riepilogo run — {len(results)} video, "
        f"{total} error{'e' if total == 1 else 'i'} "
        f"({clean} video pulit{'o' if clean == 1 else 'i'})",
        "",
        "| Video | Errori | Tipi rilevati |",
        "|---|---|---|",
    ]
    for name, errors in results.items():
        if errors:
            counts: dict[str, int] = {}
            for e in errors:
                counts[e.label] = counts.get(e.label, 0) + 1
            types = ", ".join(f"{lbl} ×{n}" for lbl, n in counts.items())
            lines.append(f"| {name} | {len(errors)} | {types} |")
        else:
            lines.append(f"| {name} | 0 | ✅ nessun errore |")
    for name, errors in results.items():
        if not errors:
            continue
        lines += ["", f"### {name}"]
        for e in errors:
            lines.append(
                f"- {e.label} `{fmt_time(e.start)}–{fmt_time(e.end)}` "
                f"(conf {e.confidence:.2f}): {e.description}")
    return "\n".join(lines)
