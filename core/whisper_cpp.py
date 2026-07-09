"""Trascrizione audio con whisper.cpp e rilevamento errori testuali."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from core.models import EditError

CACHE_DIR = Path.home() / ".cache" / "whisper.cpp"
DEFAULT_MODEL_NAME = "ggml-large-v3-turbo-q5_0.bin"
DEFAULT_MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{DEFAULT_MODEL_NAME}"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


TRIGGER_RE = re.compile(
    r"\b("
    r"lo ripeto|ripeto|aspetta|rifacciamo|da capo|taglia|tagliamo|"
    r"scusa|sbagliato|ho sbagliato|errore|riparto|riprovo|un attimo"
    r")\b",
    re.IGNORECASE,
)


def find_default_model() -> Path | None:
    env = os.environ.get("WHISPER_CPP_MODEL", "").strip()
    candidates = [
        Path(env) if env else None,
        Path.home() / ".cache" / "whisper.cpp" / "ggml-large-v3-turbo-q5_0.bin",
        Path.home() / ".cache" / "whisper.cpp" / "ggml-large-v3-turbo-q8_0.bin",
        Path.home() / ".cache" / "whisper.cpp" / "ggml-large-v3-turbo.bin",
        Path.home() / ".cache" / "whisper.cpp" / "ggml-base.bin",
        Path.home() / ".cache" / "whisper.cpp" / "ggml-small.bin",
        Path("/opt/homebrew/share/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin"),
        Path("/opt/homebrew/share/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin"),
        Path("/opt/homebrew/share/whisper.cpp/models/ggml-large-v3-turbo.bin"),
        Path("/opt/homebrew/share/whisper.cpp/models/ggml-base.bin"),
        Path("/opt/homebrew/share/whisper.cpp/models/ggml-small.bin"),
        Path("/usr/local/share/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin"),
        Path("/usr/local/share/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin"),
        Path("/usr/local/share/whisper.cpp/models/ggml-large-v3-turbo.bin"),
        Path("/usr/local/share/whisper.cpp/models/ggml-base.bin"),
        Path("/usr/local/share/whisper.cpp/models/ggml-small.bin"),
    ]
    for path in candidates:
        if path and path.exists():
            return path
    return None


def ensure_default_model(log=print) -> Path | None:
    found = find_default_model()
    if found is not None:
        return found
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / DEFAULT_MODEL_NAME
    log(f"Scarico modello whisper.cpp {DEFAULT_MODEL_NAME} in {target}...")
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, target)
    except Exception as err:
        log(f"Download modello whisper.cpp fallito: {err}")
        return None
    return target if target.exists() else None


def _to_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000.0 if value > 10000 else float(value)
    text = str(value or "").strip().strip("[]")
    if not text:
        return 0.0
    if ":" not in text:
        return float(text.replace(",", "."))
    parts = text.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def _parse_json(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("transcription") or data.get("segments") or []
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ts = item.get("timestamps") or {}
        start = item.get("start", ts.get("from", 0))
        end = item.get("end", ts.get("to", start))
        segments.append(TranscriptSegment(_to_seconds(start), _to_seconds(end), text))
    return segments


def transcribe_video(
    video: Path,
    workdir: Path,
    model_path: str = "",
    language: str = "it",
    log=print,
) -> list[TranscriptSegment]:
    whisper_bin = shutil.which("whisper-cli")
    if whisper_bin is None:
        log("whisper-cli non trovato: installa whisper.cpp oppure usa la pipeline omni.")
        return []

    model = Path(model_path).expanduser() if model_path.strip() else ensure_default_model(log=log)
    if model is None or not model.exists():
        log(
            "Modello whisper.cpp non trovato. Inserisci il path nella UI oppure imposta "
            "WHISPER_CPP_MODEL=/path/to/ggml-large-v3-turbo-q5_0.bin"
        )
        return []

    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = workdir / "audio_16khz.wav"
    out_base = workdir / "transcript"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", str(audio_path),
        ],
        capture_output=True,
        check=False,
    )
    if not audio_path.exists():
        log("Impossibile estrarre l'audio per whisper.cpp.")
        return []

    cmd = [
        whisper_bin,
        "-m", str(model),
        "-f", str(audio_path),
        "-l", language,
        "-oj",
        "-ojf",
        "-of", str(out_base),
        "-np",
    ]
    log(f"Trascrivo audio con whisper.cpp ({model.name})...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "errore sconosciuto"
        log(f"Errore whisper.cpp: {msg}")
        return []

    json_path = out_base.with_suffix(".json")
    if not json_path.exists():
        log("whisper.cpp non ha prodotto il JSON atteso.")
        return []
    try:
        segments = _parse_json(json_path)
    except Exception as err:
        log(f"Errore parsing transcript whisper.cpp: {err}")
        return []
    log(f"Trascrizione completata: {len(segments)} segmenti.")
    return segments


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sàèéìòù]", " ", text.lower())).strip()


def detect_transcript_errors(segments: list[TranscriptSegment], video_duration: float) -> list[EditError]:
    errors: list[EditError] = []
    for seg in segments:
        if TRIGGER_RE.search(seg.text):
            errors.append(EditError(
                type="missed_cut",
                start=max(0.0, seg.start - 1.0),
                end=min(video_duration, seg.end + 4.0),
                description=f"Possibile taglio mancato: frase di ripresa nel parlato («{seg.text[:120]}»).",
                confidence=0.82,
            ))

    for prev, cur in zip(segments, segments[1:]):
        gap = cur.start - prev.end
        prev_norm = _norm(prev.text)
        cur_norm = _norm(cur.text)
        if prev_norm and cur_norm:
            ratio = SequenceMatcher(None, prev_norm, cur_norm).ratio()
            if ratio >= 0.72 and gap <= 10.0:
                errors.append(EditError(
                    type="repeated_phrase",
                    start=prev.start,
                    end=cur.end,
                    description=f"Possibile frase ripetuta: «{prev.text[:80]}» / «{cur.text[:80]}».",
                    confidence=min(0.95, 0.55 + ratio * 0.4),
                ))
        if gap >= 5.0 and prev.end > 2.0 and cur.start < video_duration - 2.0:
            errors.append(EditError(
                type="audio_glitch",
                start=prev.end,
                end=cur.start,
                description=f"Silenzio o vuoto audio anomalo di circa {gap:.1f} secondi.",
                confidence=min(0.9, 0.55 + gap / 20.0),
            ))
    return errors
