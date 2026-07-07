"""Divisione del video in finestre temporali: frame campionati + audio WAV."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

WINDOW_SECONDS = 20.0
OVERLAP_SECONDS = 2.0
FRAME_EVERY_SECONDS = 3.0
FRAME_MAX_SIDE = 448


@dataclass
class Window:
    index: int
    start: float  # secondi assoluti nel video
    duration: float
    frame_paths: list[Path] = field(default_factory=list)
    frame_times: list[float] = field(default_factory=list)  # assoluti
    audio_path: Path | None = None


def probe_duration(video: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(res.stdout.strip())


def has_audio(video: Path) -> bool:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(res.stdout.strip())


def make_windows(video: Path, workdir: Path, log=print) -> list[Window]:
    """Crea le finestre, estraendo frame jpeg e segmento audio per ciascuna."""
    duration = probe_duration(video)
    audio_ok = has_audio(video)
    if not audio_ok:
        log("Attenzione: il video non ha traccia audio; analisi solo visiva.")
    windows: list[Window] = []
    step = WINDOW_SECONDS - OVERLAP_SECONDS
    start = 0.0
    idx = 0
    while start < duration:
        win_dur = min(WINDOW_SECONDS, duration - start)
        if win_dur < 1.0 and idx > 0:
            break
        win = Window(index=idx, start=start, duration=win_dur)
        wdir = workdir / f"win_{idx:04d}"
        wdir.mkdir(parents=True, exist_ok=True)

        # Frame: uno ogni FRAME_EVERY_SECONDS all'interno della finestra
        t = 0.0
        f_i = 0
        while t < win_dur:
            frame_path = wdir / f"frame_{f_i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start + t:.3f}",
                 "-i", str(video), "-frames:v", "1",
                 "-vf", f"scale='min({FRAME_MAX_SIDE},iw)':-2",
                 "-q:v", "4", str(frame_path)],
                capture_output=True, check=False,
            )
            if frame_path.exists():
                win.frame_paths.append(frame_path)
                win.frame_times.append(start + t)
                f_i += 1
            t += FRAME_EVERY_SECONDS

        # Audio: segmento WAV 16 kHz mono
        if audio_ok:
            audio_path = wdir / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                 "-t", f"{win_dur:.3f}", "-i", str(video),
                 "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
                capture_output=True, check=False,
            )
            if audio_path.exists():
                win.audio_path = audio_path

        if win.frame_paths:
            windows.append(win)
            idx += 1
        start += step
    return windows
