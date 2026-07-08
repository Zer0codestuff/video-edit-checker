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


def _extract_all_frames(video: Path, out_dir: Path) -> list[tuple[float, Path]]:
    """Estrae tutti i frame del video in un solo passaggio ffmpeg.

    Un processo per frame (seek ripetuto) teneva la CPU occupata per minuti
    a GPU ferma; un'unica decodifica sequenziale è molto più veloce.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", (f"fps=1/{FRAME_EVERY_SECONDS:g},"
                 f"scale='min({FRAME_MAX_SIDE},iw)':-2"),
         "-q:v", "4", str(out_dir / "frame_%05d.jpg")],
        capture_output=True, check=False,
    )
    return [(i * FRAME_EVERY_SECONDS, p)
            for i, p in enumerate(sorted(out_dir.glob("frame_*.jpg")))]


def make_windows(video: Path, workdir: Path, log=print,
                 with_audio: bool = True) -> list[Window]:
    """Crea le finestre riusando i frame estratti in un unico passaggio.

    L'audio per finestra viene estratto solo se richiesto (pipeline omni);
    la pipeline ibrida usa whisper.cpp sull'audio intero.
    """
    duration = probe_duration(video)
    audio_ok = with_audio and has_audio(video)
    if with_audio and not audio_ok:
        log("Attenzione: il video non ha traccia audio; analisi solo visiva.")
    frames = _extract_all_frames(video, workdir / "frames")
    if not frames:
        raise RuntimeError("ffmpeg non ha estratto alcun frame dal video")

    windows: list[Window] = []
    step = WINDOW_SECONDS - OVERLAP_SECONDS
    start = 0.0
    idx = 0
    while start < duration:
        win_dur = min(WINDOW_SECONDS, duration - start)
        if win_dur < 1.0 and idx > 0:
            break
        win = Window(index=idx, start=start, duration=win_dur)
        for t, path in frames:
            if start - 0.01 <= t < start + win_dur:
                win.frame_times.append(t)
                win.frame_paths.append(path)

        if audio_ok:
            wdir = workdir / f"win_{idx:04d}"
            wdir.mkdir(parents=True, exist_ok=True)
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
