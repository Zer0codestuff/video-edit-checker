"""Euristiche leggere per errori visivi ricorrenti."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from core.analyzer import EditError
from core.windows import FRAME_EVERY_SECONDS, Window

BLACK_MEAN_THRESHOLD = 10.0
BLACK_STD_THRESHOLD = 8.0
FREEZE_DIFF_THRESHOLD = 2.2


def _gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L").resize((96, 54))


def _brightness_stats(path: Path) -> tuple[float, float]:
    img = _gray(path)
    stat = ImageStat.Stat(img)
    return float(stat.mean[0]), float(stat.stddev[0])


def _diff_mean(a: Path, b: Path) -> float:
    diff = ImageChops.difference(_gray(a), _gray(b))
    return float(ImageStat.Stat(diff).mean[0])


def detect_visual_heuristics(windows: list[Window], log=print) -> list[EditError]:
    errors: list[EditError] = []
    frames: list[tuple[float, Path]] = []
    for win in windows:
        frames.extend(zip(win.frame_times, win.frame_paths))
    frames.sort(key=lambda item: item[0])

    black_times: list[float] = []
    for t, path in frames:
        try:
            mean, std = _brightness_stats(path)
        except Exception:
            continue
        if mean <= BLACK_MEAN_THRESHOLD and std <= BLACK_STD_THRESHOLD:
            black_times.append(t)

    for t in black_times:
        errors.append(EditError(
            type="black_screen",
            start=t,
            end=t + FRAME_EVERY_SECONDS,
            description="Frame quasi completamente nero rilevato con analisi luminanza.",
            confidence=0.88,
        ))

    freeze_start: float | None = None
    last_t: float | None = None
    for (prev_t, prev_path), (cur_t, cur_path) in zip(frames, frames[1:]):
        try:
            diff = _diff_mean(prev_path, cur_path)
        except Exception:
            continue
        if diff <= FREEZE_DIFF_THRESHOLD and (cur_t - prev_t) <= FRAME_EVERY_SECONDS + 1.0:
            if freeze_start is None:
                freeze_start = prev_t
            last_t = cur_t
        elif freeze_start is not None and last_t is not None:
            if last_t - freeze_start >= FRAME_EVERY_SECONDS:
                errors.append(EditError(
                    type="frozen_frame",
                    start=freeze_start,
                    end=last_t + FRAME_EVERY_SECONDS,
                    description="Sequenza di frame quasi identici rilevata con confronto pixel.",
                    confidence=0.78,
                ))
            freeze_start = None
            last_t = None

    if freeze_start is not None and last_t is not None and last_t - freeze_start >= FRAME_EVERY_SECONDS:
        errors.append(EditError(
            type="frozen_frame",
            start=freeze_start,
            end=last_t + FRAME_EVERY_SECONDS,
            description="Sequenza di frame quasi identici rilevata con confronto pixel.",
            confidence=0.78,
        ))

    if errors:
        log(f"Euristiche visive: {len(errors)} candidati trovati.")
    return errors
