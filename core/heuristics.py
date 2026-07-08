"""Euristiche leggere per errori visivi ricorrenti."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from core.analyzer import EditError
from core.windows import FRAME_EVERY_SECONDS, Window

BLACK_MEAN_THRESHOLD = 10.0
BLACK_STD_THRESHOLD = 8.0
# Sotto i 5 secondi uno schermo nero e' quasi sempre una transizione voluta.
BLACK_MIN_DURATION_SECONDS = 5.0
# Un vero freeze produce frame identici (diff ~0.1-0.3 da rumore JPEG).
# Soglie piu' alte scambiano slide statiche per frame congelati.
FREEZE_DIFF_THRESHOLD = 0.6
FREEZE_MIN_DURATION = 2 * FRAME_EVERY_SECONDS


def _gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L").resize((96, 54))


def _brightness_stats(path: Path) -> tuple[float, float]:
    img = _gray(path)
    stat = ImageStat.Stat(img)
    return float(stat.mean[0]), float(stat.stddev[0])


def _diff_mean(a: Path, b: Path) -> float:
    diff = ImageChops.difference(_gray(a), _gray(b))
    return float(ImageStat.Stat(diff).mean[0])


def _collect_frames(windows: list[Window]) -> list[tuple[float, Path]]:
    """Frame di tutte le finestre, ordinati e senza duplicati ai bordi overlap."""
    frames: list[tuple[float, Path]] = []
    for win in windows:
        frames.extend(zip(win.frame_times, win.frame_paths))
    frames.sort(key=lambda item: item[0])
    deduped: list[tuple[float, Path]] = []
    for t, path in frames:
        if deduped and abs(t - deduped[-1][0]) < 0.5:
            continue
        deduped.append((t, path))
    return deduped


def detect_visual_heuristics(windows: list[Window], log=print) -> list[EditError]:
    errors: list[EditError] = []
    frames = _collect_frames(windows)

    black_times: list[float] = []
    for t, path in frames:
        try:
            mean, std = _brightness_stats(path)
        except Exception:
            continue
        if mean <= BLACK_MEAN_THRESHOLD and std <= BLACK_STD_THRESHOLD:
            black_times.append(t)

    black_segments: list[list[float]] = []
    for t in black_times:
        if black_segments and t - black_segments[-1][1] <= FRAME_EVERY_SECONDS + 0.5:
            black_segments[-1][1] = t
        else:
            black_segments.append([t, t])
    for seg_start, seg_last in black_segments:
        seg_end = seg_last + FRAME_EVERY_SECONDS
        if seg_end - seg_start <= BLACK_MIN_DURATION_SECONDS:
            continue
        errors.append(EditError(
            type="black_screen",
            start=seg_start,
            end=seg_end,
            description="Schermo nero prolungato rilevato con analisi luminanza.",
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
            if last_t - freeze_start >= FREEZE_MIN_DURATION:
                errors.append(EditError(
                    type="frozen_frame",
                    start=freeze_start,
                    end=last_t + FRAME_EVERY_SECONDS,
                    description="Sequenza di frame quasi identici rilevata con confronto pixel.",
                    confidence=0.78,
                ))
            freeze_start = None
            last_t = None

    if freeze_start is not None and last_t is not None and last_t - freeze_start >= FREEZE_MIN_DURATION:
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


def verify_visual_errors(errors: list[EditError], windows: list[Window],
                         log=print) -> list[EditError]:
    """Scarta black_screen/frozen_frame non confermati dai pixel dei frame reali.

    I modelli vision piccoli tendono ad allucinare errori visivi: qui ogni
    segnalazione viene ricontrollata in modo deterministico sui frame estratti.
    """
    frames = _collect_frames(windows)
    if not frames:
        return errors

    kept: list[EditError] = []
    dropped = 0
    for err in errors:
        if err.type not in {"black_screen", "frozen_frame"}:
            kept.append(err)
            continue
        lo = err.start - FRAME_EVERY_SECONDS
        hi = err.end + FRAME_EVERY_SECONDS
        relevant = [p for t, p in frames if lo <= t <= hi]
        ok = False
        try:
            if err.type == "black_screen":
                for p in relevant:
                    mean, std = _brightness_stats(p)
                    if mean <= BLACK_MEAN_THRESHOLD and std <= BLACK_STD_THRESHOLD:
                        ok = True
                        break
            else:
                for a, b in zip(relevant, relevant[1:]):
                    if _diff_mean(a, b) <= FREEZE_DIFF_THRESHOLD:
                        ok = True
                        break
        except Exception:
            ok = True  # in dubbio, non scartare
        if ok:
            kept.append(err)
        else:
            dropped += 1
    if dropped:
        log(f"Verifica pixel: scartati {dropped} errori visivi non confermati.")
    return kept
