"""Parsing condiviso della risposta JSON del modello in EditError."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core.models import ERROR_TYPES, VISUAL_ERROR_TYPES, EditError
from core.windows import Window

TimeMode = Literal["absolute_heuristic", "relative_clip"]


@dataclass(frozen=True)
class ParsePolicy:
    """Regole di post-processing specifiche per pipeline."""

    time_mode: TimeMode
    # Tipi ammessi in uscita; gli altri diventano "other" (o vengono rimappati).
    allowed_types: frozenset[str]
    # Se True, repeated_phrase / audio_glitch → "other" (modelli senza audio).
    remap_audio_types: bool = False
    # Se True, abbassa la confidence di repeated_phrase senza citazione.
    penalize_unquoted_repeat: bool = False


OMNI_POLICY = ParsePolicy(
    time_mode="absolute_heuristic",
    allowed_types=frozenset(ERROR_TYPES.keys()),
    remap_audio_types=False,
    penalize_unquoted_repeat=True,
)

VISION_POLICY = ParsePolicy(
    time_mode="absolute_heuristic",
    allowed_types=VISUAL_ERROR_TYPES,
    remap_audio_types=True,
)

VIDEO_POLICY = ParsePolicy(
    time_mode="relative_clip",
    allowed_types=VISUAL_ERROR_TYPES,
    remap_audio_types=True,
)


def normalize_times(
    start: float,
    end: float,
    win: Window,
    mode: TimeMode,
) -> tuple[float, float]:
    """Converte i timestamp del modello in secondi assoluti nel video."""
    if mode == "relative_clip":
        # Il prompt chiede tempi relativi alla clip (0..duration). Se end e'
        # gia' fuori range, li trattiamo come assoluti e non risommiamo.
        if end <= win.duration + 1.0:
            start += win.start
            end += win.start
    else:  # absolute_heuristic
        # Il modello a volte usa secondi relativi: correggi se fuori range.
        if end <= win.duration and start < win.start:
            start += win.start
            end += win.start

    win_end = win.start + win.duration
    start = max(win.start, min(start, win_end))
    end = max(start, min(end, win_end))
    return start, end


def parse_errors(data: dict, win: Window, policy: ParsePolicy) -> list[EditError]:
    """Trasforma il JSON del modello in EditError secondo la policy."""
    errors: list[EditError] = []
    for item in data.get("errors", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            etype = str(item.get("type", "other")).strip().lower()
            if policy.remap_audio_types and etype in {"repeated_phrase", "audio_glitch"}:
                etype = "other"
            if etype not in policy.allowed_types:
                etype = "other"
            default_start = 0.0 if policy.time_mode == "relative_clip" else win.start
            start = float(item.get("start", default_start))
            end = float(item.get("end", start))
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue

        start, end = normalize_times(start, end, win, policy.time_mode)
        desc = str(item.get("description", "")).strip()

        # I modelli omni piccoli tendono a inventare frasi ripetute: il prompt
        # impone di citare testualmente le parole duplicate; senza citazione
        # abbassa la confidence sotto la soglia di default (0.5).
        if policy.penalize_unquoted_repeat and etype == "repeated_phrase":
            if not re.search(r"[«»\"']", desc):
                conf = min(conf, 0.4)

        errors.append(EditError(
            type=etype,
            start=start,
            end=end,
            description=desc,
            confidence=max(0.0, min(conf, 1.0)),
        ))
    return errors
