"""Analisi di una finestra: frame + audio inviati a llama-server, output JSON."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from core.llama_server import HOST
from core.windows import FRAME_EVERY_SECONDS, Window

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "editing_errors.txt"

ERROR_TYPES = {
    "black_screen": "⬛ Schermo nero",
    "frozen_frame": "🧊 Frame congelato",
    "missed_cut": "✂️ Taglio mancato",
    "repeated_phrase": "🔁 Frase ripetuta",
    "audio_glitch": "🔇 Problema audio",
    "other": "⚠️ Altro",
}


@dataclass
class EditError:
    type: str
    start: float
    end: float
    description: str
    confidence: float

    @property
    def label(self) -> str:
        return ERROR_TYPES.get(self.type, ERROR_TYPES["other"])


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _build_prompt(win: Window) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    frame_times = ", ".join(f"{t:.1f}" for t in win.frame_times)
    return template.format(
        frame_interval=int(FRAME_EVERY_SECONDS),
        win_start=win.start,
        win_end=win.start + win.duration,
        frame_times=frame_times,
    )


def _extract_json(text: str) -> dict:
    """Estrae il primo oggetto JSON dalla risposta (anche dentro ```json fence)."""
    if not text:
        return {"errors": []}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {"errors": []}


# Schema JSON imposto al modello via grammatica llama.cpp.
# Mantenuto semplice per non confondere i modelli piccoli.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ERROR_TYPES.keys())},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "start", "end", "description", "confidence"],
            },
        },
    },
    "required": ["errors"],
}


def analyze_window(win: Window, timeout: float = 600.0, log=print) -> list[EditError]:
    content: list[dict] = [{"type": "text", "text": _build_prompt(win)}]
    for fp, ft in zip(win.frame_paths, win.frame_times):
        # Etichetta testuale prima di ogni frame: aiuta il modello a mappare
        # correttamente immagine -> timestamp assoluto.
        content.append({"type": "text", "text": f"[Frame at second {ft:.1f}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(fp)}"},
        })
    if win.audio_path is not None:
        content.append({
            "type": "input_audio",
            "input_audio": {"data": _b64(win.audio_path), "format": "wav"},
        })

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
        "temperature": 0.0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "schema": RESPONSE_SCHEMA,
        },
    }
    try:
        r = requests.post(f"{HOST}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
        # Alcuni modelli "thinking" mettono l'output in reasoning_content
        text = message.get("content") or message.get("reasoning_content") or ""
    except Exception as err:
        log(f"Finestra {win.index}: errore di inferenza ({err}); salto.")
        return []

    data = _extract_json(text)
    errors: list[EditError] = []
    for item in data.get("errors", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            etype = str(item.get("type", "other")).strip().lower()
            if etype not in ERROR_TYPES:
                etype = "other"
            start = float(item.get("start", win.start))
            end = float(item.get("end", start))
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        # Il modello a volte usa secondi relativi alla finestra: correggi se fuori range
        win_end = win.start + win.duration
        if end <= win.duration and start < win.start:
            start += win.start
            end += win.start
        start = max(win.start, min(start, win_end))
        end = max(start, min(end, win_end))
        # I modelli omni piccoli tendono a inventare frasi ripetute: il prompt
        # impone di citare testualmente le parole duplicate; se la descrizione
        # non contiene una citazione, abbassa la confidence sotto la soglia
        # di default (0.5) cosi' il finto errore viene filtrato.
        desc = str(item.get("description", "")).strip()
        if etype == "repeated_phrase" and not re.search(r"[«»\"']", desc):
            conf = min(conf, 0.4)
        errors.append(EditError(
            type=etype,
            start=start,
            end=end,
            description=desc,
            confidence=max(0.0, min(conf, 1.0)),
        ))
    return errors
