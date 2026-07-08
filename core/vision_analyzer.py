"""Analisi vision-only di una finestra con llama-server."""

from __future__ import annotations

import requests

from core.analyzer import EditError, RESPONSE_SCHEMA, _b64, _extract_json
from core.llama_server import HOST
from core.windows import Window


def _build_prompt(win: Window) -> str:
    frame_times = ", ".join(f"{t:.1f}" for t in win.frame_times)
    return (
        "Sei un assistente esperto di montaggio video. Ricevi solo frame video, "
        "senza audio. Devi rilevare esclusivamente errori visivi di montaggio.\n\n"
        f"La finestra copre i secondi {win.start:.1f}-{win.start + win.duration:.1f}. "
        f"I frame disponibili sono ai secondi assoluti: {frame_times}.\n\n"
        "Cerca: black_screen, frozen_frame, tagli visivi strani, frame corrotti, "
        "inquadrature palesemente accidentali. Non inventare problemi audio o frasi ripetute.\n"
        "IMPORTANTE: nella maggior parte delle finestre NON c'e' alcun errore, e la "
        "risposta corretta e' {\"errors\": []}. Segnala black_screen SOLO se il frame "
        "e' davvero quasi tutto nero: se contiene slide, testo, grafica o persone "
        "visibili NON e' uno schermo nero. Basati esclusivamente su cio' che vedi "
        "nelle immagini, non sulle etichette testuali.\n"
        "Rispondi solo con JSON: {\"errors\": [{\"type\": \"black_screen\", "
        "\"start\": 1.0, \"end\": 2.0, \"description\": \"...\", \"confidence\": 0.8}]}.\n"
        "Se non vedi errori, rispondi {\"errors\": []}."
    )


def analyze_window_vision(win: Window, timeout: float = 600.0, log=print) -> list[EditError]:
    content: list[dict] = [{"type": "text", "text": _build_prompt(win)}]
    for fp, ft in zip(win.frame_paths, win.frame_times):
        content.append({"type": "text", "text": f"[Frame al secondo {ft:.1f}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(fp)}"},
        })

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1200,
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
        text = message.get("content") or message.get("reasoning_content") or ""
    except Exception as err:
        log(f"Finestra {win.index}: errore vision-only ({err}); salto.")
        return []

    data = _extract_json(text)
    errors: list[EditError] = []
    win_end = win.start + win.duration
    for item in data.get("errors", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            etype = str(item.get("type", "other")).strip().lower()
            if etype in {"repeated_phrase", "audio_glitch"}:
                etype = "other"
            start = float(item.get("start", win.start))
            end = float(item.get("end", start))
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        if end <= win.duration and start < win.start:
            start += win.start
            end += win.start
        errors.append(EditError(
            type=etype if etype in {"black_screen", "frozen_frame", "missed_cut", "other"} else "other",
            start=max(win.start, min(start, win_end)),
            end=max(start, min(end, win_end)),
            description=str(item.get("description", "")).strip(),
            confidence=max(0.0, min(conf, 1.0)),
        ))
    return errors
