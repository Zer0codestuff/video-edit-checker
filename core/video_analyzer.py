"""Analisi video nativa: la clip mp4 della finestra inviata a llama-server.

Usa il content type `input_video` (llama.cpp >= giugno 2026): il server
decodifica la clip via ffmpeg a 4 fps e inserisce marcatori temporali
"[XmY.Ys]" ogni 5 secondi, relativi all'inizio della clip. Rispetto alla
pipeline ibrida il modello vede ~12x piu frame e coglie meglio freeze,
tagli bruschi e glitch di pochi decimi di secondo.
"""

from __future__ import annotations

from core.analyzer import EditError, RESPONSE_SCHEMA, _b64, _extract_json
from core.llama_server import HOST
from core.windows import Window

import requests


def _build_prompt(win: Window) -> str:
    return (
        "You are an expert video editor reviewing raw footage for VISUAL "
        f"editing errors. You receive a short video clip of about "
        f"{win.duration:.0f} seconds, with no audio, sampled at 4 frames per "
        "second.\n\n"
        "Timestamp markers like [0m5.0s] indicate seconds elapsed from the "
        "START of the clip. In the start/end fields use seconds RELATIVE to "
        f"the start of the clip (from 0 to {win.duration:.0f}).\n\n"
        "Scan the clip from beginning to end and check every part of the "
        "timeline. Error types to detect:\n"
        "- \"black_screen\": the screen is completely or almost completely "
        "black for one second or more\n"
        "- \"frozen_frame\": the image stays perfectly still for several "
        "seconds (no motion at all, as if paused)\n"
        "- \"missed_cut\": obvious accidental shots or visual dead time\n"
        "- \"other\": corrupted frames, abrupt visual glitches, any other "
        "obvious visual editing error\n"
        "Never invent audio problems or repeated sentences: you cannot hear "
        "anything.\n\n"
        "CRITICAL RULES:\n"
        "- Many clips contain NO errors: if the whole clip looks normal, "
        "reply {\"errors\": []}. But if any part of the clip turns black or "
        "freezes, you MUST report it with its start/end seconds.\n"
        "- Report \"black_screen\" ONLY if the screen is truly almost "
        "entirely black. If it shows slides, text, graphics or visible "
        "people, it is NOT a black screen.\n"
        "- Set \"confidence\" honestly: 0.9+ only for unmistakable errors.\n\n"
        "Reply with ONLY one valid JSON object, no other text: "
        "{\"errors\": [{\"type\": \"black_screen\", \"start\": 1.0, "
        "\"end\": 2.0, \"description\": \"<short explanation written in "
        "Italian>\", \"confidence\": 0.8}]}\n"
        "If you see no errors, reply {\"errors\": []}."
    )


def analyze_window_video(win: Window, timeout: float = 900.0, log=print) -> list[EditError]:
    if win.clip_path is None:
        log(f"Finestra {win.index}: clip mp4 mancante; salto.")
        return []

    content = [
        {"type": "text", "text": _build_prompt(win)},
        {
            "type": "input_video",
            "input_video": {"data": _b64(win.clip_path), "format": "mp4"},
        },
    ]
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1200,
        "temperature": 0.0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "schema": RESPONSE_SCHEMA,
        },
        # Disattiva il ragionamento nei modelli thinking ibridi (MiniCPM-o
        # 4.5): senza, il modello brucia tutti i token in reasoning_content.
        # Ignorato dai modelli senza template thinking.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{HOST}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
    except Exception as err:
        log(f"Finestra {win.index}: errore video nativo ({err}); salto. "
            "Se l'errore dice 'video input is not supported', aggiorna "
            "llama.cpp con: python install.py")
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
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        # I tempi richiesti al modello sono relativi alla clip: converti in
        # assoluti, a meno che non siano gia' chiaramente assoluti.
        if end <= win.duration + 1.0:
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
