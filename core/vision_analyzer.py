"""Analisi vision-only di una finestra con llama-server."""

from __future__ import annotations

import requests

from core.analyzer import EditError, RESPONSE_SCHEMA, _b64, _extract_json
from core.llama_server import HOST
from core.windows import Window


def _build_prompt(win: Window) -> str:
    frame_times = ", ".join(f"{t:.1f}" for t in win.frame_times)
    return (
        "You are an expert video editor reviewing raw footage for VISUAL "
        "editing errors. You receive only video frames, no audio.\n\n"
        f"The window covers seconds {win.start:.1f}-{win.start + win.duration:.1f} "
        f"of the video. The frames are at these absolute seconds: {frame_times}. "
        "Each frame is preceded by a label \"[Frame at second X]\": use EXACTLY "
        "those values for start/end.\n\n"
        "Error types to detect:\n"
        "- \"black_screen\": frame completely or almost completely black\n"
        "- \"frozen_frame\": identical image across several consecutive frames\n"
        "- \"missed_cut\": obvious accidental shots or visual dead time\n"
        "- \"other\": corrupted frames, strange glitches, any other obvious "
        "visual editing error\n"
        "Never invent audio problems or repeated sentences: you cannot hear "
        "anything.\n\n"
        "CRITICAL RULES:\n"
        "- Most windows contain NO errors: the correct answer is usually "
        "{\"errors\": []}. Only report errors you are reasonably confident about.\n"
        "- Report \"black_screen\" ONLY if the frame is truly almost entirely "
        "black. If it shows slides, text, graphics or visible people, it is "
        "NOT a black screen.\n"
        "- Judge only from what you actually see in the images, never from the "
        "text labels alone.\n"
        "- Set \"confidence\" honestly: 0.9+ only for unmistakable errors.\n\n"
        "Reply with ONLY one valid JSON object, no other text: "
        "{\"errors\": [{\"type\": \"black_screen\", \"start\": 1.0, \"end\": 2.0, "
        "\"description\": \"<short explanation written in Italian>\", "
        "\"confidence\": 0.8}]}\n"
        "If you see no errors, reply {\"errors\": []}."
    )


def analyze_window_vision(win: Window, timeout: float = 600.0, log=print) -> list[EditError]:
    content: list[dict] = [{"type": "text", "text": _build_prompt(win)}]
    for fp, ft in zip(win.frame_paths, win.frame_times):
        content.append({"type": "text", "text": f"[Frame at second {ft:.1f}]"})
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
