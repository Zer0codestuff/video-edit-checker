"""Pipeline vision-only: solo frame a llama-server (audio via whisper.cpp)."""

from __future__ import annotations

from pathlib import Path

from core.llm_client import b64, call_chat, extract_json
from core.models import EditError
from core.parse_errors import VISION_POLICY, parse_errors
from core.windows import Window

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "vision_errors.txt"


def _build_prompt(win: Window) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    frame_times = ", ".join(f"{t:.1f}" for t in win.frame_times)
    return template.format(
        win_start=win.start,
        win_end=win.start + win.duration,
        frame_times=frame_times,
    )


def _build_content(win: Window) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": _build_prompt(win)}]
    for fp, ft in zip(win.frame_paths, win.frame_times):
        content.append({"type": "text", "text": f"[Frame at second {ft:.1f}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64(fp)}"},
        })
    return content


def analyze_window_vision(win: Window, timeout: float = 600.0, log=print) -> list[EditError]:
    text = call_chat(
        _build_content(win),
        timeout=timeout,
        max_tokens=1200,
        log=log,
        error_label=f"finestra {win.index} vision-only",
    )
    if text is None:
        return []
    return parse_errors(extract_json(text), win, VISION_POLICY)
