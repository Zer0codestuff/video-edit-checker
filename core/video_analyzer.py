"""Pipeline video nativa: clip mp4 della finestra inviata a llama-server.

Usa il content type `input_video` (llama.cpp >= giugno 2026): il server
decodifica la clip via ffmpeg a 4 fps e inserisce marcatori temporali
"[XmY.Ys]" ogni 5 secondi, relativi all'inizio della clip. Rispetto alla
pipeline ibrida il modello vede ~12x piu frame e coglie meglio freeze,
tagli bruschi e glitch di pochi decimi di secondo.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.llm_client import b64, call_chat, extract_json
from core.models import EditError
from core.parse_errors import VIDEO_POLICY, parse_errors
from core.windows import Window

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "video_errors.txt"


def _build_prompt(win: Window, lang: LanguagePack) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        win_duration=win.duration,
        language_rule=lang.language_rule,
        description_language=lang.description_name,
    )


def analyze_window_video(win: Window, timeout: float = 900.0, log=print,
                         language: str | LanguagePack = "it") -> list[EditError]:
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    if win.clip_path is None:
        log(f"Finestra {win.index}: clip mp4 mancante; salto.")
        return []

    content = [
        {"type": "text", "text": _build_prompt(win, lang)},
        {
            "type": "input_video",
            "input_video": {"data": b64(win.clip_path), "format": "mp4"},
        },
    ]
    text = call_chat(
        content,
        timeout=timeout,
        max_tokens=1200,
        enable_thinking=False,
        log=log,
        error_label=f"finestra {win.index} video nativo",
    )
    if text is None:
        return []
    return parse_errors(extract_json(text), win, VIDEO_POLICY)


def video_analyzer_for(language: str | LanguagePack):
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    return partial(analyze_window_video, language=lang)
