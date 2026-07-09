"""Pipeline omni: frame + audio inviati a llama-server, output JSON."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.llm_client import b64, call_chat, extract_json
from core.models import EditError
from core.parse_errors import OMNI_POLICY, parse_errors
from core.windows import FRAME_EVERY_SECONDS, Window

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "editing_errors.txt"


def _build_prompt(win: Window, lang: LanguagePack) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    frame_times = ", ".join(f"{t:.1f}" for t in win.frame_times)
    return template.format(
        frame_interval=int(FRAME_EVERY_SECONDS),
        win_start=win.start,
        win_end=win.start + win.duration,
        frame_times=frame_times,
        speech_language=lang.speech_name,
        description_language=lang.description_name,
        missed_cut_examples=lang.missed_cut_examples,
    )


def _build_content(win: Window, lang: LanguagePack) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": _build_prompt(win, lang)}]
    for fp, ft in zip(win.frame_paths, win.frame_times):
        # Etichetta testuale prima di ogni frame: aiuta il modello a mappare
        # correttamente immagine -> timestamp assoluto.
        content.append({"type": "text", "text": f"[Frame at second {ft:.1f}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64(fp)}"},
        })
    if win.audio_path is not None:
        content.append({
            "type": "input_audio",
            "input_audio": {"data": b64(win.audio_path), "format": "wav"},
        })
    return content


def analyze_window(win: Window, timeout: float = 600.0, log=print,
                   language: str | LanguagePack = "it") -> list[EditError]:
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    text = call_chat(
        _build_content(win, lang),
        timeout=timeout,
        max_tokens=2000,
        log=log,
        error_label=f"finestra {win.index}",
    )
    if text is None:
        return []
    return parse_errors(extract_json(text), win, OMNI_POLICY)


def analyzer_for(language: str | LanguagePack):
    """Callable compatibile con ThreadPoolExecutor: (win, log=...) -> errors."""
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    return partial(analyze_window, language=lang)
