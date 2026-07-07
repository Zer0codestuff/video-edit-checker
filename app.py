"""Video Edit Checker — UI Gradio.

Analizza video locali o YouTube con un modello omni locale (llama.cpp)
per trovare errori di montaggio (schermo nero, tagli mancati, frasi ripetute...).

Avvio:  python app.py
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from core.analyzer import EditError, analyze_window
from core.ingest import collect_local_files, download_youtube
from core.llama_server import DEFAULT_MODEL_LABEL, MODELS, SERVER
from core.report import (export_csv, export_json, extract_thumbnail,
                         filter_errors, fmt_time, merge_errors)
from core.windows import make_windows

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class VideoResult:
    name: str
    video_path: Path
    errors: list[EditError] = field(default_factory=list)
    thumbnails: list[tuple[str, str]] = field(default_factory=list)  # (path, caption)
    json_path: Path | None = None
    csv_path: Path | None = None


RESULTS: dict[str, VideoResult] = {}


def _table_rows(res: VideoResult) -> list[list]:
    return [
        [e.label, fmt_time(e.start), fmt_time(e.end), e.description, f"{e.confidence:.2f}"]
        for e in res.errors
    ]


def run_analysis(files, urls_text, model_label, min_confidence,
                 progress=gr.Progress()):
    logs: list[str] = []

    def log(msg: str):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def logs_text() -> str:
        return "\n".join(logs[-200:])

    run_dir = RUNS_DIR / time.strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raccogli input
    videos = collect_local_files(files)
    if (urls_text or "").strip():
        progress(0.0, desc="Download da YouTube...")
        videos += download_youtube(urls_text, log=log)
    if not videos:
        log("Nessun video da analizzare: carica un file o inserisci un URL.")
        yield logs_text(), gr.update(choices=[], value=None), None, [], [], None, None
        return

    log(f"{len(videos)} video in coda.")
    yield logs_text(), gr.update(), None, [], [], None, None

    # 2. Avvia il modello
    progress(0.02, desc="Avvio modello...")
    try:
        SERVER.ensure(MODELS[model_label], log=log)
    except Exception as err:
        log(f"ERRORE avvio modello: {err}")
        yield logs_text(), gr.update(), None, [], [], None, None
        return
    yield logs_text(), gr.update(), None, [], [], None, None

    # 3. Analizza in sequenza
    total = len(videos)
    for v_i, video in enumerate(videos):
        name = video.stem
        log(f"--- Analizzo '{name}' ({v_i + 1}/{total}) ---")
        vdir = run_dir / f"video_{v_i:02d}_{name[:40]}"
        vdir.mkdir(parents=True, exist_ok=True)

        try:
            wins = make_windows(video, vdir / "windows", log=log)
        except Exception as err:
            log(f"ERRORE ffmpeg su '{name}': {err}")
            continue
        log(f"{len(wins)} finestre da analizzare.")

        raw_errors: list[EditError] = []
        for w_i, win in enumerate(wins):
            frac = (v_i + (w_i / max(1, len(wins)))) / total
            progress(0.05 + 0.9 * frac,
                     desc=f"{name}: finestra {w_i + 1}/{len(wins)}")
            found = analyze_window(win, log=log)
            if found:
                for e in found:
                    log(f"  {e.label} @ {fmt_time(e.start)}-{fmt_time(e.end)} "
                        f"(conf {e.confidence:.2f}): {e.description}")
            raw_errors.extend(found)
            if w_i % 3 == 0:
                yield logs_text(), gr.update(), None, [], [], None, None

        errors = filter_errors(merge_errors(raw_errors), float(min_confidence))
        log(f"'{name}': {len(errors)} errori dopo merge e filtro (soglia {min_confidence}).")

        res = VideoResult(name=name, video_path=video, errors=errors)
        for i, e in enumerate(errors):
            thumb = extract_thumbnail(video, (e.start + e.end) / 2,
                                      vdir / "thumbs" / f"err_{i:02d}.jpg")
            if thumb:
                res.thumbnails.append(
                    (str(thumb), f"{e.label} @ {fmt_time(e.start)} — {e.description}"))
        res.json_path = export_json(name, errors, vdir / f"{name}_report.json")
        res.csv_path = export_csv(name, errors, vdir / f"{name}_report.csv")
        RESULTS[name] = res

        yield (logs_text(),
               gr.update(choices=list(RESULTS.keys()), value=name),
               str(res.video_path), _table_rows(res), res.thumbnails,
               str(res.json_path), str(res.csv_path))

    log("Analisi completata.")
    last = list(RESULTS.values())[-1] if RESULTS else None
    yield (logs_text(),
           gr.update(choices=list(RESULTS.keys()),
                     value=last.name if last else None),
           str(last.video_path) if last else None,
           _table_rows(last) if last else [],
           last.thumbnails if last else [],
           str(last.json_path) if last and last.json_path else None,
           str(last.csv_path) if last and last.csv_path else None)


def select_video(name):
    res = RESULTS.get(name)
    if res is None:
        return None, [], [], None, None
    return (str(res.video_path), _table_rows(res), res.thumbnails,
            str(res.json_path) if res.json_path else None,
            str(res.csv_path) if res.csv_path else None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Video Edit Checker") as demo:
        gr.Markdown(
            "# 🎬 Video Edit Checker\n"
            "Trova errori di montaggio (schermo nero, tagli mancati, frasi ripetute...) "
            "con un modello multimodale **locale** (llama.cpp, audio + video)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files_in = gr.File(label="Video locali", file_count="multiple",
                                   file_types=["video"], type="filepath")
                urls_in = gr.Textbox(
                    label="URL YouTube (uno per riga, anche playlist)",
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...",
                    lines=3)
                model_in = gr.Dropdown(choices=list(MODELS.keys()),
                                       value=DEFAULT_MODEL_LABEL, label="Modello")
                conf_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                    label="Soglia confidence")
                run_btn = gr.Button("🔍 Analizza", variant="primary")
                logs_out = gr.Textbox(label="Log", lines=14, interactive=False)
            with gr.Column(scale=2):
                video_sel = gr.Dropdown(choices=[], label="Risultati per video",
                                        interactive=True)
                player = gr.Video(label="Video", interactive=False)
                table = gr.Dataframe(
                    headers=["Tipo", "Inizio", "Fine", "Descrizione", "Confidence"],
                    label="Errori trovati", interactive=False, wrap=True)
                gallery = gr.Gallery(label="Screenshot degli errori", columns=3,
                                     height="auto")
                with gr.Row():
                    json_out = gr.File(label="Report JSON")
                    csv_out = gr.File(label="Report CSV")

        outputs = [logs_out, video_sel, player, table, gallery, json_out, csv_out]
        run_btn.click(run_analysis, [files_in, urls_in, model_in, conf_in], outputs)
        video_sel.change(select_video, [video_sel],
                         [player, table, gallery, json_out, csv_out])
    return demo


if __name__ == "__main__":
    if shutil.which("llama-server") is None:
        raise SystemExit("llama-server non trovato: installa con `brew install llama.cpp`")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg non trovato: installa con `brew install ffmpeg`")
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
