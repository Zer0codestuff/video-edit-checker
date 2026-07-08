"""Video Edit Checker — UI Gradio.

Analizza video locali o YouTube con un modello omni locale (llama.cpp)
per trovare errori di montaggio (schermo nero, tagli mancati, frasi ripetute...).

Avvio:  python app.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from core.analyzer import EditError, analyze_window
from core.binaries import has_whisper, missing_required, setup_path
from core.heuristics import detect_visual_heuristics, verify_visual_errors
from core.ingest import collect_local_files, download_youtube
from core.llama_server import (BATCH_PRESETS, CTX_PER_SLOT,
                               DEFAULT_BATCH_PRESET, DEFAULT_MODEL_LABEL,
                               DEFAULT_VIDEO_MODEL_LABEL,
                               DEFAULT_VISION_MODEL_LABEL, MODELS, N_PARALLEL,
                               SERVER, VIDEO_MODELS, VISION_MODELS)
from core.report import (export_csv, export_json, extract_thumbnail,
                         filter_errors, fmt_time, merge_errors)
from core.video_analyzer import analyze_window_video
from core.vision_analyzer import analyze_window_vision
from core.whisper_cpp import (detect_transcript_errors, find_default_model,
                              transcribe_video)
from core.windows import make_windows, probe_duration

RUNS_DIR = Path(__file__).resolve().parent / "runs"
PIPELINES = [
    "Omni VLM (audio + visione, singolo modello)",
    "Vision + whisper.cpp (leggero, modulare)",
    "Video nativo + whisper.cpp (clip mp4 al modello, sperimentale)",
]


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


def run_analysis(files, urls_text, pipeline_label, model_label, vision_model_label,
                 video_model_label, whisper_model_path, min_confidence,
                 n_parallel, batch_preset, ctx_per_slot,
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
        yield logs_text(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    log(f"{len(videos)} video in coda.")
    yield logs_text(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 2. Avvia il modello richiesto
    use_hybrid = pipeline_label == PIPELINES[1]
    use_video = pipeline_label == PIPELINES[2]
    progress(0.02, desc="Avvio modello...")
    perf = dict(n_parallel=int(n_parallel), ctx_per_slot=int(ctx_per_slot),
                batch_preset=batch_preset)
    log(f"Prestazioni: {perf['n_parallel']} slot paralleli, "
        f"{perf['ctx_per_slot']} token/slot, batch '{batch_preset}'.")
    try:
        if use_video:
            video_hf, video_mmproj, video_jinja = VIDEO_MODELS[video_model_label]
            SERVER.ensure(video_hf, mmproj_url=video_mmproj,
                          jinja=video_jinja, log=log, **perf)
        elif use_hybrid:
            SERVER.ensure(VISION_MODELS[vision_model_label], log=log, **perf)
        else:
            SERVER.ensure(MODELS[model_label], log=log, **perf)
    except Exception as err:
        log(f"ERRORE avvio modello: {err}")
        yield logs_text(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return
    yield logs_text(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 3. Analizza in sequenza
    total = len(videos)
    for v_i, video in enumerate(videos):
        name = video.stem
        log(f"--- Analizzo '{name}' ({v_i + 1}/{total}) ---")
        vdir = run_dir / f"video_{v_i:02d}_{name[:40]}"
        vdir.mkdir(parents=True, exist_ok=True)

        use_modular = use_hybrid or use_video
        try:
            wins = make_windows(video, vdir / "windows", log=log,
                                with_audio=not use_modular,
                                with_clips=use_video)
        except Exception as err:
            log(f"ERRORE ffmpeg su '{name}': {err}")
            continue
        log(f"{len(wins)} finestre da analizzare.")

        raw_errors: list[EditError] = []
        if use_video:
            analyze = analyze_window_video
        elif use_hybrid:
            analyze = analyze_window_vision
        else:
            analyze = analyze_window
        # Tante richieste LLM in volo quanti sono gli slot llama-server +
        # whisper su CPU in parallelo, per non lasciare ferma la GPU.
        with ThreadPoolExecutor(max_workers=SERVER.n_parallel) as llm_pool, \
                ThreadPoolExecutor(max_workers=1) as bg_pool:
            transcript_future = None
            if use_modular:
                if use_video:
                    log("Pipeline video nativa: euristiche visive + whisper.cpp "
                        "+ clip mp4 al modello video.")
                else:
                    log("Pipeline ibrida: euristiche visive + whisper.cpp + modello vision-only.")
                raw_errors.extend(detect_visual_heuristics(wins, log=log))
                transcript_future = bg_pool.submit(
                    transcribe_video,
                    video,
                    vdir / "whisper",
                    model_path=whisper_model_path or "",
                    language="it",
                    log=log,
                )

            futures = [llm_pool.submit(analyze, win, log=log) for win in wins]
            for w_i, fut in enumerate(futures):
                frac = (v_i + (w_i / max(1, len(wins)))) / total
                progress(0.05 + 0.9 * frac,
                         desc=f"{name}: finestra {w_i + 1}/{len(wins)}")
                found = fut.result()
                if found:
                    for e in found:
                        log(f"  {e.label} @ {fmt_time(e.start)}-{fmt_time(e.end)} "
                            f"(conf {e.confidence:.2f}): {e.description}")
                raw_errors.extend(found)
                if w_i % 3 == 0:
                    yield logs_text(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            if transcript_future is not None:
                segments = transcript_future.result()
                if segments:
                    raw_errors.extend(detect_transcript_errors(segments, probe_duration(video)))

        raw_errors = verify_visual_errors(raw_errors, wins, log=log)
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
            "con modelli locali: omni VLM oppure vision-only + whisper.cpp."
        )
        default_whisper = find_default_model()
        with gr.Row():
            with gr.Column(scale=1):
                files_in = gr.File(label="Video locali", file_count="multiple",
                                   file_types=["video"], type="filepath")
                urls_in = gr.Textbox(
                    label="URL YouTube (uno per riga, anche playlist)",
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...",
                    lines=3)
                pipeline_in = gr.Radio(choices=PIPELINES, value=PIPELINES[0],
                                       label="Pipeline")
                model_in = gr.Dropdown(choices=list(MODELS.keys()),
                                       value=DEFAULT_MODEL_LABEL,
                                       label="Modello omni (audio + visione)")
                vision_model_in = gr.Dropdown(choices=list(VISION_MODELS.keys()),
                                              value=DEFAULT_VISION_MODEL_LABEL,
                                              label="Modello vision-only")
                video_model_in = gr.Dropdown(choices=list(VIDEO_MODELS.keys()),
                                             value=DEFAULT_VIDEO_MODEL_LABEL,
                                             label="Modello video nativo")
                whisper_model_in = gr.Textbox(
                    label="Path modello whisper.cpp",
                    value=str(default_whisper) if default_whisper else "",
                    placeholder="vuoto = download automatico ggml-large-v3-turbo-q5_0.bin",
                    lines=1,
                )
                conf_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                    label="Soglia confidence")
                with gr.Accordion("⚡ Prestazioni (GPU)", open=False):
                    gr.Markdown(
                        "Su GPU dedicate potenti alza slot e batch per saturare la "
                        "scheda; su iGPU o poca VRAM lascia i default. Cambiare "
                        "questi valori riavvia il modello alla prossima analisi."
                    )
                    parallel_in = gr.Slider(
                        1, 8, value=N_PARALLEL, step=1,
                        label="Finestre analizzate in parallelo (slot llama-server)",
                        info="2 per iGPU, 4-8 per GPU dedicate. Ogni slot usa VRAM aggiuntiva.")
                    batch_in = gr.Dropdown(
                        choices=list(BATCH_PRESETS.keys()),
                        value=DEFAULT_BATCH_PRESET,
                        label="Preset batch (velocita' di elaborazione del prompt)",
                        info="Batch grandi accelerano immagini/audio ma usano piu' VRAM.")
                    ctx_in = gr.Slider(
                        4096, 32768, value=min(32768, CTX_PER_SLOT), step=4096,
                        label="Contesto per slot (token)",
                        info="8192 basta per le finestre standard; alza solo se compaiono errori di contesto.")
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
        run_btn.click(
            run_analysis,
            [files_in, urls_in, pipeline_in, model_in, vision_model_in,
             video_model_in, whisper_model_in, conf_in,
             parallel_in, batch_in, ctx_in],
            outputs,
        )
        video_sel.change(select_video, [video_sel],
                         [player, table, gallery, json_out, csv_out])
    return demo


if __name__ == "__main__":
    setup_path()
    missing = missing_required()
    if missing:
        raise SystemExit(
            f"Mancano i binari: {', '.join(missing)}. "
            "Esegui prima l'installazione: python install.py"
        )
    if not has_whisper():
        print("Avviso: whisper-cli non trovato; la pipeline ibrida non avra' "
              "l'analisi audio. Esegui `python install.py` per installarlo.")
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
