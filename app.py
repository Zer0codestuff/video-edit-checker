"""Video Edit Checker — UI Gradio.

Analizza video locali o YouTube con un modello omni locale (llama.cpp)
per trovare errori di montaggio (schermo nero, tagli mancati, frasi ripetute...).

Avvio:  python app.py
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import gradio as gr

from core.analyzer import analyzer_for
from core.binaries import has_whisper, missing_required, setup_path
from core.constants import WINDOW_FUTURE_TIMEOUT_SECONDS
from core.heuristics import detect_visual_heuristics, verify_visual_errors
from core.ingest import collect_local_files, download_youtube
from core.language import DEFAULT_LANGUAGE_LABEL, LANGUAGE_CHOICES, resolve_language
from core.llama_server import (BATCH_PRESETS, CTX_PER_SLOT,
                               DEFAULT_BATCH_PRESET, DEFAULT_MODEL_LABEL,
                               DEFAULT_VIDEO_MODEL_LABEL,
                               DEFAULT_VISION_MODEL_LABEL, MODELS, N_PARALLEL,
                               SERVER, VIDEO_MODELS, VISION_MODELS)
from core.models import EditError
from core.report import (batch_summary_md, export_batch, export_csv,
                         export_json, extract_thumbnail, filter_errors,
                         fmt_time, merge_errors)
from core.video_analyzer import video_analyzer_for
from core.vision_analyzer import vision_analyzer_for
from core.whisper_cpp import (DEFAULT_WHISPER_MODEL_LABEL, WHISPER_MODELS,
                              detect_transcript_errors, kill_orphan_whisper,
                              stop_tracked_whisper, transcribe_video)
from core.windows import make_windows, probe_duration

RUNS_DIR = Path(__file__).resolve().parent / "runs"


class Pipeline(str, Enum):
    OMNI = "Omni VLM (audio + visione, singolo modello)"
    HYBRID = "Vision + whisper.cpp (leggero, modulare)"
    VIDEO = "Video nativo + whisper.cpp (clip mp4 al modello, sperimentale)"


PIPELINES = [p.value for p in Pipeline]

CSS = """
.gradio-container {max-width: 1600px !important}
#summary-box {border: 1px solid var(--border-color-primary); border-radius: 8px;
              padding: 12px 16px; max-height: 520px; overflow-y: auto}
#logs-box textarea {font-family: var(--font-mono); font-size: 12px}
footer {display: none !important}
"""


@dataclass
class VideoResult:
    key: str
    name: str
    video_path: Path
    errors: list[EditError] = field(default_factory=list)
    thumbnails: list[tuple[str, str]] = field(default_factory=list)  # (path, caption)
    json_path: Path | None = None
    csv_path: Path | None = None


# Chiave = id univoco della run corrente (es. "00_short_25s"), non solo lo stem.
RESULTS: dict[str, VideoResult] = {}

# Flag condiviso: il pulsante Annulla lo setta; run_analysis lo controlla
# tra un video/finestra e l'altro e ferma whisper/LLM in coda.
_cancel_event = threading.Event()


def request_cancel():
    """Segnala l'annullamento della run in corso e ferma whisper subito."""
    _cancel_event.set()
    stop_tracked_whisper(log=print)
    kill_orphan_whisper(log=print)


def _result_key(index: int, stem: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", stem)[:40].strip(" .") or "video"
    return f"{index:02d}_{safe}"


def _table_rows(res: VideoResult) -> list[list]:
    return [
        [e.label, fmt_time(e.start), fmt_time(e.end), e.description, f"{e.confidence:.2f}"]
        for e in res.errors
    ]


def _pipeline_from_label(label: str) -> Pipeline:
    try:
        return Pipeline(label)
    except ValueError:
        return Pipeline.OMNI


def run_analysis(files, urls_text, pipeline_label, model_label, vision_model_label,
                 video_model_label, whisper_model_label, language_label,
                 min_confidence, n_parallel, batch_preset, ctx_per_slot,
                 progress=gr.Progress()):
    logs: list[str] = []
    RESULTS.clear()
    _cancel_event.clear()
    lang = resolve_language(language_label)

    def log(msg: str):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def logs_text() -> str:
        return "\n".join(logs[-200:])

    def partial():
        """Yield intermedio: aggiorna solo i log."""
        return (logs_text(),) + (gr.update(),) * 9

    def cancelled() -> bool:
        return _cancel_event.is_set()

    # Evita whisper-cli orfani da run interrotte / Analizza rilanciato.
    kill_orphan_whisper(log=log)

    run_dir = RUNS_DIR / time.strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raccogli input
    videos = collect_local_files(files)
    if (urls_text or "").strip():
        progress(0.0, desc="Download da YouTube...")
        videos += download_youtube(urls_text, log=log)
    if not videos:
        log("Nessun video da analizzare: carica un file o inserisci un URL.")
        yield partial()
        return

    log(f"{len(videos)} video in coda. Lingua analisi: {language_label} ({lang.code}).")
    yield partial()

    pipeline = _pipeline_from_label(pipeline_label)
    use_hybrid = pipeline is Pipeline.HYBRID
    use_video = pipeline is Pipeline.VIDEO
    use_modular = use_hybrid or use_video
    perf = dict(n_parallel=int(n_parallel), ctx_per_slot=int(ctx_per_slot),
                batch_preset=batch_preset)
    log(f"Prestazioni: {perf['n_parallel']} slot paralleli, "
        f"{perf['ctx_per_slot']} token/slot, batch '{batch_preset}'.")

    def ensure_llm(log_fn=log) -> None:
        """Avvia (o riavvia) llama-server col modello della pipeline corrente."""
        if use_video:
            video_hf, video_mmproj, video_jinja = VIDEO_MODELS[video_model_label]
            SERVER.ensure(video_hf, mmproj_url=video_mmproj,
                          jinja=video_jinja, log=log_fn, **perf)
        elif use_hybrid:
            SERVER.ensure(VISION_MODELS[vision_model_label], log=log_fn, **perf)
        else:
            SERVER.ensure(MODELS[model_label], log=log_fn, **perf)

    # Omni: serve il modello subito. Hybrid/video: prima whisper (GPU libera),
    # poi llama-server per i frame — evita contendere VRAM/iGPU.
    if not use_modular:
        progress(0.02, desc="Avvio modello...")
        try:
            ensure_llm()
        except Exception as err:
            log(f"ERRORE avvio modello: {err}")
            yield partial()
            return
        yield partial()
    else:
        log("Pipeline modulare: prima trascrizione whisper.cpp, "
            "poi avvio del modello per i frame.")
        yield partial()

    # 3. Analizza in sequenza
    run_results: dict[str, list[EditError]] = {}
    total = len(videos)
    aborted = False
    for v_i, video in enumerate(videos):
        if cancelled():
            log("Analisi annullata dall'utente.")
            aborted = True
            break
        name = video.stem
        key = _result_key(v_i, name)
        log(f"--- Analizzo '{name}' ({v_i + 1}/{total}) ---")
        # Nome cartella sicuro per Windows: niente caratteri riservati e
        # niente spazi/punti finali (WinError 3 in caso contrario).
        safe = re.sub(r'[<>:"/\\|?*]', "_", name)[:40].strip(" .") or "video"
        vdir = run_dir / f"video_{v_i:02d}_{safe}"
        vdir.mkdir(parents=True, exist_ok=True)

        try:
            wins = make_windows(video, vdir / "windows", log=log,
                                with_audio=not use_modular,
                                with_clips=use_video)
        except Exception as err:
            log(f"ERRORE ffmpeg su '{name}': {err}")
            continue
        if cancelled():
            log("Analisi annullata dall'utente.")
            aborted = True
            break
        log(f"{len(wins)} finestre da analizzare.")

        raw_errors: list[EditError] = []
        if use_video:
            analyze = video_analyzer_for(lang)
        elif use_hybrid:
            analyze = vision_analyzer_for(lang)
        else:
            analyze = analyzer_for(lang)

        if use_modular:
            if use_video:
                log("Pipeline video nativa: euristiche → whisper.cpp → clip mp4 al modello.")
            else:
                log("Pipeline ibrida: euristiche → whisper.cpp → modello vision-only.")
            raw_errors.extend(detect_visual_heuristics(
                wins, log=log, language=lang))
            if cancelled():
                log("Analisi annullata dall'utente.")
                aborted = True
                break
            # Libera VRAM: niente llama-server durante whisper (soprattutto iGPU).
            if SERVER.is_running():
                log("Fermo llama-server durante la trascrizione whisper.")
                SERVER.stop()
            progress(0.05 + 0.15 * (v_i / max(1, total)),
                     desc=f"{name}: trascrizione whisper...")
            yield partial()
            try:
                segments = transcribe_video(
                    video,
                    vdir / "whisper",
                    model_label=whisper_model_label or DEFAULT_WHISPER_MODEL_LABEL,
                    language=lang.code,
                    log=log,
                )
            except Exception as err:
                log(f"Trascrizione whisper fallita ({err}); salto analisi audio.")
                stop_tracked_whisper(log=log)
                segments = []
            if cancelled():
                log("Analisi annullata: interrompo dopo whisper.")
                stop_tracked_whisper(log=log)
                aborted = True
                break
            if segments:
                raw_errors.extend(detect_transcript_errors(
                    segments, probe_duration(video), language=lang))
            yield partial()
            progress(0.05 + 0.25 * ((v_i + 0.5) / max(1, total)),
                     desc=f"{name}: avvio modello vision...")
            try:
                ensure_llm()
            except Exception as err:
                log(f"ERRORE avvio modello: {err}")
                aborted = True
                break
            yield partial()

        # Finestre LLM in parallelo sugli slot llama-server.
        # shutdown(wait=False, cancel_futures=True): un future in timeout
        # non deve bloccare l'intera analisi in attesa del worker.
        llm_pool = ThreadPoolExecutor(max_workers=SERVER.n_parallel)
        try:
            futures = [llm_pool.submit(analyze, win, log=log) for win in wins]
            for w_i, fut in enumerate(futures):
                if cancelled():
                    log("Analisi annullata dall'utente: fermo le finestre in coda.")
                    for pending in futures[w_i:]:
                        pending.cancel()
                    aborted = True
                    break
                frac = (v_i + (w_i / max(1, len(wins)))) / total
                progress(0.30 + 0.65 * frac,
                         desc=f"{name}: finestra {w_i + 1}/{len(wins)}")
                try:
                    found = fut.result(timeout=WINDOW_FUTURE_TIMEOUT_SECONDS)
                except FuturesTimeout:
                    log(f"Finestra {w_i + 1}/{len(wins)}: timeout "
                        f"({WINDOW_FUTURE_TIMEOUT_SECONDS:.0f}s); salto.")
                    found = []
                    # Annulla le finestre ancora in coda; il worker corrente
                    # puo' continuare in background ma non blocchiamo lo shutdown.
                    for pending in futures[w_i:]:
                        pending.cancel()
                except Exception as err:
                    log(f"Finestra {w_i + 1}/{len(wins)}: errore ({err}); salto.")
                    found = []
                if found:
                    for e in found:
                        log(f"  {e.label} @ {fmt_time(e.start)}-{fmt_time(e.end)} "
                            f"(conf {e.confidence:.2f}): {e.description}")
                raw_errors.extend(found)
                if w_i % 3 == 0:
                    yield partial()
        finally:
            stop_tracked_whisper(log=log)
            llm_pool.shutdown(wait=False, cancel_futures=True)

        if aborted or cancelled():
            aborted = True
            break

        raw_errors = verify_visual_errors(raw_errors, wins, log=log)
        errors = filter_errors(merge_errors(raw_errors), float(min_confidence))
        log(f"'{name}': {len(errors)} errori dopo merge e filtro (soglia {min_confidence}).")

        res = VideoResult(key=key, name=name, video_path=video, errors=errors)
        for i, e in enumerate(errors):
            thumb = extract_thumbnail(video, (e.start + e.end) / 2,
                                      vdir / "thumbs" / f"err_{i:02d}.jpg")
            if thumb:
                res.thumbnails.append(
                    (str(thumb), f"{e.label} @ {fmt_time(e.start)} — {e.description}"))
        res.json_path = export_json(name, errors, vdir / f"{name}_report.json")
        res.csv_path = export_csv(name, errors, vdir / f"{name}_report.csv")
        RESULTS[key] = res
        # Stessa chiave univoca di RESULTS: evita collisioni su stem duplicati
        # nel riepilogo playlist / report combinato.
        run_results[key] = errors

        yield (logs_text(),
               gr.update(choices=list(RESULTS.keys()), value=key),
               str(res.video_path), _table_rows(res), res.thumbnails,
               str(res.json_path), str(res.csv_path),
               batch_summary_md(run_results), gr.update(), gr.update())

    # 4. Riepilogo finale della run (playlist)
    if aborted or cancelled():
        log("Run interrotta.")
        stop_tracked_whisper(log=log)
        kill_orphan_whisper(log=log)
    else:
        log("Analisi completata.")
    summary = batch_summary_md(run_results)
    batch_json = batch_csv = None
    if run_results:
        batch_json, batch_csv = export_batch(
            run_results, run_dir / "playlist_report.json",
            run_dir / "playlist_report.csv")
        total_err = sum(len(v) for v in run_results.values())
        log(f"Riepilogo: {total_err} errori totali in {len(run_results)} video. "
            "Report combinato pronto nella scheda Riepilogo.")

    last = list(RESULTS.values())[-1] if RESULTS else None
    yield (logs_text(),
           gr.update(choices=list(RESULTS.keys()),
                     value=last.key if last else None),
           str(last.video_path) if last else None,
           _table_rows(last) if last else [],
           last.thumbnails if last else [],
           str(last.json_path) if last and last.json_path else None,
           str(last.csv_path) if last and last.csv_path else None,
           summary,
           str(batch_json) if batch_json else None,
           str(batch_csv) if batch_csv else None)


def select_video(name):
    res = RESULTS.get(name)
    if res is None:
        return None, [], [], None, None
    return (str(res.video_path), _table_rows(res), res.thumbnails,
            str(res.json_path) if res.json_path else None,
            str(res.csv_path) if res.csv_path else None)


def _toggle_pipeline(pipeline_label):
    """Mostra solo i controlli rilevanti per la pipeline scelta."""
    pipeline = _pipeline_from_label(pipeline_label)
    return (gr.update(visible=pipeline is Pipeline.OMNI),
            gr.update(visible=pipeline is Pipeline.HYBRID),
            gr.update(visible=pipeline is Pipeline.VIDEO),
            gr.update(visible=pipeline in {Pipeline.HYBRID, Pipeline.VIDEO}))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Video Edit Checker") as demo:
        gr.Markdown(
            "# 🎬 Video Edit Checker\n"
            "Trova errori di montaggio (schermo nero, tagli mancati, frasi ripetute...) "
            "con modelli locali: omni VLM oppure vision-only + whisper.cpp."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Sorgenti")
                files_in = gr.File(label="Video locali", file_count="multiple",
                                   file_types=["video"], type="filepath")
                urls_in = gr.Textbox(
                    label="URL YouTube (uno per riga, anche playlist)",
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...",
                    lines=3)
                gr.Markdown("### 🧠 Modello")
                pipeline_in = gr.Radio(choices=PIPELINES, value=Pipeline.OMNI.value,
                                       label="Pipeline")
                model_in = gr.Dropdown(choices=list(MODELS.keys()),
                                       value=DEFAULT_MODEL_LABEL,
                                       label="Modello omni (audio + visione)")
                vision_model_in = gr.Dropdown(choices=list(VISION_MODELS.keys()),
                                              value=DEFAULT_VISION_MODEL_LABEL,
                                              label="Modello vision-only",
                                              visible=False)
                video_model_in = gr.Dropdown(choices=list(VIDEO_MODELS.keys()),
                                             value=DEFAULT_VIDEO_MODEL_LABEL,
                                             label="Modello video nativo",
                                             visible=False)
                whisper_model_in = gr.Dropdown(
                    choices=list(WHISPER_MODELS.keys()),
                    value=DEFAULT_WHISPER_MODEL_LABEL,
                    label="Modello whisper.cpp (audio)",
                    info="Scaricato automaticamente in ~/.cache/whisper.cpp/ al primo uso.",
                    visible=False,
                )
                language_in = gr.Radio(
                    choices=list(LANGUAGE_CHOICES.keys()),
                    value=DEFAULT_LANGUAGE_LABEL,
                    label="Lingua del video",
                    info="Imposta la lingua del parlato e delle descrizioni nel report "
                         "(whisper + prompt del modello).",
                )
                conf_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                    label="Soglia confidence",
                                    info="Alza per meno falsi positivi, abbassa per piu' segnalazioni.")
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
                with gr.Row():
                    run_btn = gr.Button("🔍 Analizza", variant="primary", size="lg")
                    cancel_btn = gr.Button("⏹ Annulla", variant="stop", size="lg")
                logs_out = gr.Textbox(label="Log", lines=14, interactive=False,
                                      elem_id="logs-box", autoscroll=True)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🎞️ Dettaglio video"):
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
                    with gr.Tab("📊 Riepilogo run"):
                        summary_out = gr.Markdown(
                            "Il riepilogo di tutti i video della run (playlist) "
                            "comparira' qui alla fine dell'analisi.",
                            elem_id="summary-box")
                        with gr.Row():
                            batch_json_out = gr.File(label="Report combinato JSON")
                            batch_csv_out = gr.File(label="Report combinato CSV")

        pipeline_in.change(_toggle_pipeline, [pipeline_in],
                           [model_in, vision_model_in, video_model_in,
                            whisper_model_in])

        outputs = [logs_out, video_sel, player, table, gallery, json_out, csv_out,
                   summary_out, batch_json_out, batch_csv_out]
        run_event = run_btn.click(
            run_analysis,
            [files_in, urls_in, pipeline_in, model_in, vision_model_in,
             video_model_in, whisper_model_in, language_in, conf_in,
             parallel_in, batch_in, ctx_in],
            outputs,
        )
        # Gradio interrompe il generator di Analizza; request_cancel ferma
        # anche whisper-cli e le finestre LLM ancora in coda.
        cancel_btn.click(request_cancel, cancels=[run_event])
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
    build_ui().launch(server_name="127.0.0.1", server_port=7860, css=CSS)
