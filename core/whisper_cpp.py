"""Trascrizione audio con whisper.cpp e rilevamento errori testuali."""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from core.language import LanguagePack, resolve_language
from core.models import EditError

CACHE_DIR = Path.home() / ".cache" / "whisper.cpp"
_HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Catalogo UI: etichetta -> nome file ggml su Hugging Face.
WHISPER_MODELS: dict[str, str] = {
    "Medium Q8 (default, ~785 MB)": "ggml-medium-q8_0.bin",
    "Medium Q5 (~514 MB)": "ggml-medium-q5_0.bin",
    "Small Q8 (~250 MB, piu veloce)": "ggml-small-q8_0.bin",
    "Base Q8 (~80 MB, leggero)": "ggml-base-q8_0.bin",
    "Large v3 Turbo Q5 (~550 MB)": "ggml-large-v3-turbo-q5_0.bin",
    "Large v3 Turbo Q8 (~850 MB)": "ggml-large-v3-turbo-q8_0.bin",
}
DEFAULT_WHISPER_MODEL_LABEL = "Medium Q8 (default, ~785 MB)"
DEFAULT_MODEL_NAME = WHISPER_MODELS[DEFAULT_WHISPER_MODEL_LABEL]

# Processi whisper avviati da questa app (per cleanup su cancel/nuova run).
_active_lock = threading.Lock()
_active_procs: set[subprocess.Popen] = set()


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _model_url(filename: str) -> str:
    return f"{_HF_BASE}/{filename}"


def _search_model_file(filename: str) -> Path | None:
    """Cerca un file ggml in cache, env e path tipici di sistema."""
    env = os.environ.get("WHISPER_CPP_MODEL", "").strip()
    candidates = [
        Path(env) if env and Path(env).name == filename else None,
        CACHE_DIR / filename,
        Path("/opt/homebrew/share/whisper.cpp/models") / filename,
        Path("/usr/local/share/whisper.cpp/models") / filename,
    ]
    for path in candidates:
        if path and path.exists():
            return path
    return None


def find_default_model() -> Path | None:
    """Path del modello default (medium Q8) se gia' presente, altrimenti None."""
    env = os.environ.get("WHISPER_CPP_MODEL", "").strip()
    if env and Path(env).exists():
        return Path(env)
    return _search_model_file(DEFAULT_MODEL_NAME)


def resolve_whisper_model(
    model_label: str = "",
    model_path: str = "",
    log=print,
) -> Path | None:
    """Risolve il path del modello: path esplicito, oppure label del catalogo.

    Se manca in cache, lo scarica da Hugging Face.
    """
    if (model_path or "").strip():
        path = Path(model_path).expanduser()
        if path.exists():
            return path
        log(f"Path whisper non trovato: {path}")
        return None

    label = (model_label or "").strip() or DEFAULT_WHISPER_MODEL_LABEL
    filename = WHISPER_MODELS.get(label, DEFAULT_MODEL_NAME)
    found = _search_model_file(filename)
    if found is not None:
        return found

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / filename
    log(f"Scarico modello whisper.cpp {filename} in {target}...")
    try:
        urllib.request.urlretrieve(_model_url(filename), target)
    except Exception as err:
        log(f"Download modello whisper.cpp fallito: {err}")
        return None
    return target if target.exists() else None


def ensure_default_model(log=print) -> Path | None:
    """Compat: scarica/restituisce il modello default (medium Q8)."""
    return resolve_whisper_model(DEFAULT_WHISPER_MODEL_LABEL, log=log)


def _register(proc: subprocess.Popen) -> None:
    with _active_lock:
        _active_procs.add(proc)


def _unregister(proc: subprocess.Popen) -> None:
    with _active_lock:
        _active_procs.discard(proc)


def _terminate_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def stop_tracked_whisper(log=print) -> int:
    """Termina i whisper-cli avviati da questa istanza dell'app."""
    with _active_lock:
        procs = list(_active_procs)
        _active_procs.clear()
    n = 0
    for proc in procs:
        if proc.poll() is None:
            _terminate_proc(proc)
            n += 1
    if n:
        log(f"Terminati {n} processi whisper-cli ancora in esecuzione.")
    return n


def kill_orphan_whisper(log=print) -> int:
    """Uccide eventuali whisper-cli.exe orfani sul sistema (Windows/Linux/macOS).

    Usato a inizio analisi e in cleanup: evita che run interrotte continuino
    a consumare CPU/GPU in background.
    """
    killed = 0
    try:
        if os.name == "nt":
            # /IM matcha il nome immagine; /F forza. Fallisce silenziosamente
            # se non ci sono processi (exit code 128).
            res = subprocess.run(
                ["taskkill", "/IM", "whisper-cli.exe", "/F"],
                capture_output=True, text=True, timeout=15,
            )
            # Conta le righe "SUCCESS" tipiche di taskkill.
            out = (res.stdout or "") + (res.stderr or "")
            killed = out.lower().count("success")
            if killed == 0 and res.returncode == 0 and "whisper-cli" in out.lower():
                killed = 1
        else:
            res = subprocess.run(
                ["pkill", "-x", "whisper-cli"],
                capture_output=True, text=True, timeout=10,
            )
            # pkill: 0 = almeno uno, 1 = nessuno
            if res.returncode == 0:
                killed = 1
    except Exception:
        pass
    # Anche i Popen tracciati (stesso processo o gia' morti).
    stop_tracked_whisper(log=lambda *_: None)
    if killed:
        log(f"Chiusi processi whisper-cli orfani ({killed}).")
    return killed


atexit.register(lambda: stop_tracked_whisper(log=lambda *_: None))


def _to_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000.0 if value > 10000 else float(value)
    text = str(value or "").strip().strip("[]")
    if not text:
        return 0.0
    if ":" not in text:
        return float(text.replace(",", "."))
    parts = text.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def _parse_json(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("transcription") or data.get("segments") or []
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ts = item.get("timestamps") or {}
        start = item.get("start", ts.get("from", 0))
        end = item.get("end", ts.get("to", start))
        segments.append(TranscriptSegment(_to_seconds(start), _to_seconds(end), text))
    return segments


def detect_whisper_backend(whisper_bin: str | Path) -> str:
    """Rileva il backend della build whisper-cli: cuda, vulkan, metal o cpu.

    Su Windows le release ufficiali CUDA espongono ggml-cuda*.dll accanto
    all'eseguibile; senza quella DLL la trascrizione resta su CPU/BLAS.
    """
    binary = Path(whisper_bin)
    parent = binary.parent
    # Layout tipici: DLL accanto all'exe, in lib/, oppure in ../lib
    # quando l'exe e' in bin/ o Release/. Non salire oltre: da una cartella
    # temp piatta parent.parent/lib puo' puntare a path non correlati.
    search_dirs = [parent, parent / "lib"]
    if parent.name.lower() in {"bin", "release", "debug", "x64", "win-x64"}:
        search_dirs.append(parent.parent / "lib")
    checks = (
        ("cuda", ("ggml-cuda*", "libggml-cuda*")),
        ("vulkan", ("ggml-vulkan*", "libggml-vulkan*")),
        ("metal", ("ggml-metal*", "libggml-metal*")),
    )
    for d in search_dirs:
        if not d.is_dir():
            continue
        for backend, patterns in checks:
            # Consuma i match: un generator Path.glob e' sempre truthy.
            if any(f for pat in patterns for f in d.glob(pat)):
                return backend
    return "cpu"


def _backend_label(backend: str) -> str:
    return {
        "cuda": "CUDA (NVIDIA GPU)",
        "vulkan": "Vulkan GPU",
        "metal": "Metal GPU",
        "cpu": "CPU/BLAS",
    }.get(backend, backend)


def transcribe_video(
    video: Path,
    workdir: Path,
    model_path: str = "",
    model_label: str = "",
    language: str = "it",
    log=print,
) -> list[TranscriptSegment]:
    whisper_bin = shutil.which("whisper-cli")
    if whisper_bin is None:
        log("whisper-cli non trovato: installa whisper.cpp oppure usa la pipeline omni.")
        return []

    model = resolve_whisper_model(model_label=model_label, model_path=model_path, log=log)
    if model is None or not model.exists():
        log(
            "Modello whisper.cpp non trovato. Scegline uno dal menu oppure imposta "
            "WHISPER_CPP_MODEL=/path/to/ggml-medium-q8_0.bin"
        )
        return []

    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = workdir / "audio_16khz.wav"
    out_base = workdir / "transcript"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", str(audio_path),
        ],
        capture_output=True,
        check=False,
    )
    if not audio_path.exists():
        log("Impossibile estrarre l'audio per whisper.cpp.")
        return []

    backend = detect_whisper_backend(whisper_bin)
    cmd = [
        whisper_bin,
        "-m", str(model),
        "-f", str(audio_path),
        "-l", language,
        "-oj",
        "-ojf",
        "-of", str(out_base),
        "-np",
    ]
    # Non passare mai -ng/--no-gpu: su build CUDA/Vulkan deve usare la GPU.
    # Flash-attn di default crasha molti driver Vulkan (AMD/Intel/NVIDIA):
    # come in WhisperDrop lo disabilitiamo su Vulkan salvo opt-in esplicito.
    if backend == "vulkan":
        opt_in = os.environ.get("VEC_VULKAN_FLASH_ATTN", "").strip().lower()
        if opt_in not in {"1", "true", "yes", "on"}:
            cmd.append("--no-flash-attn")
    log(f"Trascrivo audio con whisper.cpp ({model.name}, "
        f"backend {_backend_label(backend)})...")
    # Popen (non run): cosi' possiamo terminare il processo su cancel/timeout
    # o a inizio di una nuova analisi, senza lasciare orfani.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(whisper_bin).parent),
    )
    _register(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            log("Timeout whisper.cpp (3600s); termino il processo.")
            _terminate_proc(proc)
            return []
    finally:
        _unregister(proc)

    if proc.returncode != 0:
        err_text = (stderr or "").strip()
        msg = err_text.splitlines()[-1] if err_text else "errore sconosciuto"
        log(f"Errore whisper.cpp: {msg}")
        return []

    json_path = out_base.with_suffix(".json")
    if not json_path.exists():
        log("whisper.cpp non ha prodotto il JSON atteso.")
        return []
    try:
        segments = _parse_json(json_path)
    except Exception as err:
        log(f"Errore parsing transcript whisper.cpp: {err}")
        return []
    log(f"Trascrizione completata: {len(segments)} segmenti.")
    return segments


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sàèéìòù]", " ", text.lower())).strip()


def detect_transcript_errors(
    segments: list[TranscriptSegment],
    video_duration: float,
    language: str | LanguagePack = "it",
) -> list[EditError]:
    lang = language if isinstance(language, LanguagePack) else resolve_language(language)
    errors: list[EditError] = []
    for seg in segments:
        if lang.trigger_re.search(seg.text):
            errors.append(EditError(
                type="missed_cut",
                start=max(0.0, seg.start - 1.0),
                end=min(video_duration, seg.end + 4.0),
                description=lang.missed_cut_desc.format(quote=seg.text[:120]),
                confidence=0.82,
            ))

    for prev, cur in zip(segments, segments[1:]):
        gap = cur.start - prev.end
        prev_norm = _norm(prev.text)
        cur_norm = _norm(cur.text)
        if prev_norm and cur_norm:
            ratio = SequenceMatcher(None, prev_norm, cur_norm).ratio()
            if ratio >= 0.72 and gap <= 10.0:
                errors.append(EditError(
                    type="repeated_phrase",
                    start=prev.start,
                    end=cur.end,
                    description=lang.repeated_phrase_desc.format(
                        a=prev.text[:80], b=cur.text[:80]),
                    confidence=min(0.95, 0.55 + ratio * 0.4),
                ))
        if gap >= 5.0 and prev.end > 2.0 and cur.start < video_duration - 2.0:
            errors.append(EditError(
                type="audio_glitch",
                start=prev.end,
                end=cur.start,
                description=lang.audio_gap_desc.format(gap=gap),
                confidence=min(0.9, 0.55 + gap / 20.0),
            ))
    return errors
