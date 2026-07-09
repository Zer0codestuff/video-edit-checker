"""Gestione del processo llama-server (llama.cpp) con modelli omni GGUF."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from pathlib import Path

import requests

# Modelli GGUF Gemma 4 (omni-modali: audio + visione). Le versioni QAT
# di Unsloth usano meta' memoria a parita' di qualita' e shippano un
# drafter MTP per speculative decoding.
MODELS: dict[str, str] = {
    "Gemma 4 E2B QAT (leggero + MTP, ~3 GB RAM)": "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL",
    "Gemma 4 E4B QAT (piu accurato + MTP, ~5 GB RAM)": "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
    "Gemma 4 E2B (default, ~8 GB RAM)": "ggml-org/gemma-4-E2B-it-GGUF",
    "Gemma 4 E4B (medio, >=16 GB RAM)": "ggml-org/gemma-4-E4B-it-GGUF",
}
DEFAULT_MODEL_LABEL = "Gemma 4 E2B QAT (leggero + MTP, ~3 GB RAM)"

# Modelli che shippano un drafter MTP (speculative decoding) nel repo HF.
# I flag MTP richiedono una build recente di llama.cpp; su build piu' vecchie
# il drafter viene semplicemente ignorato e il modello gira normalmente.
# Nota: con mmproj (vision) MTP e' ancora sperimentale; forziamo -np 1.
MTP_MODELS: set[str] = {
    "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL",
}

# Vision-only: (repo_hf:quant, jinja). jinja=True per modelli thinking
# (Qwen3.5) cosi' il client puo' passare enable_thinking=false.
VISION_MODELS: dict[str, tuple[str, bool]] = {
    "InternVL3.5 4B Q4_K_M (default, accurato, ~3 GB)":
        ("bartowski/OpenGVLab_InternVL3_5-4B-GGUF:Q4_K_M", False),
    "InternVL3.5 2B Q4_K_M (bilanciato, ~1.3 GB)":
        ("bartowski/OpenGVLab_InternVL3_5-2B-GGUF:Q4_K_M", False),
    "InternVL3.5 1B Q4_K_M (leggero, ~0.5 GB)":
        ("bartowski/OpenGVLab_InternVL3_5-1B-GGUF:Q4_K_M", False),
    "Qwen3.5 4B UD-Q4_K_XL (Unsloth, thinking off, ~3 GB)":
        ("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL", True),
    "Qwen3.5 4B MTP UD-Q4_K_XL (Unsloth + MTP, sperimentale vision)":
        ("unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL", True),
    "LiquidAI LFM2.5-VL 1.6B Q8 (vision-only, leggero)":
        ("LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0", False),
    "LiquidAI LFM2.5-VL 1.6B Q4 (vision-only, minimo consumo)":
        ("LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0", False),
    "SmolVLM2 500M Video (vision-only, velocissimo)":
        ("ggml-org/SmolVLM2-500M-Video-Instruct-GGUF", False),
    "Qwen2.5-VL 3B (vision-only, piu accurato)":
        ("ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", False),
}
DEFAULT_VISION_MODEL_LABEL = "InternVL3.5 4B Q4_K_M (default, accurato, ~3 GB)"

# Modelli per la pipeline video nativa: llama-server decodifica le clip mp4
# internamente (input_video, richiede build >= giugno 2026 + ffmpeg nel PATH)
# campionando i frame a 4 fps. Valori: (repo_hf, mmproj_url | None, jinja).
# MiniCPM-o 4.5 tiene il proiettore vision in una sottocartella non standard,
# quindi l'auto-download del mmproj non lo trova: serve l'URL esplicito.
# Essendo un modello thinking ibrido richiede anche --jinja, cosi' il payload
# puo' disattivare il ragionamento con enable_thinking=false (senza, brucia
# tutti i token in reasoning_content e il content resta vuoto).
# NB: la parte audio di MiniCPM-o funziona solo col framework llama.cpp-omni
# di OpenBMB, non con llama-server mainline: qui e' usato come modello vision.
_MINICPM_O_MMPROJ = ("https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf"
                     "/resolve/main/vision/MiniCPM-o-4_5-vision-F16.gguf")
VIDEO_MODELS: dict[str, tuple[str, str | None, bool]] = {
    "MiniCPM-o 4.5 Q4_K_M (video nativo 8B, ~7 GB RAM)":
        ("openbmb/MiniCPM-o-4_5-gguf:Q4_K_M", _MINICPM_O_MMPROJ, True),
    "MiniCPM-V 4.6 Q4_K_M (video nativo, leggero, ~2 GB RAM)":
        ("openbmb/MiniCPM-V-4.6-gguf:Q4_K_M", None, False),
    "MiniCPM-V 4.6 Q8_0 (video nativo, leggero, meno quantizzato)":
        ("openbmb/MiniCPM-V-4.6-gguf:Q8_0", None, False),
}
DEFAULT_VIDEO_MODEL_LABEL = "MiniCPM-o 4.5 Q4_K_M (video nativo 8B, ~7 GB RAM)"

HOST = "http://127.0.0.1:8090"
PORT = 8090

# Valori di default (sovrascrivibili da env e dalla UI, sezione Prestazioni).
# 2 slot vanno bene su iGPU; su GPU dedicate potenti alzare a 4-8.
N_PARALLEL = max(1, int(os.environ.get("VEC_PARALLEL", "2")))
CTX_PER_SLOT = max(4096, int(os.environ.get("VEC_CTX_PER_SLOT", "8192")))

# Preset batch: (logical batch -b, physical batch -ub). Batch piu' grandi
# accelerano il prompt processing (molte immagini/audio) a costo di piu' VRAM.
BATCH_PRESETS: dict[str, tuple[int, int]] = {
    "Conservativo (iGPU / poca VRAM)": (2048, 512),
    "Bilanciato (GPU dedicata)": (4096, 1024),
    "Aggressivo (GPU potente, >=12 GB VRAM)": (8192, 2048),
}
DEFAULT_BATCH_PRESET = "Conservativo (iGPU / poca VRAM)"

_ROOT = Path(__file__).resolve().parent.parent
_LOG_PATH = _ROOT / "runs" / "llama-server.log"


def _process_name(pid: str) -> str:
    """Nome del processo per un PID, stringa vuota se non disponibile."""
    try:
        if os.name == "nt":
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            line = (res.stdout or "").strip().splitlines()
            if line and line[0].startswith('"'):
                # "llama-server.exe","1234","..."
                return line[0].split(",")[0].strip('"').lower()
        else:
            res = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="],
                capture_output=True, text=True, timeout=5,
            )
            return (res.stdout or "").strip().lower()
    except Exception:
        return ""
    return ""


def _tail_log(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


class LlamaServer:
    """Avvia e gestisce una singola istanza di llama-server.

    Al cambio modello il server viene riavviato. Il primo avvio scarica
    il GGUF da Hugging Face (cache in ~/.cache/llama.cpp/).
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.current_key: str | None = None
        self.n_parallel: int = N_PARALLEL  # slot effettivi dell'istanza corrente
        self._log_fh = None
        atexit.register(self.stop)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.current_key = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _kill_port_holders(self, log=print) -> None:
        """Uccide eventuali processi llama-server che tengono la porta PORT.

        Non tocca altri processi sulla stessa porta (es. tool di debug).
        """
        try:
            pids: list[str] = []
            if os.name == "nt":
                res = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in res.stdout.splitlines():
                    parts = line.split()
                    if (len(parts) >= 5 and parts[0] == "TCP"
                            and parts[1].endswith(f":{PORT}")
                            and parts[3].upper() == "LISTENING"):
                        pids.append(parts[4])
            else:
                res = subprocess.run(
                    ["lsof", "-tiTCP", f":{PORT}", "-sTCP:LISTEN"],
                    capture_output=True, text=True, timeout=5,
                )
                pids = [p for p in res.stdout.split() if p]
            killed = False
            for pid in set(pids):
                name = _process_name(pid)
                if "llama-server" not in name and "llama_server" not in name:
                    log(f"Porta {PORT} occupata da '{name or '?'}' (PID {pid}): "
                        "non la chiudo automaticamente.")
                    continue
                log(f"Killo llama-server {pid} che tiene la porta {PORT}.")
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True, timeout=10)
                else:
                    subprocess.run(["kill", pid], timeout=5)
                killed = True
            if killed:
                time.sleep(2)
        except Exception:
            pass

    def ensure(self, hf_model: str, mmproj_url: str | None = None,
               jinja: bool = False, n_parallel: int | None = None,
               ctx_per_slot: int | None = None,
               batch_preset: str | None = None, log=print) -> None:
        """Garantisce che il server giri con il modello richiesto.

        `mmproj_url` serve per i repo che tengono il proiettore multimodale
        in un percorso non standard (es. MiniCPM-o 4.5): viene scaricato e
        cachato da llama-server come il modello principale.
        `jinja` abilita il template jinja del modello: necessario per i
        modelli thinking ibridi che devono ricevere enable_thinking=false.
        `n_parallel`/`ctx_per_slot`/`batch_preset` sovrascrivono i default
        (UI, sezione Prestazioni); al cambio il server viene riavviato.
        """
        n_parallel = max(1, int(n_parallel or N_PARALLEL))
        ctx_per_slot = max(4096, int(ctx_per_slot or CTX_PER_SLOT))
        batch, ubatch = BATCH_PRESETS.get(batch_preset or "",
                                          BATCH_PRESETS[DEFAULT_BATCH_PRESET])
        use_mtp = hf_model in MTP_MODELS
        # MTP + mmproj / -np>1 non e' ancora supportato in modo affidabile
        # (nota Unsloth/llama.cpp): con vision forziamo 1 slot.
        if use_mtp and n_parallel > 1:
            log(f"MTP: riduco gli slot paralleli da {n_parallel} a 1 "
                "(mmproj/MTP non supporta -np > 1).")
            n_parallel = 1
        key = (f"{hf_model}|{mmproj_url or ''}|{jinja}"
               f"|{n_parallel}|{ctx_per_slot}|{batch}|{ubatch}|mtp={use_mtp}")
        self.n_parallel = n_parallel
        if self.is_running() and self.current_key == key:
            return
        self.stop()
        self._kill_port_holders(log=log)
        # Slot paralleli per tenere piena la GPU tra una finestra e l'altra;
        # il contesto totale scala con gli slot. L'encoder mmproj resta sulla
        # GPU: su CPU satura ~9 core e lascia la GPU ferma durante la
        # codifica delle immagini.
        cmd = [
            "llama-server",
            "-hf", hf_model,
            "--port", str(PORT),
            "-ngl", "999",
            "-c", str(ctx_per_slot * n_parallel),
            "-np", str(n_parallel),
            "-b", str(batch),
            "-ub", str(ubatch),
            "--reasoning-budget", "0",
            "--no-webui",
        ]
        if mmproj_url:
            cmd += ["--mmproj-url", mmproj_url]
        if jinja:
            cmd += ["--jinja"]
        # MTP speculative decoding: il drafter e' auto-scoperto da -hf.
        # Richiede build llama.cpp recente (>= b9500 circa).
        if use_mtp:
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
            log("MTP speculative decoding attivo per questo modello.")
        env = os.environ.copy()
        hf_token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            log("HF_TOKEN rilevato: usato per modelli Hugging Face gated.")
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(_LOG_PATH, "a", encoding="utf-8", errors="replace")
        self._log_fh.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"avvio {hf_model} =====\n")
        self._log_fh.flush()
        log(f"Avvio llama-server con {hf_model} (il primo avvio scarica il modello)...")
        log(f"Log llama-server: {_LOG_PATH}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.current_key = key
        self._wait_ready(log=log)

    def _wait_ready(self, timeout_s: float = 1800.0, log=print) -> None:
        """Attende che il server risponda (timeout lungo: al primo avvio scarica il GGUF)."""
        deadline = time.time() + timeout_s
        last_note = 0.0
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = _tail_log(_LOG_PATH)
                hint = f"\nUltime righe di {_LOG_PATH}:\n{tail}" if tail else ""
                raise RuntimeError(
                    "llama-server è terminato durante l'avvio. "
                    "Controlla RAM disponibile o riprova con il modello più leggero."
                    + hint
                )
            try:
                r = requests.get(f"{HOST}/health", timeout=1)
                if r.status_code == 200:
                    log("llama-server pronto.")
                    return
            except Exception:
                pass
            if time.time() - last_note > 30:
                log("In attesa del modello (download/caricamento in corso)...")
                last_note = time.time()
            time.sleep(1)
        tail = _tail_log(_LOG_PATH)
        hint = f"\nUltime righe di {_LOG_PATH}:\n{tail}" if tail else ""
        raise RuntimeError("Timeout in attesa di llama-server." + hint)


SERVER = LlamaServer()
