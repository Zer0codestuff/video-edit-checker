"""Gestione del processo llama-server (llama.cpp) con modelli omni GGUF."""

from __future__ import annotations

import atexit
import os
import subprocess
import time

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
MTP_MODELS: set[str] = {
    "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
}

VISION_MODELS: dict[str, str] = {
    "LiquidAI LFM2.5-VL 1.6B Q8 (vision-only, leggero)": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0",
    "LiquidAI LFM2.5-VL 1.6B Q4 (vision-only, minimo consumo)": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0",
    "SmolVLM2 500M Video (vision-only, velocissimo)": "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
    "Qwen2.5-VL 3B (vision-only, piu accurato)": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
}
DEFAULT_VISION_MODEL_LABEL = "LiquidAI LFM2.5-VL 1.6B Q8 (vision-only, leggero)"

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


class LlamaServer:
    """Avvia e gestisce una singola istanza di llama-server.

    Al cambio modello il server viene riavviato. Il primo avvio scarica
    il GGUF da Hugging Face (cache in ~/.cache/llama.cpp/).
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.current_key: str | None = None
        self.n_parallel: int = N_PARALLEL  # slot effettivi dell'istanza corrente
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

    def _kill_port_holders(self, log=print) -> None:
        """Uccide eventuali processi esterni che tengono la porta PORT."""
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
            for pid in set(pids):
                log(f"Killo processo {pid} che tiene la porta {PORT}.")
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True, timeout=10)
                else:
                    subprocess.run(["kill", pid], timeout=5)
            if pids:
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
        key = (f"{hf_model}|{mmproj_url or ''}|{jinja}"
               f"|{n_parallel}|{ctx_per_slot}|{batch}|{ubatch}")
        self.n_parallel = n_parallel
        if self.is_running() and self.current_key == key:
            return
        self.stop()
        self._kill_port_holders(log=log)
        # Flag conservativi per Mac 8 GB: contesto ridotto, batch piccoli,
        # encoder multimodale su CPU (evita OOM Metal), thinking disabilitato.
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
        if hf_model in MTP_MODELS:
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
            log("MTP speculative decoding attivo per questo modello.")
        log(f"Avvio llama-server con {hf_model} (il primo avvio scarica il modello)...")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.current_key = key
        self._wait_ready(log=log)

    def _wait_ready(self, timeout_s: float = 1800.0, log=print) -> None:
        """Attende che il server risponda (timeout lungo: al primo avvio scarica il GGUF)."""
        deadline = time.time() + timeout_s
        last_note = 0.0
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    "llama-server è terminato durante l'avvio. "
                    "Controlla RAM disponibile o riprova con il modello più leggero."
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
        raise RuntimeError("Timeout in attesa di llama-server.")


SERVER = LlamaServer()
