"""Gestione del processo llama-server (llama.cpp) con modelli omni GGUF."""

from __future__ import annotations

import atexit
import os
import subprocess
import time

import requests

# Modelli GGUF ufficiali ggml-org con input audio+visione, dal più leggero
# (per Mac/PC con poca RAM) al più potente (per workstation con molta RAM/VRAM).
MODELS: dict[str, str] = {
    "Qwen2.5-Omni-3B Q8 (consigliato per test, ~6 GB RAM)": "ggml-org/Qwen2.5-Omni-3B-GGUF:Q8_0",
    "Gemma 4 E2B QAT (leggero + MTP, ~3 GB RAM)": "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL",
    "Gemma 4 E2B (default, ~8 GB RAM)": "ggml-org/gemma-4-E2B-it-GGUF",
    "Qwen2.5-Omni-3B Q4 (piu leggero, meno affidabile)": "ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M",
    "Gemma 4 E4B (medio, >=16 GB RAM)": "ggml-org/gemma-4-E4B-it-GGUF",
    "Qwen2.5-Omni-7B (medio, >=16 GB RAM)": "ggml-org/Qwen2.5-Omni-7B-GGUF:Q4_K_M",
    "Qwen3-Omni-30B-A3B (potente, >=32 GB RAM)": "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF",
}
DEFAULT_MODEL_LABEL = "Qwen2.5-Omni-3B Q8 (consigliato per test, ~6 GB RAM)"

# Modelli che shippano un drafter MTP (speculative decoding) nel repo HF.
# I flag MTP richiedono una build recente di llama.cpp; su build piu' vecchie
# il drafter viene semplicemente ignorato e il modello gira normalmente.
MTP_MODELS: set[str] = {
    "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL",
}

VISION_MODELS: dict[str, str] = {
    "LiquidAI LFM2.5-VL 1.6B Q8 (vision-only, leggero)": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0",
    "LiquidAI LFM2.5-VL 1.6B Q4 (vision-only, minimo consumo)": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0",
    "SmolVLM2 500M Video (vision-only, velocissimo)": "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
    "Qwen2.5-VL 3B (vision-only, piu accurato)": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
}
DEFAULT_VISION_MODEL_LABEL = "LiquidAI LFM2.5-VL 1.6B Q8 (vision-only, leggero)"

HOST = "http://127.0.0.1:8090"
PORT = 8090

# Richieste simultanee al server (slot llama.cpp + thread nell'app).
# 2 va bene su iGPU; su GPU dedicate potenti alzare con VEC_PARALLEL=4.
N_PARALLEL = max(1, int(os.environ.get("VEC_PARALLEL", "2")))
CTX_PER_SLOT = max(4096, int(os.environ.get("VEC_CTX_PER_SLOT", "8192")))


class LlamaServer:
    """Avvia e gestisce una singola istanza di llama-server.

    Al cambio modello il server viene riavviato. Il primo avvio scarica
    il GGUF da Hugging Face (cache in ~/.cache/llama.cpp/).
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.current_hf: str | None = None
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
            self.current_hf = None

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

    def ensure(self, hf_model: str, log=print) -> None:
        """Garantisce che il server giri con il modello richiesto."""
        if self.is_running() and self.current_hf == hf_model:
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
            "-c", str(CTX_PER_SLOT * N_PARALLEL),
            "-np", str(N_PARALLEL),
            "-b", "2048",
            "-ub", "512",
            "--reasoning-budget", "0",
            "--no-webui",
        ]
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
        self.current_hf = hf_model
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
