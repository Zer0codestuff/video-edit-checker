"""Gestione del processo llama-server (llama.cpp) con modelli omni GGUF."""

from __future__ import annotations

import atexit
import subprocess
import time

import requests

# Modelli GGUF ufficiali ggml-org con input audio+visione, dal più leggero
# (per Mac/PC con poca RAM) al più potente (per workstation con molta RAM/VRAM).
MODELS: dict[str, str] = {
    "Gemma 4 E2B (default, ~8 GB RAM)": "ggml-org/gemma-4-E2B-it-GGUF",
    "Qwen2.5-Omni-3B Q4 (leggerissimo, meno affidabile)": "ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M",
    "Gemma 4 E4B (medio, ≥16 GB RAM)": "ggml-org/gemma-4-E4B-it-GGUF",
    "Qwen2.5-Omni-7B (medio, ≥16 GB RAM)": "ggml-org/Qwen2.5-Omni-7B-GGUF:Q4_K_M",
    "Qwen3-Omni-30B-A3B (potente, ≥32 GB RAM)": "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF",
}
DEFAULT_MODEL_LABEL = "Gemma 4 E2B (default, ~8 GB RAM)"

HOST = "http://127.0.0.1:8090"
PORT = 8090


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

    def ensure(self, hf_model: str, log=print) -> None:
        """Garantisce che il server giri con il modello richiesto."""
        if self.is_running() and self.current_hf == hf_model:
            return
        self.stop()
        # Flag conservativi per Mac 8 GB: contesto ridotto, batch piccoli,
        # encoder multimodale su CPU (evita OOM Metal), thinking disabilitato.
        cmd = [
            "llama-server",
            "-hf", hf_model,
            "--port", str(PORT),
            "-ngl", "999",
            "-c", "4096",
            "-b", "1024",
            "-ub", "256",
            "--no-mmproj-offload",
            "--reasoning-budget", "0",
            "--no-webui",
        ]
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
