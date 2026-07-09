"""Client HTTP condiviso verso llama-server + utilita' di encoding/parsing."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

from core.llama_server import HOST
from core.models import RESPONSE_SCHEMA

LogFn = Callable[[str], None]

# Retry solo su errori di connessione / 5xx precoci.
# NON ritentare i Timeout: llama-server potrebbe ancora generare e un
# secondo POST raddoppierebbe il carico GPU (fino a 3x con max retries).
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 1.5


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_json(text: str) -> dict:
    """Estrae il primo oggetto JSON dalla risposta (anche dentro ```json fence)."""
    if not text:
        return {"errors": []}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {"errors": []}


def _is_retryable(err: Exception) -> bool:
    if isinstance(err, requests.Timeout):
        return False
    if isinstance(err, requests.ConnectionError):
        return True
    if isinstance(err, requests.HTTPError) and err.response is not None:
        return err.response.status_code >= 500
    return False


def call_chat(
    content: list[dict],
    *,
    timeout: float = 600.0,
    max_tokens: int = 2000,
    enable_thinking: bool | None = None,
    log: LogFn = print,
    error_label: str = "inferenza",
) -> str | None:
    """Invia un messaggio multimodale a llama-server.

    Ritorna il testo della risposta, oppure None se la richiesta fallisce
    (errore gia' loggato). Riprova fino a 2 volte su ConnectionError/5xx;
    i Timeout non vengono ritentati.
    """
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "schema": RESPONSE_SCHEMA,
        },
    }
    if enable_thinking is not None:
        # Disattiva il ragionamento nei modelli thinking ibridi (es. MiniCPM-o):
        # senza, bruciano tutti i token in reasoning_content. Ignorato dai
        # modelli senza template thinking.
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{HOST}/v1/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            message = r.json()["choices"][0]["message"]
            # Alcuni modelli "thinking" mettono l'output in reasoning_content
            return message.get("content") or message.get("reasoning_content") or ""
        except Exception as err:
            last_err = err
            if attempt < _MAX_RETRIES and _is_retryable(err):
                log(f"Errore {error_label} (tentativo {attempt + 1}/"
                    f"{_MAX_RETRIES + 1}: {err}); riprovo...")
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                continue
            break

    log(f"Errore {error_label} ({last_err}); salto.")
    return None
