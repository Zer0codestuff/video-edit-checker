"""Risoluzione dei binari esterni (ffmpeg, llama-server, whisper-cli).

install.py scarica i binari nella cartella `tools/` del progetto;
qui vengono aggiunti al PATH del processo, cosi' l'app funziona anche
senza installazioni di sistema.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"

REQUIRED = ("ffmpeg", "ffprobe", "llama-server")
OPTIONAL = ("whisper-cli",)

_EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def setup_path() -> None:
    """Antepone al PATH le cartelle di tools/ che contengono i binari."""
    if not TOOLS_DIR.exists():
        return
    dirs: list[str] = []
    for name in REQUIRED + OPTIONAL:
        for hit in TOOLS_DIR.rglob(name + _EXE_SUFFIX):
            if hit.is_file():
                d = str(hit.parent)
                if d not in dirs:
                    dirs.append(d)
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


def missing_required() -> list[str]:
    return [name for name in REQUIRED if shutil.which(name) is None]


def has_whisper() -> bool:
    return shutil.which("whisper-cli") is not None
