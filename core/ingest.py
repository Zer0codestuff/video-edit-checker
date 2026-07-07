"""Input dei video: file locali e URL/playlist YouTube (via yt-dlp)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOWNLOADS_DIR = Path(__file__).resolve().parent.parent / "downloads"


def collect_local_files(paths: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for p in paths or []:
        path = Path(p)
        if path.exists():
            out.append(path)
    return out


def download_youtube(urls_text: str, log=print) -> list[Path]:
    """Scarica video/playlist YouTube (max 480p) e ritorna i path dei file mp4.

    Ogni riga di `urls_text` è un URL (video singolo o playlist).
    """
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    if not urls:
        return []
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for url in urls:
        log(f"Scarico: {url}")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bv*[height<=480]+ba/b[height<=480]/b",
            "--merge-output-format", "mp4",
            "--restrict-filenames",
            "--no-overwrites",
            "-o", str(DOWNLOADS_DIR / "%(title)s_%(id)s.%(ext)s"),
            "--print", "after_move:filepath",
            "--no-simulate",
            url,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            log(f"Timeout nel download di {url}, salto.")
            continue
        if res.returncode != 0:
            log(f"Errore yt-dlp per {url}: {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'sconosciuto'}")
            continue
        for line in res.stdout.splitlines():
            line = line.strip()
            if line and Path(line).exists():
                downloaded.append(Path(line))
                log(f"Scaricato: {Path(line).name}")
    return downloaded
