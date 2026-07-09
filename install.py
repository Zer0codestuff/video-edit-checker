"""Installer automatico di Video Edit Checker.

Uso:  python install.py    (Windows: doppio clic su install.bat)

Rileva sistema operativo e GPU, poi prepara tutto il necessario:
  1. virtualenv .venv con le dipendenze Python
  2. ffmpeg/ffprobe            (estrazione frame e audio)
  3. llama.cpp (llama-server)  (build CUDA per NVIDIA, Vulkan per AMD/Intel)
  4. whisper.cpp (whisper-cli) (trascrizione audio, pipeline ibrida)

I binari finiscono nella cartella locale `tools/` del progetto: nessuna
modifica al PATH di sistema. Rilanciare lo script e' sempre sicuro:
salta cio' che e' gia' installato.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
VENV = ROOT / ".venv"

FFMPEG_WIN_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_LINUX_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def log(msg: str) -> None:
    print(f"[install] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n[install] ERRORE: {msg}", flush=True)
    sys.exit(1)


def venv_python() -> Path:
    return VENV / ("Scripts" if platform.system() == "Windows" else "bin") / (
        "python.exe" if platform.system() == "Windows" else "python")


def which_anywhere(name: str) -> str | None:
    """Cerca un binario nel PATH e nella cartella tools/ del progetto."""
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if platform.system() == "Windows" else "")
    if TOOLS.exists():
        for hit in TOOLS.rglob(exe):
            if hit.is_file():
                return str(hit)
    return None


def _tools_binary(name: str) -> Path | None:
    """Restituisce il path del binario in tools/ se esiste, altrimenti None."""
    exe = name + (".exe" if platform.system() == "Windows" else "")
    if TOOLS.exists():
        for hit in TOOLS.rglob(exe):
            if hit.is_file():
                return hit
    return None


def _llama_version(exe: str | Path) -> int:
    """Esegue llama-server --version e restituisce il numero di build (0 se fallisce)."""
    try:
        res = subprocess.run([str(exe), "--version"], capture_output=True,
                             text=True, timeout=15)
        text = (res.stdout or "") + (res.stderr or "")
        m = re.search(r"version:\s*(\d+)", text)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def _llama_supports_mtp(exe: str | Path) -> bool:
    """Verifica che la build supporti --spec-type (MTP speculative decoding)."""
    try:
        res = subprocess.run([str(exe), "--help"], capture_output=True,
                             text=True, timeout=15)
        text = (res.stdout or "") + (res.stderr or "")
        return "--spec-type" in text
    except Exception:
        return False


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Scarico {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "video-edit-checker-installer"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r[install]   ...{read * 100 // total}%", end="", flush=True)
    print(flush=True)


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            # Blocca path traversal (Zip Slip) e link assoluti.
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                die(f"Archivio sospetto (path traversal): {info.filename}")
            target = dest / member
            if not _is_within_directory(dest, target):
                die(f"Archivio sospetto (path fuori destinazione): {info.filename}")
        z.extractall(dest)


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            name = member.name
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                die(f"Archivio sospetto (path traversal): {name}")
            if member.issym() or member.islnk():
                die(f"Archivio sospetto (symlink/hardlink): {name}")
            target = dest / path
            if not _is_within_directory(dest, target):
                die(f"Archivio sospetto (path fuori destinazione): {name}")
        # filter="data" (Python 3.12+) rifiuta path pericolosi; fallback manuale sopra.
        try:
            t.extractall(dest, filter="data")
        except TypeError:
            t.extractall(dest)


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log(f"Estraggo {archive.name} in {dest.relative_to(ROOT)}/")
    if archive.suffix == ".zip":
        _safe_extract_zip(archive, dest)
    else:
        _safe_extract_tar(archive, dest)
    archive.unlink(missing_ok=True)


def github_latest_assets(repo: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "video-edit-checker-installer"})
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    log(f"{repo}: ultima release {data.get('tag_name', '?')}")
    return data.get("assets", [])


def pick_asset(assets: list[dict], patterns: list[str]) -> dict | None:
    for pat in patterns:
        for a in assets:
            if re.search(pat, a["name"], re.IGNORECASE):
                return a
    return None


def detect_gpu() -> str:
    """Ritorna 'nvidia', 'apple' o 'generic' (AMD/Intel -> Vulkan)."""
    if platform.system() == "Darwin":
        return "apple"
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and "GPU" in res.stdout:
                return "nvidia"
        except Exception:
            pass
    return "generic"


def try_brew(*packages: str) -> bool:
    if shutil.which("brew") is None:
        return False
    for pkg in packages:
        log(f"brew install {pkg}...")
        if subprocess.run(["brew", "install", pkg]).returncode != 0:
            return False
    return True


def setup_venv() -> None:
    if not venv_python().exists():
        log("Creo il virtualenv .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    log("Installo le dipendenze Python (gradio, yt-dlp, Pillow, requests)...")
    subprocess.run([str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(venv_python()), "-m", "pip", "install", "-q",
                    "-r", str(ROOT / "requirements.txt")], check=True)
    log("Dipendenze Python OK.")


def setup_ffmpeg(system: str) -> None:
    if which_anywhere("ffmpeg") and which_anywhere("ffprobe"):
        log("ffmpeg gia' presente, salto.")
        return
    if system == "Windows":
        archive = TOOLS / "ffmpeg.zip"
        download(FFMPEG_WIN_URL, archive)
        extract(archive, TOOLS / "ffmpeg")
    elif system == "Linux":
        archive = TOOLS / "ffmpeg.tar.xz"
        download(FFMPEG_LINUX_URL, archive)
        extract(archive, TOOLS / "ffmpeg")
    else:  # macOS
        if not try_brew("ffmpeg"):
            die("Installa Homebrew (https://brew.sh) e rilancia, oppure installa ffmpeg a mano.")
    if not which_anywhere("ffmpeg"):
        die("ffmpeg non trovato dopo l'installazione.")
    log("ffmpeg OK.")


def setup_llama(system: str, gpu: str) -> None:
    # Preferisce sempre la versione in tools/; se esiste ed e' recente
    # (supporta MTP), salta il download. Le versioni di sistema vecchie
    # (es. winget) non bastano: Gemma 4 e MTP richiedono build recenti.
    tools_exe = _tools_binary("llama-server")
    if tools_exe and _llama_supports_mtp(tools_exe):
        log(f"llama-server in tools/ aggiornato (build {_llama_version(tools_exe)}), salto.")
        return

    if system == "Darwin":
        if which_anywhere("llama-server") and not tools_exe:
            log("llama-server gia' presente di sistema, salto.")
            return
        if not try_brew("llama.cpp"):
            die("Installa Homebrew (https://brew.sh) e rilancia per ottenere llama.cpp.")
        log("llama-server OK (Metal via Homebrew).")
        return

    # Rimuovi eventuale vecchia versione in tools/llama/
    if TOOLS.exists():
        for old in TOOLS.rglob("llama-server*"):
            old.unlink(missing_ok=True)

    assets = github_latest_assets("ggml-org/llama.cpp")
    if system == "Windows":
        if gpu == "nvidia":
            # ^llama- esclude gli asset cudart-*, che contengono solo il runtime CUDA
            patterns = [r"^llama-.*bin-win-cuda-c?12.*x64\.zip", r"^llama-.*bin-win-cuda.*x64\.zip"]
        else:
            patterns = [r"^llama-.*bin-win-vulkan-x64\.zip", r"^llama-.*bin-win-cpu-x64\.zip"]
    else:  # Linux
        patterns = [r"bin-ubuntu-vulkan-x64", r"bin-ubuntu-x64"]

    asset = pick_asset(assets, patterns)
    if asset is None:
        die("Nessun binario llama.cpp adatto trovato nella release; installa manualmente.")
    archive = TOOLS / asset["name"]
    download(asset["browser_download_url"], archive)
    extract(archive, TOOLS / "llama")

    if system == "Windows" and gpu == "nvidia":
        cudart = pick_asset(assets, [r"^cudart-.*x64\.zip"])
        if cudart:
            archive = TOOLS / cudart["name"]
            download(cudart["browser_download_url"], archive)
            extract(archive, TOOLS / "llama")

    new_exe = _tools_binary("llama-server")
    if not new_exe:
        die("llama-server non trovato dopo l'estrazione.")
    ver = _llama_version(new_exe)
    mtp = _llama_supports_mtp(new_exe)
    backend = {"nvidia": "CUDA", "generic": "Vulkan/CPU"}.get(gpu, gpu)
    log(f"llama-server OK (build {ver}, backend {backend}, MTP: {'si' if mtp else 'no'}).")


def _whisper_is_cuda(exe: Path) -> bool:
    """Una build CUDA di whisper.cpp ha ggml-cuda.dll accanto al binario."""
    return any(exe.parent.glob("ggml-cuda*.dll"))


def _copy_cuda_dlls_to(dest_dir: Path) -> None:
    """Copia le DLL runtime CUDA (cudart/cublas) da tools/llama accanto a whisper-cli.

    Le build cublas di whisper.cpp non includono il runtime CUDA; lo
    riusiamo dal pacchetto cudart gia' scaricato per llama.cpp.
    """
    llama_dir = TOOLS / "llama"
    if not llama_dir.exists():
        return
    copied = 0
    for dll in llama_dir.rglob("*.dll"):
        if re.match(r"(cudart|cublas)", dll.name, re.IGNORECASE):
            shutil.copy2(dll, dest_dir / dll.name)
            copied += 1
    if copied:
        log(f"Copiate {copied} DLL runtime CUDA accanto a whisper-cli.")
    else:
        log("ATTENZIONE: DLL runtime CUDA non trovate in tools/llama; "
            "whisper-cli potrebbe non partire in modalita' GPU.")


def setup_whisper(system: str, gpu: str) -> None:
    want_cuda = system == "Windows" and gpu == "nvidia"
    existing = which_anywhere("whisper-cli")
    if existing:
        if want_cuda and not _whisper_is_cuda(Path(existing)):
            log("whisper-cli presente ma senza supporto CUDA: reinstallo la build GPU.")
            if (TOOLS / "whisper").exists():
                shutil.rmtree(TOOLS / "whisper")
        else:
            log("whisper-cli gia' presente, salto.")
            return
    if system == "Darwin":
        if not try_brew("whisper-cpp"):
            log("ATTENZIONE: whisper-cli non installato; la pipeline ibrida non avra' l'analisi audio.")
        return

    assets = github_latest_assets("ggml-org/whisper.cpp")
    if system == "Windows":
        # cublas-12.x per primo, coerente con la build CUDA 12.4 di llama.cpp
        patterns = ([r"whisper-cublas-12\..*bin-x64\.zip", r"whisper-cublas.*bin-x64\.zip"]
                    if gpu == "nvidia" else [])
        patterns += [r"whisper-blas-bin-x64\.zip", r"whisper-bin-x64\.zip"]
    else:  # Linux
        patterns = [r"whisper-bin-ubuntu-x64\.tar\.gz"]

    asset = pick_asset(assets, patterns)
    if asset is None:
        log("ATTENZIONE: nessun binario whisper.cpp adatto; la pipeline ibrida non avra' l'analisi audio.")
        return
    archive = TOOLS / asset["name"]
    download(asset["browser_download_url"], archive)
    extract(archive, TOOLS / "whisper")

    exe = _tools_binary("whisper-cli")
    if exe is None:
        log("ATTENZIONE: whisper-cli non trovato dopo l'estrazione.")
        return
    if want_cuda:
        if _whisper_is_cuda(exe):
            _copy_cuda_dlls_to(exe.parent)
            log("whisper-cli OK (build CUDA).")
        else:
            log("ATTENZIONE: scaricata una build whisper senza CUDA; l'audio verra' trascritto su CPU.")
    else:
        log("whisper-cli OK.")


def main() -> None:
    if sys.version_info < (3, 10):
        die(f"Serve Python 3.10+, trovato {platform.python_version()}.")
    system = platform.system()
    if system not in {"Windows", "Linux", "Darwin"}:
        die(f"Sistema non supportato: {system}")
    gpu = detect_gpu()
    log(f"Sistema: {system} {platform.machine()} | GPU: {gpu}")
    TOOLS.mkdir(exist_ok=True)

    setup_venv()
    setup_ffmpeg(system)
    setup_llama(system, gpu)
    setup_whisper(system, gpu)

    runner = "run.bat" if system == "Windows" else "./run.sh"
    print()
    log("Installazione completata!")
    log(f"Avvia l'app con: {runner}  (poi apri http://127.0.0.1:7860)")
    log("Al primo 'Analizza' verranno scaricati i modelli AI (una sola volta).")


if __name__ == "__main__":
    main()
