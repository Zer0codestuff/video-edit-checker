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
import os
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

# Release ufficiali whisper.cpp non includono Vulkan su Windows. Per AMD/Intel
# usiamo una build community con ggml-vulkan.dll (stesso approccio di WhisperDrop).
WHISPER_VULKAN_WIN_URL = (
    "https://github.com/eviscerations/whisper-windows-mcp/releases/download/"
    "v1.4.0/whisper-vulkan-win-x64.zip"
)
WHISPER_VULKAN_WIN_NAME = "whisper-vulkan-win-x64.zip"


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
    """Estrae un tar rifiutando path traversal; i symlink relativi restano ok.

    Le release ufficiali whisper.cpp / llama.cpp su Linux usano symlink
    versionati (libwhisper.so -> libwhisper.so.1). Bloccandoli l'installer
    falliva su Ubuntu. Accettiamo solo link il cui target resta dentro dest.
    """
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for member in members:
            name = member.name
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                die(f"Archivio sospetto (path traversal): {name}")
            target = dest / path
            if not _is_within_directory(dest, target):
                die(f"Archivio sospetto (path fuori destinazione): {name}")
            if member.issym() or member.islnk():
                link = member.linkname or ""
                # Rifiuta link assoluti o che escono da dest.
                link_path = Path(link)
                if link_path.is_absolute():
                    die(f"Archivio sospetto (symlink assoluto): {name} -> {link}")
                resolved = (target.parent / link_path).resolve()
                if not _is_within_directory(dest.resolve(), resolved):
                    die(f"Archivio sospetto (symlink fuori destinazione): {name} -> {link}")
        # filter="data" (Python 3.12+) rifiuta i symlink: usiamo l'estrazione
        # manuale gia' validata sopra, cosi' le .so versionate funzionano.
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


def _nvidia_smi_candidates() -> list[str]:
    """Path tipici di nvidia-smi (anche se non e' nel PATH)."""
    found: list[str] = []
    which = shutil.which("nvidia-smi")
    if which:
        found.append(which)
    if platform.system() == "Windows":
        for candidate in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        ):
            if candidate.is_file():
                found.append(str(candidate))
    # Dedup preservando l'ordine.
    seen: set[str] = set()
    out: list[str] = []
    for path in found:
        key = path.lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _nvidia_smi_lists_gpu() -> bool:
    for smi in _nvidia_smi_candidates():
        try:
            res = subprocess.run(
                [smi, "-L"], capture_output=True, text=True, timeout=10,
            )
            if res.returncode == 0 and "GPU" in (res.stdout or ""):
                return True
        except Exception:
            continue
    return False


def _windows_wmi_has_nvidia() -> bool:
    """Fallback se nvidia-smi non e' raggiungibile ma il driver NVIDIA e' installato."""
    if platform.system() != "Windows":
        return False
    try:
        res = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_VideoController).Name -join '`n'",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if res.returncode == 0 and "nvidia" in (res.stdout or "").lower():
            return True
    except Exception:
        pass
    return False


def detect_gpu() -> str:
    """Ritorna 'nvidia', 'apple' o 'generic' (AMD/Intel -> Vulkan)."""
    if platform.system() == "Darwin":
        return "apple"
    if _nvidia_smi_lists_gpu() or _windows_wmi_has_nvidia():
        return "nvidia"
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


def _whisper_backend(exe: Path) -> str:
    """Rileva il backend GPU della build whisper-cli (cuda / vulkan / cpu)."""
    # Stessa logica di core.whisper_cpp.detect_whisper_backend, senza
    # import circolari (install.py e' usato anche fuori dal package).
    parent = exe.parent
    search_dirs = [parent, parent / "lib"]
    if parent.name.lower() in {"bin", "release", "debug", "x64", "win-x64"}:
        search_dirs.append(parent.parent / "lib")
    for d in search_dirs:
        if not d.is_dir():
            continue
        if any(d.glob("ggml-cuda*")) or any(d.glob("libggml-cuda*")):
            return "cuda"
        if any(d.glob("ggml-vulkan*")) or any(d.glob("libggml-vulkan*")):
            return "vulkan"
    return "cpu"


def _whisper_is_cuda(exe: Path) -> bool:
    """Una build CUDA di whisper.cpp ha ggml-cuda.dll accanto al binario."""
    return _whisper_backend(exe) == "cuda"


def _ensure_llama_cudart(assets: list[dict] | None = None) -> None:
    """Assicura che tools/llama contenga le DLL runtime CUDA (cudart/cublas).

    Serve a whisper-cli: le release cublas ufficiali non le includono.
    """
    llama_dir = TOOLS / "llama"
    if llama_dir.exists():
        if any(llama_dir.rglob("cudart*.dll")):
            return
    if assets is None:
        assets = github_latest_assets("ggml-org/llama.cpp")
    cudart = pick_asset(assets, [r"^cudart-.*x64\.zip"])
    if cudart is None:
        log("ATTENZIONE: asset cudart non trovato nelle release llama.cpp.")
        return
    llama_dir.mkdir(parents=True, exist_ok=True)
    archive = TOOLS / cudart["name"]
    download(cudart["browser_download_url"], archive)
    extract(archive, llama_dir)


def _copy_cuda_dlls_to(dest_dir: Path) -> int:
    """Copia le DLL runtime CUDA (cudart/cublas) da tools/llama accanto a whisper-cli.

    Le build cublas di whisper.cpp non includono il runtime CUDA; lo
    riusiamo dal pacchetto cudart gia' scaricato per llama.cpp.
    """
    llama_dir = TOOLS / "llama"
    if not llama_dir.exists():
        return 0
    copied = 0
    for dll in llama_dir.rglob("*.dll"):
        if re.match(r"(cudart|cublas)", dll.name, re.IGNORECASE):
            target = dest_dir / dll.name
            if target.exists() and target.stat().st_size == dll.stat().st_size:
                continue
            shutil.copy2(dll, target)
            copied += 1
    if copied:
        log(f"Copiate {copied} DLL runtime CUDA accanto a whisper-cli.")
    else:
        # Gia' presenti oppure assenti in tools/llama.
        if any(dest_dir.glob("cudart*.dll")):
            log("DLL runtime CUDA gia' presenti accanto a whisper-cli.")
        else:
            log("ATTENZIONE: DLL runtime CUDA non trovate in tools/llama; "
                "whisper-cli potrebbe non partire in modalita' GPU.")
    return copied


def _whisper_in_tools(exe: Path) -> bool:
    try:
        exe.resolve().relative_to((TOOLS / "whisper").resolve())
        return True
    except (ValueError, OSError):
        return False


def _desired_whisper_backend(system: str, gpu: str) -> str:
    """Backend whisper preferito: cuda (NVIDIA Win), vulkan (AMD/Intel Win), cpu."""
    if system == "Windows" and gpu == "nvidia":
        return "cuda"
    if system == "Windows" and gpu == "generic":
        return "vulkan"
    return "cpu"


def _install_whisper_vulkan_windows() -> Path | None:
    """Scarica la build community Vulkan e la estrae in tools/whisper."""
    log(f"Scarico whisper.cpp Vulkan: {WHISPER_VULKAN_WIN_NAME}")
    if (TOOLS / "whisper").exists():
        shutil.rmtree(TOOLS / "whisper", ignore_errors=True)
    archive = TOOLS / WHISPER_VULKAN_WIN_NAME
    download(WHISPER_VULKAN_WIN_URL, archive)
    extract(archive, TOOLS / "whisper")
    return _tools_binary("whisper-cli")


def _install_whisper_official(system: str, want_cuda: bool) -> Path | None:
    """Scarica una release ufficiale ggml-org/whisper.cpp (CUDA o BLAS/CPU)."""
    assets = github_latest_assets("ggml-org/whisper.cpp")
    if system == "Windows":
        patterns = ([r"whisper-cublas-12\..*bin-x64\.zip",
                     r"whisper-cublas-12.*bin-x64\.zip",
                     r"whisper-cublas.*bin-x64\.zip"]
                    if want_cuda else [])
        patterns += [r"whisper-blas-bin-x64\.zip", r"whisper-bin-x64\.zip"]
    else:  # Linux
        patterns = [r"whisper-bin-ubuntu-x64\.tar\.gz"]

    asset = pick_asset(assets, patterns)
    if asset is None:
        log("ATTENZIONE: nessun binario whisper.cpp adatto; "
            "la pipeline ibrida non avra' l'analisi audio.")
        return None
    if want_cuda and "cublas" not in asset["name"].lower():
        log("ATTENZIONE: nessuna build whisper-cublas nelle release; "
            f"uso {asset['name']} (CPU). Su PC NVIDIA rilancia install.py piu' tardi.")
    else:
        log(f"Scarico whisper.cpp: {asset['name']}")

    if want_cuda:
        _ensure_llama_cudart()

    if (TOOLS / "whisper").exists():
        shutil.rmtree(TOOLS / "whisper", ignore_errors=True)
    archive = TOOLS / asset["name"]
    download(asset["browser_download_url"], archive)
    extract(archive, TOOLS / "whisper")
    return _tools_binary("whisper-cli")


def setup_whisper(system: str, gpu: str) -> None:
    want = _desired_whisper_backend(system, gpu)
    want_cuda = want == "cuda"
    want_vulkan = want == "vulkan"

    # Preferisci sempre tools/: un whisper di sistema (PATH) non deve
    # mascherare la build GPU locale ne' forzare un re-download a ogni run.
    tools_exe = _tools_binary("whisper-cli")
    existing_path = tools_exe or (
        Path(which_anywhere("whisper-cli")) if which_anywhere("whisper-cli") else None
    )
    if existing_path is not None:
        backend = _whisper_backend(existing_path)
        in_tools = tools_exe is not None
        needs_upgrade = (
            (want_cuda and backend != "cuda")
            or (want_vulkan and backend != "vulkan")
        )
        if needs_upgrade:
            if in_tools:
                log(f"whisper-cli in tools/ (backend {backend}): "
                    f"reinstallo la build {want}.")
                shutil.rmtree(TOOLS / "whisper", ignore_errors=True)
            else:
                log(f"whisper-cli di sistema (backend {backend}): "
                    f"scarico la build {want} in tools/.")
        else:
            if want_cuda and in_tools:
                _ensure_llama_cudart()
                _copy_cuda_dlls_to(existing_path.parent)
            log(f"whisper-cli gia' presente (backend {backend}), salto.")
            return

    if system == "Darwin":
        if not try_brew("whisper-cpp"):
            log("ATTENZIONE: whisper-cli non installato; "
                "la pipeline ibrida non avra' l'analisi audio.")
        return

    exe: Path | None = None
    if want_vulkan:
        try:
            exe = _install_whisper_vulkan_windows()
        except Exception as err:
            log(f"ATTENZIONE: download whisper Vulkan fallito ({err}); "
                "provo la build ufficiale CPU.")
            exe = None
        if exe is None or _whisper_backend(exe) != "vulkan":
            log("ATTENZIONE: build Vulkan non utilizzabile; fallback CPU ufficiale.")
            exe = _install_whisper_official(system, want_cuda=False)
    else:
        exe = _install_whisper_official(system, want_cuda=want_cuda)

    if exe is None:
        log("ATTENZIONE: whisper-cli non trovato dopo l'estrazione.")
        return
    backend = _whisper_backend(exe)
    if want_cuda:
        if backend == "cuda":
            _copy_cuda_dlls_to(exe.parent)
            log("whisper-cli OK (build CUDA / NVIDIA).")
        else:
            log("ATTENZIONE: scaricata una build whisper senza CUDA; "
                "l'audio verra' trascritto su CPU.")
    elif want_vulkan:
        if backend == "vulkan":
            log("whisper-cli OK (build Vulkan / AMD-Intel).")
        else:
            log(f"ATTENZIONE: whisper senza Vulkan (backend {backend}); "
                "l'audio verra' trascritto su CPU.")
    else:
        log(f"whisper-cli OK (backend {backend}).")


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
