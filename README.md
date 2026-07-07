# Video Edit Checker

Tool locale che analizza video per trovare **errori di montaggio** usando un modello multimodale (audio + video) quantizzato GGUF servito da **llama.cpp**. Ispirato a [MUVAD](https://github.com/Zer0codestuff/MUVAD), ma molto piu semplificato: niente dataset, niente benchmark, niente conda, solo un'interfaccia web semplice e un modello locale.

## Cosa rileva

| Tipo | Descrizione |
|------|-------------|
| Schermo nero | Frame completamente neri o quasi neri che non sono dissolvenze volute |
| Frame congelato | Immagine bloccata mentre l'audio prosegue |
| Taglio mancante | Chi parla sbaglia, dice "lo ripeto" / "aspetta" / "rifacciamo", o ci sono esitazioni/momenti morti evidenti |
| Frase ripetuta | Stessa frase o parte di frase pronunciata due volte quasi identica |
| Problema audio | Audio che salta, si interrompe, silenzio anomalo, rumori di registrazione |
| Altro | Altri evidenti errori di montaggio o registrazione |

## Requisiti

- **Python 3.11+**
- **llama.cpp** (`llama-server`): modello + server di inferenza locale
- **ffmpeg/ffprobe**: estrazione frame e audio dai video
- **yt-dlp**: download video YouTube (installato automaticamente via pip)

### Installa llama.cpp

**macOS (Homebrew):**
```bash
brew install llama.cpp
```

**Linux (build from source):**
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
# metti build/bin/llama-server nel PATH o usa il path completo
```

### Installa ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

## Setup

```bash
git clone https://github.com/Zer0codestuff/video-edit-checker.git
cd video-edit-checker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio

```bash
source .venv/bin/activate
python app.py
```

L'interfaccia web sara disponibile su **http://127.0.0.1:7860**.

## Uso

1. Carica uno o piu **video locali** (mp4, mkv, mov, ...) oppure incolla **URL YouTube** (uno per riga; funzionano anche playlist).
2. Scegli il **modello** (default: Gemma 4 E2B, il piu leggero) e la **soglia di confidence**.
3. Premi **Analizza**. Al primo avvio il modello GGUF verra scaricato da Hugging Face (cache in `~/.cache/llama.cpp/`). I download successivi saranno istantanei.
4. Consulta gli errori nella **tabella** (tipo, timestamp, descrizione, confidence) e nella **galleria di screenshot**.
5. Scarica il **report JSON o CSV** per ogni video analizzato.

## Modelli supportati

Tutti i modelli sono GGUF ufficiali di `ggml-org` con **input audio + visione** (omni-modali), serviti via `llama-server` con il flag `-hf`.

| Modello | Parametri | RAM minima | Note |
|---------|-----------|------------|------|
| `gemma-4-E2B-it-GGUF` | ~2B (E2B) | ~8 GB | Default, leggero e veloce |
| `Qwen2.5-Omni-3B-GGUF` | 3B (Q4_K_M) | ~8 GB | Molto leggero, meno affidabile |
| `gemma-4-E4B-it-GGUF` | ~4B (E4B) | ~16 GB | Buon compromesso qualita/prestazioni |
| `Qwen2.5-Omni-7B-GGUF` | 7B (Q4_K_M) | ~16 GB | Piu accurato, piu lento |
| `Qwen3-Omni-30B-A3B-Instruct-GGUF` | 30B (A3B MoE) | ~32 GB | Massima qualita, richiede workstation |

> **Nota sul modello LFM2.5-VL-1.6B:** e un modello vision-only (senza input audio). Poiche questo tool usa l'audio per rilevare frasi ripetute e tagli mancati, i modelli omni-modali (con audio) sono preferibili. LFM2.5-VL puo rilevare solo problemi visivi (schermo nero, frame congelato).

## Come funziona

1. **Input**: video locali o scaricati da YouTube via yt-dlp (max 480p).
2. **Finestre**: il video viene diviso in finestre da ~20 secondi con 2 secondi di overlap.
3. **Estrazione**: per ogni finestra si estraggono ~7 frame JPEG (uno ogni 3 secondi, max 448px) e un segmento audio WAV 16kHz mono.
4. **Inferenza**: frame + audio vengono inviati a `llama-server` tramite l'API OpenAI-compatible `/v1/chat/completions`. Il modello risponde con JSON strutturato (tipo errore, timestamp, descrizione, confidence).
5. **Aggregazione**: gli errori dalle finestre adiacenti vengono uniti e filtrati per confidence.
6. **Output**: tabella interattiva, galleria di screenshot, report JSON/CSV scaricabili.

## Struttura del progetto

```
video-edit-checker/
├── app.py                  # UI Gradio
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── analyzer.py         # Invio frame+audio a llama-server, parsing JSON
│   ├── ingest.py           # Input video: file locali e YouTube (yt-dlp)
│   ├── llama_server.py     # Gestione processo llama-server
│   ├── report.py           # Merge errori, thumbnail, export JSON/CSV
│   └── windows.py          # Divisione video in finestre, estrazione frame+audio
├── prompts/
│   └── editing_errors.txt  # Prompt per il modello
├── downloads/              # Video scaricati da YouTube (generato al runtime)
└── runs/                   # Risultati delle analisi (generato al runtime)
```

## License

MIT
