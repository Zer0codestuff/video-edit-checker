# Video Edit Checker

Tool locale che analizza video per trovare **errori di montaggio** usando modelli AI locali serviti da **llama.cpp**. Ispirato a [MUVAD](https://github.com/Zer0codestuff/MUVAD), ma molto piu semplificato: niente dataset, niente benchmark, solo un'interfaccia web semplice e modelli locali.

## Cosa rileva

| Tipo | Descrizione |
|------|-------------|
| Schermo nero | Schermo nero prolungato (**oltre 5 secondi**: sotto viene considerato una transizione voluta) |
| Frame congelato | Immagine bloccata mentre l'audio prosegue |
| Taglio mancante | Chi parla sbaglia, dice "lo ripeto" / "aspetta" / "rifacciamo", o ci sono esitazioni/momenti morti evidenti |
| Frase ripetuta | Stessa frase o parte di frase pronunciata due volte quasi identica |
| Problema audio | Audio che salta, si interrompe, silenzio anomalo, rumori di registrazione |
| Altro | Altri evidenti errori di montaggio o registrazione |

Ogni segnalazione visiva (schermo nero, frame congelato) viene **verificata sui pixel reali** dei frame prima di arrivare nel report: i falsi positivi dei modelli piccoli vengono scartati automaticamente.

## Installazione (automatica)

Serve solo **Python 3.10+** ([python.org](https://www.python.org/downloads/), su Windows spunta "Add to PATH").

```bash
git clone https://github.com/Zer0codestuff/video-edit-checker.git
cd video-edit-checker
python install.py
```

Su Windows puoi anche fare doppio clic su **`install.bat`**.

L'installer rileva sistema operativo e GPU e scarica da solo tutto il resto nella cartella locale `tools/` (nessuna modifica al sistema):

| Componente | Windows | Linux | macOS |
|------------|---------|-------|-------|
| ffmpeg | build ufficiale gyan.dev | build statica | Homebrew |
| llama.cpp | **CUDA** se c'e' una GPU NVIDIA, altrimenti **Vulkan** (AMD/Intel) | Vulkan/CPU | Homebrew (Metal) |
| whisper.cpp | build CUDA o BLAS | build ufficiale | Homebrew |

Rilanciare `install.py` e' sempre sicuro: salta quello che e' gia' presente.

## Avvio

- **Windows**: doppio clic su **`run.bat`**
- **macOS/Linux**: `./run.sh`

L'interfaccia web sara disponibile su **http://127.0.0.1:7860**. Al primo "Analizza" vengono scaricati i modelli AI da Hugging Face (solo la prima volta, cache in `~/.cache/llama.cpp/`).

## Uso

1. Carica uno o piu **video locali** (mp4, mkv, mov, ...) oppure incolla **URL YouTube** (uno per riga; funzionano anche playlist).
2. Scegli la **pipeline**:
   - **Omni VLM**: un solo modello che vede e sente (audio + visione)
   - **Vision + whisper.cpp** (consigliata sui PC leggeri): euristiche pixel + modello vision-only per i frame + whisper.cpp per l'audio
3. Premi **Analizza** e consulta gli errori nella **tabella** e nella **galleria di screenshot**.
4. Scarica il **report JSON o CSV** per ogni video.

## Modelli supportati

### Pipeline omni (audio + visione)

| Modello | Parametri | RAM minima | Note |
|---------|-----------|------------|------|
| `gemma-4-E2B-it-qat-GGUF` (Unsloth QAT) | ~2B | ~3 GB | Default, leggero + MTP speculative decoding |
| `gemma-4-E4B-it-qat-GGUF` (Unsloth QAT) | ~4B | ~5 GB | Piu' accurato + MTP, per PC con >=16 GB RAM |
| `gemma-4-E2B-it-GGUF` | ~2B | ~8 GB | Leggero e veloce (senza QAT) |
| `gemma-4-E4B-it-GGUF` | ~4B | ~16 GB | Buon compromesso (senza QAT) |

### Pipeline ibrida (vision-only + whisper.cpp)

| Modello vision | Parametri | Note |
|----------------|-----------|------|
| `LiquidAI/LFM2.5-VL-1.6B` Q8/Q4 | 1.6B | Leggerissimo |
| `SmolVLM2-500M-Video` | 0.5B | Velocissimo |
| `Qwen2.5-VL-3B-Instruct` | 3B | Piu accurato |

L'audio della pipeline ibrida usa whisper.cpp con `ggml-large-v3-turbo-q5_0` (scaricato automaticamente al primo uso).

## Prestazioni e GPU dedicate

- Le richieste al modello girano in **parallelo** (default 2 slot). Su GPU dedicate con piu VRAM puoi alzare il parallelismo:

  ```bash
  # Windows (PowerShell)
  $env:VEC_PARALLEL = "4"; .venv\Scripts\python.exe app.py
  # macOS/Linux
  VEC_PARALLEL=4 ./run.sh
  ```

- `VEC_CTX_PER_SLOT` controlla il contesto per slot (default 8192).
- Su GPU potenti conviene anche scegliere modelli piu grandi dal menu a tendina.

## Come funziona

1. **Input**: video locali o scaricati da YouTube via yt-dlp (max 480p).
2. **Frame**: estratti in un unico passaggio ffmpeg (1 ogni 3 secondi, max 448px), condivisi tra finestre da ~20 secondi con 2 secondi di overlap.
3. **Euristiche pixel** (pipeline ibrida): luminanza per gli schermi neri, confronto pixel per i frame congelati.
4. **Inferenza**: frame (+ audio per la pipeline omni) inviati a `llama-server` via API OpenAI-compatible; risposta JSON strutturata forzata da schema.
5. **Verifica**: ogni errore visivo segnalato dal modello viene ricontrollato sui pixel reali; schermi neri sotto i 5 secondi vengono scartati.
6. **Aggregazione**: merge degli errori tra finestre adiacenti, filtro per confidence, tabella + screenshot + report JSON/CSV.

## Struttura del progetto

```
video-edit-checker/
├── install.py              # Installer automatico (OS + GPU detection)
├── install.bat             # Doppio clic su Windows
├── run.bat / run.sh        # Avvio dell'app
├── app.py                  # UI Gradio
├── requirements.txt
├── core/
│   ├── analyzer.py         # Pipeline omni: frame+audio a llama-server
│   ├── vision_analyzer.py  # Pipeline ibrida: solo frame
│   ├── heuristics.py       # Euristiche pixel + verifica anti-allucinazione
│   ├── whisper_cpp.py      # Trascrizione e errori audio (whisper.cpp)
│   ├── binaries.py         # Risoluzione binari da tools/ e PATH
│   ├── ingest.py           # Input video: file locali e YouTube (yt-dlp)
│   ├── llama_server.py     # Gestione processo llama-server
│   ├── report.py           # Merge errori, thumbnail, export JSON/CSV
│   └── windows.py          # Finestre temporali, estrazione frame+audio
├── prompts/
│   └── editing_errors.txt  # Prompt per il modello omni
├── tools/                  # Binari scaricati da install.py (generato)
├── downloads/              # Video scaricati da YouTube (generato)
└── runs/                   # Risultati delle analisi (generato)
```

## Requisiti minimi

- Python 3.10+
- 8 GB di RAM (16+ consigliati per i modelli medi)
- GPU opzionale ma consigliata: NVIDIA (CUDA), AMD/Intel (Vulkan) o Apple Silicon (Metal)

## License

MIT
