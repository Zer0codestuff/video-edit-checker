# Esperimenti speech-edit (video 3.5)

Ground truth manuale su `https://youtu.be/NqG8O7aNnTs` (3.5 Gestione del rischio):

| Tempo | Errore |
|------|--------|
| 2:15 | parola in più |
| 4:00 | «potrebbe» in più |
| 6:07 | «a un soggetto» da eliminare |
| 10:00 | «fornisce» da eliminare |
| 11:13 | «ehh» |

YouTube blocca gli IP cloud → corpus sintetico in `corpus/` che riproduce gli stessi pattern.

## Risultati chiave

| Pipeline | Precision | Recall | F1 | Note |
|----------|-----------|--------|----|------|
| **baseline segmenti** (`aspetta`/`lo ripeto`, frasi duplicate) | — | **0%** | 0 | Non vede nessuno dei 5 GT |
| word repeat only | alta | media | ~0.6–0.75 | Cattura «fornisce fornisce», «anche anche» |
| n-gram only | alta | bassa | ~0.3 | Cattura «a un soggetto» |
| fillers only | — | ~0 su TTS | 0–0.2 | Dipende da ASR (ehh→ehm/m/e) |
| **full word-level** | **~0.86–1.0** | **~0.6–0.8** | **~0.7–0.89** | Miglior tradeoff |
| **full + merge fix + emm** | **1.0** | **1.0** | **1.0** | Dopo fix merge speech + filler `emm` |
| text fallback only | alta | buona | fino a 0.78 | Utile se i token BPE sono rumorosi |

## Cosa funziona

1. **Word-token whisper (`-ojf`)** + merge BPE (senza `strip()` prematuro).
2. **Ripetizioni adiacenti** unigram/n-gram con gap ≤ ~1.5s, ignorando stopword unigram.
3. **Fallback sul testo del segmento** se i token sono fusi/storti.
4. **`-mc 0` + temperatura ~0.6**: riduce il collasso «potrebbe potrebbe» → «potrebbe» (visto su medium a temp 0).

## Cosa non funziona (o poco)

1. Baseline a soli segmenti: **0 recall** sui GT del video 3.5.
2. Filler «ehh»: whisper spesso li mappa a `e`/`m`/`ehm`; regex stretta su `ehh` non basta.
3. Modelli whisper più grandi a **temp 0** “correggono” gli stutter → peggiorano il recall per questo task.
4. Download YouTube da cloud: bot-check, serve run locale sul file reale.

## Whisper model size (importante)

Per **trovare stutter**, i modelli più grandi non sono sempre meglio: Large v3 Turbo
a temp 0.8 sul corpus sintetico collassa «anche anche» / «fornisce fornisce» e
scende a F1 ~0.57. **Small/Medium** con temp 0.8 restano preferibili per questo task.

| Modello whisper | Temp | F1 tipico (corpus GT) |
|-----------------|------|------------------------|
| Small Q8 | 0.8 | **1.0** |
| Medium Q8 | 0.8 | ~0.67–0.75 |
| Large v3 Turbo Q5 | 0.8 | ~0.57 |

## Pipeline «Solo parlato»

Nuova opzione UI: whisper + euristiche pixel, **senza** VLM. Ideale quando gli
errori sono quasi tutti di parlato (come il video 3.5). Molto più veloce.


Temperatura: su **medium**, `0.0` collassa «potrebbe potrebbe»; `0.8` lo preserva (e spesso anche `ehm`). Default app: `0.8` (override `WHISPER_TEMPERATURE`).

## LLM-on-transcript (Gemma E2B QAT) — peggiore delle euristiche

Modulo sperimentale `core/transcript_llm.py` (non collegato all'app di default).

| Caso | Euristiche | LLM E2B (few-shot) |
|------|------------|--------------------|
| fornisce fornisce | TP | TP |
| potrebbe potrebbe | TP | spesso FN |
| ehm | TP | FN |
| neg_clean | OK vuoto | **FP** (allucina ripetizione) |
| neg_refrain | OK vuoto | **FP** |

Conclusione: con modelli piccoli testo-only, le regole word-level battono l'LLM su precision/recall per questo task. L'LLM resta utile solo come secondo parere opzionale con modelli più grandi.

## Loop notturno

```bash
WHISPER_TEMPERATURE=0.8 python experiments/overnight_loop.py
# oppure un ciclo solo:
python experiments/overnight_loop.py --once
```

Log: `experiments/results/overnight_loop.jsonl`

## Come rieseguire

```bash
# genera wav (serve espeak-ng) oppure riusa corpus/
python experiments/run_speech_experiments.py "Small Q8 (~250 MB, piu veloce)"
python experiments/run_speech_experiments.py "Medium Q8 (default, ~785 MB)"
```
