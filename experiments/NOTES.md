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

## Video reale (luglio 2026) — risultato onesto

Sul file scaricato localmente (`NqG8O7aNnTs`), Whisper **non** emette ripetizioni adiacenti
come nel corpus TTS. Forme tipiche nel transcript Small Q8 @ 0.8:

| GT | Cosa c'e' nell'ASR | Detector classico |
|----|--------------------|-------------------|
| 2:15 | `il il` (function-word, gap ~80ms) | FN (stopword ignorate) |
| 4:00 | `potrebbe non prevenuto potrebbe` | FN (non adiacente) |
| 6:07 | `a un soggetto` … `a un soggetto` (3 parole in mezzo) | FN (non adiacente) |
| 10:00 | `fornisce, sarebbe in grado di fornire` | FN (stem diverso) |
| 11:13 | nessun token filler; gap ~0.36s quasi silente (rms≪parlato) | FN |

**Baseline app (prima dei restart detector):** Small Q8 temp0.8 = **0/5**, 0 FP;
ensemble 0.0+0.8 = **0/5** + FP «agisce» @0:59.

**Dopo near-repeat / function-word / stem-restart (stesso ASR, no GT hardcoded):**

| Evento | Esito | Nota |
|--------|-------|------|
| 2:15 | TP | «il» adiacente, gap stretto |
| 4:00 | TP | near-unigram «potrebbe» |
| 6:07 | TP | near-ngram «a un soggetto» |
| 10:00 | TP | stem false-start «fornisce / fornire» + cue `sarebbe` |
| 11:13 | **FN** | nessun segnale ASR; energia nel gap ≈ silenzio |

FP vs i 5 GT (intero video, Small@0.8 single, E2E CLI): **2**
- «politiche di prevenzione» @1:43 — ripresa reale nel parlato (non in GT)
- «di di» @12:00 — stutter function-word reale (non in GT)

Ensemble 0.0+0.8 con i nuovi detector: ancora **4/5**, ma **5 FP** (riappare «agisce»
e altri da temp 0.0). Consigliato: **single Small@0.8**, ensemble off.

### Prove negative (non integrate)

1. **Filler acustico su gap mid-energy:** decine di candidati / video; il gap @11:13 e'
   troppo silente per un «ehh» sonoro. Non aggiunto.
2. **Best-of / entropy / no-speech-thold** su finestre ±4s intorno ai GT: nessuna
   variante ha prodotto `fornisce fornisce` o un token `ehh`.
3. **Matcher permissivi solo-tempo:** evitati; TP richiede ±3s **e** descrizione coerente.

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
| **full + near/stem/function (video reale)** | ~0.67 (4/6) | **0.8 (4/5)** | ~0.73 | Vedi sezione video reale |

## Cosa funziona

1. **Word-token whisper (`-ojf`)** + merge BPE (senza `strip()` prematuro).
2. **Ripetizioni adiacenti** unigram/n-gram con gap ≤ ~1.5s, ignorando stopword unigram.
3. **Near-repeat** (stessa parola/n-gram con materiale in mezzo, ancore lessicali forti).
4. **Function-word adiacenti** con gap ≤ ~0.35s (`il il`, `di di`).
5. **Stem false-start** con cue di ripresa (`sarebbe`/`potrebbe`/…) e virgola ASR.
6. **Fallback sul testo del segmento** se i token sono fusi/storti.
7. **`-mc 0` + temperatura ~0.8**: riduce il collasso «potrebbe potrebbe» → «potrebbe» (visto su medium a temp 0).

## Cosa non funziona (o poco)

1. Baseline a soli segmenti: **0 recall** sui GT del video 3.5.
2. Filler «ehh» sul video reale: assente dall'ASR e dal profilo energetico del gap; sul corpus TTS dipende da come Whisper mappa `ehm`/`em`.
3. Modelli whisper più grandi (Large Turbo) “correggono” gli stutter → peggiorano il recall per questo task.
4. Download YouTube da cloud: bot-check, serve run locale sul file reale.
5. Ensemble su Small nel video reale: non alza il recall, alza i FP.

## Whisper model size (importante)

Per **trovare stutter**, i modelli più grandi non sono sempre meglio: Large v3 Turbo
a temp 0.8 sul corpus sintetico collassa «anche anche» / «fornisce fornisce» e
scende a F1 ~0.57. **Small** con temp 0.8 è il migliore per questo task.

| Modello whisper | Temp | F1 tipico (corpus GT) | Overnight avg (n run, ~86) |
|-----------------|------|------------------------|----------------------------|
| Small Q8 | 0.8 | **1.0** | **0.99** (n=11) |
| Small Q8 | 0.6 | ~1.0 | **0.98** (n=7) |
| Small Q8 | 0.0 | ~0.95 | 0.96 (n=14) |
| Base Q8 | 0.8 | **0.75** | 0.75 (n=2) — perde anche/filler |
| Base Q8 | 0.0 | ~0.82 | 0.82 (n=3) |
| Base ensemble 0.0+0.8 | **0.89** | (manuale) — recupera stutter, non filler |
| Medium Q8 | 0.8 | ~0.67–0.82 | 0.76 (n=10) |
| Medium Q8 | 0.0 | ~0.67 | 0.68 (n=13) |
| Medium Q8 | 0.6 | ~0.46 | **0.46** (n=7) — peggiore |
| Large v3 Turbo Q5 | 0.8 | ~0.89 | 0.89 (n=6) |
| Large v3 Turbo Q5 | 0.0 | ~0.82–0.89 | 0.86 (n=9) |
| Large v3 Turbo Q5 | 0.6 | ~0.46 | 0.46 (n=3) |
| Large ensemble 0.0+0.8 | **0.75** | (manuale) — **peggio** del single 0.8 |

## Ensemble multi-temperatura

Su **Medium**, una sola passata a temp 0.8 perde «anche anche» (F1 0.89).
Due passate (0.0 + 0.8) unite → **F1 1.0** (2× tempo whisper).

**Perché 0.0+0.8 e non 0.6:** temp `0.6` e' una valle — collassa stutter
(come un modello “pulito”) senza recuperare «potrebbe» (che compare a 0.8).
`0.0` e `0.8` catturano sottoinsiemi diversi → l'unione li copre tutti.

Checkbox UI: «Ensemble whisper (temp 0.0 + 0.8)». Consigliato solo se usi
**Medium**; con Small di solito non serve. Su **Large** l'ensemble puo'
addirittura peggiorare (0.89 → 0.75): entrambe le temperature collassano
gli stessi stutter, quindi l'unione non aggiunge recall.

## Falsi positivi

Stress test `clean_long.wav` (parlato italiano continuo ~30s senza errori): **0 errori** con Small@0.8. I negativi del corpus (`neg_clean`, liste, ritornelli) restano a 0 FP nelle run overnight.

## Configurazione consigliata (video tipo 3.5)

1. Pipeline **Solo parlato**
2. Whisper **Small Q8**
3. Ensemble **off**
4. Lingua Italiano

Se Small non basta e passi a Medium: attiva ensemble.

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

## LLM-on-transcript (Qwen3.5-4B) — video reale 3.5

Script: `python -m experiments.eval_qwen_transcript --transcript ... [--thinking|--no-thinking]`.

Stesso transcript Small@0.8 del run UI. Match stretto ±3s vs GT manuale.

| Setup | TP/5 | FP | Tempo | Note |
|-------|------|----|-------|------|
| Euristiche word-level | **4** | 2 | <1s | baseline attuale |
| Qwen3.5-4B **thinking** (budget 256, chunk 45s) | 1 | 7 | ~3 min | trova solo «il il»; molti FP ASR |
| Qwen3.5-4B **no thinking** (chunk 45s) | 0 | 7 | ~45s | peggiore |

Problemi osservati:
1. Con thinking illimitato brucia tutti i `max_tokens` elencando le righe → content vuoto.
2. Anche con budget basso spesso risponde `{"errors":[]}` sui near-repeat evidenti (`potrebbe…potrebbe`, `a un soggetto`).
3. Non recupera `ehh` (assente dal transcript).
4. Inventa ripetizioni su token ASR spezzati (`d'arnoso`, `compagnia`).

Conclusione: su questo video Qwen3.5-4B **non batte** le euristiche; thinking non aiuta il recall e aumenta i FP.

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
