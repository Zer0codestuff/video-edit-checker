#!/usr/bin/env python3
"""Esperimenti A/B sulla detection di errori di parlato.

Usa il corpus sintetico in experiments/corpus/ (pattern GT del video 3.5)
e confronta configurazioni di pipeline. YouTube e' bloccato dagli IP cloud,
quindi il corpus riproduce i 5 errori trovati a mano:
  2:15 parola in piu', 4:00 potrebbe, 6:07 a un soggetto,
  10:00 fornisce, 11:13 ehh.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.binaries import setup_path
from core.report import merge_errors
from core.speech_edits import SpeechEditConfig, detect_speech_edit_errors
from core.whisper_cpp import detect_transcript_errors, resolve_whisper_model, transcribe_video

setup_path()
# whisper .so versionate
whisper_lib = next((p.parent for p in (ROOT / "tools").rglob("whisper-cli") if p.is_file()), None)
if whisper_lib:
    os.environ["LD_LIBRARY_PATH"] = (
        str(whisper_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )

CORPUS = Path(__file__).resolve().parent / "corpus"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Ground truth per clip: lista di (approx_time_s, kind, needle)
# kind: word_repeat | ngram_repeat | filler | none
GT = {
    "gt_extra_word": [(2.0, "word_repeat", "anche")],
    "gt_potrebbe": [(2.0, "word_repeat", "potrebbe")],
    "gt_soggetto": [(2.5, "ngram_repeat", "soggetto")],
    "gt_fornisce": [(2.0, "word_repeat", "fornisce")],
    # Corpus sintetico usa "ehm" (espeak non produce bene "ehh").
    "gt_ehh": [(1.5, "filler", "ehm")],
    "neg_clean": [],
    "neg_list": [],
    "neg_refrain": [],
    "neg_short_pause": [],
    "clean_long": [],
    "mixed_long": [
        (3.0, "word_repeat", "anche"),
        (7.0, "word_repeat", "potrebbe"),
        (11.0, "ngram_repeat", "soggetto"),
        (15.0, "word_repeat", "fornisce"),
        (18.0, "filler", "ehm"),
    ],
}

CONFIGS = {
    "baseline_segments_only": SpeechEditConfig(
        enable_word_repeat=False,
        enable_ngram_repeat=False,
        enable_fillers=False,
        enable_segment_baseline=True,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
        enable_text_fallback=False,
    ),
    "word_repeat_only": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=False,
        enable_fillers=False,
        enable_segment_baseline=False,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
        enable_text_fallback=False,
    ),
    "ngram_only": SpeechEditConfig(
        enable_word_repeat=False,
        enable_ngram_repeat=True,
        enable_fillers=False,
        enable_segment_baseline=False,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
        enable_text_fallback=False,
    ),
    "fillers_only": SpeechEditConfig(
        enable_word_repeat=False,
        enable_ngram_repeat=False,
        enable_fillers=True,
        enable_segment_baseline=False,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
        enable_text_fallback=False,
    ),
    "word_plus_filler": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=False,
        enable_fillers=True,
        enable_segment_baseline=False,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
        enable_text_fallback=False,
    ),
    "full_wordlevel": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=True,
        enable_fillers=True,
        enable_segment_baseline=False,
    ),
    "full_plus_baseline": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=True,
        enable_fillers=True,
        enable_segment_baseline=True,
    ),
    "strict_gap_0.6": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=True,
        enable_fillers=True,
        enable_segment_baseline=True,
        max_repeat_gap=0.6,
    ),
    "loose_gap_3.0": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=True,
        enable_fillers=True,
        enable_segment_baseline=True,
        max_repeat_gap=3.0,
    ),
    "text_fallback_only": SpeechEditConfig(
        enable_word_repeat=False,
        enable_ngram_repeat=False,
        enable_fillers=True,
        enable_segment_baseline=False,
        enable_text_fallback=True,
        enable_near_repeats=False,
        enable_stem_restarts=False,
        enable_function_word_repeats=False,
    ),
    "no_text_fallback": SpeechEditConfig(
        enable_word_repeat=True,
        enable_ngram_repeat=True,
        enable_fillers=True,
        enable_segment_baseline=True,
        enable_text_fallback=False,
    ),
}


@dataclass
class MatchStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _match(errors, gt_items, tol: float = 4.0) -> MatchStats:
    """Match predizioni a GT.

    Preferisce needle nella description; se l'ASR ha storpiato la parola
    (es. «ansogietto» per «a un soggetto»), accetta anche overlap temporale
    con un errore dello stesso tipo atteso.
    """
    kind_to_types = {
        "word_repeat": {"repeated_phrase"},
        "ngram_repeat": {"repeated_phrase"},
        "filler": {"missed_cut", "other"},
    }
    stats = MatchStats()
    used = set()
    for t_gt, kind, needle in gt_items:
        found = False
        # 1) match stretto su needle
        for i, e in enumerate(errors):
            if i in used:
                continue
            mid = (e.start + e.end) / 2
            if needle.lower() in e.description.lower() and abs(mid - t_gt) <= tol * 2:
                used.add(i)
                found = True
                break
        if found:
            stats.tp += 1
            continue
        # 2) overlap temporale + tipo compatibile (ASR rumoroso)
        allowed = kind_to_types.get(kind, {"repeated_phrase", "missed_cut", "other"})
        for i, e in enumerate(errors):
            if i in used or e.type not in allowed:
                continue
            mid = (e.start + e.end) / 2
            if abs(mid - t_gt) <= tol * 2 or (e.start - tol) <= t_gt <= (e.end + tol):
                used.add(i)
                found = True
                break
        if found:
            stats.tp += 1
        else:
            stats.fn += 1
    stats.fp = max(0, len(errors) - len(used))
    return stats


def transcribe_clip(wav: Path, work: Path, model_label: str):
    return transcribe_video(
        wav, work,
        model_label=model_label,
        language="it",
        speech_mode=True,
        log=lambda m: print(f"  [whisper] {m}"),
    )


def _strip_words(segments):
    """Copia segmenti senza word-token (per isolare il fallback testo)."""
    from dataclasses import replace
    return [replace(s, words=[]) for s in segments]


def main() -> None:
    model_label = sys.argv[1] if len(sys.argv) > 1 else "Small Q8 (~250 MB, piu veloce)"
    low = model_label.lower()
    if "large" in low or "turbo" in low:
        model_tag = "large"
    elif "medium" in low:
        model_tag = "medium"
    elif "base" in low:
        model_tag = "base"
    else:
        model_tag = "small"
    clips = sorted(CORPUS.glob("*.wav"))
    if not clips:
        print("Nessun clip in", CORPUS)
        sys.exit(1)

    print(f"Trascrivo {len(clips)} clip con {model_label}...")
    transcripts = {}
    for wav in clips:
        name = wav.stem
        work = OUT / f"whisper_{model_tag}" / name
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        segs = transcribe_clip(wav, work, model_label)
        dt = time.time() - t0
        words = [w for s in segs for w in s.words]
        print(f"  {name}: {len(segs)} seg, {len(words)} words in {dt:.1f}s")
        print(f"    text: {' '.join(s.text for s in segs)[:160]}")
        print(f"    words: {[w.text for w in words]}")
        transcripts[name] = segs
        (work / "words.json").write_text(
            json.dumps(
                [{"start": w.start, "end": w.end, "text": w.text}
                 for s in segs for w in s.words],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    summary = {}
    for cfg_name, cfg in CONFIGS.items():
        total = MatchStats()
        per_clip = {}
        for name, segs in transcripts.items():
            dur = max((s.end for s in segs), default=1.0) + 1.0
            segs_in = _strip_words(segs) if cfg_name == "text_fallback_only" else segs
            # text_fallback_only: forza solo testo+filler (niente word-token path)
            local_cfg = cfg
            if cfg_name == "text_fallback_only":
                local_cfg = SpeechEditConfig(
                    enable_word_repeat=True,
                    enable_ngram_repeat=True,
                    enable_fillers=True,
                    enable_segment_baseline=False,
                    enable_text_fallback=True,
                )
            errs = detect_speech_edit_errors(
                segs_in, dur, language="it", cfg=local_cfg,
                baseline_fn=detect_transcript_errors,
            )
            errs = merge_errors(errs)
            gt = GT.get(name, [])
            if not gt:
                st = MatchStats(tp=0, fp=len(errs), fn=0)
            else:
                # mixed_long: tolleranza piu' ampia sui timestamp stimati
                tol = 8.0 if name == "mixed_long" else 4.0
                st = _match(errs, gt, tol=tol)
            per_clip[name] = {
                "tp": st.tp, "fp": st.fp, "fn": st.fn,
                "n_pred": len(errs),
                "preds": [
                    {"type": e.type, "start": round(e.start, 2),
                     "end": round(e.end, 2), "desc": e.description,
                     "conf": round(e.confidence, 2)}
                    for e in errs
                ],
            }
            total.tp += st.tp
            total.fp += st.fp
            total.fn += st.fn
        summary[cfg_name] = {
            "tp": total.tp, "fp": total.fp, "fn": total.fn,
            "precision": round(total.precision, 3),
            "recall": round(total.recall, 3),
            "f1": round(total.f1, 3),
            "per_clip": per_clip,
            "config": asdict(cfg),
        }
        print(f"\n=== {cfg_name} ===")
        print(f"  P={total.precision:.3f} R={total.recall:.3f} F1={total.f1:.3f} "
              f"(tp={total.tp} fp={total.fp} fn={total.fn})")

    out_path = OUT / f"speech_experiment_summary_{model_tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvato {out_path}")

    # A parita' di F1 preferisci le config piu' complete (usate in produzione).
    prefer = {
        "full_plus_baseline": 3,
        "full_wordlevel": 2,
        "no_text_fallback": 1,
    }
    ranked = sorted(
        summary.items(),
        key=lambda kv: (-kv[1]["f1"], -kv[1]["precision"], -prefer.get(kv[0], 0)),
    )
    print("\nRanking F1:")
    for name, s in ranked:
        print(f"  {s['f1']:.3f}  P={s['precision']:.3f} R={s['recall']:.3f}  {name}")


if __name__ == "__main__":
    main()
