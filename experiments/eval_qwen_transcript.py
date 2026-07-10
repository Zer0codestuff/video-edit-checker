#!/usr/bin/env python3
"""Valuta Qwen3.5-4B (thinking) sul transcript del video reale 3.5.

Uso tipico (riusa transcript gia' prodotto da Solo parlato):
  python -m experiments.eval_qwen_transcript \\
    --transcript runs/run_.../whisper/transcript.json \\
    --thinking

Confronta vs ground truth manuale e vs euristiche word-level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.binaries import setup_path
from core.llama_server import SERVER
from core.report import filter_errors, fmt_time, merge_errors
from core.speech_edits import detect_speech_edit_errors
from core.transcript_llm import (
    DEFAULT_TRANSCRIPT_LLM,
    TRANSCRIPT_LLM_MODELS,
    analyze_transcript_with_llm,
)
from core.whisper_cpp import detect_transcript_errors, load_transcript_json

setup_path()

# Ground truth video 3.5 (solo per valutazione; non usato in detection).
GT = [
    (135.0, "parola extra", ("il", "anche", "parola", "ripet")),
    (240.0, "potrebbe", ("potrebbe",)),
    (367.0, "a un soggetto", ("soggetto",)),
    (600.0, "fornisce", ("fornisce", "fornire")),
    (673.0, "ehh", ("ehh", "ehm", "uhm", "em", "filler", "esit")),
]


def _match(errors, tol: float = 3.0) -> dict:
    used: set[int] = set()
    rows = []
    tp = fn = 0
    for t_gt, label, needles in GT:
        hit = None
        for i, e in enumerate(errors):
            if i in used:
                continue
            mid = (e.start + e.end) / 2
            if abs(mid - t_gt) > tol and not (e.start - tol <= t_gt <= e.end + tol):
                continue
            blob = e.description.lower()
            if any(n.lower() in blob for n in needles):
                hit = e
                used.add(i)
                break
        if hit is None:
            # Match temporale debole solo se un solo errore e' vicino.
            near = [
                (i, e) for i, e in enumerate(errors)
                if i not in used and abs((e.start + e.end) / 2 - t_gt) <= tol
            ]
            if len(near) == 1:
                i, hit = near[0]
                used.add(i)
        if hit is not None:
            tp += 1
            rows.append({
                "gt": label, "t": t_gt, "status": "TP",
                "pred": f"{fmt_time(hit.start)} {hit.description}",
            })
        else:
            fn += 1
            rows.append({"gt": label, "t": t_gt, "status": "FN", "pred": None})
    fp = max(0, len(errors) - len(used))
    fp_list = [
        {"start": fmt_time(e.start), "end": fmt_time(e.end),
         "desc": e.description, "conf": round(e.confidence, 2)}
        for i, e in enumerate(errors) if i not in used
    ]
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "recall": tp / len(GT),
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "per_gt": rows,
        "false_positives": fp_list,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_TRANSCRIPT_LLM,
                    choices=list(TRANSCRIPT_LLM_MODELS.keys()))
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chunk-seconds", type=float, default=45.0)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    transcript = args.transcript.expanduser().resolve()
    if not transcript.exists():
        sys.exit(f"Transcript non trovato: {transcript}")
    outdir = (args.outdir or (ROOT / "runs" / "qwen_transcript_eval")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    segments = load_transcript_json(transcript)
    dur = max((s.end for s in segments), default=0.0)
    n_words = sum(len(s.words) for s in segments)
    print(f"Transcript: {transcript}")
    print(f"Segmenti={len(segments)} parole={n_words} durata={dur:.1f}s")

    heur = detect_speech_edit_errors(
        segments, dur, language="it",
        baseline_fn=detect_transcript_errors,
    )
    heur = filter_errors(merge_errors(heur), args.min_confidence)
    heur_stats = _match(heur)
    print(f"\n=== Euristiche === TP={heur_stats['tp']}/5 "
          f"FP={heur_stats['fp']} R={heur_stats['recall']:.2f}")
    for row in heur_stats["per_gt"]:
        print(f"  {row['status']} @{row['t']:.0f}s {row['gt']}: {row['pred']}")

    hf, needs_jinja = TRANSCRIPT_LLM_MODELS[args.model]
    # Con thinking illimitato Qwen brucia i token elencando le righe.
    # Budget basso + chunk corti lasciano spazio al JSON finale.
    rb = 256 if args.thinking else 0
    print(f"\nAvvio {args.model} (jinja={needs_jinja}, "
          f"thinking={args.thinking}, reasoning_budget={rb})...")
    t0 = time.time()
    SERVER.ensure(
        hf,
        jinja=needs_jinja or args.thinking,
        n_parallel=1,
        ctx_per_slot=8192,
        batch_preset="Conservativo (iGPU / poca VRAM)",
        reasoning_budget=rb,
        log=print,
    )
    print(f"Server pronto in {time.time() - t0:.1f}s")

    t1 = time.time()
    llm_errs = analyze_transcript_with_llm(
        segments,
        language="it",
        enable_thinking=args.thinking,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=min(10.0, args.chunk_seconds / 3),
        max_tokens=1024 if args.thinking else 800,
        timeout=600.0,
        log=print,
    )
    llm_errs = filter_errors(merge_errors(llm_errs), args.min_confidence)
    dt = time.time() - t1
    llm_stats = _match(llm_errs)
    print(f"\n=== LLM {args.model} (thinking={args.thinking}) "
          f"in {dt:.1f}s ===")
    print(f"Predizioni={len(llm_errs)} TP={llm_stats['tp']}/5 "
          f"FP={llm_stats['fp']} R={llm_stats['recall']:.2f} "
          f"P={llm_stats['precision']:.2f}")
    for row in llm_stats["per_gt"]:
        print(f"  {row['status']} @{row['t']:.0f}s {row['gt']}: {row['pred']}")
    if llm_stats["false_positives"]:
        print("  FP:")
        for fp in llm_stats["false_positives"]:
            print(f"    {fp['start']}–{fp['end']} {fp['desc']} ({fp['conf']})")

    out = {
        "transcript": str(transcript),
        "model": args.model,
        "hf": hf,
        "thinking": args.thinking,
        "chunk_seconds": args.chunk_seconds,
        "elapsed_s": round(dt, 1),
        "heuristics": {
            **{k: heur_stats[k] for k in ("tp", "fn", "fp", "recall", "precision")},
            "errors": [
                {"start": e.start, "end": e.end, "type": e.type,
                 "desc": e.description, "conf": e.confidence}
                for e in heur
            ],
            "per_gt": heur_stats["per_gt"],
            "false_positives": heur_stats["false_positives"],
        },
        "llm": {
            **{k: llm_stats[k] for k in ("tp", "fn", "fp", "recall", "precision")},
            "errors": [
                {"start": e.start, "end": e.end, "type": e.type,
                 "desc": e.description, "conf": e.confidence}
                for e in llm_errs
            ],
            "per_gt": llm_stats["per_gt"],
            "false_positives": llm_stats["false_positives"],
        },
    }
    out_path = outdir / "qwen_transcript_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvato: {out_path}")


if __name__ == "__main__":
    main()
