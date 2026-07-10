#!/usr/bin/env python3
"""Loop notturno: riesegue esperimenti speech-edit e logga i risultati.

Uso:
  python experiments/overnight_loop.py
  WHISPER_TEMPERATURE=0.8 python experiments/overnight_loop.py --once
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "overnight_loop.jsonl"

MODELS = [
    "Small Q8 (~250 MB, piu veloce)",
    "Medium Q8 (default, ~785 MB)",
    "Large v3 Turbo Q5 (~550 MB)",
    "Base Q8 (~80 MB, leggero)",
]
TEMPS = ["0.0", "0.8"]  # 0.6 e' una valle: collassa stutter senza i benefici di 0.8


def run_once(model: str, temp: str) -> dict:
    env = os.environ.copy()
    env["WHISPER_TEMPERATURE"] = temp
    env["PYTHONPATH"] = str(ROOT)
    # Clear whisper cache for this model tag so temp changes take effect
    if "Large" in model or "Turbo" in model:
        tag = "large"
    elif "Medium" in model:
        tag = "medium"
    elif "Base" in model:
        tag = "base"
    else:
        tag = "small"
    cache = OUT / f"whisper_{tag}"
    if cache.exists():
        subprocess.run(["rm", "-rf", str(cache)], check=False)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "run_speech_experiments.py"), model],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    dt = time.time() - t0
    summary_path = OUT / f"speech_experiment_summary_{tag}.json"
    best = None
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        best_name, best_s = max(data.items(), key=lambda kv: (kv[1]["f1"], kv[1]["precision"]))
        best = {
            "config": best_name,
            "f1": best_s["f1"],
            "precision": best_s["precision"],
            "recall": best_s["recall"],
            "tp": best_s["tp"],
            "fp": best_s["fp"],
            "fn": best_s["fn"],
        }
    # Parse ranking lines from stdout (fallback se summary non trovato)
    ranking = [ln.strip() for ln in (proc.stdout or "").splitlines()
               if ln.strip()[:1].isdigit() and "P=" in ln]
    if best is None and ranking:
        # "0.824  P=1.000 R=0.700  full_wordlevel"
        parts = ranking[0].split()
        try:
            best = {
                "config": parts[-1],
                "f1": float(parts[0]),
                "precision": float(parts[1].split("=")[1]),
                "recall": float(parts[2].split("=")[1]),
                "tp": None, "fp": None, "fn": None,
                "from_stdout": True,
            }
        except (IndexError, ValueError):
            pass
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temp,
        "elapsed_s": round(dt, 1),
        "returncode": proc.returncode,
        "best": best,
        "ranking_tail": ranking[:5],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def main() -> None:
    once = "--once" in sys.argv
    cycle = 0
    while True:
        cycle += 1
        print(f"\n######## overnight cycle {cycle} @ {datetime.now().isoformat()} ########")
        for temp in TEMPS:
            for model in MODELS:
                print(f"--- model={model} temp={temp}")
                try:
                    row = run_once(model, temp)
                except Exception as err:
                    row = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "temperature": temp,
                        "error": str(err),
                    }
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("  ->", json.dumps(row.get("best") or row.get("error"), ensure_ascii=False))
        if once:
            break
        # Pausa tra cicli completi
        time.sleep(30)


if __name__ == "__main__":
    main()
