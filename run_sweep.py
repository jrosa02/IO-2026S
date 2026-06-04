"""
run_sweep.py — Run the GEPA benchmark experiment across multiple LLM models.

Edit MODELS and BASE_ARGS below to configure the sweep.
Each model is checked against the local Ollama installation before running.
Failed runs are logged and the sweep continues; a summary is printed at the end.

Usage:
    uv run python run_sweep.py
"""

import subprocess
import sys
import traceback
from pathlib import Path

from ollama import Client

# ---------------------------------------------------------------------------
# Configuration — edit these two sections to change the sweep
# ---------------------------------------------------------------------------

MODELS = [
    "ollama/qwen3:4b-instruct-2507-q4_K_M",
    "ollama/gemma4:e2b",
    "ollama/phi4-mini-reasoning:latest",
    "ollama/granite4:1b-h-q8_0",
]

BASE_ARGS: dict[str, str] = {
    "--agents": "gepa-ga,genetic,gepa-sa,sa",
    "--data": "data/sch1000.txt",
    "--train-instances": "5",
    "--max-metric-calls": "100",
    "--interactions-log": "interactions.json",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent


def check_model(model_str: str) -> bool:
    """Return True if *model_str* is present in the local Ollama installation."""
    name = model_str.removeprefix("ollama/")
    try:
        installed = {m.model for m in Client().list().models}
        return name in installed
    except Exception:
        return False


def run_experiment(model: str) -> bool:
    """Invoke main.py for *model*; return True on success (exit code 0)."""
    argv = [sys.executable, "main.py", "--reflection-lm", model]
    for flag, value in BASE_ARGS.items():
        argv += [flag, value]
    result = subprocess.run(argv, cwd=_HERE)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    W = 60

    print("=" * W)
    print(" MODEL SWEEP")
    print(f" agents           : {BASE_ARGS['--agents']}")
    print(f" train-instances  : {BASE_ARGS['--train-instances']}")
    print(f" max-metric-calls : {BASE_ARGS['--max-metric-calls']}")
    print("=" * W)
    print()

    # --- phase 1: availability check ---
    available: list[str] = []
    skipped: list[str] = []

    for model in MODELS:
        if check_model(model):
            print(f"  [OK]   {model}")
            available.append(model)
        else:
            print(f"  [SKIP] {model}  (not in ollama list)")
            skipped.append(model)

    print()
    if not available:
        print("No models available. Exiting.")
        return

    # --- phase 2: run experiments ---
    results: list[tuple[str, str]] = []

    for model in available:
        short = model.removeprefix("ollama/")
        print("=" * W)
        print(f" Running: {short}")
        print("=" * W)
        try:
            ok = run_experiment(model)
            status = "OK" if ok else f"FAILED (exit {1})"
        except Exception:
            traceback.print_exc()
            status = "FAILED (exception)"
        results.append((short, status))
        print()

    for model in skipped:
        results.append((model.removeprefix("ollama/"), "SKIPPED (not installed)"))

    # --- phase 3: summary ---
    col = max(len(m) for m, _ in results) + 2
    print("=" * W)
    print(" MODEL SWEEP SUMMARY")
    print("=" * W)
    for model, status in results:
        print(f" {model:<{col}} {status}")
    print("=" * W)
    succeeded = sum(1 for _, s in results if s == "OK")
    print(f" {succeeded} / {len(MODELS)} succeeded")
    print("=" * W)


if __name__ == "__main__":
    main()
