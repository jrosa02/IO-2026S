"""
render_interactions.py — Render a GEPA LLM interactions JSON log as readable text.

Usage:
    python render_interactions.py results/interactions.json
    python render_interactions.py results/interactions.json --out results/interactions.txt
"""

import argparse
import json
import sys
from pathlib import Path


def render(interactions: list[dict], file=sys.stdout) -> None:
    total = len(interactions)
    for i, entry in enumerate(interactions, 1):
        timestamp = entry.get("timestamp", "")
        model = entry.get("model", "")
        duration_ms = entry.get("duration_ms", 0)
        response = entry.get("response", "")
        messages = entry.get("messages", [])
        prompt = messages[0]["content"] if messages else ""

        print(f"{'=' * 80}", file=file)
        print(f"  Interaction {i}/{total}  |  {timestamp}  |  {model}  |  {duration_ms} ms", file=file)
        print(f"{'=' * 80}", file=file)

        print("\n── PROMPT ──────────────────────────────────────────────────────────────────\n", file=file)
        print(prompt, file=file)

        print("\n── RESPONSE ────────────────────────────────────────────────────────────────\n", file=file)
        print(response, file=file)
        print(file=file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GEPA LLM interactions log as readable text.")
    parser.add_argument("input", help="Path to interactions JSON file")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Write output to this file instead of stdout")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            render(data, file=f)
        print(f"Rendered {len(data)} interaction(s) to {args.out}")
    else:
        render(data)


if __name__ == "__main__":
    main()
