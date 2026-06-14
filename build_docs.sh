#!/usr/bin/env bash
# Build HTML (with search), API reference PDF, and user guide PDF.
# Usage: ./build_docs.sh
set -euo pipefail

OUT=docs/_build
TEX=TEXMFHOME=/usr/share/texlive/texmf-dist

# --- Sphinx HTML + API PDF ---
uv run sphinx-build -b html  docs/ "$OUT/html"  -q
uv run sphinx-build -b latex docs/ "$OUT/latex" -q
env TEXMFHOME=/usr/share/texlive/texmf-dist \
    pdflatex -interaction=nonstopmode -output-directory "$OUT/latex" "$OUT/latex/io-2026s.tex" > /dev/null 2>&1
env TEXMFHOME=/usr/share/texlive/texmf-dist \
    pdflatex -interaction=nonstopmode -output-directory "$OUT/latex" "$OUT/latex/io-2026s.tex" > /dev/null 2>&1
cp "$OUT/latex/io-2026s.pdf" "$OUT/io-2026s-api.pdf"

_latex_build() {
    local src="$1" dir="$2" out="$3"
    mkdir -p "$dir"
    local tex="$dir/$(basename "$src")"
    cp "$src" "$tex"
    env TEXMFHOME=/usr/share/texlive/texmf-dist \
        pdflatex -interaction=nonstopmode -output-directory "$dir" "$src" > /dev/null 2>&1
    # second pass for TOC and references
    env TEXMFHOME=/usr/share/texlive/texmf-dist \
        pdflatex -interaction=nonstopmode -output-directory "$dir" "$src" > /dev/null 2>&1
    cp "$dir/$(basename "${src%.tex}").pdf" "$out"
}

_latex_build docs/userguide.tex "$OUT/userguide" "$OUT/io-2026s-userguide.pdf"
_latex_build docs/problem.tex   "$OUT/problem"   "$OUT/io-2026s-problem.pdf"
_latex_build docs/results.tex   "$OUT/results"   "$OUT/io-2026s-results.pdf"

echo "HTML         : $OUT/html/index.html"
echo "API PDF      : $OUT/io-2026s-api.pdf"
echo "Guide PDF    : $OUT/io-2026s-userguide.pdf"
echo "Problem PDF  : $OUT/io-2026s-problem.pdf"
echo "Results PDF  : $OUT/io-2026s-results.pdf"
