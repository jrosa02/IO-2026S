#!/usr/bin/env python3
"""
md_to_pdf.py – Konwertuje plik Markdown na PDF

Użycie:
    uv run python md_to_pdf.py <plik.md> [wyjście.pdf]

Jeśli plik wyjściowy nie jest podany, PDF zostaje zapisany obok wejściowego
pliku z rozszerzeniem .pdf.

Wymaga: weasyprint, markdown  (uv add weasyprint markdown)
"""

from __future__ import annotations
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# CSS – styl dokumentu
# ---------------------------------------------------------------------------

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2.0cm 2.2cm 2.0cm;
    @bottom-right {
        content: "Strona " counter(page) " z " counter(pages);
        font-size: 8pt;
        color: #888;
    }
}

body {
    font-family: Arial, Helvetica, "Liberation Sans", sans-serif;
    font-size: 10pt;
    color: #1a1a1a;
    line-height: 1.5;
}

h1 {
    color: #002a6e;
    font-size: 14pt;
    border-bottom: 2px solid #002a6e;
    padding-bottom: 4pt;
    margin-top: 22pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h2 {
    color: #004299;
    font-size: 11.5pt;
    border-left: 4px solid #004299;
    padding-left: 8pt;
    margin-top: 16pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

h3 {
    color: #333;
    font-size: 10.5pt;
    margin-top: 12pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

h4, h5, h6 {
    color: #444;
    margin-top: 10pt;
    page-break-after: avoid;
}

p {
    margin: 4pt 0 7pt;
    orphans: 3;
    widows: 3;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 12pt;
    font-size: 9pt;
    page-break-inside: auto;
}

th {
    background-color: #002a6e;
    color: #ffffff;
    padding: 5pt 8pt;
    text-align: left;
    font-weight: bold;
    border: 1px solid #001a50;
}

td {
    padding: 4pt 8pt;
    border: 1px solid #c0c8d8;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #eef2fb;
}

tr:nth-child(odd) td {
    background-color: #ffffff;
}

tr { page-break-inside: avoid; }

/* Code / calculations */
pre {
    background: #f4f4f4;
    border: 1px solid #d0d0d0;
    border-left: 4px solid #004299;
    padding: 8pt 10pt;
    font-family: "Courier New", "DejaVu Sans Mono", monospace;
    font-size: 8.5pt;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 6pt 0 10pt;
    page-break-inside: avoid;
}

code {
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    background: #f0f0f0;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

/* Blockquote – used for callouts/results */
blockquote {
    background: #fffbea;
    border-left: 4px solid #e0a020;
    margin: 6pt 0 10pt 0;
    padding: 6pt 12pt;
    font-style: normal;
    color: #333;
}

blockquote p {
    margin: 2pt 0;
}

/* Lists */
ul, ol {
    margin: 4pt 0 8pt 18pt;
    padding: 0;
}

li {
    margin-bottom: 3pt;
}

/* Images */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 8pt auto;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid #c0c8d8;
    margin: 14pt 0;
}

/* Links – show URL in parentheses for print */
a {
    color: #004299;
    text-decoration: none;
    word-break: break-all;
}

/* Strong / emphasis */
strong { color: #111; }
em     { color: #444; }

/* Math equations */
p.math-display {
    text-align: center;
    margin: 6pt 30pt;
    font-size: 10.5pt;
    color: #1a1a1a;
    page-break-inside: avoid;
}
"""

# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

def wrap_html(title: str, body: str) -> str:
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="pl">
        <head>
          <meta charset="utf-8">
          <title>{title}</title>
          <style>{CSS}</style>
        </head>
        <body>
        {body}
        </body>
        </html>
    """)

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

_LATEX_SUBS = [
    (r"\\cdot",   "·"),
    (r"\\times",  "×"),
    (r"\\Delta",  "Δ"),
    (r"\\delta",  "δ"),
    (r"\\alpha",  "α"),
    (r"\\beta",   "β"),
    (r"\\gamma",  "γ"),
    (r"\\lambda", "λ"),
    (r"\\mu",     "μ"),
    (r"\\sigma",  "σ"),
    (r"\\eta",    "η"),
    (r"\\theta",  "θ"),
    (r"\\varphi", "φ"),
    (r"\\phi",    "φ"),
    (r"\\psi",    "ψ"),
    (r"\\omega",  "ω"),
    (r"\\pi",     "π"),
    (r"\\approx", "≈"),
    (r"\\leq",    "≤"),
    (r"\\geq",    "≥"),
    (r"\\neq",    "≠"),
    (r"\\pm",     "±"),
    (r"\\,",      " "),  # narrow no-break space
    (r"\\;",      " "),
    (r"\\:",      " "),
    (r"\\!",      ""),
    (r"\\quad",   "  "),
]


def _latex_to_text(latex: str) -> str:
    s = latex.strip()
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1) / (\2)", s)
    s = re.sub(r"\\text\{([^}]*)\}",             r"\1",           s)
    s = re.sub(r"_\{([^}]*)\}",                  r"<sub>\1</sub>", s)
    s = re.sub(r"_([a-zA-Z0-9])",                r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{\\circ\}",                  "°",              s)
    s = re.sub(r"\^\\circ",                       "°",              s)
    s = re.sub(r"\^\{([^}]*)\}",                 r"<sup>\1</sup>", s)
    s = re.sub(r"\^([a-zA-Z0-9])",               r"<sup>\1</sup>", s)
    for pattern, repl in _LATEX_SUBS:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return s


def latex_to_html(md_text: str) -> str:
    r"""Replace \[...\] display blocks and \(...\) inline math with plain HTML."""

    def replace_display(m: re.Match) -> str:
        return f'\n<p class="math-display">{_latex_to_text(m.group(1))}</p>\n'

    def replace_inline(m: re.Match) -> str:
        return _latex_to_text(m.group(1))

    md_text = re.sub(r"\$\$(.*?)\$\$",  replace_display, md_text, flags=re.DOTALL)
    md_text = re.sub(r"\\\[(.*?)\\\]", replace_display, md_text, flags=re.DOTALL)
    md_text = re.sub(r"\$([^\$\n]+?)\$", replace_inline, md_text)
    md_text = re.sub(r"\\\((.*?)\\\)", replace_inline,  md_text)
    return md_text


def md_to_html(md_text: str) -> str:
    import markdown  # type: ignore
    md_text = latex_to_html(md_text)
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )


def convert(src: Path, dst: Path) -> None:
    try:
        import weasyprint  # type: ignore
    except ImportError:
        print("Błąd: weasyprint nie jest zainstalowany.")
        print("  Uruchom:  uv add weasyprint")
        sys.exit(1)

    md_text = src.read_text(encoding="utf-8")
    html_body = md_to_html(md_text)
    html = wrap_html(src.stem, html_body)

    print(f"  Plik wejściowy : {src}")
    print(f"  Plik wyjściowy : {dst}")

    try:
        weasyprint.HTML(string=html, base_url=str(src.parent)).write_pdf(dst)
        print(f"  OK – PDF zapisany ({dst.stat().st_size // 1024} kB)")
    except Exception as exc:
        # Fallback: save HTML so the user can open it in a browser / print
        html_out = dst.with_suffix(".html")
        html_out.write_text(html, encoding="utf-8")
        print(f"  Błąd PDF: {exc}")
        print(f"  Zapisano HTML jako fallback: {html_out}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"Błąd: plik nie istnieje: {src}")
        sys.exit(1)
    if src.suffix.lower() not in {".md", ".markdown"}:
        print(f"Ostrzeżenie: oczekiwano pliku .md, otrzymano: {src.suffix}")

    dst = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else src.with_suffix(".pdf")

    convert(src, dst)


if __name__ == "__main__":
    main()
