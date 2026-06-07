"""Sphinx configuration for io-2026s documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

project = "io-2026s"
author = "Jan Rosa"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "io-2026s"

latex_documents = [
    ("index", "io-2026s.tex", "io-2026s API Reference", "Jan Rosa", "manual"),
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"\setcounter{tocdepth}{2}",
}

exclude_patterns = ["_build"]

# Suppress warnings caused by symbols being re-exported from src/__init__
# and also defined in their source modules (duplicate autodoc entries).
suppress_warnings = ["ref.python"]
nitpicky = False
