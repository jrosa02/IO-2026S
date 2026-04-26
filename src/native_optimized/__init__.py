"""
native_optimized — C extension interface for hot-path scheduling operations.

Compilation runs on import if .so files are absent.

Public API
----------
    evaluate(p, a, b, schedule, d)              -> int
    evaluate_swap(p, a, b, schedule, d, i, j)   -> int
    evaluate_batch(p, a, b, schedules, d)        -> np.ndarray shape (k,) int64
    evaluate_batch_swap(p, a, b, schedule, d, i_arr, j_arr)
                                                 -> np.ndarray shape (k,) int64
    batch_crossover(p1, p2, a, b)               -> np.ndarray shape (k, n) int64
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE     = Path(__file__).parent
_COMPILED = _HERE / "__compiled__"
_SETUP_PY = _HERE.parent.parent / "setup.py"
_SUFFIX   = sysconfig.get_config_var("EXT_SUFFIX")


def _load_module(so_name: str):
    so = _COMPILED / f"{so_name}{_SUFFIX}"
    if not so.exists():
        return None
    spec = importlib.util.spec_from_file_location(so_name, so)
    assert spec
    mod  = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


_evaluate_mod  = _load_module("evaluate_opt")
_crossover_mod = _load_module("crossover_opt")

if _evaluate_mod is None or _crossover_mod is None:
    logger.info("Building C extensions via setup.py …")
    subprocess.run(
        ["python", str(_SETUP_PY), "build_ext", "--inplace"],
        cwd=str(_SETUP_PY.parent),
        check=True,
    )
    _evaluate_mod  = _load_module("evaluate_opt")
    _crossover_mod = _load_module("crossover_opt")

if _evaluate_mod is None:
    raise ImportError("evaluate_opt extension not found after build. Run: python setup.py build_ext --inplace")
if _crossover_mod is None:
    raise ImportError("crossover_opt extension not found after build. Run: python setup.py build_ext --inplace")

# ---------------------------------------------------------------------------
# Public API — function references bound once at import time
# ---------------------------------------------------------------------------

evaluate            = _evaluate_mod.evaluate_opt
evaluate_swap       = _evaluate_mod.evaluate_swap_opt
evaluate_batch      = _evaluate_mod.evaluate_batch_opt
evaluate_batch_swap = _evaluate_mod.evaluate_batch_swap_opt
batch_crossover     = _crossover_mod.batch_crossover_opt