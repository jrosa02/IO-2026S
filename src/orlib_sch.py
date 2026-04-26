"""
orlib_sch.py - Loader for ORLib Common Due Date Scheduling instances
====================================================================
Parses the ``sch`` family of files from:
    https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
    (sch10, sch20, sch50, sch100, sch200, sch500, sch1000)

Problem definition
------------------
Single-machine scheduling with a *common* due date d.
Each job i has:
    p(i)  - deterministic processing time
    a(i)  - earliness penalty weight  (cost per unit early)
    b(i)  - tardiness penalty weight  (cost per unit late)

The common due date is parameterised by a tightness factor h:
    d(h) = floor(SUM_P * h),   h ∈ {0.2, 0.4, 0.6, 0.8}

The objective is to minimise:
    sum_i [ a(i) * max(0, d - C(i))  +  b(i) * max(0, C(i) - d) ]
where C(i) is the completion time of job i.

File format
-----------
    <number_of_problems>
    for each problem:
        <n>                      ← number of jobs
        <p_1> <a_1> <b_1>
        <p_2> <a_2> <b_2>
        ...
        <p_n> <a_n> <b_n>

Public API
----------
    load(source)          → SchDataset
    SchDataset            - iterable collection of SchInstance
    SchInstance           - one benchmark instance (see attributes below)
    SchJob                - named tuple: (p, a, b)

Quick start
-----------
    from orlib_sch import load

    ds = load("sch10.txt")           # from a local file path
    # or
    ds = load("https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/sch10.txt")

    for inst in ds:
        print(inst)                  # summary line
        print(inst.jobs)             # list of SchJob namedtuples
        d = inst.due_date(h=0.4)     # common due date for a given h
        cost = inst.evaluate(schedule=list(range(inst.n)), h=0.4)

    # Access as a list
    instances = ds.instances
    first = ds[0]
    print(first.to_dict())           # JSON-serialisable dict
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import urllib.request
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .native_optimized import evaluate
from .native_optimized import evaluate_swap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class SchJob(NamedTuple):
    """A single job in a common due date scheduling instance."""

    p: int  # processing time
    a: int  # earliness penalty weight
    b: int  # tardiness penalty weight

    def __repr__(self) -> str:
        return f"SchJob(p={self.p}, a={self.a}, b={self.b})"


@dataclass
class SchInstance:
    """
    One benchmark instance of the common due date scheduling problem.

    Attributes
    ----------
    index : int
        0-based position within the source file.
    n : int
        Number of jobs.
    jobs : list[SchJob]
        Job data in original order.  ``jobs[i]`` corresponds to job i.
    sum_p : int
        Sum of all processing times (= total makespan of any permutation).
    """

    index: int
    n: int
    jobs: list[SchJob]
    sum_p: int = field(init=False)

    def __post_init__(self) -> None:
        if len(self.jobs) != self.n:
            raise ValueError(f"Instance {self.index}: declared n={self.n} but got {len(self.jobs)} jobs.")
        self.sum_p = sum(j.p for j in self.jobs)
        self.p_array = np.array([job.p for job in self.jobs], dtype=np.int64)
        self.a_array = np.array([job.a for job in self.jobs], dtype=np.int64)
        self.b_array = np.array([job.b for job in self.jobs], dtype=np.int64)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def due_date(self, h: float) -> int:
        """
        Compute the common due date for tightness factor *h*.

        d = floor(SUM_P * h),   h typically in {0.2, 0.4, 0.6, 0.8}

        Parameters:
        h : float
            Due-date tightness.  h=0 → all jobs tardy; h=1 → all jobs early.
        """
        if not 0.0 <= h <= 1.0:
            raise ValueError(f"h must be in [0, 1], got {h!r}.")
        return np.floor(self.sum_p * h)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, schedule: Sequence[int] | np.ndarray, h: float) -> int:
        """
        Compute the total weighted earliness + tardiness cost of a schedule.

        Parameters
        ----------
        schedule : sequence of int
            A permutation of job indices 0 … n-1 giving the processing order.
        h : float
            Due-date tightness factor used to derive d.

        Returns
        -------
        int
            Total penalty  Σ_i [ a_i * max(0, d - C_i) + b_i * max(0, C_i - d) ]
        """
        d = int(self.sum_p * h)
        s = schedule if isinstance(schedule, np.ndarray) else np.asarray(schedule, dtype=np.int64)
        return int(evaluate(self.p_array, self.a_array, self.b_array, s, d))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dictionary.

        Schema
        ------
        {
          "index":  int,
          "n":      int,
          "sum_p":  int,
          "due_dates": {"0.2": int, "0.4": int, "0.6": int, "0.8": int},
          "jobs": [
            {"p": int, "a": int, "b": int},
            ...
          ]
        }
        """
        return {
            "index": self.index,
            "n": self.n,
            "sum_p": self.sum_p,
            "due_dates": {str(h): self.due_date(h) for h in (0.2, 0.4, 0.6, 0.8)},
            "jobs": [{"p": j.p, "a": j.a, "b": j.b} for j in self.jobs],
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        due = {h: self.due_date(h) for h in (0.2, 0.4, 0.6, 0.8)}
        return f"SchInstance(index={self.index}, n={self.n}, sum_p={self.sum_p}, due_dates={due})"

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            f"Instance #{self.index}",
            f"  Jobs (n)    : {self.n}",
            f"  sum_p       : {self.sum_p}",
            "  Due dates   : " + "  ".join(f"h={h} → d={self.due_date(h)}" for h in (0.2, 0.4, 0.6, 0.8)),
            "",
            f"  {'Job':>4}  {'p':>6}  {'a':>6}  {'b':>6}",
            f"  {'---':>4}  {'---':>6}  {'---':>6}  {'---':>6}",
        ]
        for i, job in enumerate(self.jobs):
            lines.append(f"  {i:>4}  {job.p:>6}  {job.a:>6}  {job.b:>6}")
        return "\n".join(lines)


@dataclass
class SchDataset:
    """
    A collection of :class:`SchInstance` objects parsed from one ORLib file.

    Attributes
    ----------
    source : str
        Path or URL that was loaded.
    instances : list[SchInstance]
    """

    source: str
    instances: list[SchInstance]

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[SchInstance]:
        return iter(self.instances)

    def __getitem__(self, key: int) -> SchInstance:
        return self.instances[key]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for the whole dataset."""
        return {
            "source": self.source,
            "n_instances": len(self.instances),
            "instances": [inst.to_dict() for inst in self.instances],
        }

    def to_json(self, **kwargs) -> str:
        """Serialise to a JSON string.  Extra kwargs are passed to json.dumps."""
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def save_json(self, path: str | Path, **kwargs) -> None:
        """Write the dataset to a JSON file at *path*."""
        Path(path).write_text(self.to_json(**kwargs), encoding="utf-8")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SchDataset(source={self.source!r}, n_instances={len(self.instances)})"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _token_stream(text: str) -> Iterator[str]:
    """Yield all whitespace-separated tokens from *text*, skipping blank lines."""
    yield from text.split()


def _parse(text: str, source: str) -> SchDataset:
    """
    Core parser.  Raises ``ValueError`` on malformed input.
    """
    tokens = list(_token_stream(text))
    pos = 0

    def read_int(label: str = "value") -> int:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Unexpected end of input while reading {label} at token {pos}.")
        raw = tokens[pos]
        pos += 1
        try:
            return int(raw)
        except ValueError as err:
            raise ValueError(f"Expected integer for {label} at token {pos - 1}, got {raw!r}.") from err

    n_problems = read_int("number_of_problems")
    instances: list[SchInstance] = []

    for k in range(n_problems):
        n_jobs = read_int(f"n_jobs[problem={k}]")
        jobs: list[SchJob] = []
        for i in range(n_jobs):
            p = read_int(f"p[problem={k},job={i}]")
            a = read_int(f"a[problem={k},job={i}]")
            b = read_int(f"b[problem={k},job={i}]")
            jobs.append(SchJob(p=p, a=a, b=b))
        instances.append(SchInstance(index=k, n=n_jobs, jobs=jobs))

    if pos < len(tokens):
        warnings.warn(
            f"{len(tokens) - pos} trailing token(s) ignored in {source!r}.",
            stacklevel=4,
        )

    return SchDataset(source=source, instances=instances)


# ---------------------------------------------------------------------------
# Public load function
# ---------------------------------------------------------------------------


def load(source: str | Path) -> SchDataset:
    """
    Load a common due date scheduling file and return a :class:`SchDataset`.

    Parameters
    ----------
    source : str | Path
        * A local file path  (e.g. ``"sch10.txt"`` or ``Path("data/sch10.txt")``)
        * A URL              (e.g. ``"https://…/sch10.txt"``)
        * Raw text content   (detected when the string contains a newline)

    Returns
    -------
    SchDataset

    Examples
    --------
    >>> ds = load("sch10.txt")
    >>> ds = load("https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/sch10.txt")
    >>> print(ds[0].summary())
    """
    src_str = str(source)

    # ---- raw text passed directly ----------------------------------------
    if "\n" in src_str or "\r" in src_str:
        return _parse(src_str, source="<inline>")

    # ---- URL ---------------------------------------------------------------
    if src_str.startswith(("http://", "https://", "ftp://")):
        with urllib.request.urlopen(src_str, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return _parse(text, source=src_str)

    # ---- local file --------------------------------------------------------
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse(text, source=str(path))


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect an ORLib common due date scheduling file.")
    parser.add_argument("source", help="Path or URL to a sch*.txt file.")
    parser.add_argument("--json", action="store_true", help="Dump the full dataset as JSON.")
    parser.add_argument(
        "--instance",
        type=int,
        default=None,
        metavar="K",
        help="Print a detailed summary of instance K (0-based).",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.4,
        metavar="H",
        help="Due-date tightness for evaluation examples (default: 0.4).",
    )
    args = parser.parse_args()

    ds = load(args.source)
    print(repr(ds))
    print()

    if args.json:
        print(ds.to_json())
        return

    if args.instance is not None:
        inst = ds[args.instance]
        print(inst.summary())
        # show trivial identity-schedule cost as a sanity check
        sched = list(range(inst.n))
        cost = inst.evaluate(sched, h=args.h)
        print(f"\n  Cost of identity schedule (h={args.h}): {cost}")
        return

    # default: print one-line repr per instance
    for inst in ds:
        print(inst)


if __name__ == "__main__":
    _cli()
