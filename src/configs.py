"""
configs.py — Agent hyperparameter config dataclasses with LLM prompt translation.

Each config class can serialise itself to a natural-language prompt string for
an LLM to reason about, and parse an LLM response back into a validated config.
This is the bridge between GEPA's text-parameter world and the agent's numeric
hyperparameter world.

Classes
-------
AgentConfig  — abstract base; defines to_prompt / from_prompt / bounds / clamp
SAConfig     — hyperparameters for SimulatedAnnealingAgent
GAConfig     — hyperparameters for GeneticAlgorithmAgent
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from *text*.

    Handles responses wrapped in markdown code fences (```json ... ```) as
    well as raw JSON objects.  Raises ``ValueError`` if nothing is found.
    """
    # Markdown code fence first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Bare JSON object (greedy inner braces avoided by using a stack-free heuristic)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in LLM response: {text[:300]!r}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgentConfig(ABC):
    """
    Abstract base for agent hyperparameter configs.

    Subclasses are plain dataclasses that additionally know how to render
    themselves as an annotated prompt string and parse themselves back from
    an LLM response.
    """

    @abstractmethod
    def to_prompt(self) -> str:
        """
        Render the config as a human-readable string suitable for an LLM.

        The string must contain a valid JSON object so that ``from_prompt``
        can round-trip it.  Extra prose explaining parameter semantics is
        encouraged — the LLM uses it for reflection.
        """

    @classmethod
    @abstractmethod
    def from_prompt(cls, text: str) -> Self:
        """
        Parse an LLM response and return a new config instance.

        Extracts the JSON object from *text*, maps recognised keys to fields,
        clamps all values within ``bounds()``, and fills missing keys with
        the dataclass defaults.

        Raises ``ValueError`` if no JSON object is found in *text*.
        """

    @abstractmethod
    def bounds(self) -> dict[str, tuple[float, float]]:
        """
        Return ``{field_name: (lo, hi)}`` for every optimisable numeric field.

        Used by ``clamp()`` to keep LLM-proposed values sane.
        """

    def clamp(self, values: dict) -> dict:
        """
        Clamp each key in *values* to its allowed range from ``bounds()``.

        Preserves the original Python type (int vs float) of each value.
        Keys not present in ``bounds()`` are passed through unchanged.
        """
        b = self.bounds()
        result = {}
        for k, v in values.items():
            if k in b:
                lo, hi = b[k]
                result[k] = type(v)(max(lo, min(hi, v)))
            else:
                result[k] = v
        return result


# ---------------------------------------------------------------------------
# Simulated Annealing config
# ---------------------------------------------------------------------------


@dataclass
class SAConfig(AgentConfig):
    """
    Hyperparameters for SimulatedAnnealingAgent.

    Temperature schedule:  T(step) = T0 * exp(-step * b) + c * sin(step / d)^2
    """

    T0: float = 5.0
    b: float = 0.05
    c: float = 0.0
    d: float = 50.0
    T_min: float = 1e-5
    temp_plot_file: str = "temperature_schedule.png"

    # Optimisable fields (excludes temp_plot_file which is not a hyperparameter)
    _OPTIM_FIELDS: ClassVar[tuple[str, ...]] = ("T0", "b", "c", "d", "T_min")

    def to_prompt(self) -> str:
        config_dict = {
            "T0": self.T0,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "T_min": self.T_min,
        }
        descriptions = (
            "Parameter descriptions:\n"
            "  T0    — initial temperature; higher values allow more uphill moves early on\n"
            "  b     — exponential decay rate; higher means faster cooling\n"
            "  c     — sinusoidal oscillation amplitude; adds periodic reheating bursts\n"
            "  d     — sinusoidal period in steps; controls reheating frequency\n"
            "  T_min — minimum temperature floor; prevents the search from freezing\n"
            "\n"
            "Temperature formula: T(step) = T0 * exp(-step * b) + c * sin(step / d)^2"
        )
        return f"{descriptions}\n\nCurrent config:\n{json.dumps(config_dict, indent=2)}"

    @classmethod
    def from_prompt(cls, text: str) -> Self:
        raw = _extract_json(text)
        defaults = cls()
        parsed = {
            k: float(raw[k])
            for k in cls._OPTIM_FIELDS
            if k in raw
        }
        if not parsed:
            raise ValueError(f"No recognised SAConfig fields in LLM response: {text[:200]!r}")
        merged = {k: parsed.get(k, getattr(defaults, k)) for k in cls._OPTIM_FIELDS}
        clamped = defaults.clamp(merged)
        return cls(**clamped, temp_plot_file=defaults.temp_plot_file)

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "T0": (1.0, 10_000.0),
            "b": (1e-6, 0.1),
            "c": (0.0, 100.0),
            "d": (1.0, 500.0),
            "T_min": (1e-8, 1.0),
        }


# ---------------------------------------------------------------------------
# Genetic Algorithm config
# ---------------------------------------------------------------------------


@dataclass
class GAConfig(AgentConfig):
    """Hyperparameters for GeneticAlgorithmAgent."""

    population_size: int = 10
    generations: int = 3
    mutation_rate: float = 0.01
    elite_size: int = 1
    reoptimize_every: int = 10

    _INT_FIELDS: ClassVar[tuple[str, ...]] = ("population_size", "generations", "elite_size", "reoptimize_every")
    _FLOAT_FIELDS: ClassVar[tuple[str, ...]] = ("mutation_rate",)

    def to_prompt(self) -> str:
        config_dict = {
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "reoptimize_every": self.reoptimize_every,
        }
        descriptions = (
            "Parameter descriptions:\n"
            "  population_size  — number of candidate schedules evolved per GA run\n"
            "  generations      — number of GA iterations per re-optimisation call\n"
            "  mutation_rate    — probability of applying a random swap to a child schedule\n"
            "  elite_size       — top-N individuals carried over unchanged each generation\n"
            "  reoptimize_every — steps between full GA re-runs during the episode\n"
            "\n"
            "The GA uses Order Crossover (OX) and elitism."
        )
        return f"{descriptions}\n\nCurrent config:\n{json.dumps(config_dict, indent=2)}"

    @classmethod
    def from_prompt(cls, text: str) -> Self:
        raw = _extract_json(text)
        defaults = cls()
        all_fields = cls._INT_FIELDS + cls._FLOAT_FIELDS
        parsed = {}
        for k in all_fields:
            if k in raw:
                parsed[k] = int(round(raw[k])) if k in cls._INT_FIELDS else float(raw[k])
        if not parsed:
            raise ValueError(f"No recognised GAConfig fields in LLM response: {text[:200]!r}")
        merged = {k: parsed.get(k, getattr(defaults, k)) for k in all_fields}
        clamped = defaults.clamp(merged)
        return cls(**clamped)

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "population_size": (10, 500),
            "generations": (1, 100),
            "mutation_rate": (0.0, 1.0),
            "elite_size": (1, 20),
            "reoptimize_every": (1, 20),
        }
