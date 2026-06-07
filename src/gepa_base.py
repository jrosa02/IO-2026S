"""
gepa_base.py — Generic GEPA adapter base for scheduling optimization.

Implements the GEPAAdapter protocol in terms of two abstract methods that
subclasses fill in, isolating the GEPA boilerplate from task-specific logic.

Classes
-------
SchedulingGEPAAdapter
    Abstract base satisfying the GEPAAdapter protocol.  Subclasses override:
      _run(instance, config)  -> EpisodeResult   (how to evaluate one config)
      _feedback(result)       -> str              (how to describe the result to the LLM)
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .configs import AgentConfig
from .orlib_sch import SchInstance
from .sch_env import EpisodeResult

logger = logging.getLogger(__name__)

# Type aliases matching the GEPAAdapter protocol slots for this task:
#   DataInst      = SchInstance
#   Trajectory    = EpisodeResult
#   RolloutOutput = EpisodeResult


class SchedulingGEPAAdapter:
    """
    Abstract GEPA adapter for scheduling hyperparameter optimisation.

    Satisfies the GEPAAdapter protocol via duck typing.  Subclasses must
    implement ``_run`` and ``_feedback``; everything else is handled here.

    Parameters
    ----------
    seed_config : AgentConfig
        Fallback config used when the LLM response cannot be parsed.

    """

    # Required by the GEPAAdapter protocol — None means use the default LLM proposer.
    propose_new_texts = None

    TIME_PENALTY_WEIGHT: float = 0.2

    def __init__(self, seed_config: AgentConfig) -> None:
        self.seed_config = seed_config
        self.history_: list[dict] = []  # populated by evaluate(); one entry per GEPA call
        self._call_idx: int = 0
        self._max_elapsed_s: float | None = None  # running max wall time across all _run() calls

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _run(self, instance: SchInstance, config: AgentConfig) -> EpisodeResult:
        """
        Evaluate *config* on *instance* and return the episode result.

        Must never raise — return a zero-improvement EpisodeResult on failure.
        """

    @abstractmethod
    def _feedback(self, result: EpisodeResult) -> str:
        """
        Summarise *result* as plain English feedback for the LLM.

        Should explain *why* the config performed as it did so the LLM can
        propose a targeted improvement rather than a random perturbation.
        """

    # ------------------------------------------------------------------
    # GEPAAdapter protocol — evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[SchInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EpisodeResult, EpisodeResult]:
        """
        Run *candidate* config on every instance in *batch*.

        Scores are normalised improvement percentages in [0, 1] (higher = better).
        Individual failures are caught and scored 0.0 so GEPA never sees an exception.
        """
        config = self._parse_candidate(candidate)
        outputs: list[EpisodeResult] = []
        scores: list[float] = []
        quality_scores: list[float] = []
        elapsed_times: list[float] = []

        for instance in batch:
            try:
                t0 = time.perf_counter()
                result = self._run(instance, config)
                elapsed = time.perf_counter() - t0
                quality = result.improvement_pct / 100.0
            except Exception as exc:
                warnings.warn(f"_run failed on instance {instance.index}: {exc}", stacklevel=2)
                result = _zero_result(instance)
                quality = 0.0
                elapsed = 0.0

            if elapsed > (self._max_elapsed_s or 0.0):
                self._max_elapsed_s = elapsed

            if self._max_elapsed_s:
                t_norm = elapsed / self._max_elapsed_s
                score = quality - self.TIME_PENALTY_WEIGHT * t_norm
            else:
                score = quality

            outputs.append(result)
            scores.append(score)
            quality_scores.append(quality)
            elapsed_times.append(elapsed)

        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        mean_quality = float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0
        best_so_far = max(
            (e["mean_score"] for e in self.history_), default=0.0
        )
        self.history_.append({
            "call_idx": self._call_idx,
            "status": "",  # filled in by GEPAProgressCallback once accept/reject is known
            "config_params": {k: v for k, v in vars(config).items() if not k.startswith("_")},
            "scores": scores,
            "mean_score": mean_score,
            "quality_scores": quality_scores,
            "mean_quality": mean_quality,
            "best_so_far": max(mean_score, best_so_far),
            "elapsed_s": elapsed_times,
            "mean_elapsed_s": sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0,
        })
        self._call_idx += 1

        trajectories = outputs if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # ------------------------------------------------------------------
    # GEPAAdapter protocol — make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EpisodeResult, EpisodeResult],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build the per-component reflective dataset consumed by the LLM proposer.

        Each record gives the LLM the current config text, the numeric outcome,
        and a plain-English diagnosis so it can propose a targeted mutation.
        The first record is an optimization history summary when prior calls exist.
        """
        if eval_batch.trajectories is None:
            return {"config": []}

        records: list[dict] = []

        for result in eval_batch.trajectories:
            records.append(
                {
                    "Inputs": {"config": candidate.get("config", "")},
                    "Generated Outputs": {
                        "improvement_pct": f"{result.improvement_pct:.2f}%",
                        "best_cost": str(result.best_cost),
                        "initial_cost": str(result.initial_cost),
                        "n_steps": str(result.n_steps),
                    },
                    "Feedback": self._feedback(result),
                }
            )

        return {"config": records}

    def _format_history_summary(self) -> str:
        """Format self.history_ as a compact table for LLM injection."""
        lines = ["call | result   | score  | time_s | params"]
        lines.append("-----|----------|--------|--------|-------")
        for entry in self.history_:
            params_str = ", ".join(
                f"{k}={v}" for k, v in entry["config_params"].items()
            )
            lines.append(
                f"{entry['call_idx']:4d} | "
                f"{entry.get('status', ''):8s} | "
                f"{entry['mean_score']:6.3f} | "
                f"{entry.get('mean_elapsed_s', 0.0):6.2f} | "
                f"{params_str}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_candidate(self, candidate: dict[str, str]) -> AgentConfig:
        """
        Parse the ``"config"`` component of *candidate* into an AgentConfig.

        Falls back to ``self.seed_config`` on any parse failure so the
        optimisation loop never stalls on a bad LLM response.
        """
        text = candidate.get("config", "")
        if not text:
            logger.warning("Empty config in candidate, using seed config.")
            return self.seed_config
        try:
            return self.seed_config.from_prompt(text)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Config parse failed (%s), using seed config.", exc)
            return self.seed_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zero_result(instance: SchInstance) -> EpisodeResult:
    """Return a zero-improvement EpisodeResult for failed evaluations."""
    return EpisodeResult(
        instance_index=instance.index,
        h=0.0,
        initial_cost=1,
        final_cost=1,
        best_cost=1,
        total_reward=0.0,
        n_steps=0,
        n_improvements=0,
        improvement_pct=0.0,
        best_schedule=list(range(instance.n)),
        cost_history=[],
    )
