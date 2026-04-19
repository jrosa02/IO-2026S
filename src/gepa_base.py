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

    def __init__(self, seed_config: AgentConfig) -> None:
        self.seed_config = seed_config

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

        for instance in batch:
            try:
                result = self._run(instance, config)
                score = result.improvement_pct / 100.0
            except Exception as exc:
                warnings.warn(f"_run failed on instance {instance.index}: {exc}", stacklevel=2)
                result = _zero_result(instance)
                score = 0.0
            outputs.append(result)
            scores.append(score)

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
        """
        if eval_batch.trajectories is None:
            return {"config": []}

        records = []
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
