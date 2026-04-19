"""
gepa_agent.py — GEPA-driven hyperparameter optimisation for scheduling agents.

Uses the GEPA framework (https://github.com/gepa-ai/gepa) with a local Qwen3
model served via Ollama to evolve hyperparameter configs for SA and GA agents.

Classes
-------
HyperparamAdapter(SchedulingGEPAAdapter)
    Concrete GEPA adapter for hyperparameter search.  Instantiates the target
    agent class with a candidate config, runs it on one instance, and produces
    plain-English feedback describing the convergence behaviour.

GEPAAgent(Agent)
    Wraps GEPA hyperparameter search as a standard Agent.
    - train()  runs gepa.optimize() to find the best config.
    - solve()  delegates to the base agent instantiated with best_config_.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gepa
import litellm

from .agent import Agent
from .configs import AgentConfig
from .gepa_base import SchedulingGEPAAdapter
from .orlib_sch import SchInstance
from .sch_env import EpisodeResult, SchEnv

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_REFLECTION_PROMPT = (
    "You are optimising hyperparameters for a scheduling agent that solves the "
    "single-machine common due-date problem (minimise weighted earliness + tardiness). "
    "Analyse the execution results and propose a config that improves solution quality.\n\n"
    "Current hyperparameter config:\n<curr_param>\n\n"
    "Execution results with this config:\n<side_info>\n\n"
    "Return only a JSON object with the new hyperparameter values. No prose, no markdown fences."
)

_SA_REFLECTION_PROMPT = (
    "You are optimising hyperparameters for a Simulated Annealing agent that solves the "
    "single-machine common due-date scheduling problem (minimise weighted earliness + tardiness). "
    "The temperature schedule is T(step) = T0 * exp(-step * b) + c * sin(step / d)^2. "
    "Higher T0 and smaller b allow more exploration early; c and d control periodic reheating.\n\n"
    "Current hyperparameter config:\n<curr_param>\n\n"
    "Execution results with this config:\n<side_info>\n\n"
    "Return only a JSON object with the new hyperparameter values. No prose, no markdown fences."
)

_GA_REFLECTION_PROMPT = (
    "You are optimising hyperparameters for a Genetic Algorithm agent that solves the "
    "single-machine common due-date scheduling problem (minimise weighted earliness + tardiness). "
    "The GA uses Order Crossover (OX) and elitism. Larger populations explore more but cost more; "
    "higher mutation_rate increases diversity; reoptimize_every controls how often the target is refreshed.\n\n"
    "Current hyperparameter config:\n<curr_param>\n\n"
    "Execution results with this config:\n<side_info>\n\n"
    "Return only a JSON object with the new hyperparameter values. No prose, no markdown fences."
)


# ---------------------------------------------------------------------------
# Convergence analysis helper
# ---------------------------------------------------------------------------


def _describe_convergence(cost_history: list[int]) -> str:
    """Summarise the convergence pattern of a cost history trace."""
    if len(cost_history) < 3:
        return "insufficient steps to determine convergence pattern"

    initial = cost_history[0]
    best = min(cost_history)
    total_improvement = initial - best

    if total_improvement == 0:
        return "no improvement found — agent may be stuck or temperature/mutation too low"

    n = len(cost_history)
    step_50 = next((i for i, c in enumerate(cost_history) if c <= initial - total_improvement * 0.5), n)
    step_80 = next((i for i, c in enumerate(cost_history) if c <= initial - total_improvement * 0.8), n)

    pct_50 = 100.0 * step_50 / n
    pct_80 = 100.0 * step_80 / n

    if pct_50 < 20:
        pattern = "converged very early then plateaued — consider more exploitation"
    elif pct_50 < 40:
        pattern = "converged in the first half then plateaued"
    elif pct_80 > 80:
        pattern = "steadily improving with no clear plateau — more steps may help"
    else:
        pattern = "gradual improvement with plateauing toward the end"

    final_plateau = sum(1 for c in reversed(cost_history) if c == best)
    return (
        f"{pattern} "
        f"(50% of gain by step {step_50}/{n}, "
        f"80% by step {step_80}/{n}, "
        f"final plateau: {final_plateau} steps)"
    )


# ---------------------------------------------------------------------------
# Concrete GEPA adapter
# ---------------------------------------------------------------------------


class HyperparamAdapter(SchedulingGEPAAdapter):
    """
    GEPA adapter that evaluates hyperparameter configs for a scheduling agent.

    Parameters
    ----------
    base_agent_cls : type
        The agent class to instantiate (SimulatedAnnealingAgent or GeneticAlgorithmAgent).
    seed_config : AgentConfig
        Starting config; also used as fallback when LLM output cannot be parsed.
    h : float
        Due-date tightness for all evaluation episodes.
    max_steps : int | None
        Episode length forwarded to SchEnv (None → 10 * n default).
    seed : int | None
        RNG seed forwarded to agent.solve().
    """

    def __init__(
        self,
        base_agent_cls: type,
        seed_config: AgentConfig,
        h: float = 0.4,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed_config=seed_config)
        self.base_agent_cls = base_agent_cls
        self.h = h
        self.max_steps = max_steps
        self.seed = seed

    def _run(self, instance: SchInstance, config: AgentConfig) -> EpisodeResult:
        env = SchEnv(instance, h=self.h, max_steps=self.max_steps)
        agent = self.base_agent_cls(config)
        return agent.solve(env)

    def _feedback(self, result: EpisodeResult) -> str:
        improvement = result.initial_cost - result.best_cost
        convergence = _describe_convergence(result.cost_history)
        parts = [
            f"Improvement: {result.improvement_pct:.1f}% "
            f"({improvement} cost units, {result.initial_cost} → {result.best_cost}).",
            f"Convergence: {convergence}.",
            f"Steps used: {result.n_steps}, beneficial moves: {result.n_improvements}.",
        ]
        if result.improvement_pct < 5.0:
            parts.append(
                "The config performed poorly. Consider increasing exploration "
                "(higher initial temperature / larger population) or adjusting decay."
            )
        elif result.improvement_pct > 40.0:
            parts.append(
                "Strong result. Try fine-tuning the cooling/mutation rate to see "
                "if further gains are possible without sacrificing speed."
            )
        if result.n_improvements == 0:
            parts.append(
                "Zero beneficial moves recorded — the search may be too conservative "
                "or the step budget too small."
            )
        return " ".join(parts)


# ---------------------------------------------------------------------------
# GEPAAgent
# ---------------------------------------------------------------------------


class GEPAAgent(Agent):
    """
    Scheduling agent that uses GEPA to search for the best hyperparameter config
    for a given base agent class (SA or GA), then delegates solve() to it.

    Parameters
    ----------
    base_agent_cls : type
        SimulatedAnnealingAgent or GeneticAlgorithmAgent.
    seed_config : AgentConfig
        Initial config and parse fallback (SAConfig or GAConfig).
    reflection_prompt : str | None
        Task-description prompt template for the Qwen reflection LLM.
        Must contain ``<curr_param>`` and ``<side_info>`` placeholders.
        Defaults to a sensible prompt for the given base_agent_cls.
    reflection_lm : str
        LiteLLM model string for the reflection LLM (default: ollama/qwen3:4b-instruct-2507-q4_K_M).
    max_metric_calls : int
        Total evaluation budget passed to gepa.optimize().
    h : float
        Due-date tightness used during training evaluations.
    max_steps : int | None
        Episode length per evaluation (None → SchEnv default of 10 * n).
    seed : int | None
        RNG seed for reproducibility.
    interactions_log : str | Path | None
        If provided, all raw LLM prompt/response pairs are recorded and
        written to this path as a JSON file after training completes.
    """

    def __init__(
        self,
        base_agent_cls: type,
        seed_config: AgentConfig,
        reflection_prompt: str | None = None,
        reflection_lm: str = "ollama/qwen3:4b-instruct-2507-q4_K_M",
        max_metric_calls: int = 50,
        h: float = 0.4,
        max_steps: int | None = None,
        seed: int | None = None,
        interactions_log: str | Path | None = None,
    ) -> None:
        self.base_agent_cls = base_agent_cls
        self.seed_config = seed_config
        self.reflection_lm = reflection_lm
        self.max_metric_calls = max_metric_calls
        self.h = h
        self.max_steps = max_steps
        self.seed = seed
        self.interactions_log = Path(interactions_log) if interactions_log else None
        self.best_config_: AgentConfig = seed_config

        if reflection_prompt is not None:
            self.reflection_prompt = reflection_prompt
        else:
            from .classical_agents import SimulatedAnnealingAgent, GeneticAlgorithmAgent
            if base_agent_cls is SimulatedAnnealingAgent:
                self.reflection_prompt = _SA_REFLECTION_PROMPT
            elif base_agent_cls is GeneticAlgorithmAgent:
                self.reflection_prompt = _GA_REFLECTION_PROMPT
            else:
                self.reflection_prompt = _DEFAULT_REFLECTION_PROMPT

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        instances: Sequence[SchInstance],
        *,
        h: float = 0.4,
        **kwargs,
    ) -> None:
        """
        Run GEPA hyperparameter search over *instances*.

        Populates ``self.best_config_`` with the best config found.
        The ``h`` argument overrides the constructor value for this training run.
        """
        effective_h = h or self.h
        adapter = HyperparamAdapter(
            base_agent_cls=self.base_agent_cls,
            seed_config=self.seed_config,
            h=effective_h,
            max_steps=self.max_steps,
            seed=self.seed,
        )

        interactions: list[dict[str, Any]] = []

        def _log_interaction(
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: datetime,
            end_time: datetime,
        ) -> None:
            try:
                interactions.append({
                    "timestamp": start_time.astimezone(timezone.utc).isoformat(),
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                    "response": response_obj.choices[0].message.content,
                    "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                })
            except Exception as exc:  # never let logging break training
                logger.warning("Failed to log LLM interaction: %s", exc)

        if self.interactions_log is not None:
            litellm.success_callback.append(_log_interaction)

        try:
            result = gepa.optimize(
                seed_candidate={"config": self.seed_config.to_prompt()},
                trainset=list(instances),
                adapter=adapter,
                reflection_lm=self.reflection_lm,
                reflection_prompt_template=self.reflection_prompt,
                max_metric_calls=self.max_metric_calls,
                seed=self.seed or 0,
            )
        finally:
            if self.interactions_log is not None:
                litellm.success_callback.remove(_log_interaction)
                self.interactions_log.parent.mkdir(parents=True, exist_ok=True)
                self.interactions_log.write_text(
                    json.dumps(interactions, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info("Logged %d LLM interaction(s) to %s", len(interactions), self.interactions_log)

        try:
            self.best_config_ = self.seed_config.from_prompt(result.best_candidate["config"])
            logger.info("GEPA best config: %s", self.best_config_)
        except (ValueError, KeyError) as exc:
            logger.warning("Could not parse GEPA best candidate (%s); keeping seed config.", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def solve(self, env: SchEnv, *, seed: int | None = None) -> EpisodeResult:
        """Solve *env* using the base agent instantiated with ``best_config_``."""
        agent = self.base_agent_cls(self.best_config_)
        result = agent.solve(env, seed=seed)
        self.actions = agent.actions
        self.initial_schedule = agent.initial_schedule
        self.cost_history = agent.cost_history
        return result

    @property
    def name(self) -> str:
        return f"GEPA-{self.base_agent_cls.__name__}"
