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
from datetime import UTC, datetime
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


# ---------------------------------------------------------------------------
# Null logger — silences GEPA's internal verbose output
# ---------------------------------------------------------------------------

class _NullLogger:
    def log(self, msg: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Progress callback — prints clean ASCII blocks per iteration
# ---------------------------------------------------------------------------

class GEPAProgressCallback:
    """Prints an ASCII block per GEPA iteration summarising the outcome."""

    _W = 66

    def __init__(self, adapter: HyperparamAdapter, max_metric_calls: int) -> None:
        self._adapter = adapter
        self._max_calls = max_metric_calls
        self._calls_used: int = 0
        self._iter_hist_start: int = 0
        self._pending_valset: dict | None = None
        self._best_score: float = -1.0
        self._best_iter: int = 0
        self._best_config_params: dict = {}

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _bar(self, title: str = "") -> str:
        if title:
            prefix = f"=== {title}  "
            return prefix + "=" * max(0, self._W - len(prefix))
        return "=" * self._W

    @staticmethod
    def _fmt_config(params: dict) -> str:
        return "  ".join(f"{k}={v}" for k, v in params.items())

    @staticmethod
    def _fmt_quality(quality_scores: list[float]) -> str:
        pcts = "  ".join(f"{q * 100:.1f}%" for q in quality_scores)
        mean = sum(quality_scores) / len(quality_scores) * 100 if quality_scores else 0.0
        return f"[{pcts}]  mean={mean:.1f}%"

    @staticmethod
    def _fmt_valset(scores_by_val_id: dict, avg: float) -> str:
        per = "  ".join(f"{v:.3f}" for v in scores_by_val_id.values())
        return f"composite avg {avg:.3f}  [{per}]"

    def _budget(self) -> str:
        return f"{self._calls_used} / {self._max_calls}"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_optimization_start(self, event: dict) -> None:
        print(self._bar("GEPA OPTIMIZATION START"))
        print(f"  Budget  : {self._max_calls} metric calls")
        print(self._bar())
        print()

    def on_budget_updated(self, event: dict) -> None:
        self._calls_used = event["metric_calls_used"]

    def on_iteration_start(self, event: dict) -> None:
        self._iter_hist_start = len(self._adapter.history_)
        self._pending_valset = None

    def on_valset_evaluated(self, event: dict) -> None:
        iteration = event["iteration"]
        scores = event["scores_by_val_id"]
        avg = event["average_score"]
        is_best = event["is_best_program"]

        if iteration == 0:
            history = self._adapter.history_
            h = history[0] if history else {}
            if history:
                history[0]["status"] = "SEED"
            config_params = h.get("config_params", {})
            quality_scores = h.get("quality_scores", [])
            print(self._bar("Iter 0  SEED"))
            print(f"  Config  : {self._fmt_config(config_params)}")
            if quality_scores:
                print(f"  Quality : improv%  {self._fmt_quality(quality_scores)}")
            print(f"  Valset  : {self._fmt_valset(scores, avg)}")
            print(f"  Calls   : {self._budget()}")
            print(self._bar())
            print()
            self._best_score = avg
            self._best_iter = 0
            self._best_config_params = config_params
            return

        self._pending_valset = {"scores_by_val_id": scores, "average_score": avg, "is_best_program": is_best}
        if is_best:
            self._best_score = avg
            self._best_iter = iteration
            history = self._adapter.history_
            if history:
                self._best_config_params = history[-1]["config_params"]

    def on_evaluation_skipped(self, event: dict) -> None:
        iteration = event["iteration"]
        reason = event["reason"]
        print(self._bar(f"Iter {iteration}  SKIPPED"))
        print(f"  Reason  : {reason}")
        print(f"  Calls   : {self._budget()}")
        print(self._bar())
        print()

    def on_candidate_accepted(self, event: dict) -> None:
        iteration = event["iteration"]
        history = self._adapter.history_
        s = self._iter_hist_start

        h_curr = history[s] if s < len(history) else {}
        h_sub = history[s + 1] if s + 1 < len(history) else {}
        h_val = history[s + 2] if s + 2 < len(history) else (history[-1] if history else {})

        pv_is_best = (self._pending_valset or {}).get("is_best_program", False)
        status = "BEST" if pv_is_best else "accepted"
        if h_curr:
            h_curr["status"] = "re-eval"
        if h_sub:
            h_sub["status"] = status
        if h_val and h_val is not h_sub:
            h_val["status"] = status

        config_params = h_val.get("config_params", {})
        quality_scores = h_val.get("quality_scores", [])
        elapsed = h_val.get("mean_elapsed_s", 0.0)
        score_before = h_curr.get("mean_score", 0.0)
        score_after = h_sub.get("mean_score", 0.0)
        delta = score_after - score_before
        sign = "+" if delta >= 0 else ""

        pv = self._pending_valset or {}
        is_best = pv.get("is_best_program", False)
        val_ids = pv.get("scores_by_val_id", {})
        val_avg = pv.get("average_score", 0.0)

        best_tag = "  [*** NEW BEST ***]" if is_best else ""
        print(self._bar(f"Iter {iteration}  ACCEPTED{best_tag}"))
        print(f"  Config  : {self._fmt_config(config_params)}")
        print(f"▶ SCORE  : {score_after:.3f}  (Δ{sign}{delta:.3f} vs curr-prog same batch)  [GEPA optimises this]")
        if quality_scores:
            print(f"  Quality : improv%  {self._fmt_quality(quality_scores)}")
        print(f"  Valset  : {self._fmt_valset(val_ids, val_avg)}")
        print(f"  Time    : {elapsed:.2f} s/run   Calls: {self._budget()}")
        print(self._bar())
        print()

    def on_candidate_rejected(self, event: dict) -> None:
        iteration = event["iteration"]
        history = self._adapter.history_
        s = self._iter_hist_start

        h_curr = history[s] if s < len(history) else {}
        h_sub = history[s + 1] if s + 1 < len(history) else (history[-1] if history else {})

        if h_curr:
            h_curr["status"] = "re-eval"
        if h_sub:
            h_sub["status"] = "rejected"

        config_params = h_sub.get("config_params", {})
        quality_scores = h_sub.get("quality_scores", [])
        elapsed = h_sub.get("mean_elapsed_s", 0.0)
        score_before = h_curr.get("mean_score", 0.0)
        score_after = h_sub.get("mean_score", 0.0)
        delta = score_after - score_before
        sign = "+" if delta >= 0 else ""

        print(self._bar(f"Iter {iteration}  REJECTED"))
        print(f"  Config  : {self._fmt_config(config_params)}")
        print(f"▶ SCORE  : {score_after:.3f}  (Δ{sign}{delta:.3f} vs curr-prog same batch)  [worse — GEPA rejected]")
        if quality_scores:
            print(f"  Quality : improv%  {self._fmt_quality(quality_scores)}")
        print(f"  Time    : {elapsed:.2f} s/run   Calls: {self._budget()}")
        print(self._bar())
        print()

    def on_optimization_end(self, event: dict) -> None:
        total_iter = event["total_iterations"]
        total_calls = event["total_metric_calls"]
        print(self._bar("DONE"))
        print(f"  Iterations  : {total_iter}   Metric calls: {total_calls} / {self._max_calls}")
        print(f"  Best score  : {self._best_score:.3f}  (iter {self._best_iter})")
        if self._best_config_params:
            print(f"  Best config : {self._fmt_config(self._best_config_params)}")
        print(self._bar())
        print()


def _load_prompt(name: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt file not found: {path}")


_DEFAULT_REFLECTION_PROMPT = _load_prompt("default_reflection.txt")
_SA_REFLECTION_PROMPT = _load_prompt("sa_reflection.txt")
_GA_REFLECTION_PROMPT = _load_prompt("ga_reflection.txt")


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

    N_RUNS: int = 3

    def _run(self, instance: SchInstance, config: AgentConfig) -> EpisodeResult:
        results = []
        for _ in range(self.N_RUNS):
            env = SchEnv(instance, h=self.h, max_steps=self.max_steps)
            results.append(self.base_agent_cls(config).solve(env))
        best = min(results, key=lambda r: r.best_cost)
        return EpisodeResult(
            instance_index=best.instance_index,
            h=best.h,
            initial_cost=round(sum(r.initial_cost for r in results) / self.N_RUNS),
            final_cost=round(sum(r.final_cost for r in results) / self.N_RUNS),
            best_cost=round(sum(r.best_cost for r in results) / self.N_RUNS),
            total_reward=sum(r.total_reward for r in results) / self.N_RUNS,
            n_steps=round(sum(r.n_steps for r in results) / self.N_RUNS),
            n_improvements=round(sum(r.n_improvements for r in results) / self.N_RUNS),
            improvement_pct=sum(r.improvement_pct for r in results) / self.N_RUNS,
            best_schedule=best.best_schedule,
            cost_history=best.cost_history,
        )

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
        self.history_: list[dict] = []  # populated after train(); one entry per GEPA evaluate() call

        if reflection_prompt is not None:
            self.reflection_prompt = reflection_prompt
        else:
            from .classical_agents import GeneticAlgorithmAgent, SimulatedAnnealingAgent
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

        progress_cb = GEPAProgressCallback(adapter, max_metric_calls=self.max_metric_calls)
        interactions: list[dict[str, Any]] = []

        def _log_interaction(
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: datetime,
            end_time: datetime,
        ) -> None:
            try:
                interactions.append({
                    "timestamp": start_time.astimezone(UTC).isoformat(),
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
                logger=_NullLogger(),
                callbacks=[progress_cb],
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

        self.history_ = adapter.history_

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
