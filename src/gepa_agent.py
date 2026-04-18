"""
gepa_agent.py — GEPA-driven hyperparameter search for SimulatedAnnealingAgent.

Uses the GEPA framework (https://github.com/gepa-ai/gepa) to evolve SAConfig
hyperparameters via LLM-guided reflective mutation and Pareto-efficient selection.

GEPA treats each SAConfig as a "text candidate" (a dict[str, str] of parameter
values). At each iteration it:
  1. Runs SA with the candidate config on a minibatch of scheduling instances.
  2. Passes execution traces (cost curves, diagnostics) to a reflection LLM.
  3. LLM proposes a mutated config targeting observed weaknesses.
  4. Pareto-efficient selection preserves candidates that are best on ANY instance.

Multi-objective tracking:
  - Primary score  : improvement_pct / 100  (higher = better)
  - Second objective "convergence_speed": how early in the step budget the best
    cost was found (higher = faster convergence). Tracked via objective_scores.

Mock mode (no LLM): when reflection_lm=None the adapter supplies its own
propose_new_texts that randomly perturbs one parameter within schema bounds —
useful for testing without an API key.

Public API
----------
  GEPAAgentConfig   — configuration for the search
  GEPAAgent         — Agent subclass; call train() then solve()

Example
-------
    from src.orlib_sch import load
    from src.gepa_agent import GEPAAgent, GEPAAgentConfig

    ds = load("data/sch10.txt")
    cfg = GEPAAgentConfig(
        n_instances_train=6,
        n_instances_val=4,
        max_metric_calls=40,
        reflection_lm="anthropic/claude-sonnet-4-5",   # or None for mock
    )
    agent = GEPAAgent(cfg)
    agent.train(ds.instances, h=0.4, max_steps=200)
    print("Best config:", agent.best_config_)
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agent import Agent
from .classical_agents import SAConfig, SimulatedAnnealingAgent
from .orlib_sch import SchInstance
from .sch_env import EpisodeResult, SchEnv

# ---------------------------------------------------------------------------
# Parameter schema — bounds used by mock mutator and seed initialisation
# ---------------------------------------------------------------------------

_SA_PARAM_SCHEMA: dict[str, dict[str, float]] = {
    "T0":    {"lo": 10.0, "hi": 1000.0},
    "b":     {"lo": 1e-5, "hi": 0.01},
    "c":     {"lo": 0.0, "hi": 50.0},
    "d":     {"lo": 5.0, "hi": 200.0},
    "T_min": {"lo": 1e-8, "hi": 1e-3},
}

_SEED_CANDIDATE: dict[str, str] = {
    "T0":    str(SAConfig.T0),
    "b":     str(SAConfig.b),
    "c":     str(SAConfig.c),
    "d":     str(SAConfig.d),
    "T_min": str(SAConfig.T_min),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GEPAAgentConfig:
    """Configuration for the GEPA hyperparameter search.

    Attributes
    ----------
    n_instances_train : int
        Number of instances used for minibatch acceptance tests (trainset).
    n_instances_val : int
        Number of instances for full Pareto-tracking evaluation (valset).
    sa_max_steps : int
        Step budget given to each SA run during search.
    max_metric_calls : int
        Total SA evaluations GEPA may perform. Roughly: generations * population.
    reflection_lm : str | None
        LiteLLM model string for the reflection LLM (e.g. "anthropic/claude-sonnet-4-5",
        "openai/gpt-4o"). When None, falls back to the mock random-perturbation mutator.
    candidate_selection : str
        GEPA candidate selection strategy: "pareto", "current_best", "epsilon_greedy",
        "top_k_pareto".
    run_dir : str | None
        Directory for GEPA logs. None disables file logging.
    seed : int
        RNG seed forwarded to GEPA and to SA runs.
    """

    n_instances_train: int = 6
    n_instances_val: int = 4
    sa_max_steps: int = 200
    max_metric_calls: int = 40
    reflection_lm: str | None = None
    candidate_selection: str = "pareto"
    run_dir: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------


def _find_plateau_step(cost_history: list[int]) -> int:
    """Return the step index at which cost stopped improving."""
    best = cost_history[0]
    last_improvement = 0
    for i, c in enumerate(cost_history):
        if c < best:
            best = c
            last_improvement = i
    return last_improvement


_MIN_HISTORY_LEN = 2
_LOW_IMPROVEMENT_PCT = 5.0
_HIGH_IMPROVEMENT_PCT = 50.0
_LOG_SCALE_RATIO = 100.0


def _convergence_speed(cost_history: list[int]) -> float:
    """Fraction of budget remaining when 90% of total improvement was achieved.

    Returns a value in [0, 1]; higher means faster convergence (good).
    """
    if len(cost_history) < _MIN_HISTORY_LEN:
        return 0.0
    initial, best = cost_history[0], min(cost_history)
    total_improvement = initial - best
    if total_improvement <= 0:
        return 0.0
    target = initial - 0.9 * total_improvement
    for i, c in enumerate(cost_history):
        if c <= target:
            return 1.0 - i / len(cost_history)
    return 0.0


def _diagnose_run(result: EpisodeResult, cfg: SAConfig) -> str:
    """Produce a natural-language diagnosis of one SA run for LLM reflection."""
    history = result.cost_history
    plateau = _find_plateau_step(history)
    n = len(history)
    pct = result.improvement_pct

    lines: list[str] = [
        f"improvement_pct={pct:.1f}%, best_cost={result.best_cost}, "
        f"n_improvements={result.n_improvements}, steps={result.n_steps}",
        f"Last improvement at step {plateau}/{n} "
        f"({'early' if plateau < n * 0.3 else 'mid' if plateau < n * 0.7 else 'late'} convergence).",
    ]

    if pct < _LOW_IMPROVEMENT_PCT:
        lines.append(
            "Very low improvement. Possible causes: T0 too small so bad moves are never accepted, "
            "or cooling too aggressive (b too large). Try increasing T0 or decreasing b."
        )
    elif plateau < n * 0.25:
        lines.append(
            f"Converged very early (step {plateau}/{n}). "
            "Temperature cooled too fast. Consider decreasing b or increasing c/d to add re-heating."
        )
    elif pct > _HIGH_IMPROVEMENT_PCT:
        lines.append("Good improvement achieved. Temperature schedule appears well-calibrated.")
    else:
        lines.append(
            "Moderate improvement. If plateau is too early, decrease b. "
            "If cost oscillates without improving, increase b or decrease T0."
        )

    lines.append(
        f"Current config: T0={cfg.T0}, b={cfg.b}, c={cfg.c}, d={cfg.d}, T_min={cfg.T_min}."
    )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# GEPA Adapter
# ---------------------------------------------------------------------------


class SASchedulingAdapter:
    """GEPAAdapter that evaluates SAConfig candidates on scheduling instances.

    DataInst   = SchInstance
    Trajectory = dict with cost_history, improvement_pct, config, diagnosis
    RolloutOutput = EpisodeResult
    """

    def __init__(self, h: float, sa_max_steps: int, seed: int, mock_rng: random.Random) -> None:
        self.h = h
        self.sa_max_steps = sa_max_steps
        self.seed = seed
        self._mock_rng = mock_rng

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_to_saconfig(candidate: dict[str, str]) -> SAConfig:
        """Parse a GEPA candidate dict into an SAConfig, clamping to schema bounds."""
        values: dict[str, float] = {}
        for name, bounds in _SA_PARAM_SCHEMA.items():
            raw = candidate.get(name, _SEED_CANDIDATE[name])
            try:
                v = float(raw)
            except (ValueError, TypeError):
                v = float(_SEED_CANDIDATE[name])
            values[name] = max(bounds["lo"], min(bounds["hi"], v))
        return SAConfig(
            T0=values["T0"],
            b=values["b"],
            c=values["c"],
            d=values["d"],
            T_min=values["T_min"],
        )

    def _run_on_instance(
        self, sa_cfg: SAConfig, instance: SchInstance, capture_trace: bool
    ) -> tuple[float, float, dict[str, Any] | None]:
        """Run SA on one instance; return (improvement_score, speed_score, trace_dict|None)."""
        agent = SimulatedAnnealingAgent(sa_cfg)
        env = SchEnv(instance, h=self.h, max_steps=self.sa_max_steps, seed=self.seed)
        result: EpisodeResult = agent.solve(env, seed=self.seed)

        improvement_score = result.improvement_pct / 100.0
        speed_score = _convergence_speed(result.cost_history)

        trace = None
        if capture_trace:
            trace = {
                "instance_index": instance.index,
                "n_jobs": instance.n,
                "improvement_pct": result.improvement_pct,
                "best_cost": result.best_cost,
                "initial_cost": result.initial_cost,
                "n_improvements": result.n_improvements,
                "plateau_step": _find_plateau_step(result.cost_history),
                "total_steps": result.n_steps,
                "cost_history_head": result.cost_history[:15],
                "config": dict(zip(
                    _SA_PARAM_SCHEMA.keys(),
                    [str(sa_cfg.T0), str(sa_cfg.b), str(sa_cfg.c), str(sa_cfg.d), str(sa_cfg.T_min)],
                    strict=True,
                )),
                "diagnosis": _diagnose_run(result, sa_cfg),
                "episode_result": result,
            }
        return improvement_score, speed_score, trace

    # ------------------------------------------------------------------
    # GEPAAdapter protocol
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[SchInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ):
        from gepa import EvaluationBatch  # noqa: PLC0415

        sa_cfg = self._candidate_to_saconfig(candidate)
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        outputs: list[EpisodeResult] = []
        objective_scores: list[dict[str, float]] = []

        for instance in batch:
            improvement, speed, trace = self._run_on_instance(sa_cfg, instance, capture_traces)
            scores.append(improvement)
            objective_scores.append({"improvement": improvement, "convergence_speed": speed})
            if capture_traces and trace is not None:
                trajectories.append(trace)  # type: ignore[union-attr]
                outputs.append(trace["episode_result"])
            else:
                outputs.append(None)  # type: ignore[arg-type]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    @staticmethod
    def make_reflective_dataset(
        candidate: dict[str, str],
        eval_batch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records: list[dict[str, Any]] = []
        trajectories = eval_batch.trajectories or []

        for trace in trajectories:
            records.append({
                "Inputs": {
                    "T0": candidate.get("T0", "?"),
                    "b": candidate.get("b", "?"),
                    "c": candidate.get("c", "?"),
                    "d": candidate.get("d", "?"),
                    "T_min": candidate.get("T_min", "?"),
                    "n_jobs": str(trace["n_jobs"]),
                },
                "Generated Outputs": {
                    "improvement_pct": f"{trace['improvement_pct']:.2f}%",
                    "n_improvements": str(trace["n_improvements"]),
                    "plateau_step": f"{trace['plateau_step']}/{trace['total_steps']}",
                    "cost_history_head": str(trace["cost_history_head"]),
                },
                "Feedback": trace["diagnosis"],
            })

        # Return under the first component name (single-component candidate)
        component = components_to_update[0] if components_to_update else "config"
        return {component: records}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],  # noqa: ARG002
        components_to_update: list[str],  # noqa: ARG002
    ) -> dict[str, str]:
        """Mock mutator used when reflection_lm=None.

        Randomly perturbs one numeric SA parameter within its schema bounds
        using a log-scale perturbation for parameters spanning many orders of
        magnitude (b, T_min) and a linear perturbation for the rest.
        """
        new_candidate = dict(candidate)
        param = self._mock_rng.choice(list(_SA_PARAM_SCHEMA.keys()))
        bounds = _SA_PARAM_SCHEMA[param]
        current = float(candidate.get(param, _SEED_CANDIDATE[param]))

        lo, hi = bounds["lo"], bounds["hi"]
        log_scale = (hi / max(lo, 1e-12)) > _LOG_SCALE_RATIO

        if log_scale:
            log_lo, log_hi = math.log(lo), math.log(hi)
            log_cur = math.log(max(current, lo))
            log_new = log_cur + self._mock_rng.gauss(0, (log_hi - log_lo) * 0.2)
            new_val = math.exp(max(log_lo, min(log_hi, log_new)))
        else:
            span = hi - lo
            new_val = current + self._mock_rng.gauss(0, span * 0.15)
            new_val = max(lo, min(hi, new_val))

        new_candidate[param] = str(new_val)
        return new_candidate


# ---------------------------------------------------------------------------
# GEPAAgent
# ---------------------------------------------------------------------------


class GEPAAgent(Agent):
    """Agent that uses GEPA to evolve SAConfig hyperparameters.

    Call ``train(instances, h=...)`` to run the GEPA search and store the
    best found config in ``self.best_config_``.  Then call ``solve(env)``
    to use the best config on a new instance.

    Attributes
    ----------
    best_config_ : SAConfig | None
        Set after train() completes. None before training.
    gepa_result_ : GEPAResult | None
        Full GEPA result object from the last train() call.
    """

    def __init__(self, cfg: GEPAAgentConfig | None = None) -> None:
        self.cfg = cfg or GEPAAgentConfig()
        self.best_config_: SAConfig | None = None
        self.gepa_result_ = None

    @property
    def name(self) -> str:
        return "GEPAAgent"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        instances: Sequence[SchInstance],
        *,
        h: float = 0.4,
    ) -> None:
        """Run GEPA search to find the best SAConfig for the given instances.

        Splits ``instances`` into trainset (minibatch acceptance) and valset
        (Pareto tracking) according to ``self.cfg.n_instances_train/val``.
        Stores the best found config in ``self.best_config_``.

        Parameters
        ----------
        instances : sequence of SchInstance
        h : float
            Due-date tightness forwarded to SA runs.
        """
        from gepa import optimize  # noqa: PLC0415

        instances = list(instances)
        rng = random.Random(self.cfg.seed)
        shuffled = instances[:]
        rng.shuffle(shuffled)

        n_train = min(self.cfg.n_instances_train, len(shuffled))
        n_val = min(self.cfg.n_instances_val, len(shuffled) - n_train)
        if n_val <= 0:
            n_val = n_train  # fall back: reuse trainset as valset
            valset = shuffled[:n_train]
        else:
            valset = shuffled[n_train: n_train + n_val]
        trainset = shuffled[:n_train]

        mock_rng = random.Random(self.cfg.seed + 1)
        adapter = SASchedulingAdapter(
            h=h,
            sa_max_steps=self.cfg.sa_max_steps,
            seed=self.cfg.seed,
            mock_rng=mock_rng,
        )

        use_mock = self.cfg.reflection_lm is None

        print(
            f"[GEPAAgent] Starting search: "
            f"{'mock' if use_mock else self.cfg.reflection_lm} | "
            f"trainset={len(trainset)} val={len(valset)} | "
            f"max_metric_calls={self.cfg.max_metric_calls}"
        )

        result = optimize(
            seed_candidate=_SEED_CANDIDATE,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=None if use_mock else self.cfg.reflection_lm,
            candidate_selection_strategy=self.cfg.candidate_selection,
            frontier_type="instance",
            max_metric_calls=self.cfg.max_metric_calls,
            run_dir=self.cfg.run_dir,
            seed=self.cfg.seed,
            display_progress_bar=True,
        )

        self.gepa_result_ = result
        best_candidate = result.best_candidate
        self.best_config_ = SASchedulingAdapter._candidate_to_saconfig(best_candidate)

        best_score = result.val_aggregate_scores[result.best_idx]
        print(
            f"[GEPAAgent] Search complete. "
            f"Best val score: {best_score:.4f} | "
            f"Candidates explored: {result.num_candidates} | "
            f"Best config: T0={self.best_config_.T0:.3g}, "
            f"b={self.best_config_.b:.3g}, "
            f"c={self.best_config_.c:.3g}, "
            f"d={self.best_config_.d:.3g}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def solve(self, env: SchEnv, *, seed: int | None = None) -> EpisodeResult:
        """Run SA with the best config found by train().

        Falls back to default SAConfig if train() has not been called.
        """
        cfg = self.best_config_ if self.best_config_ is not None else SAConfig()
        agent = SimulatedAnnealingAgent(cfg)
        return agent.solve(env, seed=seed)
