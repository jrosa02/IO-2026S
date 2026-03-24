"""
benchmark.py — Multi-agent benchmark runner and result aggregation.

Runs every registered agent on every instance in a dataset, collects
EpisodeResult objects, and produces summary statistics, CSV exports,
and comparison plots suitable for an academic report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .sch_env import EpisodeResult, SchEnv
from .orlib_sch import SchInstance

if TYPE_CHECKING:
    from .agent import Agent


# ---------------------------------------------------------------------------
# Aggregated result per agent
# ---------------------------------------------------------------------------


@dataclass
class AgentBenchmarkResult:
    """Aggregated benchmark results for one agent across all instances."""

    agent_name: str
    results: list[EpisodeResult]
    mean_improvement_pct: float = field(init=False)
    mean_best_cost: float = field(init=False)
    std_improvement_pct: float = field(init=False)
    mean_n_steps: float = field(init=False)

    def __post_init__(self) -> None:
        import statistics

        imps = [r.improvement_pct for r in self.results]
        self.mean_improvement_pct = statistics.mean(imps) if imps else 0.0
        self.std_improvement_pct = statistics.stdev(imps) if len(imps) > 1 else 0.0
        self.mean_best_cost = statistics.mean(r.best_cost for r in self.results) if self.results else 0.0
        self.mean_n_steps = statistics.mean(r.n_steps for r in self.results) if self.results else 0.0

    def __repr__(self) -> str:
        return (
            f"AgentBenchmarkResult(agent={self.agent_name!r}, "
            f"n_instances={len(self.results)}, "
            f"mean_improvement={self.mean_improvement_pct:.1f}%, "
            f"mean_best_cost={self.mean_best_cost:.1f})"
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Orchestrates a full benchmark experiment over a set of instances.

    Parameters
    ----------
    instances : sequence of SchInstance
    h : float
        Due-date tightness (default 0.4).
    max_steps : int | None
        Episode length per instance (default: 10 * n from SchEnv).
    seed : int | None
        RNG seed used for every agent × instance run, enabling fair comparison.
    """

    def __init__(
        self,
        instances: Sequence[SchInstance],
        h: float = 0.4,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.instances = list(instances)
        self.h = h
        self.max_steps = max_steps
        self.seed = seed

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run(
        self,
        agents: dict[str, Agent],
        verbose: bool = False,
    ) -> dict[str, AgentBenchmarkResult]:
        """
        Run each agent on every instance and collect results.

        Parameters
        ----------
        agents : dict[str, Agent]
            Mapping of display name → agent instance.
        verbose : bool
            Print per-instance progress.

        Returns
        -------
        dict[str, AgentBenchmarkResult]
        """
        benchmark_results: dict[str, AgentBenchmarkResult] = {}

        for agent_name, agent in agents.items():
            episode_results: list[EpisodeResult] = []
            for inst in self.instances:
                env = SchEnv(inst, h=self.h, max_steps=self.max_steps, seed=self.seed)
                result = agent.solve(env, seed=self.seed)
                episode_results.append(result)
                if verbose:
                    print(
                        f"  [{agent_name}] inst={inst.index}  "
                        f"best={result.best_cost}  "
                        f"improve={result.improvement_pct:.1f}%"
                    )
            benchmark_results[agent_name] = AgentBenchmarkResult(agent_name, episode_results)
            if verbose:
                abr = benchmark_results[agent_name]
                print(
                    f"  [{agent_name}] mean_improve={abr.mean_improvement_pct:.1f}%  "
                    f"mean_best={abr.mean_best_cost:.1f}\n"
                )

        return benchmark_results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, results: dict[str, AgentBenchmarkResult]) -> pd.DataFrame:
        """
        Flatten results into a tidy DataFrame (one row per agent × instance).

        Columns: agent, instance_index, initial_cost, best_cost,
                 improvement_pct, n_steps, n_improvements, total_reward
        """
        rows = [
            {
                "agent": name,
                "instance_index": r.instance_index,
                "initial_cost": r.initial_cost,
                "best_cost": r.best_cost,
                "improvement_pct": r.improvement_pct,
                "n_steps": r.n_steps,
                "n_improvements": r.n_improvements,
                "total_reward": r.total_reward,
            }
            for name, abr in results.items()
            for r in abr.results
        ]
        return pd.DataFrame(rows)

    def save_csv(self, results: dict[str, AgentBenchmarkResult], path: str | Path) -> None:
        """Write the flat results table to a CSV file."""
        df = self.to_dataframe(results)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def print_summary(self, results: dict[str, AgentBenchmarkResult]) -> None:
        """Print an ASCII summary table comparing all agents."""
        col_w = max(len(name) for name in results) + 2
        header = (
            f"{'Agent':<{col_w}} | {'Mean Improv%':>12} | {'Mean Best Cost':>14} | {'Std Improv%':>11}"
        )
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        for name, abr in results.items():
            print(
                f"{name:<{col_w}} | {abr.mean_improvement_pct:>12.2f} | "
                f"{abr.mean_best_cost:>14.1f} | {abr.std_improvement_pct:>11.2f}"
            )
        print(separator)
