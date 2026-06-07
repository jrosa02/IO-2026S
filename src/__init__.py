"""
io-2026s: Common due date scheduling with Reinforcement Learning.

Main exports for easy importing:
  - SchInstance, SchJob, SchDataset: Scheduling problem instances
  - SchEnv: Gym-like scheduling environment
  - run_episode: Batch episode runner
"""

from .agent import Agent, GreedyAgent, RandomAgent
from .benchmark import AgentBenchmarkResult, BenchmarkRunner
from .classical_agents import GAConfig, GeneticAlgorithmAgent, SAConfig, SimulatedAnnealingAgent
from .orlib_sch import SchDataset, SchInstance, SchJob, load
from .sch_env import EpisodeResult, SchEnv, run_episode
from .visualize import plot_gepa_history

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentBenchmarkResult",
    "BenchmarkRunner",
    "EpisodeResult",
    "GAConfig",
    "GeneticAlgorithmAgent",
    "GreedyAgent",
    "RandomAgent",
    "SAConfig",
    "SchDataset",
    "SchEnv",
    "SchInstance",
    "SchJob",
    "SimulatedAnnealingAgent",
    "load",
    "plot_gepa_history",
    "run_episode",
]


def main():
    """Entry point — delegates to top-level main.py CLI."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "_cli", Path(__file__).parent.parent / "main.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.main()