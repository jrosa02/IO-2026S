"""
io-2026s: Common due date scheduling with Reinforcement Learning (PPO agent)

Main exports for easy importing:
  - SchInstance, SchJob, SchDataset: Scheduling problem instances
  - SchEnv: Gym-like scheduling environment
  - PPOAgent, PPOConfig, train, evaluate: RL training and evaluation
  - run_episode: Batch episode runner
"""

from .orlib_sch import SchDataset, SchInstance, SchJob, load
from .sch_env import SchEnv, EpisodeResult, run_episode
from .agent import Agent, RandomAgent, GreedyAgent
from .classical_agents import SimulatedAnnealingAgent, SAConfig, GeneticAlgorithmAgent, GAConfig
from .benchmark import BenchmarkRunner, AgentBenchmarkResult

# RL components (from sch_rl.py) are not yet available
# from .sch_rl import (
#     PPOAgent,
#     PPOConfig,
#     RolloutBuffer,
#     ActorCritic,
#     train,
#     evaluate,
#     evaluate_batch,
#     benchmark,
#     TrainLog,
# )

__version__ = "0.1.0"
__all__ = [
    # Data structures
    "SchInstance",
    "SchJob",
    "SchDataset",
    # Environment
    "SchEnv",
    "EpisodeResult",
    "run_episode",
    # Loaders
    "load",
    # Agents
    "Agent",
    "RandomAgent",
    "GreedyAgent",
    "SimulatedAnnealingAgent",
    "SAConfig",
    "GeneticAlgorithmAgent",
    "GAConfig",
    # Benchmark
    "BenchmarkRunner",
    "AgentBenchmarkResult",
    # RL components from sch_rl.py (not yet available)
    # "PPOAgent", "PPOConfig", "ActorCritic", "RolloutBuffer", "TrainLog",
    # "train", "evaluate", "evaluate_batch", "benchmark",
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