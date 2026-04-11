"""
agent.py — Abstract Agent base class and built-in concrete agents.

All agents that operate on SchEnv should subclass :class:`Agent`.

Hierarchy::

    Agent (ABC)
    ├── RandomAgent    — uniform random swap selection
    └── GreedyAgent    — exhaustive 1-step look-ahead

Notes
-----
- Override ``act()`` for iterative (swap-based) agents.
- Override ``construct()`` for constructive agents that build an initial schedule.
- Override ``train()/save()/load()`` for learning agents (e.g. PPO).
- ``solve()`` is concrete and orchestrates a full episode via :func:`run_episode`.
"""

from __future__ import annotations

import random
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .orlib_sch import SchInstance
from .sch_env import EpisodeResult, SchEnv, run_episode

if TYPE_CHECKING:
    pass


class Agent(ABC):
    """
    Abstract base class for all scheduling agents.

    Subclass and override at least one of:
    - ``act(obs)``      — iterative step-by-step policy (swap-based)
    - ``construct(instance)`` — build an initial schedule permutation in one shot

    Training/persistence hooks (``train``, ``save``, ``load``) are no-ops by
    default and should be overridden by learning agents.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable agent name (defaults to class name)."""
        return type(self).__name__

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """
        Select a swap action given the current observation.

        Parameters
        ----------
        obs : np.ndarray
            Flat float32 observation vector of shape ``(obs_size,)``.

        Returns
        -------
        int
            Action index in ``[0, n_actions)``.
        """
        raise NotImplementedError(f"{self.name}.act() is not implemented")

    def construct(self, instance: SchInstance) -> list[int] | None:
        """
        Optionally build an initial job permutation before the iterative phase.

        Parameters
        ----------
        instance : SchInstance
            The scheduling instance to construct a schedule for.

        Returns
        -------
        list[int] | None
            A permutation of ``range(instance.n)``, or ``None`` to let the
            environment use a random permutation.
        """
        return None

    # ------------------------------------------------------------------
    # Episode orchestration
    # ------------------------------------------------------------------

    def _make_recording_policy(
        self, env: SchEnv, policy_fn: Callable
    ) -> Callable[[np.ndarray], int]:
        """Wrap a policy function to record decoded swap pairs as (i, j) tuples."""

        def wrapper(obs: np.ndarray) -> int:
            # Handle both obs-accepting and no-arg policy functions
            try:
                action = policy_fn(obs)
            except TypeError:
                action = policy_fn()
            i, j = env.decode_action(action)
            self.actions.append((i, j))
            return action

        return wrapper

    def solve(self, env: SchEnv, *, seed: int | None = None) -> EpisodeResult:
        """
        Run a complete episode and return the result.

        Calls :func:`~sch_env.run_episode` with ``self.act`` as the policy
        function.  If ``construct()`` is overridden it is called first to
        provide an initial schedule.  All actions taken are recorded as decoded
        swap pairs (i, j) in ``self.actions``.

        Parameters
        ----------
        env : SchEnv
            The environment to solve.
        seed : int | None
            Optional RNG seed forwarded to ``env.reset()``.

        Returns
        -------
        EpisodeResult
        """
        self.actions: list[tuple[int, int]] = []
        start_schedule = self.construct(env.instance)
        # Reset env to capture the initial schedule
        env.reset(seed=seed, schedule=start_schedule)
        self.initial_schedule = env.current_schedule[:]
        recording_policy = self._make_recording_policy(env, self.act)
        # Re-run with recorded initial schedule to ensure determinism
        return run_episode(env, recording_policy, seed=None, start_schedule=self.initial_schedule)

    # ------------------------------------------------------------------
    # Training lifecycle (no-ops for non-learning agents)
    # ------------------------------------------------------------------

    def train(
        self,
        instances: Sequence[SchInstance],
        *,
        h: float = 0.4,
        **kwargs,
    ) -> None:
        """
        Train the agent on a collection of instances.

        Default implementation is a no-op.  Override in learning agents.
        """

    def save(self, path: Path | str) -> None:
        """
        Persist agent state to *path*.

        Default implementation is a no-op.  Override in learning agents.
        """

    def load(self, path: Path | str) -> None:
        """
        Restore agent state from *path*.

        Default implementation is a no-op.  Override in learning agents.
        """

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.name}()"


# ---------------------------------------------------------------------------
# Built-in concrete agents
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """
    Agent that selects swap actions uniformly at random.

    Delegates to the environment's own RNG via ``env.action_space_sample``
    to stay reproducible when the environment is seeded.
    """

    def solve(self, env: SchEnv, *, seed: int | None = None) -> EpisodeResult:
        """Run an episode with a random policy using the env's own RNG."""
        self.actions = []
        env.reset(seed=seed)
        self.initial_schedule = env.current_schedule[:]
        recording_policy = self._make_recording_policy(env, env.action_space_sample)
        return run_episode(env, recording_policy, seed=None, start_schedule=self.initial_schedule)

    def act(self, obs: np.ndarray) -> int:
        """Not used directly; ``solve()`` delegates to env's sampler."""
        raise NotImplementedError("RandomAgent uses env.action_space_sample — call solve(env) instead")


class GreedyAgent(Agent):
    """
    Agent that greedily selects the swap that minimises cost at each step.

    At every step, all ``n*(n-1)/2`` candidate swaps are evaluated by
    computing the resulting cost via :meth:`SchInstance.evaluate`.  The
    move with the lowest cost is chosen.  Ties are broken by lowest index.

    This is an O(n² * step) algorithm — suitable for small instances.
    """

    def solve(self, env: SchEnv, *, seed: int | None = None) -> EpisodeResult:
        """Run a greedy episode with exhaustive 1-step look-ahead."""
        self.actions = []
        start_schedule = self.construct(env.instance)
        env.reset(seed=seed, schedule=start_schedule)
        self.initial_schedule = env.current_schedule[:]

        def greedy_policy(obs: np.ndarray) -> int:
            best_action = 0
            best_cost = float("inf")
            schedule = env.current_schedule  # returns a copy
            for action in range(env.n_actions):
                i, j = env.decode_action(action)
                schedule[i], schedule[j] = schedule[j], schedule[i]
                cost = env.instance.evaluate(schedule, env.h)
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
                schedule[i], schedule[j] = schedule[j], schedule[i]  # undo
            return best_action

        recording_policy = self._make_recording_policy(env, greedy_policy)
        return run_episode(env, recording_policy, seed=None, start_schedule=self.initial_schedule)

    def act(self, obs: np.ndarray) -> int:
        """Not directly usable without env access; call ``solve(env)`` instead."""
        raise NotImplementedError("GreedyAgent requires env access — call solve(env) instead")
