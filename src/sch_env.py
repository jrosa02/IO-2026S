"""
sch_env.py — Scheduling Environment (modify → evaluate loop)
=============================================================
Wraps a :class:`~orlib_sch.SchInstance` as a self-contained Gym-style
environment whose *action space* is the set of pairwise-swap moves on a
job permutation.

Design
------
State
    A permutation o of {0 … n-1} representing the current processing order,
    together with scalar context (current cost, time-step, due-date tightness h).

Action
    An integer in [0, n*(n-1)/2) that selects a unique pair (i, j) with i < j.
    Applying the action swaps positions i and j in the current permutation.

Reward
    r_t = (cost_before - cost_after) / normaliser
    Positive when the swap improves the schedule, negative when it worsens it.
    The normaliser is the initial (random) cost so rewards stay near [-1, 1].

Episode
    Runs for ``max_steps`` steps (default: 10 * n).
    Terminates early only if ``max_steps`` is reached (no natural terminal state).

Observation vector (flat float32, length 3*n + 3)
    [  p_o(0)/P, a_o(0)/A, b_o(0)/B,          <- job features in current order
       p_o(1)/P, a_o(1)/A, b_o(1)/B,
       ...
       p_o(n-1)/P, a_o(n-1)/A, b_o(n-1)/B,
       current_cost / initial_cost,             ← progress signal
       step / max_steps,                        ← time budget used
       h                                        ← due-date tightness
    ]
    P, A, B = max p_i, max a_i, max b_i  (instance-level normalisation)

Usage
-----
    from orlib_sch import load
    from sch_env import SchEnv

    ds   = load("sch10.txt")
    env  = SchEnv(ds[0], h=0.4, max_steps=100, seed=42)

    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space_sample()       # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Final cost: {info['cost']}  Best cost: {info['best_cost']}")

    # --- Batch runner (iterates over all instances in a dataset) ---
    from sch_env import run_episode
    result = run_episode(env, policy_fn=env.action_space_sample)
    print(result)
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from orlib_sch import SchInstance

# ---------------------------------------------------------------------------
# Action encoding helpers
# ---------------------------------------------------------------------------


def _build_swap_pairs(n: int) -> list[tuple[int, int]]:
    """Return all (i, j) pairs with i < j, in lexicographic order."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SchEnv:
    """
    Gym-compatible environment for single-machine common due date scheduling.

    The environment wraps one :class:`~orlib_sch.SchInstance` and exposes
    a swap-based action space so an agent can iteratively improve a schedule.

    Parameters
    ----------
    instance : SchInstance
    h : float
        Due-date tightness (default 0.4).
    max_steps : int | None
        Episode length.  Defaults to ``10 * n``.
    seed : int | None
        RNG seed for reproducible resets.
    reward_shaping : bool
        If True (default), scale rewards by 1/initial_cost.
        If False, reward = raw cost improvement (integer).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        instance: SchInstance,
        h: float = 0.4,
        max_steps: int | None = None,
        seed: int | None = None,
        reward_shaping: bool = True,
    ) -> None:
        self.instance = instance
        self.h = h
        self.n = instance.n
        self.d = instance.due_date(h)
        self.max_steps = max_steps if max_steps is not None else 10 * self.n
        self.reward_shaping = reward_shaping

        # All possible swap moves
        self._pairs: list[tuple[int, int]] = _build_swap_pairs(self.n)
        self.n_actions: int = len(self._pairs)  # = n*(n-1)//2

        # Observation size: 3*n job features + 3 scalars
        self.obs_size: int = 3 * self.n + 3

        # Per-instance normalisation constants (avoid division by zero)
        self._max_p = max(j.p for j in instance.jobs) or 1
        self._max_a = max(j.a for j in instance.jobs) or 1
        self._max_b = max(j.b for j in instance.jobs) or 1

        # RNG
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Mutable state (initialised properly on reset())
        self._schedule: list[int] = list(range(self.n))
        self._cost: int = 0
        self._initial_cost: int = 1  # set on reset
        self._best_cost: int = 0
        self._best_schedule: list[int] = list(range(self.n))
        self._step_count: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        schedule: list[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int | None
            Override the RNG seed for this episode.
        schedule : list[int] | None
            Start from a specific permutation instead of a random one.

        Returns
        -------
        obs : np.ndarray  shape (obs_size,)
        info : dict
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        if schedule is not None:
            self._schedule = list(schedule)
        else:
            self._schedule = list(range(self.n))
            self._rng.shuffle(self._schedule)

        self._cost = self._compute_cost(self._schedule)
        self._initial_cost = max(self._cost, 1)  # guard against zero
        self._best_cost = self._cost
        self._best_schedule = self._schedule[:]
        self._step_count = 0
        self._done = False

        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply a swap action and return the Gym 5-tuple.

        Parameters
        ----------
        action : int
            Index into the swap-pair list [0, n_actions).

        Returns
        -------
        obs         : np.ndarray
        reward      : float
        terminated  : bool   - always False (no natural terminal state)
        truncated   : bool   - True when step budget is exhausted
        info        : dict
        """
        if self._done:
            raise RuntimeError("Call reset() before step().")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action {action} out of range [0, {self.n_actions}).")

        cost_before = self._cost

        # Apply swap in-place
        i, j = self._pairs[action]
        self._schedule[i], self._schedule[j] = self._schedule[j], self._schedule[i]
        self._cost = self._compute_cost(self._schedule)

        # Track best seen
        if self._cost < self._best_cost:
            self._best_cost = self._cost
            self._best_schedule = self._schedule[:]

        delta = cost_before - self._cost
        reward = delta / self._initial_cost if self.reward_shaping else float(delta)

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        terminated = False
        if truncated:
            self._done = True

        return self._observe(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Action space helpers
    # ------------------------------------------------------------------

    def action_space_sample(self) -> int:
        """Sample a uniformly random action."""
        return self._rng.randrange(self.n_actions)

    def decode_action(self, action: int) -> tuple[int, int]:
        """Return the (i, j) position pair for a given action index."""
        return self._pairs[action]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cost(self, schedule: list[int]) -> int:
        """Evaluate the cost of *schedule* against this instance."""
        return self.instance.evaluate(schedule, self.h)

    def _observe(self) -> np.ndarray:
        """Build the flat float32 observation vector."""
        obs = np.empty(self.obs_size, dtype=np.float32)
        for rank, job_idx in enumerate(self._schedule):
            job = self.instance.jobs[job_idx]
            base = rank * 3
            obs[base] = job.p / self._max_p
            obs[base + 1] = job.a / self._max_a
            obs[base + 2] = job.b / self._max_b
        # Scalar context
        obs[-3] = self._cost / self._initial_cost
        obs[-2] = self._step_count / self.max_steps
        obs[-1] = self.h
        return obs

    def _info(self) -> dict[str, Any]:
        return {
            "cost": self._cost,
            "best_cost": self._best_cost,
            "best_schedule": self._best_schedule[:],
            "step": self._step_count,
            "initial_cost": self._initial_cost,
            "improvement": self._initial_cost - self._best_cost,
            "improvement_pct": (100.0 * (self._initial_cost - self._best_cost) / self._initial_cost),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def current_schedule(self) -> list[int]:
        return self._schedule[:]

    @property
    def current_cost(self) -> int:
        return self._cost

    @property
    def best_cost(self) -> int:
        return self._best_cost

    def __repr__(self) -> str:
        return (
            f"SchEnv(instance={self.instance.index}, n={self.n}, "
            f"h={self.h}, max_steps={self.max_steps}, "
            f"n_actions={self.n_actions}, obs_size={self.obs_size})"
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Summary statistics for one completed episode."""

    instance_index: int
    h: float
    initial_cost: int
    final_cost: int
    best_cost: int
    total_reward: float
    n_steps: int
    n_improvements: int
    improvement_pct: float
    best_schedule: list[int]

    def __repr__(self) -> str:
        return (
            f"EpisodeResult("
            f"inst={self.instance_index}, h={self.h}, "
            f"initial={self.initial_cost}, best={self.best_cost}, "
            f"improve={self.improvement_pct:.1f}%, steps={self.n_steps})"
        )


def run_episode(
    env: SchEnv,
    policy_fn: Callable[[np.ndarray], int],
    *,
    seed: int | None = None,
    start_schedule: list[int] | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """
    Run a single episode using *policy_fn* and return an :class:`EpisodeResult`.

    Parameters
    ----------
    env : SchEnv
        The environment to run in.
    policy_fn : callable(obs) -> int
        Maps an observation array to an action index.
        For a random policy pass ``env.action_space_sample`` (no call brackets).
    seed : int | None
        Passed to ``env.reset()``.
    start_schedule : list[int] | None
        Optional fixed starting permutation.
    verbose : bool
        Print per-step info.

    Returns
    -------
    EpisodeResult
    """
    obs, info = env.reset(seed=seed, schedule=start_schedule)
    initial_cost = info["initial_cost"]
    total_reward = 0.0
    n_improvements = 0
    prev_cost = info["cost"]

    done = False
    while not done:
        # Call policy_fn with observation if it's a callable
        # Handle both lambda obs: ... and method references like env.action_space_sample
        try:
            action = policy_fn(obs)
        except TypeError:
            # If policy_fn doesn't accept obs, call it without arguments
            action = policy_fn()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["cost"] < prev_cost:
            n_improvements += 1
        prev_cost = info["cost"]
        done = terminated or truncated
        if verbose:
            print(
                f"  step={info['step']:4d}  cost={info['cost']:8d}"
                f"  reward={reward:+.4f}  best={info['best_cost']:8d}"
            )

    return EpisodeResult(
        instance_index=env.instance.index,
        h=env.h,
        initial_cost=initial_cost,
        final_cost=info["cost"],
        best_cost=info["best_cost"],
        total_reward=total_reward,
        n_steps=info["step"],
        n_improvements=n_improvements,
        improvement_pct=info["improvement_pct"],
        best_schedule=info["best_schedule"],
    )


# ---------------------------------------------------------------------------
# Multi-instance batch runner
# ---------------------------------------------------------------------------


@dataclass
class DatasetRunConfig:
    """Configuration for batch running episodes across a dataset."""
    h: float = 0.4
    max_steps: int | None = None
    seed: int | None = None
    verbose: bool = False


def run_dataset(
    instances: Sequence[SchInstance],
    policy_fn_factory: Callable[[SchEnv], Callable[[np.ndarray], int]],
    cfg: DatasetRunConfig | None = None,
) -> list[EpisodeResult]:
    """
    Run one episode per instance in *instances* and collect results.

    Parameters
    ----------
    instances : sequence of SchInstance
    policy_fn_factory : callable(env) -> policy_fn
        Called once per instance to produce a per-episode policy.
        For a random policy: ``lambda env: env.action_space_sample``
    cfg : DatasetRunConfig | None
        Configuration for the run. If None, uses defaults.

    Returns
    -------
    list[EpisodeResult]

    Example
    -------
    >>> results = run_dataset(ds.instances, lambda env: env.action_space_sample)
    >>> avg_imp = sum(r.improvement_pct for r in results) / len(results)
    """
    if cfg is None:
        cfg = DatasetRunConfig()
    results = []
    for inst in instances:
        env = SchEnv(inst, h=cfg.h, max_steps=cfg.max_steps, seed=cfg.seed)
        policy_fn = policy_fn_factory(env)
        result = run_episode(env, policy_fn, seed=cfg.seed, verbose=cfg.verbose)
        if cfg.verbose:
            print(result)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from orlib_sch import load

    ds = load("data/sch10.txt")
    env = SchEnv(ds[0], h=0.4, max_steps=50, seed=0)
    print(env)
    
    print("\nSingle episode with random policy:")
    result = run_episode(env, policy_fn=env.action_space_sample, seed=0)
    print(result)

    print("\nBatch run over 2 instances:")
    cfg = DatasetRunConfig(h=0.4, max_steps=50, seed=1, verbose=False)
    results = run_dataset(
        ds.instances[:2],
        lambda env: env.action_space_sample,
        cfg=cfg,
    )
    for r in results:
        print(" ", r)
