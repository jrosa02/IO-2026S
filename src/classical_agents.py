"""
classical_agents.py — Classical metaheuristic scheduling agents.

Agents
------
SimulatedAnnealingAgent
    Iterative improvement via simulated annealing.  Accepts worsening swaps
    with probability exp(-delta/T) where T decreases geometrically each step.
    Config parameters exposed for GEPA search: T0, cooling, T_min.

GeneticAlgorithmAgent
    Population-based evolutionary search over job permutations.
    Uses Order Crossover (OX) to combine parents while preserving relative order,
    and swap-mutation with configurable mutation_rate.  Best individual is kept
    via elitism across all generations.

Both agents subclass :class:`~agent.Agent` and store:
- ``self.actions``        — list of (i, j) swap pairs applied
- ``self.cost_history``   — best cost recorded at each step / generation
- ``self.initial_schedule`` — permutation at the start of solve()

"""
from __future__ import annotations

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy.random import Generator

from .agent import Agent
from .configs import GAConfig, SAConfig
from .native_optimized import batch_crossover as _batch_crossover_opt
from .native_optimized import evaluate_batch_swap as _evaluate_batch_swap
from .sch_env import SchEnv, EpisodeResult


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------


class SimulatedAnnealingAgent(Agent):

    def __init__(self, cfg: SAConfig = SAConfig()):
        self.cfg = cfg
        self._np_rng = np.random.default_rng()

    def _plot_temperature_schedule(self, temperatures):
        """Plot and save the temperature evolution over steps."""
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Temperature")
        plt.title(f"Annealing Schedule: T0={self.cfg.T0}, b={self.cfg.b}, c={self.cfg.c}, d={self.cfg.d}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.cfg.temp_plot_file, dpi=150, bbox_inches='tight')
        plt.close()

    def solve(self, env: SchEnv, *, seed: int | None = None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        env.reset()
        self._solve_init(env)

        initial_cost = env.current_cost
        current_cost = initial_cost
        total_reward = 0.0
        temperatures = np.ndarray((env.max_steps))

        inst = env.instance
        d    = int(inst.sum_p * env.h)
        p, a, b = inst.p_array, inst.a_array, inst.b_array
        _I = np.empty(4, dtype=np.int64)
        _J = np.empty(4, dtype=np.int64)

        for step in range(env.max_steps):
            T = (self.cfg.T0 * math.exp(-step * self.cfg.b) +
                 self.cfg.c * (math.sin(step / self.cfg.d))**2)
            T = max(T, self.cfg.T_min)
            temperatures[step] = T

            looping = 4
            while True:
                acts = env.action_space_samples(looping)
                for k in range(4):
                    _I[k], _J[k] = env.decode_action(acts[k])
                costs = _evaluate_batch_swap(p, a, b, env._schedule, d, _I, _J)
                for k in range(4):
                    delta = int(costs[k]) - current_cost
                    if delta < 0 or self._np_rng.random() < math.exp(-delta / T):
                        action = acts[k]
                        i, j = int(_I[k]), int(_J[k])
                        break
                else:
                    continue
                break

            _, reward, _, _, _ = env.step(action)
            current_cost = env.current_cost
            total_reward += reward
            self.actions.append((i, j))
            self.cost_history.append(env.best_cost)

        # self._plot_temperature_schedule(temperatures)

        best_cost = env.best_cost
        n_improvements = sum(
            1 for a, b in zip(self.cost_history, self.cost_history[1:]) if b < a
        )
        return EpisodeResult(
            instance_index=env.instance.index,
            h=env.h,
            initial_cost=initial_cost,
            final_cost=env.current_cost,
            best_cost=best_cost,
            total_reward=total_reward,
            n_steps=len(self.actions),
            n_improvements=n_improvements,
            improvement_pct=(100.0 * (initial_cost - best_cost) / initial_cost),
            best_schedule=env.best_schedule,
            cost_history=self.cost_history[:],
        )

# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------


class GeneticAlgorithmAgent(Agent):
    """Online GA that takes one action per step until max_steps is reached."""

    def __init__(self, cfg: GAConfig = GAConfig()):
        self.cfg = cfg
        self._np_rng = np.random.default_rng()

    def solve(self, env: SchEnv, *, seed: int | None = None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        env.reset()
        self._solve_init(env)
        initial_cost = env.current_cost
        best_cost_so_far = initial_cost
        self.cost_history.append(initial_cost)

        n = env.n

        # ------------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------------
        def cost(schedule):
            return env.instance.evaluate(schedule, env.h)

        def random_perm(_np_rng: Generator):
            p = np.arange(n, dtype=np.int64)
            _np_rng.shuffle(p)
            return p

        def mutate(p: np.ndarray) -> None:
            if self._np_rng.random() < self.cfg.mutation_rate:
                # Fisher-Yates pick
                i = self._np_rng.integers(0, n)
                j = self._np_rng.integers(n - 1)
                if j >= i:
                    j += 1
                p[i], p[j] = p[j], p[i]

        def run_ga(current_schedule) -> tuple[list, int]:
            cur = np.asarray(current_schedule, dtype=np.int64)
            half = self.cfg.population_size // 2
            population: list[np.ndarray] = (
                [random_perm(self._np_rng) for _ in range(half)]
                + [cur.copy() for _ in range(self.cfg.population_size - half)]
            )
            for p in population[half:]:
                if self._np_rng.random() < 0.5:
                    mutate(p)

            best_schedule = cur.copy()
            best_cost = cost(cur)
            pool_k = max(2, self.cfg.population_size // 2)
            n_ch = self.cfg.population_size - self.cfg.elite_size

            for _ in range(self.cfg.generations):
                costs = [cost(p) for p in population]
                order = np.argsort(costs)

                if costs[order[0]] < best_cost:
                    best_cost = costs[order[0]]
                    best_schedule = population[order[0]].copy()

                elite = [population[order[i]].copy() for i in range(self.cfg.elite_size)]
                pool = [population[order[i]] for i in range(pool_k)]

                # Batch: generate all parent indices and cut points at once
                pi = self._np_rng.integers(0, pool_k, n_ch)
                pj = self._np_rng.integers(0, pool_k, n_ch)
                pts = self._np_rng.integers(0, n, (n_ch, 2))
                same = pts[:, 0] == pts[:, 1]
                if same.any():
                    pts[same, 1] = (pts[same, 1] + 1) % n
                a_pts = np.minimum(pts[:, 0], pts[:, 1]).astype(np.int32)
                b_pts = np.maximum(pts[:, 0], pts[:, 1]).astype(np.int32)

                # Single C call for the entire generation's crossover
                children_mat = _batch_crossover_opt(
                    np.stack([pool[i] for i in pi]),
                    np.stack([pool[i] for i in pj]),
                    a_pts, b_pts,
                )
                children = list(children_mat)  # row views into children_mat
                for child in children:
                    mutate(child)

                population = elite + children

            return best_schedule.tolist(), best_cost

        # ------------------------------------------------------------------
        # Main loop: take exactly env.max_steps actions
        # ------------------------------------------------------------------
        current = env.current_schedule
        target, target_cost = run_ga(current)   # initial target

        total_reward = 0.0

        for step in range(env.max_steps):
            if current == target or step % self.cfg.reoptimize_every == 0:
                target, target_cost = run_ga(current)

            if current == target:
                action = env.action_space_samples()
                i, j = env.decode_action(action)
            else:
                i, j = next(
                    (i, current.index(target[i])) for i in range(n) if current[i] != target[i]
                )
                action = env.encode_action(i, j)

            _, reward, _, _, _ = env.step(action)
            self.actions.append(env.decode_action(action))
            total_reward += reward

            current = env.current_schedule
            if env.current_cost < best_cost_so_far:
                best_cost_so_far = env.current_cost
            self.cost_history.append(best_cost_so_far)

        # ------------------------------------------------------------------
        # Build EpisodeResult
        # ------------------------------------------------------------------
        n_improvements = sum(
            1 for a, b in zip(self.cost_history, self.cost_history[1:]) if b < a
        )
        improvement_pct = (100.0 * (initial_cost - best_cost_so_far) / initial_cost) if initial_cost > 0 else 0.0

        return EpisodeResult(
            instance_index=env.instance.index,
            h=env.h,
            initial_cost=initial_cost,
            final_cost=env.current_cost,
            best_cost=best_cost_so_far,
            total_reward=total_reward,
            n_steps=len(self.actions),
            n_improvements=n_improvements,
            improvement_pct=improvement_pct,
            best_schedule=env.best_schedule,
            cost_history=self.cost_history[:],
        )