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

from .agent import Agent
from .configs import GAConfig, SAConfig


@njit(cache=True)
def _crossover_jit(p1: np.ndarray, p2: np.ndarray, a: int, b: int) -> np.ndarray:
    """Order Crossover (OX) – compiled, single pass, no intermediate list."""
    n = len(p1)
    child = np.empty(n, dtype=np.int64)
    child[a:b] = p1[a:b]

    used = np.zeros(n, dtype=np.bool_)
    for k in range(a, b):
        used[p1[k]] = True

    p2_ptr = 0
    for k in range(n):
        if k < a or k >= b:
            while used[p2[p2_ptr]]:
                p2_ptr += 1
            child[k] = p2[p2_ptr]
            p2_ptr += 1
    return child
from .sch_env import SchEnv, EpisodeResult


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------


class SimulatedAnnealingAgent(Agent):

    def __init__(self, cfg: SAConfig = SAConfig()):
        self.cfg = cfg

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
            random.seed(seed)

        env.reset()
        self._solve_init(env)

        initial_cost = env.current_cost
        current_cost = initial_cost
        total_reward = 0.0
        temperatures = []

        for step in range(env.max_steps):
            T = (self.cfg.T0 * math.exp(-step * self.cfg.b) +
                 self.cfg.c * (math.sin(step / self.cfg.d))**2)
            T = max(T, self.cfg.T_min)
            temperatures.append(T)

            while True:
                action = env.action_space_sample()
                i, j = env.decode_action(action)
                temp_schedule = env.current_schedule
                temp_schedule[i], temp_schedule[j] = temp_schedule[j], temp_schedule[i]
                new_cost = env.instance.evaluate(temp_schedule, env.h)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    break

            _, reward, _, _, _ = env.step(action)
            current_cost = env.current_cost
            total_reward += reward
            self.actions.append((i, j))
            self.cost_history.append(env.best_cost)

        self._plot_temperature_schedule(temperatures)

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

    def solve(self, env: SchEnv, *, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

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

        def random_perm():
            p = list(range(n))
            random.shuffle(p)
            return p

        def crossover(p1, p2):
            a, b = sorted(random.sample(range(n), 2))
            return _crossover_jit(
                np.asarray(p1, dtype=np.int64),
                np.asarray(p2, dtype=np.int64),
                a, b,
            ).tolist()

        def mutate(p):
            if random.random() < self.cfg.mutation_rate:
                i, j = random.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            return p

        def run_ga(current_schedule):
            """Evolve a population starting from `current_schedule` and
               return the best schedule found."""
            # Initialize population: half random, half mutated copies of current
            population = [random_perm() for _ in range(self.cfg.population_size // 2)]
            population += [list(current_schedule) for _ in range(self.cfg.population_size - len(population))]
            # add some mutations to the copies
            for i in range(len(population) // 2, len(population)):
                if random.random() < 0.5:
                    population[i] = mutate(population[i])

            best_schedule = current_schedule[:]
            best_cost = cost(current_schedule)

            for _ in range(self.cfg.generations):
                scored = [(cost(p), p) for p in population]
                scored.sort(key=lambda x: x[0])
                if scored[0][0] < best_cost:
                    best_cost = scored[0][0]
                    best_schedule = scored[0][1][:]

                # elitism
                new_pop = [p[:] for (_, p) in scored[:self.cfg.elite_size]]
                pool = [p for (_, p) in scored[:len(scored)//2]]
                while len(new_pop) < self.cfg.population_size:
                    p1, p2 = random.sample(pool, 2)
                    child = crossover(p1, p2)
                    child = mutate(child)
                    new_pop.append(child)
                population = new_pop

            return best_schedule, best_cost

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
                action = env.action_space_sample()
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