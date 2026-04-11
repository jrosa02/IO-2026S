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
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .agent import Agent


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

@dataclass
class SAConfig:
    T0: float = 100.0
    b: float = 0.005
    c: float = 10
    d: float = 5
    T_min: float = 1e-5
    temp_plot_file: str = "temperature_schedule.png"


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

        obs, _ = env.reset()

        self.actions = []
        self.cost_history = []
        self.initial_schedule = list(env._schedule)

        current_cost = env._compute_cost(env._schedule)
        best_cost = current_cost

        temperatures = []

        for step in range(env.max_steps):
            T = (self.cfg.T0 * math.exp(-step * self.cfg.b) +
                 self.cfg.c * (math.sin(step / self.cfg.d))**2)
            T = max(T, self.cfg.T_min)
            temperatures.append(T)
            i, j = 0, 0
            accept = False
            while not accept:
                action = random.randrange(len(env._pairs))
                i, j = env._pairs[action]

                temp_schedule = list(env._schedule)
                temp_schedule[i], temp_schedule[j] = temp_schedule[j], temp_schedule[i]
                new_cost = env._compute_cost(temp_schedule)

                delta = new_cost - current_cost

                if delta < 0:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True

            if accept:
                obs, reward, term, trunc, _ = env.step(action)
                current_cost = env._compute_cost(env._schedule)
                self.actions.append((i, j))
                if current_cost < best_cost:
                    best_cost = current_cost

            self.cost_history.append(best_cost)

        # Plot and save the temperature schedule
        self._plot_temperature_schedule(temperatures)

        info = env._info()
        return EpisodeResult(
            instance_index=env.instance.index,
            h=env.h,
            initial_cost=env._initial_cost,
            final_cost=self.cost_history[-1],
            best_cost=min(self.cost_history),
            total_reward=sum(env._compute_cost(env._schedule) - self.cost_history[i]
                           for i in range(len(self.cost_history) - 1)),
            n_steps=len(self.actions),
            n_improvements=0,
            improvement_pct=(100.0 * (env._initial_cost - min(self.cost_history)) / env._initial_cost),
            best_schedule=env._best_schedule[:],
        )

# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    population_size: int = 100
    generations: int = 10          # GA generations per re-optimization
    mutation_rate: float = 0.1
    elite_size: int = 5
    reoptimize_every: int = 2      # how many steps between full GA runs


class GeneticAlgorithmAgent(Agent):
    """Online GA that takes one action per step until max_steps is reached."""

    def __init__(self, cfg: GAConfig = GAConfig()):
        self.cfg = cfg

    def solve(self, env: SchEnv, *, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        obs, _ = env.reset()
        self.actions = []
        self.cost_history = []
        self.initial_schedule = list(env._schedule)
        initial_cost = env._initial_cost
        best_cost_so_far = initial_cost
        self.cost_history.append(initial_cost)

        n = env.n

        # ------------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------------
        def cost(schedule):
            return env._compute_cost(schedule)

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
        current = list(env._schedule)
        target, target_cost = run_ga(current)   # initial target

        total_reward = 0.0

        for step in range(env.max_steps):
            # If current schedule equals target (or we have no improvement left),
            # find a new target (re-optimize) or perform a random step.
            if current == target or step % self.cfg.reoptimize_every == 0:
                target, target_cost = run_ga(current)

            # If we are already at the target, do a random swap to escape local optimum
            if current == target:
                # random action (swap)
                action = random.randrange(len(env._pairs))
                i, j = env._pairs[action]
            else:
                # Find first position where current differs from target
                for i in range(n):
                    if current[i] != target[i]:
                        j = current.index(target[i])
                        break
                # ensure ordered pair
                a, b = (i, j) if i < j else (j, i)
                action = env._pairs.index((a, b))

            # Execute the action
            obs, reward, term, trunc, _ = env.step(action)
            self.actions.append(env._pairs[action])
            total_reward += reward

            # Update current schedule and cost history
            current = list(env._schedule)
            current_cost = env._compute_cost(current)
            if current_cost < best_cost_so_far:
                best_cost_so_far = current_cost
            self.cost_history.append(best_cost_so_far)

        # ------------------------------------------------------------------
        # Build EpisodeResult
        # ------------------------------------------------------------------
        final_cost = env._compute_cost(env._schedule)
        improvement_pct = (100.0 * (initial_cost - best_cost_so_far) / initial_cost) if initial_cost > 0 else 0.0

        return EpisodeResult(
            instance_index=env.instance.index,
            h=env.h,
            initial_cost=initial_cost,
            final_cost=final_cost,
            best_cost=best_cost_so_far,
            total_reward=total_reward,
            n_steps=len(self.actions),
            n_improvements=0,        # could be computed but not critical
            improvement_pct=improvement_pct,
            best_schedule=env._best_schedule[:],
        )