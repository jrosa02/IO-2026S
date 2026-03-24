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

To be implemented.
"""
