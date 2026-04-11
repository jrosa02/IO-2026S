#!/usr/bin/env python3
"""
Sanity checks for SimulatedAnnealingAgent and GeneticAlgorithmAgent.

Run with: pytest sanity/sanity_classical_agents.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load, SchEnv
from src.classical_agents import (
    GeneticAlgorithmAgent,
    GAConfig,
    SAConfig,
    SimulatedAnnealingAgent,
)
from src.sch_env import EpisodeResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset():
    return load("data/sch10.txt")


@pytest.fixture
def instance(dataset):
    return dataset[0]


@pytest.fixture
def env(instance):
    return SchEnv(instance, h=0.4, max_steps=30, seed=42)


@pytest.fixture
def sa_agent():
    return SimulatedAnnealingAgent(SAConfig(T0=50.0, b=0.1, c=5, d=3, T_min=1e-4))


@pytest.fixture
def ga_agent():
    return GeneticAlgorithmAgent(GAConfig(population_size=10, generations=3, mutation_rate=0.2))


# ---------------------------------------------------------------------------
# SimulatedAnnealingAgent
# ---------------------------------------------------------------------------


def test_sa_solve_returns_episode_result(sa_agent, env):
    result = sa_agent.solve(env, seed=42)
    assert isinstance(result, EpisodeResult)


def test_sa_result_fields_valid(sa_agent, env):
    result = sa_agent.solve(env, seed=42)
    assert result.initial_cost > 0
    assert result.best_cost > 0
    assert result.best_cost <= result.initial_cost
    assert result.improvement_pct >= 0.0
    assert result.n_steps >= 0


def test_sa_records_actions(sa_agent, env):
    sa_agent.solve(env, seed=42)
    assert hasattr(sa_agent, "actions")
    assert isinstance(sa_agent.actions, list)
    for swap in sa_agent.actions:
        assert len(swap) == 2


def test_sa_records_cost_history(sa_agent, env):
    sa_agent.solve(env, seed=42)
    assert hasattr(sa_agent, "cost_history")
    assert len(sa_agent.cost_history) == env.max_steps
    # Cost history must be monotonically non-increasing (tracking best)
    for i in range(1, len(sa_agent.cost_history)):
        assert sa_agent.cost_history[i] <= sa_agent.cost_history[i - 1]


def test_sa_records_initial_schedule(sa_agent, env, instance):
    sa_agent.solve(env, seed=42)
    assert hasattr(sa_agent, "initial_schedule")
    assert sorted(sa_agent.initial_schedule) == list(range(instance.n))


def test_sa_best_schedule_is_valid_permutation(sa_agent, env, instance):
    result = sa_agent.solve(env, seed=42)
    assert sorted(result.best_schedule) == list(range(instance.n))


def test_sa_best_cost_matches_best_schedule(sa_agent, env, instance):
    result = sa_agent.solve(env, seed=42)
    recomputed = instance.evaluate(result.best_schedule, h=0.4)
    assert recomputed == result.best_cost


def test_sa_reproducibility(instance):
    cfg = SAConfig(T0=50.0, b=0.1, c=5, d=3, T_min=1e-4)
    env1 = SchEnv(instance, h=0.4, max_steps=20, seed=42)
    r1 = SimulatedAnnealingAgent(cfg).solve(env1, seed=7)

    env2 = SchEnv(instance, h=0.4, max_steps=20, seed=42)
    r2 = SimulatedAnnealingAgent(cfg).solve(env2, seed=7)

    assert r1.best_cost == r2.best_cost


def test_sa_instance_index_in_result(sa_agent, env, instance):
    result = sa_agent.solve(env, seed=42)
    assert result.instance_index == instance.index


def test_sa_h_in_result(sa_agent, env):
    result = sa_agent.solve(env, seed=42)
    assert result.h == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# GeneticAlgorithmAgent
# ---------------------------------------------------------------------------


def test_ga_solve_returns_episode_result(ga_agent, env):
    result = ga_agent.solve(env, seed=42)
    assert isinstance(result, EpisodeResult)


def test_ga_result_fields_valid(ga_agent, env):
    result = ga_agent.solve(env, seed=42)
    assert result.initial_cost > 0
    assert result.best_cost > 0
    assert result.best_cost <= result.initial_cost
    assert result.improvement_pct >= 0.0
    assert result.n_steps == env.max_steps


def test_ga_records_actions(ga_agent, env):
    ga_agent.solve(env, seed=42)
    assert hasattr(ga_agent, "actions")
    assert len(ga_agent.actions) == env.max_steps
    for swap in ga_agent.actions:
        assert len(swap) == 2


def test_ga_records_cost_history(ga_agent, env):
    ga_agent.solve(env, seed=42)
    assert hasattr(ga_agent, "cost_history")
    # One entry per step plus the initial cost
    assert len(ga_agent.cost_history) == env.max_steps + 1
    # Cost history tracks best — must be non-increasing
    for i in range(1, len(ga_agent.cost_history)):
        assert ga_agent.cost_history[i] <= ga_agent.cost_history[i - 1]


def test_ga_records_initial_schedule(ga_agent, env, instance):
    ga_agent.solve(env, seed=42)
    assert hasattr(ga_agent, "initial_schedule")
    assert sorted(ga_agent.initial_schedule) == list(range(instance.n))


def test_ga_best_schedule_is_valid_permutation(ga_agent, env, instance):
    result = ga_agent.solve(env, seed=42)
    assert sorted(result.best_schedule) == list(range(instance.n))


def test_ga_best_cost_matches_best_schedule(ga_agent, env, instance):
    result = ga_agent.solve(env, seed=42)
    recomputed = instance.evaluate(result.best_schedule, h=0.4)
    assert recomputed == result.best_cost


def test_ga_reproducibility(instance):
    cfg = GAConfig(population_size=10, generations=3, mutation_rate=0.2)
    env1 = SchEnv(instance, h=0.4, max_steps=10, seed=42)
    r1 = GeneticAlgorithmAgent(cfg).solve(env1, seed=7)

    env2 = SchEnv(instance, h=0.4, max_steps=10, seed=42)
    r2 = GeneticAlgorithmAgent(cfg).solve(env2, seed=7)

    assert r1.best_cost == r2.best_cost


def test_ga_instance_index_in_result(ga_agent, env, instance):
    result = ga_agent.solve(env, seed=42)
    assert result.instance_index == instance.index


def test_ga_h_in_result(ga_agent, env):
    result = ga_agent.solve(env, seed=42)
    assert result.h == pytest.approx(0.4)
