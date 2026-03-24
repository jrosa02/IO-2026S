#!/usr/bin/env python3
"""
Sanity checks for Agent — verify all agent types work with SchEnv.

Run with: pytest sanity/sanity_agent.py -v
"""

import sys
from pathlib import Path

import pytest

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load, SchEnv, Agent, RandomAgent, GreedyAgent, ConstructiveRandomAgent
from src.sch_env import EpisodeResult


@pytest.fixture
def dataset():
    """Load test dataset."""
    return load("data/sch10.txt")


@pytest.fixture
def env(dataset):
    """Create test environment."""
    return SchEnv(dataset[0], h=0.4, max_steps=50, seed=42)


def test_agent_abstract_base():
    """Check that Agent is abstract and has required interface."""
    assert hasattr(Agent, "act")
    assert hasattr(Agent, "construct")
    assert hasattr(Agent, "solve")
    assert hasattr(Agent, "train")
    assert hasattr(Agent, "save")
    assert hasattr(Agent, "load")


def test_agent_name_property():
    """Check that agent.name property works."""
    r = RandomAgent()
    assert r.name == "RandomAgent"

    g = GreedyAgent()
    assert g.name == "GreedyAgent"

    c = ConstructiveRandomAgent()
    assert c.name == "ConstructiveRandomAgent"


def test_agent_repr():
    """Check that agent repr works."""
    r = RandomAgent()
    assert "RandomAgent" in repr(r)


def test_random_agent_solve(env):
    """Check that RandomAgent.solve() returns a valid EpisodeResult."""
    agent = RandomAgent()
    result = agent.solve(env, seed=42)

    assert isinstance(result, EpisodeResult)
    assert result.initial_cost > 0
    assert result.best_cost > 0
    assert result.improvement_pct >= 0
    assert 0 < result.n_steps <= env.max_steps


def test_random_agent_reproducibility(dataset):
    """Check that RandomAgent produces consistent results with same seed."""
    instance = dataset[0]

    env1 = SchEnv(instance, h=0.4, max_steps=50, seed=42)
    r1 = RandomAgent().solve(env1, seed=99)

    env2 = SchEnv(instance, h=0.4, max_steps=50, seed=42)
    r2 = RandomAgent().solve(env2, seed=99)

    assert r1.best_cost == r2.best_cost
    assert r1.n_steps == r2.n_steps


def test_greedy_agent_solve(env):
    """Check that GreedyAgent.solve() returns a valid EpisodeResult."""
    agent = GreedyAgent()
    result = agent.solve(env, seed=42)

    assert isinstance(result, EpisodeResult)
    assert result.initial_cost > 0
    assert result.best_cost > 0
    assert result.best_cost <= result.initial_cost
    assert result.improvement_pct >= 0


def test_greedy_agent_better_than_random(dataset):
    """Check that GreedyAgent typically achieves better cost than RandomAgent."""
    instance = dataset[0]

    env_random = SchEnv(instance, h=0.4, max_steps=30, seed=42)
    r_random = RandomAgent().solve(env_random, seed=42)

    env_greedy = SchEnv(instance, h=0.4, max_steps=30, seed=42)
    r_greedy = GreedyAgent().solve(env_greedy, seed=42)

    # Greedy should generally find a better schedule
    assert r_greedy.best_cost <= r_random.best_cost


def test_constructive_random_agent_construct(dataset):
    """Check that ConstructiveRandomAgent.construct() returns a valid schedule."""
    instance = dataset[0]
    agent = ConstructiveRandomAgent(seed=42)

    schedule = agent.construct(instance)

    assert schedule is not None
    assert len(schedule) == instance.n
    assert set(schedule) == set(range(instance.n))


def test_constructive_random_agent_solve(dataset):
    """Check that ConstructiveRandomAgent.solve() uses constructed schedule."""
    instance = dataset[0]
    agent = ConstructiveRandomAgent(seed=42)

    env = SchEnv(instance, h=0.4, max_steps=50, seed=42)
    result = agent.solve(env, seed=42)

    assert isinstance(result, EpisodeResult)
    assert result.initial_cost > 0
    assert result.best_cost > 0


def test_constructive_better_initial(dataset):
    """Check that constructive agent starts from a better initial cost."""
    instance = dataset[0]

    # Random init: expect random initial cost
    env_random = SchEnv(instance, h=0.4, max_steps=1, seed=42)
    r_random = RandomAgent().solve(env_random, seed=42)
    random_initial = r_random.initial_cost

    # Constructive init: should have lower initial cost
    env_const = SchEnv(instance, h=0.4, max_steps=1, seed=42)
    r_const = ConstructiveRandomAgent(seed=42).solve(env_const, seed=42)
    const_initial = r_const.initial_cost

    # Constructive heuristic should produce a better starting schedule
    assert const_initial <= random_initial


def test_agent_train_noop():
    """Check that Agent.train() is a no-op for non-learning agents."""
    agent = RandomAgent()
    result = agent.train([], h=0.4)
    assert result is None


def test_agent_save_load_noop(tmp_path):
    """Check that Agent.save/load are no-ops for non-learning agents."""
    agent = RandomAgent()
    path = tmp_path / "agent.pkl"

    # Should not error
    agent.save(path)
    agent.load(path)


def test_agent_act_not_implemented():
    """Check that Agent.act() raises NotImplementedError by default."""
    agent = RandomAgent()
    with pytest.raises(NotImplementedError):
        agent.act(__import__("numpy").array([1.0, 2.0]))


def test_greedy_agent_act_not_implemented():
    """Check that GreedyAgent.act() raises NotImplementedError (requires env)."""
    agent = GreedyAgent()
    with pytest.raises(NotImplementedError):
        agent.act(__import__("numpy").array([1.0, 2.0]))


def test_all_agents_on_multiple_instances(dataset):
    """Check that all agents work across different instances."""
    for i, instance in enumerate(dataset.instances[:2]):
        for agent_class in [RandomAgent, GreedyAgent, ConstructiveRandomAgent]:
            env = SchEnv(instance, h=0.4, max_steps=20, seed=42)
            agent = agent_class(seed=42) if agent_class == ConstructiveRandomAgent else agent_class()
            result = agent.solve(env, seed=42)

            assert result.best_cost > 0
            assert result.n_steps > 0


def test_agent_construct_default_none():
    """Check that Agent.construct() returns None by default."""
    agent = RandomAgent()
    result = agent.construct(None)  # type: ignore
    assert result is None
