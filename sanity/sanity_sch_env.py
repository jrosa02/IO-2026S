#!/usr/bin/env python3
"""
Sanity checks for SchEnv — verify core functionality works.

Run with: pytest sanity/demo_sch_env.py -v
"""

import sys
from pathlib import Path

import pytest

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load, SchEnv, run_episode


@pytest.fixture
def dataset():
    """Load test dataset."""
    return load("data/sch10.txt")


@pytest.fixture
def env(dataset):
    """Create test environment."""
    return SchEnv(dataset[0], h=0.4, max_steps=50, seed=42)


def test_imports():
    """Check that imports work."""
    assert load is not None
    assert SchEnv is not None
    assert run_episode is not None


def test_load_dataset(dataset):
    """Check that dataset loads."""
    assert len(dataset.instances) > 0


def test_env_creation(env):
    """Check that SchEnv has required attributes."""
    assert env.obs_size > 0
    assert env.n_actions > 0
    assert env.max_steps > 0


def test_env_reset(env):
    """Check that env.reset() returns valid obs and info."""
    obs, info = env.reset()
    assert obs is not None
    assert info is not None
    assert len(obs) == env.obs_size


def test_properties(env):
    """Check that properties are accessible."""
    env.reset()
    cost = env.current_cost
    best = env.best_cost
    assert isinstance(cost, int)
    assert isinstance(best, int)


def test_step(env):
    """Check that env.step() returns valid output."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(info, dict)


def test_run_episode(env):
    """Check that run_episode() completes."""
    result = run_episode(env, policy_fn=env.action_space_sample, seed=42)
    assert result.initial_cost > 0
    assert result.best_cost > 0
    assert result.improvement_pct >= 0


def test_multiple_instances(dataset):
    """Check that env works with multiple instances."""
    for i, instance in enumerate(dataset.instances[:3]):
        env = SchEnv(instance, h=0.4, max_steps=10, seed=42)
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == env.obs_size