#!/usr/bin/env python3
"""
Sanity checks for benchmark.py — BenchmarkRunner and AgentBenchmarkResult.

Run with: pytest sanity/sanity_benchmark.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load, RandomAgent, GreedyAgent
from src.benchmark import BenchmarkRunner, AgentBenchmarkResult
from src.sch_env import EpisodeResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset():
    return load("data/sch10.txt")


@pytest.fixture
def instances(dataset):
    return dataset.instances[:2]


@pytest.fixture
def agents():
    return {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
    }


@pytest.fixture
def runner(instances):
    return BenchmarkRunner(instances, h=0.4, max_steps=20, seed=42)


@pytest.fixture
def results(runner, agents):
    return runner.run(agents)


# ---------------------------------------------------------------------------
# AgentBenchmarkResult
# ---------------------------------------------------------------------------


def test_agent_benchmark_result_fields(results):
    for name, abr in results.items():
        assert abr.agent_name == name
        assert isinstance(abr.mean_improvement_pct, float)
        assert isinstance(abr.mean_best_cost, (int, float))
        assert isinstance(abr.std_improvement_pct, float)
        assert isinstance(abr.mean_n_steps, (int, float))


def test_agent_benchmark_result_aggregation():
    """Stats are computed correctly from synthetic EpisodeResults."""
    from src.sch_env import EpisodeResult

    fake_results = [
        EpisodeResult(
            instance_index=i, h=0.4, initial_cost=100, final_cost=80,
            best_cost=70, total_reward=1.0, n_steps=10,
            n_improvements=3, improvement_pct=float(10 * i),
            best_schedule=list(range(5)),
        )
        for i in range(1, 4)  # improvement_pct: 10, 20, 30
    ]
    abr = AgentBenchmarkResult("TestAgent", fake_results)
    assert abr.mean_improvement_pct == pytest.approx(20.0)
    assert abr.mean_best_cost == pytest.approx(70.0)
    assert abr.mean_n_steps == pytest.approx(10.0)


def test_agent_benchmark_result_repr(results):
    abr = results["Random"]
    r = repr(abr)
    assert "Random" in r
    assert "mean_improvement" in r


# ---------------------------------------------------------------------------
# BenchmarkRunner.run
# ---------------------------------------------------------------------------


def test_run_returns_all_agents(results, agents):
    assert set(results.keys()) == set(agents.keys())


def test_run_result_counts(results, instances):
    for abr in results.values():
        assert len(abr.results) == len(instances)


def test_run_results_are_episode_results(results):
    for abr in results.values():
        for r in abr.results:
            assert isinstance(r, EpisodeResult)


def test_run_best_cost_non_negative(results):
    for abr in results.values():
        for r in abr.results:
            assert r.best_cost >= 0


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_shape(runner, results, instances, agents):
    df = runner.to_dataframe(results)
    assert len(df) == len(instances) * len(agents)


def test_to_dataframe_columns(runner, results):
    df = runner.to_dataframe(results)
    expected = {"agent", "instance_index", "initial_cost", "best_cost",
                "improvement_pct", "n_steps", "n_improvements", "total_reward"}
    assert expected.issubset(set(df.columns))


def test_to_dataframe_agent_values(runner, results, agents):
    df = runner.to_dataframe(results)
    assert set(df["agent"].unique()) == set(agents.keys())


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------


def test_save_csv_creates_file(runner, results, instances, agents):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.csv"
        runner.save_csv(results, path)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        # header + one row per agent×instance
        assert len(lines) == 1 + len(instances) * len(agents)


def test_save_csv_has_header(runner, results):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.csv"
        runner.save_csv(results, path)
        header = path.read_text().splitlines()[0]
        assert "agent" in header
        assert "improvement_pct" in header


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


def test_print_summary_no_crash(runner, results, capsys):
    runner.print_summary(results)
    captured = capsys.readouterr()
    assert "Random" in captured.out
    assert "Greedy" in captured.out
