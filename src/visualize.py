"""
visualize.py — Visualization of Scheduling Solutions
=====================================================
Provides functions to create publication-quality visualizations of final schedules,
including Gantt charts, cost breakdowns, and improvement tracking.

Features
--------
    - Gantt chart showing job execution timeline with earliness/tardiness
    - Cost breakdown (earliness vs tardiness penalties)
    - Schedule comparison (before/after optimization)
    - Training curves (loss, improvement over time)
    - Instance heatmaps and statistics

Requirements
    pip install matplotlib numpy

Usage
-----
    from orlib_sch import load
    from sch_env import SchEnv, run_episode
    from visualize import (
        plot_gantt_chart,
        plot_cost_breakdown,
        plot_schedule_comparison,
        plot_training_curves,
    )

    # Generate a schedule
    ds = load("sch10.txt")
    env = SchEnv(ds[0], h=0.4)
    initial_schedule = list(range(env.n))
    result = run_episode(env, env.action_space_sample)

    # Visualize
    plot_gantt_chart(
        env.instance,
        result.best_schedule,
        h=0.4,
        save_path="gantt.png"
    )
    plot_cost_breakdown(env.instance, result.best_schedule, h=0.4)
    plot_schedule_comparison(
        env.instance,
        initial_schedule,
        result.best_schedule,
        h=0.4
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np

from src.benchmark import AgentBenchmarkResult

try:
    from .orlib_sch import SchInstance
except ImportError:
    from orlib_sch import SchInstance

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "early": "#2ecc71",      # Green
    "late": "#e74c3c",       # Red
    "on_time": "#f1c40f",    # Yellow
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_job_timeline(instance: SchInstance, schedule: Sequence[int], h: float) -> list[dict]:
    """
    Compute timeline data for each job in the schedule.

    Returns a list of dicts with keys:
        job_idx, rank, start, end, duration, color, early_cost, late_cost
    """
    d = instance.due_date(h)
    jobs_data = []
    time = 0

    for rank, job_idx in enumerate(schedule):
        job = instance.jobs[job_idx]
        start = time
        end = time + job.p

        # Determine color based on due-date relationship
        if end == d:
            color = _PALETTE["on_time"]
        elif end < d:
            color = _PALETTE["early"]
        else:
            color = _PALETTE["late"]

        jobs_data.append({
            "job_idx": job_idx,
            "rank": rank,
            "start": start,
            "end": end,
            "duration": job.p,
            "color": color,
            "early_cost": job.a * max(0, d - end),
            "late_cost": job.b * max(0, end - d),
        })
        time = end

    return jobs_data


def _draw_gantt_on_ax(ax: matplotlib.axes.Axes, instance: SchInstance, schedule: Sequence[int], h: float) -> None:
    """Draw a Gantt chart onto an existing axes."""
    d = instance.due_date(h)
    n = len(schedule)
    jobs_data = _compute_job_timeline(instance, schedule, h)

    # Create bars
    for data in jobs_data:
        rect = Rectangle(
            (data["start"], data["rank"] - 0.4),
            data["duration"],
            0.8,
            facecolor=data["color"],
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Add job index and processing time text
        mid_x = data["start"] + data["duration"] / 2
        ax.text(
            mid_x,
            data["rank"],
            f"J{data['job_idx']}\n(p={data['duration']})",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Due date vertical line
    ax.axvline(d, color="navy", linewidth=2.5, linestyle="--", label=f"Due date d={d}")

    # Formatting
    y_positions = list(range(n))
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(-0.5, instance.sum_p + 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Rank {i}" for i in y_positions])
    ax.set_xlabel("Completion Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Execution Order", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.8)

    # Legend
    early_patch = mpatches.Patch(color=_PALETTE["early"], label="Early (penalty: earliness)")
    late_patch = mpatches.Patch(color=_PALETTE["late"], label="Late (penalty: tardiness)")
    on_time_patch = mpatches.Patch(color=_PALETTE["on_time"], label="On-time")
    ax.legend(
        handles=[early_patch, late_patch, on_time_patch],
        loc="upper right",
        fontsize=10,
    )


# ---------------------------------------------------------------------------
# Gantt Chart Visualization
# ---------------------------------------------------------------------------


def plot_gantt_chart(
    instance: SchInstance,
    schedule: Sequence[int],
    h: float = 0.4,
    *,
    figsize: tuple[float, float] = (14, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a Gantt chart showing job execution timeline with due date marking.

    Visualizes:
        - Each job as a horizontal bar colored by earliness (green) or tardiness (red)
        - Due date as a vertical line
        - Numeric job indices and processing times on each bar

    Parameters
    ----------
    instance : SchInstance
        The scheduling problem instance.
    schedule : sequence of int
        Job permutation (order of execution).
    h : float
        Due-date tightness factor.
    figsize : tuple
        Figure size (width, height).
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    d = instance.due_date(h)
    total_cost = instance.evaluate(schedule, h)

    # Draw Gantt chart using shared helper
    _draw_gantt_on_ax(ax, instance, schedule, h)

    # Add title with total cost
    title = f"Gantt Chart  (n={instance.n}, h={h}, d={d})"
    title += f"\nTotal Cost = {total_cost}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved Gantt chart to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# Cost Breakdown Visualization
# ---------------------------------------------------------------------------


def plot_cost_breakdown(
    results: "dict[str, AgentBenchmarkResult]",
    instance: SchInstance,
    h: float = 0.4,
    *,
    instance_index: int = 0,
    figsize: tuple[float, float] = (14, 10),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Multi-agent cost breakdown for a single instance.

    Aggregates per-job costs across all agents' best schedules for
    *instance_index*, then plots four subplots:

    1. Per-job mean cost bar chart (±std across agents, sorted descending).
    2. Deletion candidate pie chart — top-*k* costliest jobs exploded out,
       remainder grouped as "Others".
    3. Cumulative cost band — mean ± std across agents over schedule position.
    4. Deletion candidate table — ranked by mean cost with early/late split
       and how often each job appears in top-k across agents.

    Parameters
    ----------
    results : dict[str, AgentBenchmarkResult]
        Output of BenchmarkRunner.run().
    instance : SchInstance
        The instance to analyse (should match *instance_index*).
    h : float
        Due-date tightness factor.
    instance_index : int
        Index into each agent's result list.
    top_k : int
        Number of jobs flagged as deletion candidates (default 5).
    figsize : tuple
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    # ------------------------------------------------------------------
    # Aggregate per-job costs across agents
    # ------------------------------------------------------------------
    # job_data[job_id] = {"early": [...], "late": [...]}
    # aggregated only over instance_index for bar/heatmap/scatter
    job_data: dict[int, dict[str, list[float]]] = {
        j: {"early": [], "late": []} for j in range(instance.n)
    }

    # per_job_traces[agent_name] = list of per-instance per-job-cost arrays
    # each inner list has length = n jobs (cost at schedule position k)
    per_job_traces: dict[str, list[list[float]]] = {name: [] for name in results}

    for name, agent_result in results.items():
        # heatmap / bar / scatter: use instance_index only
        sched_ref = agent_result.results[instance_index].best_schedule
        for entry in _compute_job_timeline(instance, sched_ref, h):
            job_data[entry["job_idx"]]["early"].append(entry["early_cost"])
            job_data[entry["job_idx"]]["late"].append(entry["late_cost"])

        # per-job traces: one line per instance
        for ep in agent_result.results:
            timeline = _compute_job_timeline(instance, ep.best_schedule, h)
            per_job_traces[name].append(
                [entry["early_cost"] + entry["late_cost"] for entry in timeline]
            )

    n_agents = len(results)
    job_ids = list(range(instance.n))
    mean_early = np.array([np.mean(job_data[j]["early"]) for j in job_ids])
    std_early  = np.array([np.std(job_data[j]["early"])  for j in job_ids])
    mean_late  = np.array([np.mean(job_data[j]["late"])  for j in job_ids])
    std_late   = np.array([np.std(job_data[j]["late"])   for j in job_ids])
    mean_total = mean_early + mean_late
    std_total  = np.sqrt(std_early**2 + std_late**2)

    # Sort jobs by mean total cost descending
    order = np.argsort(mean_total)[::-1]
    sorted_ids    = [job_ids[i] for i in order]
    sorted_mean_e = mean_early[order]
    sorted_std_e  = std_early[order]
    sorted_mean_l = mean_late[order]
    sorted_std_l  = std_late[order]
    sorted_mean_t = mean_total[order]
    sorted_std_t  = std_total[order]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    # --- Subplot 1: per-job mean cost bar (sorted) ---
    ax = axes[0]
    x = np.arange(instance.n)
    ax.bar(x, sorted_mean_e, 0.7, label="Earliness", color="#3498db",
           yerr=sorted_std_e, capsize=2, error_kw={"linewidth": 0.8})
    ax.bar(x, sorted_mean_l, 0.7, bottom=sorted_mean_e, label="Tardiness",
           color="#e67e22", yerr=sorted_std_l, capsize=2,
           error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{j}" for j in sorted_ids], fontsize=7, rotation=45)
    ax.set_xlabel("Job (sorted by mean cost)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean penalty cost", fontsize=10, fontweight="bold")
    ax.set_title("Per-Job Cost Across Agents (±std)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Subplot 2: per-job cost heatmap (jobs × agents) ---
    ax = axes[1]
    agent_names = list(results.keys())
    # matrix: rows = jobs sorted by mean cost, cols = agents
    heatmap = np.zeros((instance.n, n_agents), dtype=float)
    for a_idx, agent_result in enumerate(results.values()):
        schedule = agent_result.results[instance_index].best_schedule
        timeline = _compute_job_timeline(instance, schedule, h)
        for entry in timeline:
            heatmap[order.tolist().index(entry["job_idx"]), a_idx] = (
                entry["early_cost"] + entry["late_cost"]
            )
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels(agent_names, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(instance.n))
    ax.set_yticklabels([f"J{j}" for j in sorted_ids], fontsize=7)
    ax.set_xlabel("Agent", fontsize=10, fontweight="bold")
    ax.set_ylabel("Job (sorted by mean cost ↓)", fontsize=10, fontweight="bold")
    ax.set_title("Per-Job Cost Heatmap", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cost")

    # --- Subplot 3: per-job cost — one translucent line per instance, coloured by agent ---
    ax = axes[2]
    xs = np.arange(1, instance.n + 1)
    for a_idx, (name, traces) in enumerate(per_job_traces.items()):
        color = cmap(a_idx % 10)
        for inst_trace in traces:
            ax.plot(xs, inst_trace, color=color, linewidth=1.0, alpha=0.35)
        # bold mean line on top
        if traces:
            mean_trace = np.mean(traces, axis=0)
            ax.plot(xs, mean_trace, color=color, linewidth=2.2,
                    label=name, zorder=3)
    ax.set_xlabel("Schedule position", fontsize=10, fontweight="bold")
    ax.set_ylabel("Per-job cost", fontsize=10, fontweight="bold")
    ax.set_title("Per-Job Cost per Instance (mean bold)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Subplot 4: earliness vs tardiness scatter (one point per job) ---
    ax = axes[3]
    ax.scatter(mean_early, mean_late, s=60, color="#8e44ad", alpha=0.8, zorder=3)
    for j, (xe, xl) in enumerate(zip(mean_early, mean_late)):
        ax.annotate(
            f"J{j}", (xe, xl),
            textcoords="offset points", xytext=(4, 4),
            fontsize=7, color="#2c3e50",
        )
    # Quadrant guidelines at medians
    ax.axvline(float(np.median(mean_early)), color="#3498db", linewidth=0.8,
               linestyle="--", alpha=0.6, label="median earliness")
    ax.axhline(float(np.median(mean_late)),  color="#e67e22", linewidth=0.8,
               linestyle="--", alpha=0.6, label="median tardiness")
    ax.set_xlabel("Mean earliness cost", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean tardiness cost", fontsize=10, fontweight="bold")
    ax.set_title("Job Penalty Profile (Earliness vs Tardiness)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    all_finals = [traces[-1][-1] for traces in per_job_traces.values() if traces]
    mean_total_cost = float(np.mean(all_finals)) if all_finals else 0
    fig.suptitle(
        f"Cost Breakdown: Instance {instance.index}, n={instance.n}, h={h}  |  "
        f"{n_agents} agent(s)  |  Mean total cost = {mean_total_cost:.0f}",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved cost breakdown to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, axes


# ---------------------------------------------------------------------------
# Schedule Comparison Visualization
# ---------------------------------------------------------------------------


def plot_schedule_comparison(
    results: "dict[str, AgentBenchmarkResult]",
    instance: SchInstance,
    h: float = 0.4,
    *,
    instance_index: int = 0,
    figsize: tuple[float, float] = (16, 7),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Compare the initial schedule against the best agent's optimized schedule.

    Picks the agent with the lowest ``best_cost`` on *instance_index*, then
    draws two Gantt charts: initial (identity order) on top, best schedule
    below.  Each subplot title shows the agent name and cost score.

    Parameters
    ----------
    results : dict[str, AgentBenchmarkResult]
        Output of BenchmarkRunner.run().
    instance : SchInstance
        The scheduling problem instance to visualize.
    h : float
        Due-date tightness factor.
    instance_index : int
        Which per-instance result to use when selecting the best agent.
    figsize : tuple
        Figure size (width, height).
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    best_agent_name = min(
        results,
        key=lambda name: results[name].results[instance_index].best_cost,
    )
    best_result = results[best_agent_name].results[instance_index]
    optimized_schedule = best_result.best_schedule
    initial_schedule = list(range(instance.n))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    n = len(initial_schedule)
    initial_cost = instance.evaluate(initial_schedule, h)
    final_cost = best_result.best_cost

    for ax, schedule, label in [
        (ax1, initial_schedule, f"Initial  |  Cost = {initial_cost}"),
        (ax2, optimized_schedule, f"{best_agent_name}  |  Cost = {final_cost}"),
    ]:
        _draw_gantt_on_ax(ax, instance, schedule, h)
        ax.set_yticks(list(range(min(n, 20))))
        ax.set_ylabel("Job Order", fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold")

    ax2.set_xlabel("Completion Time", fontsize=10, fontweight="bold")

    improvement = initial_cost - final_cost
    improvement_pct = 100.0 * improvement / max(initial_cost, 1)

    fig.suptitle(
        f"Schedule Comparison: Instance {instance.index}, n={instance.n}, h={h}\n"
        f"Best agent: {best_agent_name}  |  "
        f"Improvement: {improvement:+d} ({improvement_pct:+.1f}%)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    early_patch = mpatches.Patch(color="#2ecc71", label="Early")
    late_patch = mpatches.Patch(color="#e74c3c", label="Late")
    ax1.legend(handles=[early_patch, late_patch], loc="upper right", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved schedule comparison to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, [ax1, ax2]


# ---------------------------------------------------------------------------
# Training Curves Visualization
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_log: dict | object,
    *,
    figsize: tuple[float, float] = (14, 10),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot training metrics and evaluation curves from a training log.

    Assumes train_log has attributes or dict keys:
        - loss_total, loss_policy, loss_value, loss_entropy (lists of floats)
        - eval_update, eval_best, eval_improve (lists/arrays)

    Parameters
    ----------
    train_log : dict or object (e.g., TrainLog)
        Training statistics with loss curves and evaluation metrics.
    figsize : tuple
        Figure size (width, height).
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Helper to extract values (dict or object attribute)
    def get_value(obj, key):
        if isinstance(obj, dict):
            return obj.get(key, [])
        else:
            return getattr(obj, key, [])

    # --- Subplot 1: Loss curves ---
    ax = axes[0]
    loss_total = get_value(train_log, "loss_total")
    loss_policy = get_value(train_log, "loss_policy")
    loss_value = get_value(train_log, "loss_value")
    loss_entropy = get_value(train_log, "loss_entropy")
    updates = list(range(1, len(loss_total) + 1))

    if loss_total:
        ax.plot(updates, loss_total, linewidth=2, label="Total", color="#2c3e50")
        ax.plot(updates, loss_policy, linewidth=1.5, label="Policy", alpha=0.7, color="#e74c3c")
        ax.plot(updates, loss_value, linewidth=1.5, label="Value", alpha=0.7, color="#3498db")
        ax.plot(updates, loss_entropy, linewidth=1.5, label="Entropy", alpha=0.7, color="#f39c12")
    ax.set_xlabel("Update Step", fontsize=10, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax.set_title("Training Loss Curves", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Subplot 2: Best cost over evaluation steps ---
    ax = axes[1]
    eval_updates = get_value(train_log, "eval_update")
    eval_best = get_value(train_log, "eval_best")
    if eval_best:
        ax.plot(eval_updates, eval_best, marker="o", linewidth=2, markersize=6, color="#e74c3c")
        ax.fill_between(eval_updates, eval_best, alpha=0.2, color="#e74c3c")
    ax.set_xlabel("Update Step", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Best Cost", fontsize=10, fontweight="bold")
    ax.set_title("Evaluation: Best Cost Trend", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # --- Subplot 3: Improvement percentage over time ---
    ax = axes[2]
    eval_improve = get_value(train_log, "eval_improve")
    if eval_improve:
        ax.plot(eval_updates, eval_improve, marker="s", linewidth=2, markersize=6, color="#27ae60")
        ax.fill_between(eval_updates, eval_improve, alpha=0.2, color="#27ae60")
    ax.set_xlabel("Update Step", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Improvement (%)", fontsize=10, fontweight="bold")
    ax.set_title("Evaluation: Solution Improvement", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # --- Subplot 4: Summary statistics ---
    ax = axes[3]
    ax.axis("off")
    stats_text = "Training Summary\n" + "=" * 40 + "\n"
    if loss_total:
        stats_text += f"Total updates: {len(loss_total)}\n"
        stats_text += f"Final loss: {loss_total[-1]:.4f}\n"
    if eval_best:
        stats_text += f"\nEvaluation intervals: {len(eval_best)}\n"
        stats_text += f"Best avg cost: {min(eval_best):.1f}\n"
        stats_text += f"Final avg cost: {eval_best[-1]:.1f}\n"
    if eval_improve:
        stats_text += f"\nBest improvement: {max(eval_improve):.2f}%\n"
        stats_text += f"Final improvement: {eval_improve[-1]:.2f}%\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("PPO Training Progress", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved training curves to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, axes


# ---------------------------------------------------------------------------
# Agent Action Animation
# ---------------------------------------------------------------------------


def animate_agent_actions(
    agent,
    instance: SchInstance,
    h: float = 0.4,
    *,
    interval: int = 500,
    figsize: tuple[float, float] = (14, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animate the sequence of swaps taken by an agent during optimization.

    Replays the schedule evolution by applying each swap in ``agent.actions``
    step-by-step and redrawing the Gantt chart at each frame.

    Parameters
    ----------
    agent : Agent
        An agent with ``.initial_schedule`` and ``.actions`` attributes
        (recorded by Agent.solve()).
    instance : SchInstance
        The scheduling problem instance.
    h : float
        Due-date tightness factor.
    interval : int
        Milliseconds between frames (default 500).
    figsize : tuple
        Figure size (width, height).
    save_path : str | Path | None
        If provided, save the animation to this path (e.g., "animation.gif").
    show : bool
        If True, display the animation.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object (can be saved or displayed).
    """
    if not hasattr(agent, "actions") or not hasattr(agent, "initial_schedule"):
        raise ValueError("Agent must have .actions and .initial_schedule attributes")

    if not agent.actions:
        raise ValueError("Agent has no recorded actions — call agent.solve() first")

    # Reconstruct schedule states by applying swaps incrementally
    schedules = [agent.initial_schedule[:]]  # Frame 0: initial schedule

    current = agent.initial_schedule[:]
    for i, j in agent.actions:
        current = current[:]  # Make a copy before mutating
        current[i], current[j] = current[j], current[i]
        schedules.append(current)

    # Precompute costs
    costs = [instance.evaluate(sched, h) for sched in schedules]
    initial_cost = costs[0]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    # Draw first frame (initialization)
    def init_func():
        ax.clear()
        return []

    # Update function for each frame
    def update(frame_idx):
        ax.clear()
        schedule = schedules[frame_idx]
        cost = costs[frame_idx]

        # Draw Gantt chart
        _draw_gantt_on_ax(ax, instance, schedule, h)

        # Update title with step info
        n_steps = len(agent.actions)
        delta = cost - initial_cost
        ax.set_title(
            f"Step {frame_idx}/{n_steps}  |  Cost: {cost}  (Δ {delta:+d})",
            fontsize=14,
            fontweight="bold",
        )

        return []

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(schedules),
        init_func=init_func,
        interval=interval,
        blit=False,
        repeat=True,
    )

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        try:
            anim.save(str(save_path), dpi=100)
            print(f"[visualize] Saved animation to {save_path}")
        except Exception as e:
            print(f"[visualize] Failed to save animation: {e}")

    if show:
        plt.show()
    plt.close(fig)

    return anim


# ---------------------------------------------------------------------------
# Benchmark comparison plots
# ---------------------------------------------------------------------------


def plot_convergence_curves(
    results: dict[str, AgentBenchmarkResult],
    title: str = "Cost convergence",
    figsize: tuple[float, float] = (12, 5),
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot mean convergence curve ± std band across all instances for each agent.

    For each agent all per-instance ``cost_history`` traces are aligned to a
    common step axis by padding shorter traces with their final value.  The
    mean cost at each step is drawn as the main line; a shaded band shows
    ±1 standard deviation across instances.

    Parameters
    ----------
    results : dict[str, AgentBenchmarkResult]
        Mapping of agent name → AgentBenchmarkResult as returned by BenchmarkRunner.run().
    title : str
    figsize : tuple
    show : bool
    save_path : str | None

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for idx, (name, agent_result) in enumerate(results.items()):
        histories = [r.cost_history for r in agent_result.results if r.cost_history]
        if not histories:
            continue

        max_len = max(len(h) for h in histories)
        # Pad shorter traces with their last value so all have equal length
        padded = np.array(
            [h + [h[-1]] * (max_len - len(h)) for h in histories],
            dtype=float,
        )  # shape: (n_instances, max_len)

        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        xs = np.arange(max_len)
        color = cmap(idx % 10)

        best = padded.min(axis=0)

        ax.plot(xs, mean, label=f"{name} (mean)", color=color, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2, label=f"{name} (±1σ)")
        ax.plot(xs, best, color=color, linewidth=1.0, linestyle="--", alpha=0.7, label=f"{name} (best)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_agent_comparison(
    results: dict,
    figsize: tuple[float, float] = (12, 6),
    show: bool = True,
    save_path: str | None = None,
):
    """
    Side-by-side comparison of agents: bar chart of mean improvement% and
    box plot of best-cost distribution.

    Parameters
    ----------
    results : dict[str, AgentBenchmarkResult]
        Output of BenchmarkRunner.run().
    figsize : tuple
    show : bool
    save_path : str | None

    Returns
    -------
    plt.Figure
    """
    names = list(results.keys())
    means = [results[n].mean_improvement_pct for n in names]
    stds = [results[n].std_improvement_pct for n in names]
    best_costs = [[r.best_cost for r in results[n].results] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(names))]

    # --- Left: mean improvement % bar chart ---
    bars = ax1.barh(names, means, xerr=stds, color=colors, capsize=4, alpha=0.85)
    ax1.set_xlabel("Mean improvement % (higher is better)")
    ax1.set_title("Schedule improvement by agent")
    ax1.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, means):
        ax1.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )

    # --- Right: best-cost box plot ---
    bp = ax2.boxplot(best_costs, labels=names, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Best cost (lower is better)")
    ax2.set_title("Best cost distribution")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# GEPA optimisation history plots
# ---------------------------------------------------------------------------


def plot_gepa_history(
    gepa_agent,
    *,
    seed_result: "EpisodeResult | None" = None,
    best_result: "EpisodeResult | None" = None,
    figsize: tuple[float, float] = (16, 12),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Visualise the GEPA hyperparameter optimisation history.

    Produces four subplots:

    1. **Score per call** — scatter of mean improvement% at every GEPA
       evaluate() call, with the cumulative-best line overlaid.
    2. **Per-instance score spread** — box plots (one per call) showing
       variance across the training batch.
    3. **Config parameter evolution** — one line per hyperparameter, showing
       how the LLM mutates values over the search.
    4. **Baseline vs best** — bar chart comparing seed-config and best-config
       improvement% side by side (requires *seed_result* and *best_result*).

    Parameters
    ----------
    gepa_agent : GEPAAgent
        A trained GEPAAgent with a populated ``history_`` attribute.
    seed_result : EpisodeResult | None
        Episode result from the seed config on a test instance, for comparison.
    best_result : EpisodeResult | None
        Episode result from the best found config on the same instance.
    figsize : tuple
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    history = gepa_agent.history_
    if not history:
        raise ValueError("GEPAAgent.history_ is empty — call train() first.")

    call_idxs = [e["call_idx"] for e in history]
    mean_scores = [e["mean_score"] * 100.0 for e in history]  # back to percent
    best_so_far = [e["best_so_far"] * 100.0 for e in history]
    all_scores = [
        [s * 100.0 for s in e["scores"]] for e in history
    ]

    # Collect param names (exclude file-path strings)
    param_keys = [
        k for k, v in history[0]["config_params"].items()
        if isinstance(v, (int, float))
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    # --- Subplot 1: mean score per call + cumulative best ---
    ax = axes[0]
    ax.scatter(call_idxs, mean_scores, s=40, color="#3498db", alpha=0.7, zorder=3, label="mean score")
    ax.plot(call_idxs, best_so_far, color="#e74c3c", linewidth=2.0, label="cumulative best")
    ax.set_xlabel("GEPA call index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean improvement %", fontsize=10, fontweight="bold")
    ax.set_title("Score per GEPA evaluation call", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if best_so_far:
        ax.annotate(
            f"best: {max(best_so_far):.1f}%",
            xy=(call_idxs[best_so_far.index(max(best_so_far))], max(best_so_far)),
            xytext=(10, -15), textcoords="offset points",
            fontsize=8, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
        )

    # --- Subplot 2: per-instance score spread as box plots ---
    ax = axes[1]
    bp = ax.boxplot(
        all_scores,
        positions=call_idxs,
        widths=max(1, len(call_idxs) // 20) * 0.6,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
    ax.plot(call_idxs, mean_scores, color="#2c3e50", linewidth=1.2, alpha=0.5, label="mean")
    ax.set_xlabel("GEPA call index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Improvement % (per instance)", fontsize=10, fontweight="bold")
    ax.set_title("Per-instance score spread across calls", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Subplot 3: config parameter evolution ---
    ax = axes[2]
    for p_idx, key in enumerate(param_keys):
        values = [e["config_params"].get(key) for e in history]
        if any(v is None for v in values):
            continue
        # Normalise each param to [0,1] for readability on a shared axis
        lo, hi = min(values), max(values)
        norm = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in values]
        color = cmap(p_idx % 10)
        ax.plot(call_idxs, norm, color=color, linewidth=1.6, label=key, marker=".", markersize=4)
    ax.set_xlabel("GEPA call index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Normalised parameter value [0, 1]", fontsize=10, fontweight="bold")
    ax.set_title("Hyperparameter evolution (each param normalised)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # --- Subplot 4: baseline vs best comparison (or summary if no results given) ---
    ax = axes[3]
    if seed_result is not None and best_result is not None:
        labels = ["Seed config", "GEPA best"]
        improv = [seed_result.improvement_pct, best_result.improvement_pct]
        colors = ["#95a5a6", "#27ae60"]
        bars = ax.bar(labels, improv, color=colors, alpha=0.85, width=0.4)
        for bar, val in zip(bars, improv):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_ylabel("Improvement %", fontsize=10, fontweight="bold")
        ax.set_title("Seed config vs GEPA best", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    else:
        # Show summary stats text instead
        ax.axis("off")
        n_calls = len(history)
        best_pct = max(best_so_far)
        first_pct = mean_scores[0] if mean_scores else 0.0
        gain = best_pct - first_pct
        best_params = history[best_so_far.index(max(best_so_far))]["config_params"]
        lines = [
            "GEPA search summary",
            "=" * 32,
            f"Total evaluate() calls : {n_calls}",
            f"First mean score       : {first_pct:.2f}%",
            f"Best mean score        : {best_pct:.2f}%",
            f"Gain over seed         : {gain:+.2f}%",
            "",
            "Best config params:",
        ]
        for k, v in best_params.items():
            if isinstance(v, float):
                lines.append(f"  {k:20s}: {v:.6g}")
            else:
                lines.append(f"  {k:20s}: {v}")
        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    agent_label = getattr(gepa_agent, "name", type(gepa_agent).__name__)
    fig.suptitle(
        f"GEPA optimisation history — {agent_label}  ({len(history)} calls)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved GEPA history to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, list(axes)


# ---------------------------------------------------------------------------
# Demo / Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    try:
        from .orlib_sch import load
        from .sch_env import SchEnv
        from .agent import GreedyAgent
    except ImportError:
        # Fallback for direct execution
        from orlib_sch import load
        from sch_env import SchEnv
        from agent import GreedyAgent

    print("=" * 60)
    print("Visualization Demo")
    print("=" * 60)

    # Load a small dataset
    try:
        ds = load("data/sch10.txt")
    except FileNotFoundError:
        print("Note: data/sch10.txt not found. Skipping demo.")
        sys.exit(0)

    # Run optimization with GreedyAgent on first instance
    inst = ds[0]
    env = SchEnv(inst, h=0.4, max_steps=100, seed=42)

    agent = GreedyAgent()
    result = agent.solve(env, seed=42)

    print(f"\nInstance {inst.index}:")
    print(f"  n={inst.n}, sum_p={inst.sum_p}")
    print(f"  Initial cost: {agent.initial_schedule[0]}")  # Compute from initial_schedule
    print(f"  Final cost:   {result.best_cost}")
    print(f"  Improvement:  {result.improvement_pct:.1f}%")

    # Create visualizations
    print("\nGenerating visualizations...")

    # plot_gantt_chart(inst, result.best_schedule, h=0.4, show=True)
    # print("  ✓ Gantt chart")

    # plot_cost_breakdown(inst, result.best_schedule, h=0.4, show=True)
    # print("  ✓ Cost breakdown")

    # plot_schedule_comparison(inst, agent.initial_schedule, result.best_schedule, h=0.4, show=True)
    # print("  ✓ Schedule comparison")

    animate_agent_actions(env, inst)
    print("  ✓ Agent actions animation")

    print("\nDemo complete!")
