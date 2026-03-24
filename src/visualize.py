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
    instance: SchInstance,
    schedule: Sequence[int],
    h: float = 0.4,
    *,
    figsize: tuple[float, float] = (14, 10),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create a cost breakdown visualization with multiple subplots.

    Subplots:
        1. Stacked bar chart of per-job costs (earliness vs tardiness)
        2. Cumulative cost over the schedule timeline
        3. Cost distribution across jobs

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
    axes : list of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    d = instance.due_date(h)
    total_cost = instance.evaluate(schedule, h)

    # Collect per-job cost data
    time = 0
    job_costs_early = []
    job_costs_late = []
    job_indices = []
    completions = []
    cumulative_costs = []
    cumulative = 0

    for job_idx in schedule:
        job = instance.jobs[job_idx]
        time += job.p
        early = max(0, d - time)
        late = max(0, time - d)
        cost_early = job.a * early
        cost_late = job.b * late
        job_cost = cost_early + cost_late

        job_indices.append(job_idx)
        job_costs_early.append(cost_early)
        job_costs_late.append(cost_late)
        completions.append(time)
        cumulative += job_cost
        cumulative_costs.append(cumulative)

    # --- Subplot 1: Stacked bar chart of per-job costs ---
    ax = axes[0]
    x = np.arange(len(schedule))
    width = 0.7
    bars_early = ax.bar(x, job_costs_early, width, label="Earliness cost", color="#3498db")
    bars_late = ax.bar(x, job_costs_late, width, bottom=job_costs_early, label="Tardiness cost", color="#e67e22")

    ax.set_xlabel("Job Order Position", fontsize=10, fontweight="bold")
    ax.set_ylabel("Penalty Cost", fontsize=10, fontweight="bold")
    ax.set_title("Per-Job Cost Breakdown", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Subplot 2: Cumulative cost over time ---
    ax = axes[1]
    ax.plot(completions, cumulative_costs, marker="o", linewidth=2, markersize=6, color="#e74c3c")
    ax.axhline(total_cost, color="black", linestyle="--", linewidth=1, label=f"Final cost = {total_cost}")
    ax.fill_between(completions, cumulative_costs, alpha=0.3, color="#e74c3c")
    ax.set_xlabel("Completion Time", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Cost", fontsize=10, fontweight="bold")
    ax.set_title("Cumulative Cost Over Time", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Subplot 3: Cost distribution (pie chart) ---
    ax = axes[2]
    total_early = sum(job_costs_early)
    total_late = sum(job_costs_late)
    sizes = [total_early, total_late]
    labels = [f"Earliness\n{total_early}", f"Tardiness\n{total_late}"]
    colors = ["#3498db", "#e67e22"]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9, "fontweight": "bold"},
    )
    ax.set_title("Cost Composition", fontsize=11, fontweight="bold")

    # --- Subplot 4: Top 10 highest-cost jobs ---
    ax = axes[3]
    job_total_costs = [job_costs_early[i] + job_costs_late[i] for i in range(len(schedule))]
    top_k = min(10, len(schedule))
    top_indices = np.argsort(job_total_costs)[-top_k:][::-1]
    top_jobs = [schedule[i] for i in top_indices]
    top_costs = [job_total_costs[i] for i in top_indices]
    top_y_labels = [f"J{j}" for j in top_jobs]

    ax.barh(range(len(top_jobs)), top_costs, color="#9b59b6")
    ax.set_yticks(range(len(top_jobs)))
    ax.set_yticklabels(top_y_labels)
    ax.set_xlabel("Total Cost", fontsize=10, fontweight="bold")
    ax.set_title(f"Top {top_k} Highest-Cost Jobs", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Cost Analysis: Instance {instance.index}, n={instance.n}, h={h}\n"
        f"Total Cost = {total_cost}",
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
    instance: SchInstance,
    initial_schedule: Sequence[int],
    optimized_schedule: Sequence[int],
    h: float = 0.4,
    *,
    figsize: tuple[float, float] = (16, 7),
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Compare two schedules side-by-side using Gantt charts.

    Parameters
    ----------
    instance : SchInstance
        The scheduling problem instance.
    initial_schedule : sequence of int
        The starting schedule (typically random or identity).
    optimized_schedule : sequence of int
        The final optimized schedule.
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
    axes : list of matplotlib.axes.Axes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    n = len(initial_schedule)

    for ax, schedule, title_suffix in [
        (ax1, initial_schedule, "Initial"),
        (ax2, optimized_schedule, "Optimized"),
    ]:
        # Draw Gantt chart using shared helper
        _draw_gantt_on_ax(ax, instance, schedule, h)

        # Override some formatting for comparison view
        ax.set_yticks(list(range(min(n, 20))))  # Limit labels for readability
        ax.set_ylabel("Job Order", fontsize=10, fontweight="bold")

        cost = instance.evaluate(schedule, h)
        ax.set_title(f"{title_suffix} Schedule (Cost = {cost})", fontsize=11, fontweight="bold")

    ax2.set_xlabel("Completion Time", fontsize=10, fontweight="bold")

    # Overall title with improvement
    initial_cost = instance.evaluate(initial_schedule, h)
    final_cost = instance.evaluate(optimized_schedule, h)
    improvement = initial_cost - final_cost
    improvement_pct = 100.0 * improvement / max(initial_cost, 1)

    fig.suptitle(
        f"Schedule Comparison: Instance {instance.index}, n={instance.n}, h={h}\n"
        f"Initial Cost = {initial_cost}  →  Final Cost = {final_cost}  "
        f"(Improvement: {improvement:+d}, {improvement_pct:+.1f}%)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    # Legend
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
# Multi-instance Summary Grid
# ---------------------------------------------------------------------------


def plot_instance_summary_grid(
    instances_with_schedules: list[tuple[SchInstance, list[int]]],
    h: float = 0.4,
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create a grid of small Gantt charts for multiple instances.

    Parameters
    ----------
    instances_with_schedules : list of (SchInstance, schedule)
        Pairs of instances and their optimized schedules.
    h : float
        Due-date tightness factor.
    figsize : tuple | None
        Figure size. If None, computed automatically based on number of instances.
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes (flattened)
    """
    n_instances = len(instances_with_schedules)
    grid_cols = 3
    grid_rows = (n_instances + grid_cols - 1) // grid_cols

    if figsize is None:
        figsize = (grid_cols * 5, grid_rows * 4)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (instance, schedule) in enumerate(instances_with_schedules):
        ax = axes[idx]
        cost = instance.evaluate(schedule, h)

        # Draw Gantt chart using shared helper
        _draw_gantt_on_ax(ax, instance, schedule, h)

        # Compact formatting for grid view
        ax.set_title(f"Instance {instance.index} (Cost={cost})", fontsize=9, fontweight="bold")
        ax.set_ylabel("Job Order", fontsize=8)
        ax.set_xlabel("Time", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend().remove()  # Remove legend to save space

    # Hide unused subplots
    for idx in range(n_instances, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Instance Summary Grid (h={h})", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved instance grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return fig, axes[:n_instances]


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
    agent_histories: dict[str, list[int]],
    title: str = "Cost convergence",
    figsize: tuple[float, float] = (12, 5),
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot cost-vs-step convergence curves for multiple agents.

    Parameters
    ----------
    agent_histories : dict[str, list[int]]
        Mapping of agent name → cost_history list recorded during solve().
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

    for idx, (name, history) in enumerate(agent_histories.items()):
        if not history:
            continue
        # Normalise x to [0, 1] so curves with different lengths are comparable
        n = len(history)
        xs = [i for i in range(n)]
        ax.plot(xs, history, label=name, color=cmap(idx % 10), linewidth=1.8)

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
