"""
main.py — CLI entry point for io-2026s scheduling benchmark.

Usage:
    python main.py [OPTIONS]

Options:
    --data PATH             Path to dataset file  [default: data/sch10.txt]
    --n-instances INT       Number of instances to benchmark  [default: 5]
    --h FLOAT               Due-date tightness parameter  [default: 0.4]
    --max-steps INT         Max steps per episode  [default: 50]
    --seed INT              RNG seed  [default: 42]
    --out-dir PATH          Output directory for CSV and plots  [default: results]
    --no-plots              Skip plot generation
    --agents LIST           Comma-separated agents to run: random,greedy,sa,genetic,gepa
                            [default: random,greedy,sa,genetic]
    -v, --verbose           Show per-instance progress
    -h, --help              Show this message and exit

GEPA-specific options (only used when 'gepa' is in --agents):
    --gepa-lm MODEL         LiteLLM model for GEPA reflection LLM, e.g.
                            'anthropic/claude-haiku-4-5' or 'openai/gpt-4o-mini'.
                            Omit (or pass 'mock') to use the mock random mutator.
                            [default: mock]
    --gepa-calls INT        Max SA evaluations GEPA may perform  [default: 40]
    --gepa-sa-steps INT     SA step budget used during GEPA search  [default: 200]
    --gepa-train-n INT      Instances reserved for GEPA training (taken from the
                            front of the dataset, before benchmark instances).
                            0 means train on the same instances used for benchmarking.
                            [default: 0]
    --gepa-run-dir PATH     Directory for GEPA logs/checkpoints.  [default: none]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Benchmark scheduling agents on common due-date instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/sch10.txt", metavar="PATH",
                        help="Path to dataset file")
    parser.add_argument("--n-instances", type=int, default=5, metavar="INT",
                        help="Number of instances to benchmark")
    parser.add_argument("--h", type=float, default=0.4, metavar="FLOAT",
                        help="Due-date tightness parameter")
    parser.add_argument("--max-steps", type=int, default=50, metavar="INT",
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, metavar="INT",
                        help="RNG seed")
    parser.add_argument("--out-dir", default="results", metavar="PATH",
                        help="Output directory for CSV and plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--agents", default="random,greedy,sa,genetic", metavar="LIST",
                        help="Comma-separated agents: random,greedy,sa,genetic,gepa")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-instance progress")

    gepa = parser.add_argument_group("GEPA options (used when 'gepa' is in --agents)")
    gepa.add_argument("--gepa-lm", default="mock", metavar="MODEL",
                      help="LiteLLM reflection model, e.g. 'anthropic/claude-haiku-4-5'. "
                           "'mock' uses random perturbation (no API key needed).")
    gepa.add_argument("--gepa-calls", type=int, default=40, metavar="INT",
                      help="Max SA evaluations GEPA may perform")
    gepa.add_argument("--gepa-sa-steps", type=int, default=200, metavar="INT",
                      help="SA step budget per GEPA evaluation")
    gepa.add_argument("--gepa-train-n", type=int, default=0, metavar="INT",
                      help="Extra instances for GEPA training (0 = train on benchmark set)")
    gepa.add_argument("--gepa-run-dir", default=None, metavar="PATH",
                      help="Directory for GEPA logs (None = disabled)")

    return parser.parse_args()


def build_agents(agent_names: list[str], seed: int, gepa_args) -> dict:
    from src.agent import GreedyAgent, RandomAgent
    from src.classical_agents import GeneticAlgorithmAgent, SimulatedAnnealingAgent
    from src.gepa_agent import GEPAAgent, GEPAAgentConfig

    reflection_lm = None if gepa_args.gepa_lm == "mock" else gepa_args.gepa_lm
    gepa_cfg = GEPAAgentConfig(
        sa_max_steps=gepa_args.gepa_sa_steps,
        max_metric_calls=gepa_args.gepa_calls,
        reflection_lm=reflection_lm,
        run_dir=gepa_args.gepa_run_dir,
        seed=seed,
    )

    mapping = {
        "random":  ("Random",  RandomAgent()),
        "greedy":  ("Greedy",  GreedyAgent()),
        "sa":      ("SA",      SimulatedAnnealingAgent()),
        "genetic": ("Genetic", GeneticAlgorithmAgent()),
        "gepa":    ("GEPA-SA", GEPAAgent(gepa_cfg)),
    }

    agents = {}
    for name in agent_names:
        key = name.strip().lower()
        if key not in mapping:
            valid = ", ".join(mapping)
            print(f"Unknown agent '{name}'. Choose from: {valid}", file=sys.stderr)
            sys.exit(1)
        label, agent = mapping[key]
        agents[label] = agent
    return agents


def train_agents(agents: dict, train_instances, h: float) -> None:
    """Call train() on any agent that requires it (e.g. GEPAAgent)."""
    from src.gepa_agent import GEPAAgent

    for label, agent in agents.items():
        if isinstance(agent, GEPAAgent):
            print(f"Training {label} on {len(train_instances)} instance(s)...")
            agent.train(train_instances, h=h)
            print(f"  Best config found: {agent.best_config_}")
            if agent.gepa_result_ is not None:
                front = agent.gepa_result_.objective_pareto_front or {}
                if front:
                    fmt = ", ".join(f"{k}={v:.3f}" for k, v in front.items())
                    print(f"  Pareto front: {fmt}")


def main():
    args = parse_args()

    from src.benchmark import BenchmarkRunner
    from src.orlib_sch import load
    import src.visualize as viz

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data}...")
    ds = load(args.data)

    agent_names = [n.strip() for n in args.agents.split(",")]
    needs_gepa = any(n.lower() == "gepa" for n in agent_names)

    # Partition instances: GEPA training set + benchmark set.
    # With --gepa-train-n 0, GEPA trains on the same set it is benchmarked on.
    gepa_train_n = args.gepa_train_n if needs_gepa else 0
    total_needed = gepa_train_n + args.n_instances
    if len(ds.instances) < total_needed:
        print(
            f"Warning: dataset has {len(ds.instances)} instances but "
            f"{total_needed} requested (--gepa-train-n {gepa_train_n} + "
            f"--n-instances {args.n_instances}). Adjusting.",
            file=sys.stderr,
        )
        gepa_train_n = max(0, len(ds.instances) - args.n_instances)
        total_needed = gepa_train_n + args.n_instances

    benchmark_instances = ds.instances[gepa_train_n: gepa_train_n + args.n_instances]
    gepa_train_instances = ds.instances[:gepa_train_n] if gepa_train_n > 0 else benchmark_instances

    agents = build_agents(agent_names, args.seed, args)

    if needs_gepa:
        train_agents(agents, gepa_train_instances, h=args.h)

    n = len(benchmark_instances)
    inst0 = benchmark_instances[0]
    print(f"\nBenchmarking {len(agents)} agent(s) on {n} instance(s) "
          f"(h={args.h}, max_steps={args.max_steps}, seed={args.seed})...")
    runner = BenchmarkRunner(benchmark_instances, h=args.h, max_steps=args.max_steps, seed=args.seed)
    results = runner.run(agents, verbose=args.verbose)

    runner.print_summary(results)

    csv_path = out_dir / "benchmark.csv"
    runner.save_csv(results, csv_path)
    print(f"CSV saved to {csv_path}")

    if args.no_plots:
        return

    print("Generating plots...")

    viz.plot_cost_breakdown(results, inst0, h=args.h, show=False,
                            save_path=str(out_dir / "cost_breakdown.png"))
    print(f"  {out_dir}/cost_breakdown.png")

    viz.plot_schedule_comparison(results, inst0, h=args.h, show=False,
                                 save_path=str(out_dir / "schedule_comparison.png"))
    print(f"  {out_dir}/schedule_comparison.png")

    viz.plot_convergence_curves(results, show=False,
                                save_path=str(out_dir / "convergence.png"))
    print(f"  {out_dir}/convergence.png")

    viz.plot_agent_comparison(results, show=False,
                              save_path=str(out_dir / "agent_comparison.png"))
    print(f"  {out_dir}/agent_comparison.png")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
