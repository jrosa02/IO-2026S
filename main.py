"""
main.py — CLI entry point for io-2026s scheduling benchmark.

Usage:
    python main.py [OPTIONS]

Options:
    --data PATH           Path to dataset file  [default: data/sch10.txt]
    --n-instances INT     Number of instances to benchmark  [default: 5]
    --h FLOAT             Due-date tightness parameter  [default: 0.4]
    --max-steps INT       Max steps per episode  [default: 50]
    --seed INT            RNG seed  [default: 42]
    --out-dir PATH        Output directory for CSV and plots  [default: results]
    --no-plots            Skip plot generation
    --agents LIST         Comma-separated agents to run:
                          random, greedy, sa, genetic, gepa-sa, gepa-ga
                          [default: random,greedy,sa,genetic]
    --train-instances INT Number of instances used for GEPA training  [default: 3]
    --max-metric-calls INT GEPA evaluation budget  [default: 50]
    --reflection-lm STR   LiteLLM model string for GEPA reflection LLM
                          [default: ollama/qwen3:4b-instruct-2507-q4_K_M]
    -v, --verbose         Show per-instance progress
    -h, --help            Show this message and exit
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
                        help="Comma-separated agents: random,greedy,sa,genetic,gepa-sa,gepa-ga")
    parser.add_argument("--train-instances", type=int, default=3, metavar="INT",
                        help="Instances used to train GEPA agents before benchmarking")
    parser.add_argument("--max-metric-calls", type=int, default=50, metavar="INT",
                        help="GEPA evaluation budget (total agent×instance runs)")
    parser.add_argument("--reflection-lm", default="ollama/qwen3:4b-instruct-2507-q4_K_M", metavar="STR",
                        help="LiteLLM model string for GEPA reflection LLM")
    parser.add_argument("--interactions-log", default=None, metavar="PATH",
                        help="Write raw GEPA LLM prompt/response pairs to this JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-instance progress")
    return parser.parse_args()


def build_agents(
    agent_names: list[str],
    seed: int,
    max_metric_calls: int,
    reflection_lm: str,
    interactions_log: str | None,
) -> dict:
    from src.agent import RandomAgent, GreedyAgent
    from src.classical_agents import SimulatedAnnealingAgent, GeneticAlgorithmAgent
    from src.configs import SAConfig, GAConfig
    from src.gepa_agent import GEPAAgent

    mapping = {
        "random":  ("Random",  RandomAgent()),
        "greedy":  ("Greedy",  GreedyAgent()),
        "sa":      ("SA",      SimulatedAnnealingAgent()),
        "genetic": ("Genetic", GeneticAlgorithmAgent()),
        "gepa-sa": ("GEPA-SA", GEPAAgent(
            base_agent_cls=SimulatedAnnealingAgent,
            seed_config=SAConfig(),
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            seed=seed,
            interactions_log=interactions_log,
        )),
        "gepa-ga": ("GEPA-GA", GEPAAgent(
            base_agent_cls=GeneticAlgorithmAgent,
            seed_config=GAConfig(),
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            seed=seed,
            interactions_log=interactions_log,
        )),
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


def main():
    args = parse_args()

    from src.orlib_sch import load
    from src.benchmark import BenchmarkRunner
    import src.visualize as viz

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data}...")
    ds = load(args.data)
    n = min(args.n_instances, len(ds.instances))
    instances = ds.instances[:n]
    inst0 = instances[0]

    agents = build_agents(
        args.agents.split(","),
        seed=args.seed,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=args.reflection_lm,
        interactions_log=args.interactions_log,
    )

    # Train any GEPA agents before benchmarking
    from src.gepa_agent import GEPAAgent
    gepa_agents = {label: agent for label, agent in agents.items() if isinstance(agent, GEPAAgent)}
    if gepa_agents:
        train_n = min(args.train_instances, n)
        train_instances = instances[:train_n]
        for label, agent in gepa_agents.items():
            print(f"Training {label} with GEPA on {train_n} instance(s) "
                  f"(max_metric_calls={args.max_metric_calls}, lm={args.reflection_lm})...")
            agent.train(train_instances, h=args.h)
            print(f"  {label} best config: {agent.best_config_}")

    print(f"Benchmarking {len(agents)} agent(s) on {n} instance(s) "
          f"(h={args.h}, max_steps={args.max_steps}, seed={args.seed})...")
    runner = BenchmarkRunner(instances, h=args.h, max_steps=args.max_steps, seed=args.seed)
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
