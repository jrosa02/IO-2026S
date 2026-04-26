# Single-Machine Scheduling with LLMs

A research project applying LLMs and classical metaheuristics to the **common due-date single-machine scheduling problem** (ORLib `sch*` instances). Agents optimize job permutations to minimize weighted earliness + tardiness cost.

## Quick Start

### Prerequisites
- Python 3.9+
- `uv` (fast package manager) — [install here](https://docs.astral.sh/uv/)
- GCC (for C extension compilation)

### Installation

```bash
# Install dependencies and compile C extensions
uv sync
```

This automatically:
- Installs Python dependencies
- Builds optimized C extensions for cost evaluation and crossover operations

## Running the Benchmark

### Basic usage (defaults: sch10, 5 instances, random+greedy+constructive agents)
```bash
python main.py
```

### With classical metaheuristics
```bash
python main.py --agents random,greedy,sa,genetic
```

### Custom dataset and parameters
```bash
python main.py --data data/sch10.txt --n-instances 10 --max-steps 100 --no-plots
```

### Full option list
```bash
python main.py --help
```

**Output:** Prints ASCII results table, exports `benchmark.csv`, and (by default) shows convergence plots.

## Performance Profiling

### Using `pyinstrument` (detailed call traces)

```bash
# Profile SA and Genetic agents on larger instances
uv run pyinstrument -r html -o logs.html main.py --agents sa,genetic --data ./data/sch1000.txt --max-steps 10000
```

This generates an interactive HTML report (`logs.html`) with flame graph and call stack analysis.

### Using `py-spy` (low-overhead sampling profiler)

```bash
# Profile Genetic agent with 500Hz sampling, saves flamegraph SVG
uv run py-spy record -r 500 -o profile.svg -- python main.py --agents genetic --interactions-log ./logs --data ./data/sch1000.txt --max-steps 20000
```

Options:
- `-r 500` — Sample at 500Hz (adjust for finer/coarser granularity)
- `-o profile.svg` — Output SVG flamegraph (open in browser)
- `--interactions-log ./logs` — Log agent interactions to directory

Use `py-spy` for low-overhead, wall-clock time profiling; use `pyinstrument` for detailed call-level analysis.

## Testing

### Run all tests
```bash
python -m pytest sanity/ -v
```

### Run specific test file
```bash
python -m pytest sanity/sanity_sch_env.py -v
```

### Run a single test
```bash
python -m pytest sanity/sanity_agent.py::test_greedy_agent_solve -v
```

Tests use small instances (10 jobs, 20–30 steps) for speed (~2s total).

## Code Quality

### Linting
```bash
uv run ruff check src/ sanity/
```

### Auto-format
```bash
uv run ruff format src/ sanity/
```

## Available Agents

| Agent | Type | Notes |
|---|---|---|
| **RandomAgent** | Iterative | Uniform random swaps |
| **GreedyAgent** | Iterative | 1-step lookahead |
| **ConstructiveRandomAgent** | Constructive | Heuristic sort + random swaps |
| **SimulatedAnnealingAgent** | Classical | Oscillating temperature schedule |
| **GeneticAlgorithmAgent** | Classical | OX crossover, re-optimization every N steps |
| **PPOAgent** | Learning | Actor-critic (stub) |
| **GEPAAgent** | Meta | LLM hyperparameter evolution (stub) |

## Project Structure

```
.
├── src/
│   ├── agent.py                 # Agent ABC & basic agents
│   ├── sch_env.py               # Gym-style environment
│   ├── orlib_sch.py             # Problem data structures
│   ├── classical_agents.py       # SA, GA implementations
│   ├── native_optimized/        # C extensions (cost eval, crossover)
│   │   ├── evaluate_opt.c       # Fast cost evaluation
│   │   ├── crossover_opt.c      # Batch OX crossover
│   │   └── __compiled__/        # Compiled .so files
│   ├── benchmark.py             # Benchmark runner
│   └── visualize.py             # Convergence plotting
├── sanity/                       # Unit tests
├── data/                         # ORLib instances (sch10.txt, sch100.txt, etc.)
├── main.py                       # CLI entrypoint
├── setup.py                      # C extension build config
└── pyproject.toml               # Python project metadata
```

## Data

Problem instances are in `data/` (ORLib format):
- `sch10.txt` — 10 jobs (fast, for testing)
- `sch100.txt` — 100 jobs
- `sch1000.txt` — 1000 jobs

## C Optimizations

The hot path (cost evaluation) uses compiled C extensions:

- **`evaluate_opt.c`** — Fast cost evaluation + swap deltas
- **`evaluate_simd.c`** — Auto-specialized SIMD version per problem size
- **`crossover_opt.c`** — Batch OX crossover

Compiled .so files are cached in `src/native_optimized/__compiled__/` and auto-rebuilt if missing.

## Results

Results are saved as:
- `benchmark.csv` — Aggregate statistics (mean, std improvement %)
- `convergence_*.png` — Per-agent convergence traces
- `logs.html` — Profiling report (if using `pyinstrument`)

## Architecture Overview

```
ORLib file  →  SchDataset  →  SchInstance(s)
                                    ↓
                                SchEnv
                                    ↓
                            Agent.solve()
                                    ↓
                            BenchmarkRunner
                                    ↓
                    benchmark.csv + plots
```

Each agent operates on a **SchEnv** (Gym-style environment) with:
- **State:** Job permutation + cost scalars
- **Action:** Pairwise swap (from n*(n-1)/2 possible swaps)
- **Reward:** Cost improvement, shaped by initial cost

## Common Issues

**C extensions won't compile:**
```bash
# Ensure GCC is installed
gcc --version
# On Ubuntu/Debian:
sudo apt-get install build-essential
# Rebuild
uv sync
```

**Tests can't find imports:**
```bash
# Ensure venv is activated
source .venv/bin/activate
python -m pytest sanity/ -v
```

**Memory error on large instances:**
- Use smaller `--max-steps` (e.g., 100 instead of 10000)
- Try `--n-instances 1` to test a single instance first

## References

- ORLib scheduling instances: [http://people.brunel.ac.uk/~mastjjb/jeb/orlib/schinfo.html](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/schinfo.html)
- CLAUDE.md — detailed architecture and agent inventory
- PROBLEM.md — formal problem definition
