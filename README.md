# Schedule Optimization Using LLMs
*Project for the "Computational Intelligence Course*

## How to run
After cloning the repo, set up your virtual envrionment based on the `uv.lock` file. If you're using uv, make sure to run `uv sync`. Activate four venv.

To run the optimization with a basic agent, in a terminal inside your venv, run:
```
pytohn main.py --agents sa
```

Replace `sa` with any of the following options to run use different agents:
* sa - simulated annealing
* genetic - a simple genetic algorithm
* greedy - full search; can take a very long time on larger datasets
* random - for tests only.

To run with GEPA, in a terminal inside your venv, run:
```
python main.py --agents sa,genetic,gepa --n-instances 5 \
               --gepa-train-n 10 --gepa-lm anthropic/claude-haiku-4-5 \
               --gepa-calls 60 --gepa-sa-steps 300 \
               --gepa-run-dir results/gepa_logs
```

## Results
All results are visualized and saved in `results\`. 