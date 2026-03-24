"""
gepa_agent.py — GEPA-driven hyperparameter search agent.

GEPA (Genetic-Pareto) is a framework that uses an LLM to evolve textual
parameters via reflection and Pareto-efficient evolutionary search.
See: https://github.com/gepa-ai/gepa

Here, GEPA's "text parameter" is a JSON config dict for a base agent class
(SimulatedAnnealingAgent or PPOAgent).  Each candidate config is evaluated
by running the base agent on a set of scheduling instances; fitness is
mean improvement % over the initial schedule.

Classes
-------
GEPAConfig (dataclass)
    Configuration for the GEPA search:
    - base_agent_cls  : type        — SimulatedAnnealingAgent or PPOAgent
    - config_schema   : dict        — search space definition per parameter:
                                      {name: {"type": float, "lo": ..., "hi": ...}}
    - n_generations   : int         — number of evolution generations (default 10)
    - population_size : int         — configs evaluated per generation (default 8)
    - llm_client      : Any | None  — LLM client for mutation; None → mock mode

GEPAAgent (Agent)
    Wraps GEPA to search hyperparameters of a base agent.

    train(instances, *, h, **kwargs)
        Evolution loop:
        1. Initialise population of config dicts sampled within schema bounds.
        2. Evaluate each config: run base_agent on all instances; capture trace.
        3. LLM (or mock mutator) reads execution trace → proposes config mutation.
        4. Pareto selection on (mean_cost, std_cost) front.
        5. Repeat n_generations; store self.best_config_.

    _mock_mutate(config) -> dict
        Fallback mutator (no LLM): randomly perturbs one numeric parameter
        within its schema bounds.

    _evaluate_config(config, instances, h) -> (float, str)
        Instantiate base agent with config, run on all instances, return
        (mean improvement %, execution trace string for LLM reflection).

    solve(env, *, seed) -> EpisodeResult
        Instantiate base_agent_cls(**self.best_config_) and delegate to its
        solve(env, seed=seed).

Notes
-----
- LLM integration: plug in any client exposing a .complete(prompt) -> str
  interface (OpenAI, Anthropic, Ollama, etc.).  When llm_client is None,
  _mock_mutate() is used instead — useful for testing without an API key.
- GEPA dependency: ``pip install gepa``  (only imported inside gepa_agent.py).

To be implemented.
"""
