"""
rl_ppo.py — Proximal Policy Optimization (PPO) scheduling agent.

Classes
-------
PPOConfig (dataclass)
    Hyperparameters for PPO training:
    - hidden       : list[int]  — MLP hidden layer sizes (default [128, 128])
    - lr           : float      — Adam learning rate (default 3e-4)
    - clip_eps     : float      — PPO clipping epsilon (default 0.2)
    - gamma        : float      — discount factor (default 0.99)
    - gae_lambda   : float      — GAE lambda for advantage estimation (default 0.95)
    - n_epochs     : int        — update epochs per rollout (default 4)
    - batch_size   : int        — mini-batch size (default 64)
    - entropy_coef : float      — entropy bonus coefficient (default 0.01)
    - value_coef   : float      — value loss coefficient (default 0.5)
    - max_grad_norm: float      — gradient clipping norm (default 0.5)

PPOAgent (Agent)
    Deep RL agent using PPO with an actor-critic architecture from rl_nets.py.

    act(obs)
        Sample an action from the categorical policy distribution.

    train(instances, *, h, n_episodes, max_steps, **kwargs)
        PPO training loop: collect rollouts via SchEnv, compute GAE advantages,
        run n_epochs of mini-batch updates with clipped surrogate loss.
        Returns per-episode mean reward history for convergence plotting.

    save(path) / load(path)
        Persist and restore policy + value network state dicts and PPOConfig.

Notes
-----
- Depends on rl_nets.PolicyNet and rl_nets.ValueNet.
- GEPA-searchable parameters: lr, clip_eps, entropy_coef, gae_lambda.
- Trained agent records self.cost_history during solve() via inherited mechanism.

To be implemented.
"""
