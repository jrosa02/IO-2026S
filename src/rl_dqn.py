"""
rl_dqn.py — Deep Q-Network (DQN) scheduling agent.

Classes
-------
DQNConfig (dataclass)
    Hyperparameters for DQN training:
    - hidden            : list[int] — MLP hidden layer sizes (default [128, 128])
    - lr                : float     — Adam learning rate (default 1e-3)
    - gamma             : float     — discount factor (default 0.99)
    - epsilon_start     : float     — initial exploration rate (default 1.0)
    - epsilon_min       : float     — minimum exploration rate (default 0.05)
    - epsilon_decay     : float     — per-episode epsilon multiplier (default 0.995)
    - buffer_size       : int       — replay buffer capacity (default 10_000)
    - batch_size        : int       — mini-batch size (default 64)
    - target_update_freq: int       — steps between target network syncs (default 100)

ReplayBuffer
    Circular buffer storing (obs, action, reward, next_obs, done) transitions
    used for experience replay during DQN training.

DQNAgent (Agent)
    Deep RL agent using DQN with experience replay and a target network.

    act(obs)
        Epsilon-greedy selection over Q(s, a) during training;
        pure greedy (epsilon=0) after load().

    train(instances, *, h, n_episodes, max_steps, **kwargs)
        DQN training loop: collect transitions into ReplayBuffer, sample
        mini-batches, apply Bellman update, sync target network periodically.
        Returns per-episode mean reward history.

    save(path) / load(path)
        Persist and restore Q-network state dict and DQNConfig.

Notes
-----
- Depends on rl_nets.QNet.
- Unlike PPO, DQN is off-policy — replay buffer allows data reuse.
- Trained agent records self.cost_history during solve() via inherited mechanism.

To be implemented.
"""
