"""
rl_nets.py — Shared PyTorch neural network modules for RL agents.

Modules
-------
MLP
    Configurable multi-layer perceptron backbone with a variable number of
    hidden layers and a pluggable activation function (default: Tanh).

PolicyNet
    Actor network mapping an observation vector to logits over the discrete
    action space (used by PPOAgent for categorical sampling).

ValueNet
    Critic network mapping an observation vector to a scalar state-value
    estimate V(s) (used by PPOAgent for advantage computation).

QNet
    Q-network mapping an observation vector to Q(s, a) for all actions
    simultaneously (used by DQNAgent for epsilon-greedy selection and
    Bellman updates).

Notes
-----
- All modules accept ``obs_size`` and ``n_actions`` matching the SchEnv dims.
- Default hidden layer sizes are [128, 128] for all networks.
- No agent logic lives here — only nn.Module definitions.

To be implemented.
"""
