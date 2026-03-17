"""
sch_rl.py — Proximal Policy Optimisation (PPO) agent for SchEnv
================================================================
Trains a small actor-critic network to improve scheduling solutions
by iteratively selecting swap moves in :class:`~sch_env.SchEnv`.

Architecture
------------
A shared MLP backbone feeds two heads:
    - Actor  - categorical distribution over n*(n-1)/2 swap actions
    - Critic - scalar value estimate V(s)

    Input (obs_size)
        ↓
    Linear(obs_size → hidden) + LayerNorm + ReLU
        ↓
    Linear(hidden → hidden)   + LayerNorm + ReLU
        ↓ ────────────────────────────────────────┐
    Linear(hidden → n_actions)               Linear(hidden → 1)
    softmax → π(a|s)                         → V(s)

Algorithm: PPO-clip (Schulman et al. 2017)
    • Collects a rollout buffer of T steps across parallel envs
    • Runs K epochs of mini-batch gradient updates
    • Entropy bonus encourages exploration
    • GAE(λ) advantage estimation

Usage
-----
    from orlib_sch import load
    from sch_env   import SchEnv
    from sch_rl    import PPOAgent, PPOConfig, train, evaluate

    ds    = load("sch10.txt")
    env   = SchEnv(ds[0], h=0.4, max_steps=100)

    cfg   = PPOConfig()                     # or customise fields
    agent = PPOAgent(env.obs_size, env.n_actions, cfg)
    train(agent, ds.instances, cfg, h=0.4)  # trains in-place
    agent.save("agent.pt")

    # Later:
    agent2 = PPOAgent.load("agent.pt")
    result = evaluate(agent2, env)
    print(result)

Requirements
------------
    pip install torch
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from orlib_sch import SchInstance
from sch_env import EpisodeResult, SchEnv, run_episode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """
    All hyper-parameters for PPO training in one place.

    Fields
    ------
    hidden_size : int
        Width of each hidden layer in the actor-critic network.
    n_layers : int
        Number of hidden layers (minimum 1).
    lr : float
        Initial Adam learning rate.
    lr_decay : bool
        Whether to linearly decay lr to zero over training.
    gamma : float
        Discount factor for returns.
    gae_lambda : float
        GAE smoothing parameter λ.
    clip_eps : float
        PPO clipping radius ε.
    value_coef : float
        Coefficient for the value-function loss term.
    entropy_coef : float
        Coefficient for the entropy bonus term.
    max_grad_norm : float
        Gradient clipping threshold.
    rollout_steps : int
        Number of environment steps collected per update cycle (T).
    n_epochs : int
        Number of optimisation passes over the rollout buffer per update.
    batch_size : int
        Mini-batch size during the optimisation pass.
    max_steps_per_episode : int | None
        Overrides ``SchEnv.max_steps`` if set.
    n_updates : int
        Total number of update cycles (total steps = n_updates * rollout_steps).
    eval_interval : int
        Run a greedy evaluation episode every this many updates.
    eval_instances : int
        Number of instances to evaluate on (picked from the start of the dataset).
    seed : int | None
        Global RNG seed.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"auto"``.
    """

    hidden_size: int = 128
    n_layers: int = 2
    lr: float = 3e-4
    lr_decay: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    n_epochs: int = 4
    batch_size: int = 64
    max_steps_per_episode: int | None = None
    n_updates: int = 200
    eval_interval: int = 20
    eval_instances: int = 5
    seed: int | None = 42
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic MLP.

    Parameters
    ----------
    obs_size : int
    n_actions : int
    cfg : PPOConfig
    """

    def __init__(self, obs_size: int, n_actions: int, cfg: PPOConfig) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        # Build shared backbone
        layers: list[nn.Module] = []
        in_size = obs_size
        for _ in range(cfg.n_layers):
            layers += [
                nn.Linear(in_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.ReLU(),
            ]
            in_size = cfg.hidden_size
        self.backbone = nn.Sequential(*layers)

        # Heads
        self.actor_head = nn.Linear(cfg.hidden_size, n_actions)
        self.critic_head = nn.Linear(cfg.hidden_size, 1)

        # Orthogonal initialisation (empirically helps PPO)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller output scale for policy head
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (B, obs_size)

        Returns
        -------
        logits : (B, n_actions)
        value  : (B,)
        """
        h = self.backbone(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action.

        Returns
        -------
        action   : (B,) int64
        log_prob : (B,) float32
        value    : (B,) float32
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during the optimisation pass to compute updated log-probs/entropy.

        Returns
        -------
        log_prob : (B,)
        entropy  : (B,)
        value    : (B,)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Transition data helper
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """Single transition data."""
    obs: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Stores one rollout (T steps, single environment) and computes
    GAE advantages + discounted returns on demand.

    Parameters
    ----------
    size : int
        Number of steps to store.
    obs_size : int
    device : torch.device
    """

    def __init__(self, size: int, obs_size: int, device: torch.device) -> None:
        self.size = size
        self.obs_size = obs_size
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._obs = np.zeros((self.size, self.obs_size), dtype=np.float32)
        self._actions = np.zeros(self.size, dtype=np.int64)
        self._rewards = np.zeros(self.size, dtype=np.float32)
        self._values = np.zeros(self.size, dtype=np.float32)
        self._log_probs = np.zeros(self.size, dtype=np.float32)
        self._dones = np.zeros(self.size, dtype=np.float32)
        self._ptr = 0

    def add(self, trans: Transition) -> None:
        i = self._ptr
        self._obs[i] = trans.obs
        self._actions[i] = trans.action
        self._rewards[i] = trans.reward
        self._values[i] = trans.value
        self._log_probs[i] = trans.log_prob
        self._dones[i] = float(trans.done)
        self._ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GAE(λ) advantage and discounted return computation.

        Returns
        -------
        advantages : (T,) float32
        returns    : (T,) float32
        """
        T = self._ptr
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self._values[t + 1]
            next_done = self._dones[t]  # 1 if episode ended at t
            delta = self._rewards[t] + gamma * next_val * (1.0 - next_done) - self._values[t]
            gae = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
        returns = advantages + self._values[:T]
        return advantages, returns

    def to_tensors(self, advantages: np.ndarray, returns: np.ndarray) -> dict[str, torch.Tensor]:
        T = self._ptr
        return {
            "obs": torch.from_numpy(self._obs[:T]).to(self.device),
            "actions": torch.from_numpy(self._actions[:T]).to(self.device),
            "log_probs": torch.from_numpy(self._log_probs[:T]).to(self.device),
            "advantages": torch.from_numpy(advantages).to(self.device),
            "returns": torch.from_numpy(returns).to(self.device),
        }


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """
    Proximal Policy Optimisation agent.

    Parameters
    ----------
    obs_size : int
    n_actions : int
    cfg : PPOConfig
    """

    def __init__(self, obs_size: int, n_actions: int, cfg: PPOConfig) -> None:
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = cfg.resolve_device()

        self.net = ActorCritic(obs_size, n_actions, cfg).to(self.device)
        self.optimiser = Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def policy_fn(self, obs: np.ndarray) -> int:
        """Stochastic policy callable compatible with :func:`~sch_env.run_episode`."""
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.net.act(t, deterministic=False)
        return int(action.item())

    def greedy_policy_fn(self, obs: np.ndarray) -> int:
        """Deterministic (greedy) policy callable."""
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.net.act(t, deterministic=True)
        return int(action.item())

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env: SchEnv,
        buffer: RolloutBuffer,
        obs: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, bool, float]:
        """
        Fill *buffer* with ``cfg.rollout_steps`` transitions.

        Handles episode boundaries: resets env automatically when an episode ends.

        Parameters
        ----------
        env   : SchEnv  (already reset once externally)
        buffer: RolloutBuffer (reset before this call)
        obs   : current observation
        done  : whether the env is currently at an episode boundary

        Returns
        -------
        obs       : last observation after collecting
        done      : whether last step was terminal
        last_val  : critic estimate at the last state (for GAE bootstrap)
        """
        self.net.eval()
        for _ in range(self.cfg.rollout_steps):
            if done:
                obs, _ = env.reset()

            t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.net.act(t_obs)
            a = int(action.item())
            lp = float(log_prob.item())
            v = float(value.item())

            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            transition = Transition(obs, a, reward, v, lp, done)

            buffer.add(transition)
            obs = next_obs

        # Bootstrap value for GAE
        t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_val = self.net.act(t_obs)
        last_val_f = float(last_val.item()) * (1.0 - float(done))

        return obs, done, last_val_f

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Run K epochs of PPO mini-batch updates on *data*.

        Returns a dict of scalar metrics for logging.
        """
        self.net.train()
        cfg = self.cfg
        T = data["obs"].shape[0]

        # Normalise advantages (per mini-batch or over full rollout)
        adv = data["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_pg_loss = 0.0
        total_val_loss = 0.0
        total_ent_loss = 0.0
        total_loss_sum = 0.0
        n_batches = 0

        indices = np.arange(T)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, cfg.batch_size):
                idx = indices[start : start + cfg.batch_size]
                idx_t = torch.from_numpy(idx).to(self.device)

                b_obs = data["obs"][idx_t]
                b_actions = data["actions"][idx_t]
                b_old_lp = data["log_probs"][idx_t]
                b_adv = adv[idx_t]
                b_returns = data["returns"][idx_t]

                new_lp, entropy, new_val = self.net.evaluate_actions(b_obs, b_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_lp - b_old_lp)
                pg_loss1 = -b_adv * ratio
                pg_loss2 = -b_adv * ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                val_loss = F.mse_loss(new_val, b_returns)

                # Entropy bonus (maximise → minimise negative)
                ent_loss = -entropy.mean()

                loss = pg_loss + cfg.value_coef * val_loss + cfg.entropy_coef * ent_loss

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimiser.step()

                total_pg_loss += float(pg_loss.item())
                total_val_loss += float(val_loss.item())
                total_ent_loss += float(ent_loss.item())
                total_loss_sum += float(loss.item())
                n_batches += 1

        nb = max(n_batches, 1)
        return {
            "loss_total": total_loss_sum / nb,
            "loss_policy": total_pg_loss / nb,
            "loss_value": total_val_loss / nb,
            "loss_entropy": total_ent_loss / nb,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save network weights and config to a .pt file."""
        torch.save(
            {
                "obs_size": self.obs_size,
                "n_actions": self.n_actions,
                "cfg": self.cfg,
                "state_dict": self.net.state_dict(),
            },
            str(path),
        )
        print(f"[PPOAgent] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PPOAgent:
        """Load a previously saved agent."""
        ckpt = torch.load(str(path), map_location="cpu")
        agent = cls(ckpt["obs_size"], ckpt["n_actions"], ckpt["cfg"])
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        print(f"[PPOAgent] Loaded from {path}")
        return agent

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.net.parameters())
        return (
            f"PPOAgent(obs={self.obs_size}, actions={self.n_actions}, "
            f"params={n_params:,}, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainLog:
    """Accumulated training statistics."""

    update: list[int] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    loss_policy: list[float] = field(default_factory=list)
    loss_value: list[float] = field(default_factory=list)
    loss_entropy: list[float] = field(default_factory=list)
    eval_best: list[float] = field(default_factory=list)  # avg best cost
    eval_improve: list[float] = field(default_factory=list)  # avg % improvement
    eval_update: list[int] = field(default_factory=list)


def train(
    agent: PPOAgent,
    instances: Sequence[SchInstance],
    cfg: PPOConfig,
    *,
    h: float = 0.4,
    verbose: bool = True,
) -> TrainLog:
    """
    Full PPO training loop.

    The agent trains across all *instances* by cycling through them:
    each update cycle uses one instance (round-robin with shuffle).

    Parameters
    ----------
    agent     : PPOAgent   (modified in-place)
    instances : sequence of SchInstance to train on
    cfg       : PPOConfig
    h         : float      due-date tightness
    verbose   : bool       print progress every eval_interval updates

    Returns
    -------
    TrainLog
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = agent.device
    log = TrainLog()
    buffer = RolloutBuffer(cfg.rollout_steps, agent.obs_size, device)

    # LR scheduler
    scheduler = None
    if cfg.lr_decay:
        scheduler = LinearLR(
            agent.optimiser,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=cfg.n_updates,
        )

    # Build a cycling list of envs (one per instance)
    envs = [
        SchEnv(
            inst,
            h=h,
            max_steps=cfg.max_steps_per_episode,
            seed=cfg.seed,
        )
        for inst in instances
    ]
    inst_order = list(range(len(envs)))
    random.shuffle(inst_order)
    inst_cycle_idx = 0

    # Initialise first env
    active_env = envs[inst_order[inst_cycle_idx]]
    obs, _ = active_env.reset()
    done = False

    t0 = time.perf_counter()

    for update in range(1, cfg.n_updates + 1):
        # Cycle to a fresh instance every rollout to expose the agent to variety
        env_idx = inst_order[inst_cycle_idx % len(inst_order)]
        active_env = envs[env_idx]
        inst_cycle_idx += 1
        if inst_cycle_idx >= len(inst_order):
            inst_cycle_idx = 0
            random.shuffle(inst_order)

        # Reset for a fresh episode at the start of each rollout
        obs, _ = active_env.reset()
        done = False

        # Collect rollout
        buffer.reset()
        obs, done, last_val = agent.collect_rollout(active_env, buffer, obs, done)

        # Compute GAE
        advantages, returns = buffer.compute_returns_and_advantages(last_val, cfg.gamma, cfg.gae_lambda)
        data = buffer.to_tensors(advantages, returns)

        # PPO update
        metrics = agent.update(data)

        if scheduler is not None:
            scheduler.step()

        # Logging
        log.update.append(update)
        for k, _v in metrics.items():
            getattr(log, k.replace("loss_", "loss_"))[:]  # just ensure key
        log.loss_total.append(metrics["loss_total"])
        log.loss_policy.append(metrics["loss_policy"])
        log.loss_value.append(metrics["loss_value"])
        log.loss_entropy.append(metrics["loss_entropy"])

        # Periodic evaluation
        if update % cfg.eval_interval == 0 or update == cfg.n_updates:
            eval_insts = list(instances)[: cfg.eval_instances]
            results = evaluate_batch(agent, eval_insts, h=h, cfg=cfg)
            avg_best = sum(r.best_cost for r in results) / len(results)
            avg_imp = sum(r.improvement_pct for r in results) / len(results)
            log.eval_best.append(avg_best)
            log.eval_improve.append(avg_imp)
            log.eval_update.append(update)

            if verbose:
                elapsed = time.perf_counter() - t0
                lr_now = agent.optimiser.param_groups[0]["lr"]
                print(
                    f"[{update:4d}/{cfg.n_updates}]"
                    f"  loss={metrics['loss_total']:+.4f}"
                    f"  pol={metrics['loss_policy']:+.4f}"
                    f"  val={metrics['loss_value']:.4f}"
                    f"  ent={metrics['loss_entropy']:+.4f}"
                    f"  |  eval_best={avg_best:.1f}"
                    f"  improve={avg_imp:.1f}%"
                    f"  lr={lr_now:.2e}"
                    f"  {elapsed:.1f}s"
                )

    return log


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate(
    agent: PPOAgent,
    env: SchEnv,
    *,
    deterministic: bool = True,
    seed: int | None = None,
) -> EpisodeResult:
    """
    Run one greedy (or stochastic) episode and return an :class:`~sch_env.EpisodeResult`.

    Parameters
    ----------
    agent         : PPOAgent
    env           : SchEnv
    deterministic : bool   - greedy argmax if True, sampled if False
    seed          : int | None
    """
    policy = agent.greedy_policy_fn if deterministic else agent.policy_fn
    return run_episode(env, policy, seed=seed)


def evaluate_batch(
    agent: PPOAgent,
    instances: Sequence[SchInstance],
    *,
    h: float = 0.4,
    cfg: PPOConfig | None = None,
    deterministic: bool = True,
    seed: int | None = None,
) -> list[EpisodeResult]:
    """Run greedy evaluation across multiple instances."""
    results = []
    for inst in instances:
        max_steps = cfg.max_steps_per_episode if cfg else None
        env = SchEnv(inst, h=h, max_steps=max_steps)
        result = evaluate(agent, env, deterministic=deterministic, seed=seed)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Quick benchmark: PPO vs random baseline
# ---------------------------------------------------------------------------


def benchmark(
    instances: Sequence[SchInstance],
    cfg: PPOConfig | None = None,
    h: float = 0.4,
    verbose: bool = True,
) -> dict[str, float | PPOAgent]:
    """
    Train a PPO agent and compare it to a random-swap baseline.

    Returns a dict with keys:
        random_avg_best, random_avg_improve_pct,
        ppo_avg_best,    ppo_avg_improve_pct,
        best_agent (the agent with lowest cost)
    """
    if cfg is None:
        cfg = PPOConfig()

    # Build a representative env to get obs_size / n_actions
    ref_env = SchEnv(instances[0], h=h, max_steps=cfg.max_steps_per_episode)

    # --- Random baseline ---
    random_results = []
    for inst in instances:
        env = SchEnv(inst, h=h, max_steps=cfg.max_steps_per_episode, seed=cfg.seed)
        r = run_episode(env, env.action_space_sample, seed=cfg.seed)
        random_results.append(r)

    rand_best_avg = sum(r.best_cost for r in random_results) / len(random_results)
    rand_imp_avg = sum(r.improvement_pct for r in random_results) / len(random_results)

    if verbose:
        print(f"[Random baseline]  avg_best={rand_best_avg:.1f}  avg_improve={rand_imp_avg:.1f}%")

    # --- PPO agent ---
    agent = PPOAgent(ref_env.obs_size, ref_env.n_actions, cfg)
    if verbose:
        print(agent)

    train(agent, instances, cfg, h=h, verbose=verbose)

    ppo_results = evaluate_batch(agent, list(instances), h=h, cfg=cfg)
    ppo_best_avg = sum(r.best_cost for r in ppo_results) / len(ppo_results)
    ppo_imp_avg = sum(r.improvement_pct for r in ppo_results) / len(ppo_results)

    if verbose:
        print(f"[PPO]              avg_best={ppo_best_avg:.1f}  avg_improve={ppo_imp_avg:.1f}%")
        print(
            f"[Delta]            best Δ={rand_best_avg - ppo_best_avg:+.1f}"
            f"  improve Δ={ppo_imp_avg - rand_imp_avg:+.1f}pp"
        )

    # Determine best agent (PPO has lower average cost than random baseline)
    best_agent = agent if ppo_best_avg < rand_best_avg else None
    
    if verbose and best_agent:
        print(f"[Best agent] PPO with avg_best={ppo_best_avg:.1f}")

    return {
        "random_avg_best": rand_best_avg,
        "random_avg_improve_pct": rand_imp_avg,
        "ppo_avg_best": ppo_best_avg,
        "ppo_avg_improve_pct": ppo_imp_avg,
        "best_agent": best_agent,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from orlib_sch import load

    print("=" * 60)
    print("sch_rl.py  -  PPO for common due date scheduling")
    print("=" * 60)

    sch_ds = load("data/sch10.txt")

    cfg = PPOConfig(
        hidden_size=64,
        n_layers=3,
        n_updates=5000,
        rollout_steps=128,
        eval_interval=20,
        eval_instances=3,
        seed=42,
        lr=0.01
    )

    stats = benchmark(sch_ds.instances, cfg=cfg, h=0.4, verbose=True)
    print("\nFinal stats:", {k: v for k, v in stats.items() if k != "agent"})
