"""
Deep Q-Network (DQN) Agent
--------------------------
• Two networks: online (policy) + target (soft-updated)
• Experience replay with prioritised sampling option
• ε-greedy exploration with linear / exponential decay
• Saves / loads weights automatically
"""

import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config.settings import AgentConfig


# ──────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Dueling DQN architecture:
      shared backbone → [value stream | advantage stream] → Q-values
    """

    def __init__(self, state_size: int, action_size: int, hidden: list[int]):
        super().__init__()
        layers = []
        in_dim = state_size
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        v    = self.value_head(feat)                    # (B, 1)
        adv  = self.adv_head(feat)                      # (B, A)
        q    = v + adv - adv.mean(dim=1, keepdim=True)  # (B, A)
        return q


# ──────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Uniform experience replay."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_valid_actions, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            np.array(next_valid_actions, dtype=np.bool_),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, next_valid_actions, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(next_valid_actions, dtype=np.bool_),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Manages online / target networks, replay buffer, ε-decay, and learning.
    """

    def __init__(
        self,
        state_size:  int,
        action_size: int,
        cfg: AgentConfig | None = None,
    ):
        self.state_size  = state_size
        self.action_size = action_size
        self.cfg = cfg or AgentConfig()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

        # Networks
        self.online = QNetwork(state_size, action_size, self.cfg.hidden_layers).to(self.device)
        self.target = QNetwork(state_size, action_size, self.cfg.hidden_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer    = ReplayBuffer(self.cfg.buffer_size)

        self.epsilon    = self.cfg.epsilon_start
        self.step_count = 0

        print(f"[DQNAgent] device={self.device} | "
              f"state={state_size} action={action_size} | "
              f"params={sum(p.numel() for p in self.online.parameters()):,}")

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, valid_actions: list[bool] | None = None) -> int:
        """epsilon-greedy action selection, optionally restricted to a valid-action mask."""
        allowed = (
            [i for i, ok in enumerate(valid_actions) if ok]
            if valid_actions is not None
            else list(range(self.action_size))
        )
        if not allowed:  # should not happen (action 0 is always legal), but guard anyway
            allowed = list(range(self.action_size))

        if random.random() < self.epsilon:
            return random.choice(allowed)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(s).squeeze(0)
        if valid_actions is not None:
            mask = torch.tensor(valid_actions, device=q.device)
            q = q.masked_fill(~mask, float("-inf"))
        return int(q.argmax().item())

    def store(self, state, action, reward, next_state, next_valid_actions, done):
        if len(next_valid_actions) != self.action_size:
            raise ValueError(
                f"Expected {self.action_size} next-state action-mask entries, "
                f"got {len(next_valid_actions)}."
            )
        self.buffer.push(state, action, reward, next_state, next_valid_actions, done)

    def learn(self) -> float | None:
        """Sample a mini-batch and perform one gradient update."""
        if len(self.buffer) < self.cfg.batch_size:
            return None
        if len(self.buffer) < self.cfg.min_buffer_before_learn:
            return None

        states, actions, rewards, next_states, next_valid_actions, dones = self.buffer.sample(self.cfg.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_valid_actions = torch.BoolTensor(next_valid_actions).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q
        current_q = self.online(states).gather(1, actions)

        # Double DQN target
        with torch.no_grad():
            online_next_q = self.online(next_states).masked_fill(~next_valid_actions, float("-inf"))
            best_actions = online_next_q.argmax(dim=1, keepdim=True)
            target_next_q = self.target(next_states).masked_fill(~next_valid_actions, float("-inf"))
            next_q       = target_next_q.gather(1, best_actions)
            target_q     = rewards + self.cfg.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        # Soft update target network
        if self.step_count % self.cfg.target_update_freq == 0:
            self._soft_update()

        # Epsilon decay
        self.epsilon = max(
            self.cfg.epsilon_min,
            self.epsilon * self.cfg.epsilon_decay
        )

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online":      self.online.state_dict(),
            "target":      self.target.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "step_count":  self.step_count,
            "cfg":         self.cfg,
            "state_size":  self.state_size,
            "action_size": self.action_size,
        }, path)
        print(f"[DQNAgent] Saved -> {path}")

    def load(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        saved_state_size = ckpt.get("state_size")
        saved_action_size = ckpt.get("action_size")
        if (
            saved_state_size is not None and saved_state_size != self.state_size
        ) or (
            saved_action_size is not None and saved_action_size != self.action_size
        ):
            raise RuntimeError(
                "Checkpoint is not compatible with the current observation/action shape. "
                "Retrain from scratch with the direct-phase action environment."
            )
        try:
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with the current model shape. "
                "It was trained with a different action space or observation size. "
                "Retrain from scratch with the direct-phase action environment."
            ) from exc
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        print(f"[DQNAgent] Loaded <- {path}  (epsilon={self.epsilon:.4f})")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _soft_update(self):
        τ = self.cfg.tau
        for online_p, target_p in zip(self.online.parameters(), self.target.parameters()):
            target_p.data.copy_(τ * online_p.data + (1 - τ) * target_p.data)
