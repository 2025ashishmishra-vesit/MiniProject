"""
EpisodeMetrics — accumulates per-step data and returns a summary dict.
"""

import numpy as np


class EpisodeMetrics:
    def __init__(self):
        self.rewards    = []
        self.queues     = []
        self.losses     = []
        self.mode_rl    = 0
        self.mode_fixed = 0
        self.switch_requested = 0
        self.switch_applied = 0
        self.switch_blocked = 0

    def update(self, reward: float, queues: dict, mode: str, loss: float | None, info: dict | None = None):
        self.rewards.append(reward)
        self.queues.append(sum(queues.values()))
        if loss is not None:
            self.losses.append(loss)
        if mode == "rl":
            self.mode_rl += 1
        else:
            self.mode_fixed += 1
        if info:
            self.switch_requested += int(info.get("switch_requested", 0))
            self.switch_applied += int(info.get("switch_applied", 0))
            self.switch_blocked += int(info.get("switch_blocked", 0))

    def summary(self) -> dict:
        total_steps = self.mode_rl + self.mode_fixed
        return {
            "total_reward": float(np.sum(self.rewards)),
            "avg_reward":   float(np.mean(self.rewards)),
            "min_reward":   float(np.min(self.rewards)),
            "avg_queue":    float(np.mean(self.queues)),
            "max_queue":    float(np.max(self.queues)),
            "avg_loss":     float(np.mean(self.losses)) if self.losses else 0.0,
            "rl_steps":     self.mode_rl,
            "fixed_steps":  self.mode_fixed,
            "rl_frac":      self.mode_rl / max(total_steps, 1),
            "switch_requested": self.switch_requested,
            "switch_applied": self.switch_applied,
            "switch_blocked": self.switch_blocked,
            "switch_rate": self.switch_applied / max(total_steps, 1),
        }
