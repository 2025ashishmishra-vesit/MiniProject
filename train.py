"""
Training script.

Examples:
    python train.py
    python train.py --gui
    python train.py --network 4_way --scenario-selection random
    python train.py --network all --episodes 300
    python train.py --resume models/checkpoints/ep_100.pt
"""

import argparse
import json
import os
import time

import numpy as np

from agent.dqn_agent import DQNAgent
from config.settings import AgentConfig, EnvConfig, TrainConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv
from utils.logger import TrainingLogger
from utils.metrics import EpisodeMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Train SUMO RL Traffic Controller")
    p.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--eval", action="store_true", help="Evaluation mode only (no training)")
    p.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    p.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N episodes")
    p.add_argument("--eval-every", type=int, default=None, help="Run training-set evaluation every N episodes; 0 disables it")
    p.add_argument(
        "--network",
        choices=["default", "3_way", "4_way", "all"],
        default="default",
        help="Scenario set. default keeps the original single SUMO config.",
    )
    p.add_argument(
        "--scenario-selection",
        choices=["cycle", "random"],
        default="cycle",
        help="Choose scenarios cyclically or randomly at episode reset.",
    )
    p.add_argument(
        "--control-mode",
        choices=["rl", "hybrid"],
        default="rl",
        help="rl gives the agent every decision; hybrid uses fixed timing when queues are small.",
    )
    p.add_argument(
        "--reward-mode",
        choices=["delta_pressure", "absolute_pressure"],
        default=None,
        help="Reward formula to use.",
    )
    p.add_argument("--throughput-reward-weight", type=float, default=None)
    p.add_argument("--switch-penalty", type=float, default=None)
    p.add_argument("--min-green-time", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--best-model-dir", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--final-model", type=str, default=None)
    return p.parse_args()


def evaluate(env: SumoTrafficEnv, agent: DQNAgent, n_eval: int = 3) -> dict:
    """Run n_eval episodes with greedy policy and return averaged metrics."""

    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    all_rewards, all_queues = [], []
    mode_counts = {
        "rl": 0,
        "fixed": 0,
        "emergency": 0
    }

    try:
        for _ in range(n_eval):
            state = env.reset()

            ep_reward = 0.0
            ep_queues = []
            done = False

            while not done:
                action = agent.select_action(
                    state,
                    valid_actions=env.valid_actions_mask()
                )

                next_state, reward, done, info = env.step(action)

                ep_reward += reward
                ep_queues.append(sum(info["queues"].values()))

                mode = info.get("mode", "unknown")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

                state = next_state

            all_rewards.append(ep_reward)
            all_queues.append(np.mean(ep_queues))

    finally:
        # Always restore training epsilon
        agent.epsilon = saved_eps

    return {
        "eval_reward": float(np.mean(all_rewards)),
        "eval_avg_queue": float(np.mean(all_queues)),
        "eval_mode_rl": mode_counts.get("rl", 0),
        "eval_mode_fixed": mode_counts.get("fixed", 0),
        "eval_mode_emergency": mode_counts.get("emergency", 0),
    }


def train():
    args = parse_args()

    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()
    train_cfg = TrainConfig()

    env_cfg.scenarios = build_scenarios(args.network, split="train")
    env_cfg.scenario_selection = args.scenario_selection
    env_cfg.control_mode = args.control_mode
    if args.reward_mode is not None:
        env_cfg.reward_mode = args.reward_mode
    if args.throughput_reward_weight is not None:
        env_cfg.throughput_reward_weight = args.throughput_reward_weight
    if args.switch_penalty is not None:
        env_cfg.switch_penalty = args.switch_penalty
    if args.min_green_time is not None:
        env_cfg.min_green_time = args.min_green_time

    if args.gui:
        train_cfg.gui = True
    if args.episodes:
        train_cfg.num_episodes = args.episodes
    if args.save_every is not None:
        train_cfg.save_every = args.save_every
    if args.eval_every is not None:
        train_cfg.eval_every = args.eval_every
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.best_model_dir:
        train_cfg.best_model_dir = args.best_model_dir
    if args.log_dir:
        train_cfg.log_dir = args.log_dir

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(train_cfg.best_model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    env = SumoTrafficEnv(cfg=env_cfg, gui=train_cfg.gui)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, cfg=agent_cfg)
    logger = TrainingLogger(log_dir=train_cfg.log_dir)

    start_episode = 0
    if args.resume:
        agent.load(args.resume)
        start_episode = int(os.path.basename(args.resume).split("ep_")[1].split(".")[0])
        print(f"[train] Resuming from episode {start_episode}")

    if args.eval:
        print("[train] Evaluation mode")
        metrics = evaluate(env, agent, n_eval=5)
        print(json.dumps(metrics, indent=2))
        env.close()
        return

    best_eval_reward = float("-inf")
    history = []

    print(f"\n{'=' * 60}")
    print(" SUMO RL Traffic Controller - Training")
    print(f" Episodes : {train_cfg.num_episodes}")
    print(f" Device   : {agent.device}")
    if env_cfg.scenarios:
        print(f" Scenarios: {len(env_cfg.scenarios)} ({args.network}, {args.scenario_selection})")
    else:
        print(" Scenarios: default single SUMO config")
    print(f" Control  : {env_cfg.control_mode}")
    print(" Actions  : 0-3 select configured green-phase indices (masked when unavailable)")
    print(
        f" Reward   : {env_cfg.reward_mode} | "
        f"throughput={env_cfg.throughput_reward_weight} | "
        f"switch_penalty={env_cfg.switch_penalty} | "
        f"min_green={env_cfg.min_green_time}s"
    )
    print(f"{'=' * 60}\n")

    for episode in range(start_episode, train_cfg.num_episodes):
        t0 = time.time()
        state = env.reset()
        scenario_name = env.current_scenario["name"] if env.current_scenario else "default"
        metrics = EpisodeMetrics()
        done = False

        while not done:
            action = agent.select_action(state, valid_actions=env.valid_actions_mask())
            next_state, reward, done, info = env.step(action)

            agent.store(
                state,
                action,
                reward,
                next_state,
                env.valid_actions_mask(),
                done,
            )
            loss = agent.learn()

            metrics.update(reward, info["queues"], info["mode"], loss, info=info)
            state = next_state

        ep_time = time.time() - t0
        summary = metrics.summary()
        summary.update({
            "episode": episode,
            "epsilon": agent.epsilon,
            "duration": ep_time,
            "scenario": scenario_name,
        })
        history.append(summary)
        logger.log(summary)

        print(
            f"Ep {episode:4d} | "
            f"scenario={scenario_name:18s} | "
            f"reward={summary['total_reward']:8.1f} | "
            f"avg_queue={summary['avg_queue']:5.1f} | "
            f"RL%={summary['rl_frac'] * 100:5.1f} | "
            f"sw={summary['switch_rate'] * 100:5.1f}% | "
            f"eps={agent.epsilon:.4f} | "
            f"loss={summary['avg_loss']:.4f} | "
            f"{ep_time:.1f}s"
        )

        if train_cfg.save_every > 0 and (episode + 1) % train_cfg.save_every == 0:
            ckpt = os.path.join(train_cfg.checkpoint_dir, f"ep_{episode + 1}.pt")
            agent.save(ckpt)

        if train_cfg.eval_every > 0 and (episode + 1) % train_cfg.eval_every == 0:
            eval_metrics = evaluate(env, agent)
            logger.log({**eval_metrics, "episode": episode})
            print(
                f"  EVAL reward={eval_metrics['eval_reward']:.1f} "
                f"queue={eval_metrics['eval_avg_queue']:.1f}"
            )

            if eval_metrics["eval_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_reward"]
                best_path = os.path.join(train_cfg.best_model_dir, "best.pt")
                agent.save(best_path)
                print(f"  New best model saved ({best_eval_reward:.1f})")

    env.close()
    best_path = os.path.join(train_cfg.best_model_dir, "best.pt")
    if args.final_model:
        agent.save(args.final_model)
    if best_eval_reward == float("-inf"):
        # An explicit --eval-every 0 means there is no basis for selecting an
        # earlier checkpoint.  Still provide a predictable best.pt artifact,
        # clearly equal to the final model rather than silently omitting it.
        agent.save(best_path)
        print("[train] No periodic evaluation; saved final model as best.pt")
    logger.save_history(history)
    print("\n[train] Done. Best eval reward:", best_eval_reward)


if __name__ == "__main__":
    train()
