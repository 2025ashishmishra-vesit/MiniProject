"""
Fine-tune a trained DQN model using weaknesses found during evaluation.

The evaluation CSV should come from evaluate_compare.py on held-out test routes.
This script does not train on test_routes. It maps weak test scenarios back to
the matching training-split route names and oversamples those during continued
training.

Example:
    python fine_tune_from_eval.py --network all --base-model models/best/best.pt --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

from agent.dqn_agent import DQNAgent
from config.settings import AgentConfig, EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv
from utils.logger import TrainingLogger
from utils.metrics import EpisodeMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model based on evaluation results.")
    parser.add_argument(
        "--eval-csv",
        default="logs/evaluation/comparison.csv",
        help="Comparison CSV produced by evaluate_compare.py.",
    )
    parser.add_argument(
        "--base-model",
        default="models/best/best.pt",
        help="Checkpoint to resume from.",
    )
    parser.add_argument(
        "--network",
        choices=["3_way", "4_way", "all"],
        default="all",
        help="Training scenario group to fine-tune.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Fine-tuning episodes.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of weak scenario types to focus.")
    parser.add_argument(
        "--focus-multiplier",
        type=int,
        default=4,
        help="Maximum extra copies added for the weakest selected scenario.",
    )
    parser.add_argument(
        "--focus-only",
        action="store_true",
        help="Train only on weak scenarios. By default all training scenarios are retained once.",
    )
    parser.add_argument(
        "--metric",
        choices=["composite", "queue", "waiting", "throughput", "reward"],
        default="composite",
        help="Metric used to rank weak scenarios.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["hybrid", "rl"],
        default="rl",
        help="Control mode used while fine-tuning.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate.")
    parser.add_argument("--epsilon-start", type=float, default=0.10, help="Exploration epsilon after loading.")
    parser.add_argument("--epsilon-min", type=float, default=0.02, help="Minimum exploration epsilon.")
    parser.add_argument(
        "--reward-mode",
        choices=["delta_pressure", "absolute_pressure"],
        default=None,
        help="Reward formula to use while fine-tuning.",
    )
    parser.add_argument("--throughput-reward-weight", type=float, default=None)
    parser.add_argument("--switch-penalty", type=float, default=None)
    parser.add_argument("--min-green-time", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123, help="SUMO seed.")
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N episodes.")
    parser.add_argument("--gui", action="store_true", help="Run SUMO GUI.")
    parser.add_argument(
        "--output-model",
        default="models/fine_tuned/fine_tuned.pt",
        help="Final fine-tuned checkpoint path.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/fine_tuning",
        help="Directory for fine-tuning CSV/JSON logs.",
    )
    return parser.parse_args()


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def training_scenario_name(evaluated_name: str) -> str:
    parts = evaluated_name.split("/")
    if len(parts) >= 3 and parts[1] == "test":
        return f"{parts[0]}/{parts[2]}"
    return evaluated_name


def weakness_score(row: dict, metric: str) -> float:
    queue_loss = max(0.0, -to_float(row, "queue_reduction_pct"))
    waiting_loss = max(0.0, -to_float(row, "waiting_reduction_pct"))
    throughput_loss = max(0.0, -to_float(row, "throughput_improvement_pct"))

    fixed_reward = abs(to_float(row, "fixed_total_reward"))
    reward_delta = to_float(row, "reward_delta_rl_minus_fixed")
    reward_loss = 100.0 * max(0.0, -reward_delta) / max(fixed_reward, 1.0)

    if metric == "queue":
        return queue_loss
    if metric == "waiting":
        return waiting_loss
    if metric == "throughput":
        return throughput_loss
    if metric == "reward":
        return reward_loss

    return 0.30 * queue_loss + 0.35 * waiting_loss + 0.25 * throughput_loss + 0.10 * reward_loss


def load_ranked_weaknesses(eval_csv: Path, metric: str) -> list[dict]:
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {eval_csv}. Run evaluate_compare.py first."
        )

    ranked = []
    with open(eval_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["training_scenario"] = training_scenario_name(row["scenario"])
            row["weakness_score"] = weakness_score(row, metric)
            ranked.append(row)

    ranked.sort(key=lambda item: item["weakness_score"], reverse=True)
    return ranked


def build_weighted_training_scenarios(
    network: str,
    ranked_rows: list[dict],
    top_k: int,
    focus_multiplier: int,
    focus_only: bool,
) -> tuple[list[dict], list[dict]]:
    train_scenarios = build_scenarios(network, split="train")
    scenarios_by_name = {scenario["name"]: scenario for scenario in train_scenarios}

    selected = [
        row for row in ranked_rows
        if row["training_scenario"] in scenarios_by_name and row["weakness_score"] > 0
    ][:top_k]

    if not selected:
        selected = [
            row for row in ranked_rows
            if row["training_scenario"] in scenarios_by_name
        ][:top_k]

    weighted = [] if focus_only else list(train_scenarios)
    max_score = max((row["weakness_score"] for row in selected), default=0.0)
    focus_plan = []

    for row in selected:
        if max_score > 0:
            repeats = 1 + math.ceil(focus_multiplier * row["weakness_score"] / max_score)
        else:
            repeats = 1

        scenario = scenarios_by_name[row["training_scenario"]]
        weighted.extend([scenario] * repeats)
        focus_plan.append({
            "evaluated_scenario": row["scenario"],
            "training_scenario": row["training_scenario"],
            "weakness_score": row["weakness_score"],
            "extra_training_copies": repeats,
            "queue_reduction_pct": to_float(row, "queue_reduction_pct"),
            "waiting_reduction_pct": to_float(row, "waiting_reduction_pct"),
            "throughput_improvement_pct": to_float(row, "throughput_improvement_pct"),
            "reward_delta_rl_minus_fixed": to_float(row, "reward_delta_rl_minus_fixed"),
        })

    return weighted, focus_plan


def configure_agent(agent: DQNAgent, lr: float, epsilon_start: float, epsilon_min: float) -> None:
    agent.cfg.lr = lr
    agent.cfg.epsilon_min = epsilon_min
    agent.epsilon = max(agent.epsilon, epsilon_start)
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = lr


def fine_tune(args) -> None:
    ranked_rows = load_ranked_weaknesses(Path(args.eval_csv), args.metric)
    scenarios, focus_plan = build_weighted_training_scenarios(
        network=args.network,
        ranked_rows=ranked_rows,
        top_k=args.top_k,
        focus_multiplier=args.focus_multiplier,
        focus_only=args.focus_only,
    )

    if not scenarios:
        raise RuntimeError("No training scenarios selected for fine-tuning.")

    env_cfg = EnvConfig()
    env_cfg.scenarios = scenarios
    env_cfg.scenario_selection = "random"
    env_cfg.control_mode = args.control_mode
    env_cfg.seed = args.seed
    if args.reward_mode is not None:
        env_cfg.reward_mode = args.reward_mode
    if args.throughput_reward_weight is not None:
        env_cfg.throughput_reward_weight = args.throughput_reward_weight
    if args.switch_penalty is not None:
        env_cfg.switch_penalty = args.switch_penalty
    if args.min_green_time is not None:
        env_cfg.min_green_time = args.min_green_time

    agent_cfg = AgentConfig(lr=args.lr, epsilon_min=args.epsilon_min)
    env = SumoTrafficEnv(cfg=env_cfg, gui=args.gui)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, cfg=agent_cfg)

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    agent.load(args.base_model)
    configure_agent(agent, lr=args.lr, epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min)

    logger = TrainingLogger(log_dir=args.log_dir)
    output_model = Path(args.output_model)
    checkpoint_dir = output_model.parent / "checkpoints"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    focus_path = output_model.parent / "fine_tune_focus_plan.json"
    with open(focus_path, "w", encoding="utf-8") as f:
        json.dump(focus_plan, f, indent=2)

    print("\nFine-tuning focus:")
    for item in focus_plan:
        print(
            f"  {item['training_scenario']:18s} "
            f"score={item['weakness_score']:8.2f} "
            f"copies={item['extra_training_copies']}"
        )
    print(f"\nSelected scenario entries: {len(scenarios)}")
    print(f"Base model: {args.base_model}")
    print(f"Output model: {output_model}\n")

    history = []
    best_reward = float("-inf")

    try:
        for episode in range(args.episodes):
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
                f"FT {episode:4d} | "
                f"scenario={scenario_name:18s} | "
                f"reward={summary['total_reward']:8.1f} | "
                f"avg_queue={summary['avg_queue']:5.1f} | "
                f"sw={summary['switch_rate'] * 100:5.1f}% | "
                f"eps={agent.epsilon:.4f} | "
                f"loss={summary['avg_loss']:.4f} | "
                f"{ep_time:.1f}s"
            )

            if summary["total_reward"] > best_reward:
                best_reward = summary["total_reward"]
                agent.save(str(output_model))

            if (episode + 1) % args.save_every == 0:
                agent.save(str(checkpoint_dir / f"fine_tune_ep_{episode + 1}.pt"))

    finally:
        env.close()

    agent.save(str(output_model))
    logger.save_history(history)
    print(f"\n[fine-tune] Done. Best fine-tune episode reward: {best_reward:.1f}")
    print(f"[fine-tune] Focus plan saved -> {focus_path}")


if __name__ == "__main__":
    fine_tune(parse_args())
