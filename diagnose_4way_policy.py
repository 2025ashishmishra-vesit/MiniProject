"""Greedy, single-scenario diagnostic for a trained 4-way phase-selection DQN.

Example:
    python diagnose_4way_policy.py --model models/best/best.pt --scenario light
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from config.settings import AgentConfig, EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv, traci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose greedy direct-phase choices for one 4-way test scenario."
    )
    parser.add_argument("--model", required=True, help="Trained 14-state, 4-action checkpoint.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="4-way test scenario suffix (light, heavy, balanced, ...) or full name.",
    )
    parser.add_argument("--route-split", choices=["test", "train"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--switch-penalty", type=float, default=None)
    parser.add_argument("--throughput-reward-weight", type=float, default=None)
    parser.add_argument("--reward-mode", choices=["delta_pressure", "absolute_pressure"], default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to logs/diagnostics_4way/<timestamp>_<scenario>.",
    )
    return parser.parse_args()


def select_scenario(name: str, route_split: str) -> dict:
    scenarios = build_scenarios("4_way", split=route_split)
    matches = [
        scenario for scenario in scenarios
        if scenario["name"] == name or scenario["name"].rsplit("/", 1)[-1] == name
    ]
    if len(matches) != 1:
        choices = ", ".join(scenario["name"] for scenario in scenarios)
        raise ValueError(f"Expected one 4-way scenario matching {name!r}; available: {choices}")
    return matches[0]


def make_env(scenario: dict, args: argparse.Namespace) -> SumoTrafficEnv:
    cfg = EnvConfig()
    cfg.control_mode = "rl"
    cfg.seed = args.seed
    cfg.min_green_time = args.min_green_time
    if args.switch_penalty is not None:
        cfg.switch_penalty = args.switch_penalty
    if args.throughput_reward_weight is not None:
        cfg.throughput_reward_weight = args.throughput_reward_weight
    if args.reward_mode is not None:
        cfg.reward_mode = args.reward_mode
    cfg.scenarios = [scenario]
    cfg.scenario_selection = "cycle"
    return SumoTrafficEnv(cfg=cfg, gui=args.gui)


def raw_q_values(agent: DQNAgent, state: np.ndarray) -> list[float]:
    with torch.no_grad():
        values = agent.online(torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0))
    return [float(value) for value in values.squeeze(0).cpu().tolist()]


def get_served_approaches(env: SumoTrafficEnv) -> tuple[list[str], str | None]:
    """Map the active SUMO signal state back to detector approaches when possible."""
    try:
        detector_lanes = {
            approach: {
                traci.lanearea.getLaneID(detector_id)
                for detector_id in env._approach_detector_ids(approach)
            }
            for approach in env.APPROACHES
        }
        signal_state = traci.trafficlight.getRedYellowGreenState(env.tls_id)
        controlled_links = traci.trafficlight.getControlledLinks(env.tls_id)
        served_lanes = {
            link[0]
            for signal, links in zip(signal_state, controlled_links)
            if signal in "Gg"
            for link in links
            if link
        }
        return (
            [approach for approach, lanes in detector_lanes.items() if lanes & served_lanes],
            None,
        )
    except Exception as exc:  # Diagnostics must not interrupt an evaluation run.
        return [], f"could not map served approaches: {exc}"


def print_first_legal(row: dict) -> None:
    print("\nFIRST LEGAL DECISION")
    print(f"time={row['sim_time_before']:.1f}s | elapsed_green={row['elapsed_green']:.1f}s")
    print(
        f"current_action={row['current_action']} | "
        f"current_sumo_phase={row['actual_sumo_phase']} | "
        f"state_phase={row['state_phase_index']}"
    )
    print("queues: " + " ".join(f"{key}={row[f'queue_{key}']}" for key in "NSEW"))
    print("waiting: " + " ".join(f"{key}={row[f'waiting_{key}']:.1f}" for key in "NSEW"))
    print("raw Q:")
    print(" ".join(f"action{i}={row[f'raw_q_{i}']:.4f}" for i in range(4)))
    print(f"mask={row['valid_mask']}")
    print(" ".join(f"masked{i}={row[f'masked_q_{i}']}" for i in range(4)))
    print(
        f"chosen_action={row['chosen_action']} | decision={row['decision']} | "
        f"target_phase={row['target_sumo_phase']}"
    )
    print(
        "reward: "
        f"pressure={row['reward_pressure']:.3f} | "
        f"throughput={row['reward_throughput']:.3f} | "
        f"switch_penalty={row['reward_switch_penalty']:.3f} | "
        f"total={row['reward_total']:.3f}"
    )
    if row["state_phase_mismatch"]:
        print("WARNING: encoded current phase does not match the environment phase index.")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    scenario = select_scenario(args.scenario, args.route_split)
    default_output = Path("logs") / "diagnostics_4way" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_{scenario['name'].rsplit('/', 1)[-1]}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=False)

    env = make_env(scenario, args)
    agent = DQNAgent(env.state_size, env.action_size, AgentConfig())
    agent.load(str(model_path))
    agent.epsilon = 0.0

    rows: list[dict] = []
    first_legal_rows: list[dict] = []
    switch_green_durations: list[float] = []
    per_phase_switches = [0] * env.action_size
    target_counts = [0] * env.action_size
    switched_with_served_largest = 0
    switches_with_served_mapping = 0
    state_mismatches = 0
    awaiting_first_legal = True
    served_mapping_error: str | None = None

    try:
        state = env.reset()
        if len(env.green_phases) != 4:
            raise RuntimeError("This diagnostic is restricted to a 4-way scenario with four green phases.")

        done = False
        while not done:
            if args.max_decisions is not None and len(rows) >= args.max_decisions:
                break

            current_action = env._phase_index
            elapsed_green = float(env._green_elapsed)
            mask = env.valid_actions_mask()
            if not mask[current_action]:
                raise AssertionError("Invalid action mask: active phase action is masked.")
            can_switch = elapsed_green >= env.cfg.min_green_time
            expected_mask = [can_switch or action == current_action for action in range(4)]
            if mask != expected_mask:
                raise AssertionError(f"Unexpected action mask {mask}; expected {expected_mask}.")

            queues = env._get_queues()
            waiting = env._get_waiting_times()
            raw_q = raw_q_values(agent, state)
            masked_q = [value if valid else float("-inf") for value, valid in zip(raw_q, mask)]
            chosen_action = int(np.argmax(masked_q))
            target_counts[chosen_action] += 1
            decision = "HOLD" if chosen_action == current_action else "SWITCH"
            state_phase_index = int(round(float(state[12]) * (len(env.green_phases) - 1)))
            state_phase_mismatch = state_phase_index != current_action
            state_mismatches += int(state_phase_mismatch)
            actual_sumo_phase = int(traci.trafficlight.getPhase(env.tls_id))
            served_approaches, mapping_error = get_served_approaches(env)
            served_mapping_error = served_mapping_error or mapping_error
            largest_queue = max(queues.values())
            served_has_largest_queue = (
                any(queues[approach] == largest_queue for approach in served_approaches)
                if served_approaches else None
            )
            first_legal = awaiting_first_legal and can_switch

            next_state, reward, done, info = env.step(chosen_action)
            components = info["reward_components"]
            applied = bool(info["switch_applied"])
            if applied:
                switch_green_durations.append(float(info["green_time_before_switch"]))
                per_phase_switches[current_action] += 1
                if served_has_largest_queue is not None:
                    switches_with_served_mapping += 1
                    switched_with_served_largest += int(served_has_largest_queue)

            row = {
                "step": int(info["step"]),
                "scenario": scenario["name"],
                "sim_time_before": float(info["sim_time"] - env.cfg.delta_t),
                "sim_time_after": float(info["sim_time"]),
                "current_action": current_action,
                "current_sumo_phase": actual_sumo_phase,
                "actual_sumo_phase": actual_sumo_phase,
                "state_phase_normalized": float(state[12]),
                "state_phase_index": state_phase_index,
                "state_green_normalized": float(state[13]),
                "state_phase_mismatch": state_phase_mismatch,
                "elapsed_green": elapsed_green,
                "valid_mask": json.dumps(mask),
                "chosen_action": chosen_action,
                "decision": decision,
                "target_sumo_phase": int(env.green_phases[chosen_action]),
                "switch_requested": int(info["switch_requested"]),
                "switch_applied": int(info["switch_applied"]),
                "switch_blocked": int(info["switch_blocked"]),
                "first_legal_decision": first_legal,
                "served_approaches": ";".join(served_approaches),
                "served_has_largest_queue": served_has_largest_queue,
                "reward_pressure": float(components["pressure"]),
                "reward_throughput": float(components["throughput"]),
                "reward_switch_penalty": float(components["switch_penalty"]),
                "reward_total": float(reward),
            }
            row.update({f"queue_{key}": int(queues[key]) for key in "NSEW"})
            row.update({f"waiting_{key}": float(waiting[key]) for key in "NSEW"})
            row.update({f"raw_q_{index}": raw_q[index] for index in range(4)})
            row.update({
                f"masked_q_{index}": raw_q[index] if mask[index] else "-inf"
                for index in range(4)
            })
            rows.append(row)
            if first_legal:
                first_legal_rows.append(row)
                print_first_legal(row)
                awaiting_first_legal = False
            if applied:
                awaiting_first_legal = True
            state = next_state
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No decisions were recorded.")
    csv_path = output_dir / "decision_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first_switch_rows = [row for row in first_legal_rows if row["decision"] == "SWITCH"]
    first_hold_rows = [row for row in first_legal_rows if row["decision"] == "HOLD"]
    q_differences = [
        max(row[f"raw_q_{action}"] for action in range(4) if action != row["current_action"])
        - row[f"raw_q_{row['current_action']}"]
        for row in first_legal_rows
    ]
    hold_qs = [row[f"raw_q_{row['current_action']}"] for row in first_legal_rows]
    best_switch_qs = [
        max(row[f"raw_q_{action}"] for action in range(4) if action != row["current_action"])
        for row in first_legal_rows
    ]
    summary = {
        "scenario": scenario["name"],
        "model": str(model_path),
        "epsilon": agent.epsilon,
        "total_decisions": len(rows),
        "total_applied_switches": sum(row["switch_applied"] for row in rows),
        "first_legal_decisions": len(first_legal_rows),
        "switches_at_first_legal": len(first_switch_rows),
        "switches_at_first_legal_pct": 100.0 * len(first_switch_rows) / max(len(first_legal_rows), 1),
        "holds_at_first_legal": len(first_hold_rows),
        "holds_at_first_legal_pct": 100.0 * len(first_hold_rows) / max(len(first_legal_rows), 1),
        "average_green_duration_at_switch": mean(switch_green_durations) if switch_green_durations else 0.0,
        "average_hold_q_at_first_legal": mean(hold_qs) if hold_qs else 0.0,
        "average_best_switch_q_at_first_legal": mean(best_switch_qs) if best_switch_qs else 0.0,
        "average_best_switch_q_minus_hold_q": mean(q_differences) if q_differences else 0.0,
        "best_switch_q_greater_than_hold_count": sum(value > 0 for value in q_differences),
        "best_switch_q_greater_than_hold_pct": 100.0 * sum(value > 0 for value in q_differences) / max(len(q_differences), 1),
        "switches_while_served_approach_has_largest_queue": switched_with_served_largest,
        "switches_with_served_approach_mapping": switches_with_served_mapping,
        "per_phase_switch_counts": {str(action): count for action, count in enumerate(per_phase_switches)},
        "target_phase_selection_counts": {str(action): count for action, count in enumerate(target_counts)},
        "state_phase_mismatches": state_mismatches,
        "served_approach_mapping_note": served_mapping_error,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDIAGNOSTIC SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Summary JSON:  {summary_path}")


if __name__ == "__main__":
    main()
