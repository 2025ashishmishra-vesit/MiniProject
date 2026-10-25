"""
Compare a trained DQN signal controller against fixed-time control.

Examples:
    python evaluate_compare.py --network all --model models/best/best.pt
    python evaluate_compare.py --network 4_way --episodes-per-scenario 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean

from agent.dqn_agent import DQNAgent
from config.settings import AgentConfig, EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RL model vs fixed-time signal control.")
    parser.add_argument(
        "--model",
        default="models/best/best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--network",
        choices=["default", "3_way", "4_way", "all"],
        default="all",
        help="Scenario set to evaluate.",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=1,
        help="Repeated evaluation episodes for each scenario and controller.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name filter, such as balanced, heavy, or 4_way/ns_heavy.",
    )
    parser.add_argument(
        "--route-split",
        choices=["test", "train"],
        default="test",
        help="Route split to evaluate. Default is held-out test routes.",
    )
    parser.add_argument(
        "--rl-control-mode",
        choices=["hybrid", "rl"],
        default="rl",
        help="hybrid matches training behavior; rl always applies model actions.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["delta_pressure", "absolute_pressure"],
        default=None,
        help="Reward formula to use while reporting evaluation reward.",
    )
    parser.add_argument("--throughput-reward-weight", type=float, default=None)
    parser.add_argument("--switch-penalty", type=float, default=None)
    parser.add_argument("--min-green-time", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Base SUMO seed.")
    parser.add_argument("--gui", action="store_true", help="Run SUMO GUI during evaluation.")
    parser.add_argument("--strategy", choices=["both", "fixed", "rl"], default="both", help="Which controller to evaluate.")
    parser.add_argument("--debug-steps", action="store_true", help="Log before/after each RL action and simulation step near 200s.")
    parser.add_argument(
        "--log-rl-switches",
        action="store_true",
        help="Print every applied RL switch and save its timing to rl_switch_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/evaluation",
        help="Directory for CSV and JSON evaluation outputs.",
    )
    return parser.parse_args()


def make_env_cfg(
    scenario: dict | None,
    control_mode: str,
    seed: int,
    reward_mode: str | None = None,
    throughput_reward_weight: float | None = None,
    switch_penalty: float | None = None,
    min_green_time: int | None = None,
) -> EnvConfig:
    cfg = EnvConfig()
    cfg.control_mode = control_mode
    cfg.seed = seed
    if reward_mode is not None:
        cfg.reward_mode = reward_mode
    if throughput_reward_weight is not None:
        cfg.throughput_reward_weight = throughput_reward_weight
    if switch_penalty is not None:
        cfg.switch_penalty = switch_penalty
    if min_green_time is not None:
        cfg.min_green_time = min_green_time
    if scenario is not None:
        cfg.scenarios = [scenario]
        cfg.scenario_selection = "cycle"
    return cfg


def run_episode(
    env: SumoTrafficEnv,
    agent: DQNAgent | None,
    log_switches: bool = False,
    debug_steps: bool = False,
) -> dict:
    state = env.reset()
    done = False

    rewards = []
    queues = []
    vehicle_waits = []
    arrivals = []
    switch_requested = 0
    switch_applied = 0
    switch_blocked = 0
    rl_steps = 0
    fixed_steps = 0
    final_sim_time = 0.0
    switch_events = []
    time_series = []

    step_count = 0
    while not done:
        step_count += 1
        trace = debug_steps and (step_count % 25 == 0 or final_sim_time >= 175)
        if trace:
            print(f"[DEBUG] before action | step={step_count} last_sim_time={final_sim_time:.1f}", flush=True)
        action = agent.select_action(state, valid_actions=env.valid_actions_mask()) if agent is not None else 0
        if trace:
            print(f"[DEBUG] before env.step | step={step_count} action={action}", flush=True)
        state, reward, done, info = env.step(action)
        if trace:
            print(f"[DEBUG] after env.step | step={step_count} sim_time={info.get('sim_time')} phase={info.get('phase')} mode={info.get('mode')}", flush=True)

        time_series.append({
        "sim_time": info["sim_time"],

        "total_queue": info["total_queue"],

        "queue_north": info["queues"]["N"],
        "queue_south": info["queues"]["S"],
        "queue_east": info["queues"]["E"],
        "queue_west": info["queues"]["W"],

        "avg_waiting_time": info["avg_vehicle_waiting_time"],

        "arrived_vehicles": info["arrived_vehicles"],

        "phase": info["phase"],
        "mode": info["mode"],
         # Emergency information
        "emergency_active": info["emergency_active"],
        "emergency_vehicle_id": info["emergency_vehicle_id"],
        "emergency_direction": info["emergency_direction"],
        "emergency_action": info["emergency_action"],
        })

        rewards.append(reward)
        queues.append(info["total_queue"])
        vehicle_waits.append(info["avg_vehicle_waiting_time"])
        arrivals.append(info["arrived_vehicles"])
        switch_requested += int(info.get("switch_requested", 0))
        switch_applied += int(info.get("switch_applied", 0))
        switch_blocked += int(info.get("switch_blocked", 0))
        final_sim_time = info["sim_time"]
        if info["mode"] == "rl" and info.get("switch_applied", 0):
            from_action = int(info["switch_from_index"])
            to_action = int(info["switch_to_index"])
            event = {
                "sim_time": float(info["sim_time"]),
                "green_time": float(info["green_time_before_switch"]),
                "from_action": from_action,
                "to_action": to_action,
                "from_green_phase": int(env.green_phases[from_action]),
                "to_green_phase": int(env.green_phases[to_action]),
                "step": int(info["step"]),
            }
            switch_events.append(event)
            if log_switches:
                print(
                    "[SWITCH APPLIED] "
                    f"t={event['sim_time']:.1f}s | "
                    f"green_time={event['green_time']:.1f}s | "
                    f"min_green={env.cfg.min_green_time:.1f}s | "
                    f"action {event['from_action']} -> {event['to_action']} | "
                    f"green phase {event['from_green_phase']} -> {event['to_green_phase']}"
                )
        if info["mode"] == "rl":
            rl_steps += 1
        else:
            fixed_steps += 1

    total_arrived = sum(arrivals)
    throughput_rate = total_arrived / max(final_sim_time / 3600.0, 1e-9)

    return {
        "total_reward": float(sum(rewards)),
        "avg_reward": float(mean(rewards)),
        "avg_queue": float(mean(queues)),
        "max_queue": float(max(queues)),
        # Final episode-wide average accumulated waiting time
        # per vehicle.
        "avg_waiting_time": float(
            vehicle_waits[-1] if vehicle_waits else 0.0
        ),
        "max_waiting_time": float(
            max(vehicle_waits) if vehicle_waits else 0.0
        ),
        "throughput_vehicles": float(total_arrived),
        "throughput_rate_vph": float(throughput_rate),
        "sim_time": float(final_sim_time),
        "switch_requested": switch_requested,
        "switch_applied": switch_applied,
        "switch_blocked": switch_blocked,
        "switch_rate": switch_applied / max(len(rewards), 1),
        "rl_steps": rl_steps,
        "fixed_steps": fixed_steps,
        "switch_events": switch_events,
        "time_series": time_series
    }


def evaluate_strategy(
    scenario: dict | None,
    strategy: str,
    agent: DQNAgent | None,
    control_mode: str,
    episodes: int,
    base_seed: int,
    gui: bool,
    reward_mode: str | None = None,
    throughput_reward_weight: float | None = None,
    switch_penalty: float | None = None,
    min_green_time: int | None = None,
    log_switches: bool = False,
    debug_steps: bool = False,
) -> dict:
    runs = []
    all_time_series = []

    for repeat in range(episodes):
        env_cfg = make_env_cfg(
            scenario,
            control_mode,
            seed=base_seed + repeat,
            reward_mode=reward_mode,
            throughput_reward_weight=throughput_reward_weight,
            switch_penalty=switch_penalty,
            min_green_time=min_green_time,
        )
        env = SumoTrafficEnv(cfg=env_cfg, gui=gui)
        try:
            result = run_episode(
                env,
                agent,
                log_switches=log_switches,
                debug_steps=debug_steps
            )

            runs.append(result)

            for point in result["time_series"]:
                all_time_series.append({
                    "episode": repeat + 1,
                    **point
                })

        finally:
            env.close()

    scenario_name = scenario["name"] if scenario is not None else "default"
    return {
        "scenario": scenario_name,
        "strategy": strategy,
        "episodes": episodes,
        "total_reward": mean(run["total_reward"] for run in runs),
        "avg_reward": mean(run["avg_reward"] for run in runs),
        "avg_queue": mean(run["avg_queue"] for run in runs),
        "max_queue": mean(run["max_queue"] for run in runs),
        "avg_waiting_time": mean(run["avg_waiting_time"] for run in runs),
        "max_waiting_time": mean(run["max_waiting_time"] for run in runs),
        "throughput_vehicles": mean(run["throughput_vehicles"] for run in runs),
        "throughput_rate_vph": mean(run["throughput_rate_vph"] for run in runs),
        "sim_time": mean(run["sim_time"] for run in runs),
        "switch_requested": mean(run["switch_requested"] for run in runs),
        "switch_applied": mean(run["switch_applied"] for run in runs),
        "switch_blocked": mean(run["switch_blocked"] for run in runs),
        "switch_rate": mean(run["switch_rate"] for run in runs),
        "rl_steps": mean(run["rl_steps"] for run in runs),
        "fixed_steps": mean(run["fixed_steps"] for run in runs),
        "switch_events": [
    {"episode": repeat + 1, **event}
    for repeat, run in enumerate(runs)
    for event in run["switch_events"]
    ],

    "time_series": all_time_series,
    }


def build_comparison_rows(rows: list[dict]) -> list[dict]:
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], {})[row["strategy"]] = row

    comparisons = []
    for scenario, strategies in by_scenario.items():
        rl = strategies["rl_model"]
        fixed = strategies["fixed_time"]
        queue_delta = fixed["avg_queue"] - rl["avg_queue"]
        wait_delta = fixed["avg_waiting_time"] - rl["avg_waiting_time"]
        throughput_delta = rl["throughput_rate_vph"] - fixed["throughput_rate_vph"]
        comparisons.append({
            "scenario": scenario,
            "rl_avg_queue": rl["avg_queue"],
            "fixed_avg_queue": fixed["avg_queue"],
            "queue_delta_fixed_minus_rl": queue_delta,
            "queue_reduction_pct":
                100.0 * queue_delta / max(fixed["avg_queue"], 1e-9),
            "rl_avg_waiting_time": rl["avg_waiting_time"],
            "fixed_avg_waiting_time": fixed["avg_waiting_time"],
            "waiting_delta_fixed_minus_rl": wait_delta,
            "waiting_reduction_pct":
                100.0 * wait_delta /
                max(fixed["avg_waiting_time"], 1e-9),
            "rl_throughput_vehicles": rl["throughput_vehicles"],
            "fixed_throughput_vehicles": fixed["throughput_vehicles"],
            "rl_throughput_rate_vph": rl["throughput_rate_vph"],
            "fixed_throughput_rate_vph": fixed["throughput_rate_vph"],
            "throughput_delta_rl_minus_fixed_vph":
                throughput_delta,
            "throughput_improvement_pct":
                100.0 * throughput_delta /
                max(fixed["throughput_rate_vph"], 1e-9),
            "rl_switch_rate": rl["switch_rate"],
            "fixed_switch_rate": fixed["switch_rate"],
            "rl_switch_applied": rl["switch_applied"],
            "fixed_switch_applied": fixed["switch_applied"],
        })
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    scenarios = (
        build_scenarios(args.network, split=args.route_split)
        if args.network != "default"
        else [None]
    )

    # Evaluate base scenarios only.
    # Each base scenario is switched to its matching emergency route later.
    scenarios = [
        scenario
        for scenario in scenarios
        if scenario is None
        or "_emergency" not in scenario["name"]
    ]

    if args.scenario:
        scenarios = [
            scenario for scenario in scenarios
            if scenario is not None and args.scenario in scenario["name"]
        ]
        if not scenarios:
            raise ValueError(f"No scenarios matched filter: {args.scenario}")

    if args.strategy != "fixed" and not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    agent = None
    if args.strategy in ("both", "rl"):
        probe_env = SumoTrafficEnv(
            cfg=make_env_cfg(
                scenarios[0], args.rl_control_mode, args.seed,
                reward_mode=args.reward_mode,
                throughput_reward_weight=args.throughput_reward_weight,
                switch_penalty=args.switch_penalty,
                min_green_time=args.min_green_time,
            ), gui=False,
        )
        try:
            agent = DQNAgent(
                state_size=probe_env.state_size,
                action_size=probe_env.action_size,
                cfg=AgentConfig(),
            )
        finally:
            probe_env.close()
        agent.load(args.model)
        agent.epsilon = 0.0

    rows = []
    timeseries_rows = []
    for scenario in scenarios:
        scenario_name = scenario["name"] if scenario is not None else "default"

        # -----------------------------------------------------
        # USE MATCHING EMERGENCY ROUTE
        # -----------------------------------------------------

        if scenario is not None:

            original_route = Path(scenario["route_file"])

            # Avoid processing an emergency file twice
            if "_emergency" not in original_route.stem:

                base_name = original_route.name.removesuffix(".rou.xml")

                emergency_route = (
                    original_route.parent
                    / f"{base_name}_emergency.rou.xml"
                )

                if emergency_route.exists():

                    scenario["route_file"] = str(emergency_route)

                    print(
                        f"[EMERGENCY ROUTE] "
                        f"{scenario_name} -> "
                        f"{emergency_route}"
                    )

                else:

                    print(
                        f"[WARNING] Emergency route not found for "
                        f"{scenario_name}: {emergency_route}"
                    )

        if args.strategy in ("both", "rl"):
            print(f"[eval] {scenario_name}: RL model")

            rl_result = evaluate_strategy(
                scenario=scenario,
                strategy="rl_model",
                agent=agent,
                control_mode=args.rl_control_mode,
                episodes=args.episodes_per_scenario,
                base_seed=args.seed,
                gui=args.gui,
                reward_mode=args.reward_mode,
                throughput_reward_weight=args.throughput_reward_weight,
                switch_penalty=args.switch_penalty,
                min_green_time=args.min_green_time,
                log_switches=args.log_rl_switches,
                debug_steps=args.debug_steps,
            )

            rl_timeseries = rl_result.pop("time_series", [])

            for point in rl_timeseries:
                timeseries_rows.append({
                    "scenario": scenario_name,
                    "strategy": "rl_model",
                    **point
                })

            rows.append(rl_result)

        if args.strategy in ("both", "fixed"):
            print(f"[eval] {scenario_name}: fixed time")

            fixed_result = evaluate_strategy(
                scenario=scenario,
                strategy="fixed_time",
                agent=None,
                control_mode="fixed",
                episodes=args.episodes_per_scenario,
                base_seed=args.seed,
                gui=args.gui,
                reward_mode=args.reward_mode,
                throughput_reward_weight=args.throughput_reward_weight,
                switch_penalty=args.switch_penalty,
                min_green_time=args.min_green_time,
            )

            fixed_timeseries = fixed_result.pop("time_series", [])

            for point in fixed_timeseries:
                timeseries_rows.append({
                    "scenario": scenario_name,
                    "strategy": "fixed_time",
                    **point
                })

            rows.append(fixed_result)

    comparisons = build_comparison_rows(rows) if args.strategy == "both" else []

    output_dir = Path(args.output_dir)

    switch_events = []

    for row in rows:

        events = row.pop("switch_events", [])

        if row["strategy"] == "rl_model":
            switch_events.extend({
                "scenario": row["scenario"],
                **event
            } for event in events)


    # WRITE FILES ONLY AFTER THE LOOP IS COMPLETE

    write_csv(
        output_dir / "strategy_metrics.csv",
        rows
    )

    write_csv(
        output_dir / "comparison.csv",
        comparisons
    )

    print(
        f"[DEBUG] Time-series rows collected: "
        f"{len(timeseries_rows)}"
    )

    write_csv(
        output_dir / "timeseries.csv",
        timeseries_rows
    )


    if args.log_rl_switches:

        write_csv(
            output_dir / "rl_switch_events.csv",
            switch_events
        )

    if comparisons:
        print("\nScenario comparison: positive queue/waiting reduction and throughput improvement mean RL is better.")
    for row in comparisons:
        print(
            f"{row['scenario']:18s} | "
            f"queue reduction={row['queue_reduction_pct']:7.2f}% | "
            f"waiting reduction={row['waiting_reduction_pct']:7.2f}% | "
            f"throughput improvement={row['throughput_improvement_pct']:7.2f}% | "
            f"rl switch={row['rl_switch_rate'] * 100:6.2f}% | "
        )

    print(f"\n[eval] Saved results to {output_dir}")


if __name__ == "__main__":
    main()
