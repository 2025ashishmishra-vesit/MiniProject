"""Run a scenario and print per-lane and per-approach E2 detector readings.

Example:
    python check_detectors.py --network 4_way --scenario light
"""

from __future__ import annotations

import argparse

from config.settings import EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv, traci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check lane-area detector coverage.")
    parser.add_argument("--network", choices=["3_way", "4_way"], default="4_way")
    parser.add_argument("--scenario", default="light", help="Scenario suffix or full scenario name.")
    parser.add_argument("--route-split", choices=["test", "train"], default="test")
    parser.add_argument("--steps", type=int, default=120, help="Number of five-second control windows.")
    parser.add_argument("--print-every", type=int, default=5, help="Print every N control windows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def select_scenario(network: str, name: str, route_split: str) -> dict:
    matches = [
        scenario for scenario in build_scenarios(network, split=route_split)
        if scenario["name"] == name or scenario["name"].rsplit("/", 1)[-1] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {network} scenario matching {name!r}; found {len(matches)}.")
    return matches[0]


def format_approach(env: SumoTrafficEnv, approach: str) -> str:
    lane_readings = []
    total_count = 0
    total_queue = 0
    for detector_id in env._approach_detector_ids(approach):
        count = traci.lanearea.getLastStepVehicleNumber(detector_id)
        queue = traci.lanearea.getJamLengthVehicle(detector_id)
        total_count += count
        total_queue += queue
        lane_readings.append(f"{detector_id}:count={count},queue={queue}")
    lane_text = " | ".join(lane_readings) if lane_readings else "no detectors"
    return f"{approach}: {lane_text} | aggregate_count={total_count},aggregate_queue={total_queue}"


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.print_every <= 0:
        raise ValueError("--steps and --print-every must be positive.")

    scenario = select_scenario(args.network, args.scenario, args.route_split)
    cfg = EnvConfig(control_mode="fixed", seed=args.seed, scenarios=[scenario], scenario_selection="cycle")
    env = SumoTrafficEnv(cfg=cfg, gui=args.gui)
    try:
        env.reset()
        print(f"Detector check: {scenario['name']}")
        for step in range(args.steps):
            _, _, _, info = env.step(0)  # Fixed control ignores the supplied RL action.
            if step % args.print_every == 0 or step == args.steps - 1:
                print(f"time={info['sim_time']:.1f}s | SUMO total vehicles={traci.vehicle.getIDCount()}")
                for approach in env.APPROACHES:
                    print("  " + format_approach(env, approach))
    finally:
        env.close()


if __name__ == "__main__":
    main()
