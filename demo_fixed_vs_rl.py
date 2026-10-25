"""Four-way SUMO GUI: fixed-time baseline vs trained DQN with documented safety/ambulance priority.

Put this file at the MiniProject repository root and run:
    python demo_fixed_vs_rl.py --scenario balanced

The ambulance is inserted by TraCI, not by the static route XML. Emergency priority
is a rule-based override on top of the DQN, NOT a learned DQN capability.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import traci

from agent.dqn_agent import DQNAgent
from config.settings import AgentConfig, EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv

DIRECTIONS = ("E", "S", "W", "N")  # phase indices 0,1,2,3
AMBULANCES = ((70, "N", "E1"), (360, "W", "E3"))
INCOMING = {"N": "-E2", "S": "-E1", "E": "-E3", "W": "E0"}


def arguments():
    p = argparse.ArgumentParser(description="SUMO fixed-time vs supervised DQN ambulance demo")
    p.add_argument("--scenario", default="balanced")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--model", default="models/best/best.pt")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--max-red", type=float, default=65)
    p.add_argument("--max-hold", type=float, default=50)
    p.add_argument("--min-green", type=int, default=20)
    p.add_argument("--delta-t", type=int, default=5)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--rl-only", action="store_true", help="Show only the RL ambulance priority GUI")
    return p.parse_args()


def scenario_for(args):
    matches = [s for s in build_scenarios("4_way", split=args.split)
               if args.scenario in s["name"] and "emergency" not in s["name"]]
    if not matches:
        raise ValueError(f"No 4-way scenario matching {args.scenario!r}")
    return matches[0]


def inject_ambulance(vid, direction, outbound, time, events, mode):
    """Use a compatible, pre-existing ambulance vType in the balanced route file."""
    route_id = "demo_route_" + vid
    traci.route.add(route_id, [INCOMING[direction], outbound])
    traci.vehicle.add(vid, route_id, typeID="ambulance", depart="now",
                      departLane="best", departPos="base", departSpeed="0")
    traci.vehicle.setColor(vid, (255, 0, 255, 255))  # visually distinct magenta
    # Do not set the non-portable has.bluelight.device parameter.
    # The dedicated vehicle type and magenta color are sufficient for this demo.
    events.append(dict(mode=mode, sim_time=time, vehicle_id=vid, event="INSERTED",
                       direction=direction, edge=INCOMING[direction], details="magenta ambulance"))
    print(f"\n🚑 [INSERTED] t={time:.0f}s {vid} from {direction}; watch for MAGENTA vehicle.\n")


def main():
    args = arguments()
    scenario = scenario_for(args)
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"DQN checkpoint not found: {args.model}")
    dest = Path("logs/demo_comparison")
    dest.mkdir(parents=True, exist_ok=True)
    rows, events = [], []

    for mode in (("rl",) if args.rl_only else ("fixed", "rl")):
        cfg = EnvConfig()
        cfg.scenarios = [scenario]
        cfg.control_mode = mode
        cfg.seed = 42
        cfg.delta_t = args.delta_t
        cfg.min_green_time = args.min_green
        cfg.max_steps = args.steps
        env = SumoTrafficEnv(cfg=cfg, gui=not args.no_gui)
        try:
            state = env.reset()
            # The original environment pre-empts BOTH fixed and RL modes for ambulances.
            # Disable that pre-emption ONLY for the fixed-time comparison, so the
            # baseline stays truly fixed-time. In RL mode, retain repository behavior.
            if mode == "fixed":
                env._check_emergency = lambda: (False, None, None, None)
            agent = None
            if mode == "rl":
                agent = DQNAgent(state_size=env.state_size, action_size=env.action_size,
                                 cfg=AgentConfig())
                agent.load(str(args.model))
                agent.epsilon = 0.0

            inserted, observed, finished = set(), set(), set()
            previous_phase, held_since = int(env._phase_index), 0.0
            last_served = [0.0] * env.action_size
            total_arrived = 0
            print(f"\n{'='*68}\n{mode.upper()} GUI: {scenario['name']}\n{'='*68}")

            for step in range(args.steps):
                now = float(traci.simulation.getTime())
                for insertion_time, direction, outbound in AMBULANCES:
                    vid = f"demo_ambulance_{insertion_time}_{direction}"
                    if now >= insertion_time and vid not in inserted:
                        # A failed TraCI insertion must not kill the whole demo.
                        # Record the failure and keep SUMO moving.
                        inserted.add(vid)
                        try:
                            inject_ambulance(vid, direction, outbound, now, events, mode)
                        except traci.TraCIException as exc:
                            events.append(dict(mode=mode, sim_time=now, vehicle_id=vid,
                                               event="INSERTION_FAILED", direction=direction,
                                               edge=INCOMING[direction], details=str(exc)))
                            print(f"[AMBULANCE INSERTION FAILED] t={now:.0f}s {vid}: {exc}")
                            print("Continuing traffic simulation; see emergency_events.csv.")
                valid = env.valid_actions_mask()
                emergency_before = (env._check_emergency() if mode == "rl"
                                    else (False, None, None, None))
                if mode == "fixed":
                    action, reason = 0, "FIXED (no emergency override)"
                else:
                    proposed = int(agent.select_action(state, valid_actions=valid))
                    action, reason = proposed, "DQN"
                    queues = env._get_queues()
                    active = int(env._phase_index)
                    candidates = [i for i in range(min(4, env.action_size))
                                  if queues.get(DIRECTIONS[i], 0) > 0]
                    neglected = [i for i in candidates if i != active and
                                 now - last_served[i] >= args.max_red]
                    if neglected:
                        target = max(neglected, key=lambda i: (now - last_served[i],
                                                               queues[DIRECTIONS[i]]))
                        if bool(valid[target]):
                            action, reason = target, "FAIRNESS (max red)"
                    elif now - held_since >= args.max_hold:
                        other = [i for i in candidates if i != active]
                        if other:
                            target = max(other, key=lambda i: queues[DIRECTIONS[i]])
                            if bool(valid[target]):
                                action, reason = target, "FAIRNESS (max hold)"
                    if emergency_before[0]:
                        reason = "AMBULANCE PRIORITY (rule-based; overrides DQN)"

                state, reward, done, info = env.step(int(action))
                now = float(info["sim_time"])
                phase = int(env._phase_index)
                if phase != previous_phase:
                    held_since = now
                last_served[phase] = now
                previous_phase = phase
                arrived = int(info.get("arrived_vehicles", 0))
                total_arrived += arrived

                running = set(traci.vehicle.getIDList())
                for vid in sorted(inserted):
                    if vid in running:
                        edge = traci.vehicle.getRoadID(vid)
                        if vid not in observed:
                            observed.add(vid)
                            events.append(dict(mode=mode, sim_time=now, vehicle_id=vid,
                                               event="VISIBLE", direction=vid.rsplit("_", 1)[-1],
                                               edge=edge, details="Vehicle has entered simulation"))
                            print(f"🚑 [VISIBLE] t={now:.0f}s {vid} edge={edge}")
                        # Keep the GUI centered on the ambulance; do not hide other vehicles.
                        if not args.no_gui:
                            try:
                                view = traci.gui.DEFAULT_VIEW
                                x, y = traci.vehicle.getPosition(vid)
                                # traci.gui.setOffset(view, x, y)
                                # traci.gui.setZoom(view, 350)
                            except traci.TraCIException:
                                pass
                    elif vid in observed and vid not in finished:
                        finished.add(vid)
                        event = "ARRIVED" if vid in traci.simulation.getArrivedIDList() else "LEFT_SIMULATION"
                        events.append(dict(mode=mode, sim_time=now, vehicle_id=vid,
                                           event=event, direction=vid.rsplit("_", 1)[-1],
                                           edge="", details="Not present after this simulation window"))
                        print(f"🚑 [{event}] t={now:.0f}s {vid}")

                if emergency_before[0]:
                    events.append(dict(mode=mode, sim_time=now, vehicle_id=emergency_before[1],
                                       event="PRIORITY_REQUESTED", direction=emergency_before[2],
                                       edge="", details=f"requested_phase={emergency_before[3]}; actual_phase={phase}"))
                rows.append(dict(mode="fixed_time" if mode == "fixed" else "DQN_with_safety_and_emergency",
                                 scenario=scenario["name"], sim_time=now, phase_action=phase,
                                 requested_action=action, decision_source=reason,
                                 q_north=info["queues"].get("N", 0), q_south=info["queues"].get("S", 0),
                                 q_east=info["queues"].get("E", 0), q_west=info["queues"].get("W", 0),
                                 total_queue=info.get("total_queue", 0), arrivals_this_step=arrived,
                                 arrivals_total=total_arrived, emergency_active=bool(info.get("emergency_active")),
                                 emergency_vehicle_id=info.get("emergency_vehicle_id") or ""))
                if step % max(1, 30 // args.delta_t) == 0 or emergency_before[0]:
                    print(f"t={now:6.0f}s phase={phase} queued={info.get('total_queue', 0)} "
                          f"arrived={total_arrived} | {reason}")
                if done:
                    break
            print(f"{mode.upper()} complete: {total_arrived} arrivals; ambulances visible: {len(observed)}.\n")
        finally:
            env.close()

    for name, data in (("comparison_trace.csv", rows), ("emergency_events.csv", events)):
        if data:
            with (dest / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print(f"Wrote logs to {dest.resolve()}")
    print("NOTE: ambulance priority and fairness are explicit rule-based safety overrides, not learned by DQN.")


if __name__ == "__main__":
    main()
