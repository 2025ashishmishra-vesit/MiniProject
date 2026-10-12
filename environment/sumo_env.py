"""
SUMO Traffic Signal Environment.

The RL action space is fixed at four outputs across networks.  Each action
selects the green phase at the same index in the configured green_phases list.
For example, action 2 selects green_phases[2].  A 3-way network masks action
3 because it has no fourth configured green phase.

Each network declares its own green/yellow phase sequence. This lets the same
agent work with 3-way and 4-way intersections without hard-coding only two
directions. Yellow time is included inside the normal decision window.
"""

import numpy as np
import traci

from config.settings import EnvConfig


class SumoTrafficEnv:
    APPROACHES = ["N", "S", "E", "W"]
    ACTION_SIZE = 4

    def _get_emergency_vehicles(self):

        emergency_vehicles = []

        for veh_id in traci.vehicle.getIDList():

            veh_type = traci.vehicle.getTypeID(veh_id)

            if veh_type == "ambulance":

                print(
                    f"[AMBULANCE EXISTS] "
                    f"time={traci.simulation.getTime():.1f}s | "
                    f"id={veh_id} | "
                    f"edge={traci.vehicle.getRoadID(veh_id)}"
                )

                emergency_vehicles.append(veh_id)

        return emergency_vehicles



    def _get_emergency_direction(self, veh_id):

        edge = traci.vehicle.getRoadID(veh_id)

        # 3-way intersection
        if len(self.green_phases) == 3:

            if edge == "-E2":
                return "north"

            elif edge == "-E1":
                return "south"

            elif edge == "E0":
                return "east"

        # 4-way intersection
        elif len(self.green_phases) == 4:

            if edge == "-E3":
                return "east"

            elif edge == "-E1":
                return "south"

            elif edge == "E0":
                return "west"

            elif edge == "-E2":
                return "north"

        return None    


    def _get_emergency_action(self, direction):

        # 3-way intersection
        if len(self.green_phases) == 3:

            actions = {
                "east": 0,
                "south": 1,
                "north": 2,
            }

        # 4-way intersection
        else:

            actions = {
                "east": 0,
                "south": 1,
                "west": 2,
                "north": 3,
            }

        return actions.get(direction)


    def _is_emergency_close(self, veh_id, detection_distance=100.0):

        lane_id = traci.vehicle.getLaneID(veh_id)

        if not lane_id:
            return False

        # Ignore internal junction lanes
        if lane_id.startswith(":"):
            return False

        lane_length = traci.lane.getLength(lane_id)
        position = traci.vehicle.getLanePosition(veh_id)

        distance_to_junction = lane_length - position

        return distance_to_junction <= detection_distance

    def _check_emergency(self):

        emergency_vehicles = self._get_emergency_vehicles()

        for veh_id in emergency_vehicles:

            print(
                f"[DEBUG] {veh_id} "
                f"edge={traci.vehicle.getRoadID(veh_id)} "
                f"lane={traci.vehicle.getLaneID(veh_id)} "
                f"pos={traci.vehicle.getLanePosition(veh_id):.2f}"
            )

            if not self._is_emergency_close(veh_id):
                continue

            direction = self._get_emergency_direction(veh_id)

            if direction is None:
                continue

            emergency_action = self._get_emergency_action(direction)

            if emergency_action is None:
                continue

            return True, veh_id, direction, emergency_action

        return False, None, None, None


    def __init__(self, cfg: EnvConfig | None = None, gui: bool = False):
        self.cfg = cfg or EnvConfig()
        self.gui = gui

        # A phase change can never happen faster than one decision interval
        # (delta_t seconds), since the agent only gets to act once per
        # delta_t seconds of simulated time. Clamp min_green_time up to
        # delta_t so this is guaranteed by config, not just by accident.
        if self.cfg.min_green_time < self.cfg.delta_t:
            self.cfg.min_green_time = self.cfg.delta_t

        self.tls_id = self.cfg.tls_id
        self.detector_ids = dict(self.cfg.detector_ids)
        self.green_phases = list(self.cfg.green_phases)
        self.yellow_phases = list(self.cfg.yellow_phases)

        self.scenarios = list(getattr(self.cfg, "scenarios", []))
        self.scenario_selection = getattr(self.cfg, "scenario_selection", "cycle")
        self.current_scenario = None
        self._scenario_rng = np.random.default_rng(self.cfg.seed)

        self._episode_index = 0
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_mode = "rl"
        self._phase_index = 0
        self._fixed_elapsed = 0
        self._green_elapsed = 0
        self._traci_started = False
        self._green_elapsed = 0
        self._traci_started = False
        self.emergency_active = False

    def reset(self) -> np.ndarray:
        """Start or restart SUMO and return the initial state."""
        if self._traci_started:
            traci.close()

        
        self.current_scenario = self._select_scenario()
        self._apply_scenario_settings(self.current_scenario)

        sumo_binary = "sumo-gui" if self.gui else "sumo"
        traci.start(self._build_sumo_cmd(sumo_binary, self.current_scenario))

        self._traci_started = True
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_mode = "rl"
        self._phase_index = 0
        self._fixed_elapsed = 0
        self._green_elapsed = 0

        self.emergency_active = False
        # Reset per-vehicle waiting-time statistics.
        self._vehicle_waiting_history = {}
        self._episode_index += 1
        self._set_current_green_phase()

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one RL decision step."""
        previous_queues = self._get_queues()
        previous_waiting_by_approach = self._get_waiting_times()
        previous_waiting = sum(previous_waiting_by_approach.values())
        is_normal = all(q < self.cfg.normal_queue_threshold for q in previous_queues.values())

        emergency, emergency_id, direction, emergency_action = self._check_emergency()

        if self.emergency_active and not emergency:
            print(
                 f"[EMERGENCY CLEARED] "
                 f"time={traci.simulation.getTime():.1f}s | "
                 f"Returning to normal control"
            )

        self.emergency_active = False

        # Emergency currently active
        if emergency:
            self.emergency_active = True

            selected_action = emergency_action

            print(
                f"[EMERGENCY] "
                f"t={traci.simulation.getTime():.1f}s | "
                f"id={emergency_id} | "
                f"direction={direction} | "
                f"RL action={action} | "
                f"override action={emergency_action}"
            )

        else:
            selected_action = action

        control_info = {
            "arrived_vehicles": 0,
            "switch_requested": 0,
            "switch_applied": 0,
            "switch_blocked": 0,
            "green_time_before_switch": None,
            "switch_from_index": None,
            "switch_to_index": None,
        }


        if emergency:
            self._current_mode = "emergency"
            control_info = self._run_rl_control_window(selected_action)

        elif self.cfg.control_mode == "fixed":
            self._current_mode = "fixed"
            control_info = self._run_fixed_control_window()

        elif self.cfg.control_mode == "hybrid" and is_normal:
            self._current_mode = "fixed"
            control_info = self._run_fixed_control_window()

        else:
            self._current_mode = "rl"
            control_info = self._run_rl_control_window(selected_action)


        self._step_count += 1
        state = self._get_state()
        current_queues = self._get_queues()
        current_waiting_by_approach = self._get_waiting_times()
        current_waiting = sum(current_waiting_by_approach.values())
        reward_components = self._get_reward_components(
            previous_queue=sum(previous_queues.values()),
            previous_waiting=previous_waiting,
            current_queue=sum(current_queues.values()),
            current_waiting=current_waiting,
            arrived_vehicles=control_info["arrived_vehicles"],
            switch_applied=self._current_mode == "rl" and bool(control_info["switch_applied"]),
            switch_blocked=self._current_mode == "rl" and bool(control_info["switch_blocked"]),
            switch_green_time=control_info["green_time_before_switch"],
        )
        reward = reward_components["total"]
        sim_time = traci.simulation.getTime()
        self._episode_reward += reward
        done = self._step_count >= self.cfg.max_steps

        info = {
            "mode": self._current_mode,
            "queues": current_queues,
            "waiting_by_approach": current_waiting_by_approach,
            "total_queue": sum(current_queues.values()),
            "detector_waiting_time": current_waiting,
            "avg_vehicle_waiting_time": self._get_average_vehicle_waiting_time(),
            "arrived_vehicles": control_info["arrived_vehicles"],
            "switch_requested": control_info["switch_requested"],
            "switch_applied": control_info["switch_applied"],
            "switch_blocked": control_info["switch_blocked"],
            # This is deliberately the duration before resetting the green
            # timer.  The current green_elapsed value is the time spent in
            # the *new* phase after a switch window has completed.
            "green_time_before_switch": control_info["green_time_before_switch"],
            "switch_from_index": control_info["switch_from_index"],
            "switch_to_index": control_info["switch_to_index"],
            "sim_time": sim_time,
            "phase": self.green_phases[self._phase_index],
            "phase_index": self._phase_index,
            "step": self._step_count,
            "ep_reward": self._episode_reward,
            "scenario": self.current_scenario.get("name") if self.current_scenario else "default",
            "reward_components": reward_components,
            "emergency_active": self.emergency_active,
            "emergency_vehicle_id": emergency_id,
            "emergency_direction": direction,
            "emergency_action": emergency_action,
        }
        return state, reward, done, info

    def close(self):
        if self._traci_started:
            traci.close()
            self._traci_started = False

    def _select_scenario(self) -> dict | None:
        if not self.scenarios:
            return None
        if self.scenario_selection == "random":
            index = int(self._scenario_rng.integers(len(self.scenarios)))
            return self.scenarios[index]
        return self.scenarios[self._episode_index % len(self.scenarios)]

    def _apply_scenario_settings(self, scenario: dict | None) -> None:
        self.tls_id = self.cfg.tls_id
        self.detector_ids = dict(self.cfg.detector_ids)
        self.green_phases = list(self.cfg.green_phases)
        self.yellow_phases = list(self.cfg.yellow_phases)

        if not scenario:
            return

        self.tls_id = scenario.get("tls_id", self.tls_id)
        self.detector_ids.update(scenario.get("detector_ids", {}))
        self.green_phases = list(scenario.get("green_phases", self.green_phases))
        self.yellow_phases = list(scenario.get("yellow_phases", self.yellow_phases))

        if len(self.green_phases) != len(self.yellow_phases):
            raise ValueError("green_phases and yellow_phases must have the same length.")
        if not self.green_phases:
            raise ValueError("At least one green phase is required.")

    def _build_sumo_cmd(self, sumo_binary: str, scenario: dict | None) -> list[str]:
        cmd = [sumo_binary]

        if scenario:
            cmd.extend(["-n", scenario["net_file"], "-r", scenario["route_file"]])
            if scenario.get("additional_file"):
                cmd.extend(["-a", scenario["additional_file"]])
        else:
            cmd.extend(["-c", self.cfg.sumo_cfg_file])

        cmd.extend([
            "--no-step-log",
            "--waiting-time-memory", "4000",
            "--time-to-teleport", "-1",
            "--seed", str(self.cfg.seed + self._episode_index),
        ])
        return cmd

    def _set_current_green_phase(self) -> None:
        traci.trafficlight.setPhase(self.tls_id, self.green_phases[self._phase_index])

    def _run_seconds(self, seconds: int) -> int:
        arrived = 0

        for _ in range(max(0, seconds)):
            traci.simulationStep()

            # Store accumulated waiting time of every vehicle
            # currently present in the simulation.
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    waiting = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    self._vehicle_waiting_history[vehicle_id] = waiting
                except traci.TraCIException:
                    pass

            arrived += traci.simulation.getArrivedNumber()

        return arrived

    def _run_rl_control_window(self, action: int) -> dict:
        if not 0 <= action < len(self.green_phases):
            raise ValueError(
                f"Action {action} does not select a configured green phase for this "
                f"{len(self.green_phases)}-phase network."
            )

        phase_change_requested = action != self._phase_index
        control_info = {
            "arrived_vehicles": 0,
            "switch_requested": int(phase_change_requested),
            "switch_applied": 0,
            "switch_blocked": 0,
            "green_time_before_switch": None,
            "switch_from_index": None,
            "switch_to_index": None,
        }

        if phase_change_requested:
            control_info["green_time_before_switch"] = self._green_elapsed
            if self._green_elapsed < self.cfg.min_green_time:
                self._set_current_green_phase()
                control_info["arrived_vehicles"] = self._run_seconds(self.cfg.delta_t)
                control_info["switch_blocked"] = 1
                self._green_elapsed += self.cfg.delta_t
                return control_info

            yellow_seconds = min(self.cfg.yellow_time, self.cfg.delta_t)
            traci.trafficlight.setPhase(self.tls_id, self.yellow_phases[self._phase_index])
            arrived = self._run_seconds(yellow_seconds)

            control_info["switch_from_index"] = self._phase_index
            self._phase_index = action
            control_info["switch_to_index"] = self._phase_index
            self._set_current_green_phase()
            self._green_elapsed = 0
            arrived += self._run_seconds(self.cfg.delta_t - yellow_seconds)
            self._green_elapsed += self.cfg.delta_t - yellow_seconds
            control_info["arrived_vehicles"] = arrived
            control_info["switch_applied"] = 1
            return control_info

        self._set_current_green_phase()
        control_info["arrived_vehicles"] = self._run_seconds(self.cfg.delta_t)
        self._green_elapsed += self.cfg.delta_t
        return control_info

    def _run_fixed_control_window(self) -> dict:
        arrived = 0
        switches = 0
        previous_phase_index = self._phase_index
        for _ in range(self.cfg.delta_t):
            self._apply_fixed_control_second()
            if self._phase_index != previous_phase_index:
                switches += 1
                previous_phase_index = self._phase_index
            arrived += self._run_seconds(1)
            self._fixed_elapsed += 1
        return {
            "arrived_vehicles": arrived,
            "switch_requested": 0,
            "switch_applied": switches,
            "switch_blocked": 0,
            "green_time_before_switch": None,
            "switch_from_index": None,
            "switch_to_index": None,
        }

    def _apply_fixed_control_second(self) -> None:
        slot_seconds = self.cfg.fixed_green_time + self.cfg.yellow_time
        slot_position = self._fixed_elapsed % slot_seconds
        phase_index = (self._fixed_elapsed // slot_seconds) % len(self.green_phases)
        self._phase_index = phase_index

        if slot_position < self.cfg.fixed_green_time:
            traci.trafficlight.setPhase(self.tls_id, self.green_phases[phase_index])
            self._green_elapsed = slot_position
        else:
            traci.trafficlight.setPhase(self.tls_id, self.yellow_phases[phase_index])
            self._green_elapsed = 0

    def _get_state(self) -> np.ndarray:
        """Return traffic observations plus the controller's temporal state.

        Traffic measurements alone are not Markov: the same queues can need a
        different action depending on which approach is currently green and
        how long it has been served.  Including both values lets the DQN learn
        whether to keep or switch once action 1 becomes legal.
        """
        features = []

        for approach in self.APPROACHES:
            detector_ids = self._approach_detector_ids(approach)
            if not detector_ids:
                features.extend([0, 0, 0])
                continue

            lane_counts = [traci.lanearea.getLastStepVehicleNumber(det_id) for det_id in detector_ids]
            queue = sum(traci.lanearea.getLastStepHaltingNumber(det_id) for det_id in detector_ids)
            count = sum(lane_counts)
            speed = (
                sum(
                    traci.lanearea.getLastStepMeanSpeed(det_id) * lane_count
                    for det_id, lane_count in zip(detector_ids, lane_counts)
                ) / count
                if count > 0 else 0.0
            )
            features.extend([queue, speed, count])

        state = np.array(features, dtype=np.float32)
        state[0::3] /= self.cfg.max_queue
        state[1::3] /= self.cfg.max_speed
        state[2::3] /= self.cfg.max_vehicle_count

        phase_normalized = self._phase_index / max(len(self.green_phases) - 1, 1)
        green_normalized = min(self._green_elapsed / 60.0, 1.0)
        state = np.concatenate((
            state,
            np.asarray([phase_normalized, green_normalized], dtype=np.float32),
        ))

        return np.clip(state, 0.0, 1.0)

    def _get_queues(self) -> dict[str, int]:
        return {
            approach: sum(
                traci.lanearea.getJamLengthVehicle(detector_id)
                for detector_id in self._approach_detector_ids(approach)
            )
            for approach in self.APPROACHES
        }

    def _get_total_waiting_time(self) -> float:
        return sum(self._get_waiting_times().values())

    def _get_waiting_times(self) -> dict[str, float]:
        """Waiting-time totals by detector approach for diagnostics and reward."""
        waiting_times = {}
        for approach in self.APPROACHES:
            waiting_times[approach] = 0.0
            seen_vehicles = set()
            for detector_id in self._approach_detector_ids(approach):
                seen_vehicles.update(traci.lanearea.getLastStepVehicleIDs(detector_id))
            for vid in seen_vehicles:
                waiting_times[approach] += traci.vehicle.getWaitingTime(vid)
        return waiting_times

    def _get_average_vehicle_waiting_time(self) -> float:
        """
        Return average accumulated waiting time per vehicle
        observed during the current episode.
        """

        if not self._vehicle_waiting_history:
            return 0.0

        total_waiting = sum(self._vehicle_waiting_history.values())
        total_vehicles = len(self._vehicle_waiting_history)

        return total_waiting / total_vehicles


    def _approach_detector_ids(self, approach: str) -> list[str]:
        """Return the configured detector IDs for one approach.

        A string is accepted for backwards-compatible custom configurations;
        built-in 3-way and 4-way scenarios provide one E2 detector per lane.
        """
        configured = self.detector_ids.get(approach)
        if configured is None:
            return []
        if isinstance(configured, str):
            return [configured]
        return list(configured)

    def _compute_reward(
        self,
        previous_queue: float,
        previous_waiting: float,
        current_queue: float,
        current_waiting: float,
        arrived_vehicles: int,
        switch_applied: bool,
        switch_blocked: bool = False,
        switch_green_time: float | None = None,
    ) -> float:
        return self._get_reward_components(
            previous_queue,
            previous_waiting,
            current_queue,
            current_waiting,
            arrived_vehicles,
            switch_applied,
            switch_blocked,
            switch_green_time,
        )["total"]

    def _get_reward_components(
        self,
        previous_queue: float,
        previous_waiting: float,
        current_queue: float,
        current_waiting: float,
        arrived_vehicles: int,
        switch_applied: bool,
        switch_blocked: bool = False,
        switch_green_time: float | None = None,
    ) -> dict[str, float]:
        """Return the existing reward as named components for diagnostics."""
        if self.cfg.reward_mode == "absolute_pressure":
            pressure = -self.cfg.alpha * current_queue - self.cfg.beta * current_waiting
        else:
            previous_pressure = self.cfg.alpha * previous_queue + self.cfg.beta * previous_waiting
            current_pressure = self.cfg.alpha * current_queue + self.cfg.beta * current_waiting
            pressure = previous_pressure - current_pressure

        throughput = self.cfg.throughput_reward_weight * arrived_vehicles
        switch_penalty = 0.0
        green_time = max(float(switch_green_time or 0.0), 1.0)
        if switch_applied:
            switch_penalty = -self._timed_switch_penalty(green_time)
        elif switch_blocked:
            early_multiplier = max(1.0, self.cfg.min_green_time / green_time)
            switch_penalty = -self.cfg.switch_penalty * self.cfg.blocked_switch_penalty_ratio * early_multiplier
        return {
            "pressure": float(pressure),
            "throughput": float(throughput),
            "switch_penalty": float(switch_penalty),
            "total": float(pressure + throughput + switch_penalty),
        }

    def _timed_switch_penalty(self, green_time: float) -> float:
        """Return a switch cost that favours holding green beyond the minimum."""
        if not self.cfg.switch_penalty_timing_decay:
            return self.cfg.switch_penalty

        multiplier = self.cfg.min_green_time / max(green_time, self.cfg.min_green_time)
        multiplier = max(self.cfg.min_switch_penalty_multiplier, multiplier)
        return self.cfg.switch_penalty * multiplier

    @property
    def state_size(self) -> int:
        # Four approaches with queue/speed/count, current phase, and elapsed
        # green time.  Checkpoints trained with the old 12-feature state are
        # incompatible and must be retrained from scratch.
        return len(self.APPROACHES) * 3 + 2

    @property
    def action_size(self) -> int:
        return self.ACTION_SIZE

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def valid_actions_mask(self) -> list[bool]:
        """Which actions are legal right now.

        Each action is an index into green_phases.  The active phase is always
        legal, so selecting it holds green.  Before min_green_time elapses,
        every other phase is masked.  Networks with fewer than four green
        phases (such as 3-way scenarios) mask nonexistent phase actions.
        """
        can_switch = self._green_elapsed >= self.cfg.min_green_time
        return [
            action < len(self.green_phases) and (can_switch or action == self._phase_index)
            for action in range(self.ACTION_SIZE)
        ]
