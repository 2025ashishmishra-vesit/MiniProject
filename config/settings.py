"""
Central configuration for the SUMO RL Traffic Signal project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_NET_DIR = PROJECT_ROOT / "sumo_net"
NETWORKS_DIR = SUMO_NET_DIR / "networks"


SCENARIO_GROUPS = {
    "3_way": {
        "base_dir": NETWORKS_DIR / "3_way",
        "net_file": "network.net.xml",
        "additional_file": "detectors.add.xml",
        "tls_id": "J1",
        "detector_ids": {
            "N": ["det_N_0", "det_N_1"],
            "S": ["det_S_0", "det_S_1"],
            "E": ["det_E_0", "det_E_1"],
            "W": [],
        },
        "green_phases": [0, 2, 4],
        "yellow_phases": [1, 3, 5],
    },
    "4_way": {
        "base_dir": NETWORKS_DIR / "4_way",
        "net_file": "network.net.xml",
        "additional_file": "detectors.add.xml",
        "tls_id": "J1",
        "detector_ids": {
            "N": ["det_N_0", "det_N_1"],
            "S": ["det_S_0", "det_S_1"],
            "E": ["det_E_0", "det_E_1"],
            "W": ["det_W_0", "det_W_1"],
        },
        "green_phases": [0, 2, 4, 6],
        "yellow_phases": [1, 3, 5, 7],
    },
}


def build_scenarios(network: str = "default", split: str = "train") -> list[dict]:
    """Return scenario dictionaries for one network, all networks, or none."""
    if network in (None, "", "default"):
        return []
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")

    selected = SCENARIO_GROUPS.keys() if network == "all" else [network]
    scenarios = []

    for network_name in selected:
        if network_name not in SCENARIO_GROUPS:
            raise ValueError(f"Unknown network '{network_name}'. Use default, 3_way, 4_way, or all.")

        group = SCENARIO_GROUPS[network_name]
        base_dir = group["base_dir"]
        routes_dir = base_dir / ("routes" if split == "train" else "test_routes")

        for route_file in sorted(routes_dir.glob("*.rou.xml")):
            scenario_name = route_file.name.removesuffix(".rou.xml")
            display_name = (
                f"{network_name}/{scenario_name}"
                if split == "train"
                else f"{network_name}/test/{scenario_name}"
            )
            scenarios.append({
                "name": display_name,
                "network": network_name,
                "split": split,
                "net_file": str(base_dir / group["net_file"]),
                "route_file": str(route_file),
                "additional_file": str(base_dir / group["additional_file"]),
                "tls_id": group["tls_id"],
                "detector_ids": group["detector_ids"],
                "green_phases": group["green_phases"],
                "yellow_phases": group["yellow_phases"],
            })

    if not scenarios:
        raise FileNotFoundError(
            f"No {split} route files found for network '{network}'. Run scripts/generate_routes.py first."
        )
    return scenarios


@dataclass
class EnvConfig:
    # Default single-scenario config kept for backwards compatibility.
    sumo_cfg_file: str = "sumo_net/intersection.sumocfg"
    tls_id: str = "J1"

    detector_ids: dict = field(default_factory=lambda: {
        "N": ["det_N_0", "det_N_1"],
        "S": ["det_S_0", "det_S_1"],
        "E": ["det_E_0", "det_E_1"],
        "W": ["det_W_0", "det_W_1"],
    })

    scenarios: list[dict] = field(default_factory=list)
    scenario_selection: str = "cycle"  # cycle or random
    control_mode: str = "rl"  # rl, hybrid, or fixed

    green_phases: list[int] = field(default_factory=lambda: [0, 2, 4, 6])
    yellow_phases: list[int] = field(default_factory=lambda: [1, 3, 5, 7])

    reward_mode: str = "absolute_pressure"

    # Reward vehicles that successfully leave the intersection.
    throughput_reward_weight: float = 1.0

    # Discourage unnecessary phase switching.
    switch_penalty: float = 1.0

    switch_penalty_timing_decay: bool = True

    # Minimum fraction of the switching penalty that remains
    # when a green phase has already been held for a long time.
    min_switch_penalty_multiplier: float = 0.25

    blocked_switch_penalty_ratio: float = 0.3

    # Prevent the model from changing signal too frequently.
    min_green_time: int = 10
    delta_t: int = 5
    max_steps: int = 720
    seed: int = 42

    max_queue: float = 50.0
    max_speed: float = 13.89
    max_vehicle_count: float = 80.0

    normal_queue_threshold: int = 3

    fixed_green_time: int = 30
    yellow_time: int = 4

    alpha: float = 0.5
    beta: float = 0.1


@dataclass
class AgentConfig:
    hidden_layers: list = field(default_factory=lambda: [256, 256, 128])

    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_before_learn: int = 1_000
    target_update_freq: int = 4

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999993


@dataclass
class TrainConfig:
    num_episodes: int = 200
    save_every: int = 20
    eval_every: int = 10
    checkpoint_dir: str = "models/checkpoints"
    best_model_dir: str = "models/best"
    log_dir: str = "logs"
    gui: bool = False


# =========================
# Emergency vehicle settings
# =========================

EMERGENCY_PRIORITY = True

EMERGENCY_VEHICLE_TYPES = [
    "ambulance"
]

# How close the emergency vehicle should be
# before priority control starts
EMERGENCY_DETECTION_DISTANCE = 150.0

# Minimum time to keep emergency green active
EMERGENCY_MIN_GREEN = 10