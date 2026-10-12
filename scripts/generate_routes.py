"""
Generate train and held-out test route files for the 3-way and 4-way SUMO
networks.

Run from sumo_rl_traffic:
    python scripts/generate_routes.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMO_NET = ROOT / "sumo_net"
NETWORKS = SUMO_NET / "networks"


VEHICLE_TYPES = {
    "car": {"accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5", "maxSpeed": "13.9"},
    "bus": {"accel": "1.2", "decel": "4.0", "sigma": "0.5", "length": "12", "maxSpeed": "12"},
    "truck": {"accel": "1.0", "decel": "4.0", "sigma": "0.5", "length": "10", "maxSpeed": "11"},
    "DEFAULT_BIKETYPE": {"vClass": "motorcycle"},
}


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    source_net: Path
    approaches: dict[str, str]
    detector_lanes: dict[str, list[str]]


NETWORK_SPECS = {
    "3_way": NetworkSpec(
        name="3_way",
        source_net=SUMO_NET / "three_way_intersection" / "test1.net.xml",
        approaches={"S": "-E1", "N": "-E2", "E": "E0"},
        detector_lanes={
            "S": ["-E1_0", "-E1_1"],
            "N": ["-E2_0", "-E2_1"],
            "E": ["E0_0", "E0_1"],
        },
    ),
    "4_way": NetworkSpec(
        name="4_way",
        source_net=SUMO_NET / "four_way_intersection" / "test.net.xml",
        approaches={"S": "-E1", "N": "-E2", "E": "-E3", "W": "E0"},
        detector_lanes={
            "S": ["-E1_0", "-E1_1"],
            "N": ["-E2_0", "-E2_1"],
            "E": ["-E3_0", "-E3_1"],
            "W": ["E0_0", "E0_1"],
        },
    ),
}


BASE_PATTERNS = {
    "balanced": {
        "total_volume": 2400,
        "directional_weights": {},
        "vehicle_mix": {"car": 0.68, "DEFAULT_BIKETYPE": 0.20, "truck": 0.08, "bus": 0.04},
        "turn_weights": {"left": 0.20, "straight": 0.60, "right": 0.20},
        "seed": 101,
    },
    "light": {
        "total_volume": 900,
        "directional_weights": {},
        "vehicle_mix": {"car": 0.72, "DEFAULT_BIKETYPE": 0.18, "truck": 0.07, "bus": 0.03},
        "turn_weights": {"left": 0.15, "straight": 0.70, "right": 0.15},
        "seed": 102,
    },
    "medium": {
        "total_volume": 2200,
        "directional_weights": {},
        "vehicle_mix": {"car": 0.65, "DEFAULT_BIKETYPE": 0.22, "truck": 0.09, "bus": 0.04},
        "turn_weights": {"left": 0.22, "straight": 0.55, "right": 0.23},
        "seed": 103,
    },
    "heavy": {
        "total_volume": 4200,
        "directional_weights": {},
        "vehicle_mix": {"car": 0.62, "DEFAULT_BIKETYPE": 0.22, "truck": 0.11, "bus": 0.05},
        "turn_weights": {"left": 0.25, "straight": 0.50, "right": 0.25},
        "seed": 104,
    },
    "ns_heavy": {
        "total_volume": 3300,
        "directional_weights": {"N": 2.4, "S": 2.4, "E": 0.7, "W": 0.7},
        "vehicle_mix": {"car": 0.66, "DEFAULT_BIKETYPE": 0.18, "truck": 0.10, "bus": 0.06},
        "turn_weights": {"left": 0.20, "straight": 0.62, "right": 0.18},
        "seed": 105,
    },
    "ew_heavy": {
        "total_volume": 3300,
        "directional_weights": {"N": 0.7, "S": 0.7, "E": 2.4, "W": 2.4},
        "vehicle_mix": {"car": 0.66, "DEFAULT_BIKETYPE": 0.18, "truck": 0.10, "bus": 0.06},
        "turn_weights": {"left": 0.18, "straight": 0.62, "right": 0.20},
        "seed": 106,
    },
    "random_1": {
        "total_volume": 2600,
        "directional_weights": "random",
        "vehicle_mix": {"car": 0.56, "DEFAULT_BIKETYPE": 0.28, "truck": 0.11, "bus": 0.05},
        "turn_weights": "random",
        "seed": 201,
    },
    "random_2": {
        "total_volume": 3600,
        "directional_weights": "random",
        "vehicle_mix": {"car": 0.70, "DEFAULT_BIKETYPE": 0.12, "truck": 0.13, "bus": 0.05},
        "turn_weights": "random",
        "seed": 202,
    },
}


TEST_PATTERNS = {
    "balanced": {
        "total_volume": 2550,
        "directional_weights": {"N": 1.15, "S": 0.95, "E": 1.05, "W": 0.90},
        "vehicle_mix": {"car": 0.64, "DEFAULT_BIKETYPE": 0.23, "truck": 0.09, "bus": 0.04},
        "turn_weights": {"left": 0.24, "straight": 0.52, "right": 0.24},
        "seed": 1101,
    },
    "light": {
        "total_volume": 750,
        "directional_weights": {"N": 0.85, "S": 1.20, "E": 1.05, "W": 0.95},
        "vehicle_mix": {"car": 0.69, "DEFAULT_BIKETYPE": 0.22, "truck": 0.06, "bus": 0.03},
        "turn_weights": {"left": 0.20, "straight": 0.62, "right": 0.18},
        "seed": 1102,
    },
    "medium": {
        "total_volume": 2500,
        "directional_weights": {"N": 1.30, "S": 0.80, "E": 1.10, "W": 0.90},
        "vehicle_mix": {"car": 0.61, "DEFAULT_BIKETYPE": 0.25, "truck": 0.10, "bus": 0.04},
        "turn_weights": {"left": 0.27, "straight": 0.49, "right": 0.24},
        "seed": 1103,
    },
    "heavy": {
        "total_volume": 4550,
        "directional_weights": {"N": 1.10, "S": 1.25, "E": 0.85, "W": 1.00},
        "vehicle_mix": {"car": 0.58, "DEFAULT_BIKETYPE": 0.24, "truck": 0.12, "bus": 0.06},
        "turn_weights": {"left": 0.30, "straight": 0.46, "right": 0.24},
        "seed": 1104,
    },
    "ns_heavy": {
        "total_volume": 3600,
        "directional_weights": {"N": 2.8, "S": 2.1, "E": 0.8, "W": 0.6},
        "vehicle_mix": {"car": 0.62, "DEFAULT_BIKETYPE": 0.20, "truck": 0.11, "bus": 0.07},
        "turn_weights": {"left": 0.24, "straight": 0.57, "right": 0.19},
        "seed": 1105,
    },
    "ew_heavy": {
        "total_volume": 3700,
        "directional_weights": {"N": 0.6, "S": 0.8, "E": 2.7, "W": 2.0},
        "vehicle_mix": {"car": 0.63, "DEFAULT_BIKETYPE": 0.20, "truck": 0.11, "bus": 0.06},
        "turn_weights": {"left": 0.21, "straight": 0.56, "right": 0.23},
        "seed": 1106,
    },
    "random_1": {
        "total_volume": 2850,
        "directional_weights": "random",
        "vehicle_mix": {"car": 0.52, "DEFAULT_BIKETYPE": 0.31, "truck": 0.12, "bus": 0.05},
        "turn_weights": "random",
        "seed": 1201,
    },
    "random_2": {
        "total_volume": 3900,
        "directional_weights": "random",
        "vehicle_mix": {"car": 0.74, "DEFAULT_BIKETYPE": 0.10, "truck": 0.10, "bus": 0.06},
        "turn_weights": "random",
        "seed": 1202,
    },
}


ROUTE_SETS = {
    "routes": BASE_PATTERNS,
    "test_routes": TEST_PATTERNS,
}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return {key: value / total for key, value in weights.items()}


def parse_connections(net_file: Path) -> dict[str, list[tuple[str, str]]]:
    tree = ET.parse(net_file)
    root = tree.getroot()
    connections: dict[str, list[tuple[str, str]]] = {}

    for connection in root.findall("connection"):
        source = connection.get("from")
        target = connection.get("to")
        turn_dir = connection.get("dir", "s")
        if not source or not target or source.startswith(":"):
            continue
        turn_name = {"l": "left", "s": "straight", "r": "right"}.get(turn_dir, "straight")
        pair = (target, turn_name)
        if pair not in connections.setdefault(source, []):
            connections[source].append(pair)

    return connections


def movement_shares(items: list[tuple[str, str]], turn_weights: dict[str, float]) -> list[tuple[str, str, float]]:
    weighted = [(target, turn, turn_weights.get(turn, 0.01)) for target, turn in items]
    total = sum(weight for _, _, weight in weighted)
    return [(target, turn, weight / total) for target, turn, weight in weighted]


def add_vehicle_types(routes: ET.Element) -> None:
    for type_id, attrs in VEHICLE_TYPES.items():
        ET.SubElement(routes, "vType", {"id": type_id, **attrs})


def resolve_directional_weights(spec: NetworkSpec, value, rng: random.Random) -> dict[str, float]:
    if value == "random":
        return {direction: rng.uniform(0.5, 2.5) for direction in spec.approaches}
    weights = {direction: 1.0 for direction in spec.approaches}
    for direction, weight in value.items():
        if direction in weights:
            weights[direction] = weight
    return weights


def resolve_turn_weights(value, rng: random.Random) -> dict[str, float]:
    if value == "random":
        return normalize({
            "left": rng.uniform(0.10, 0.35),
            "straight": rng.uniform(0.35, 0.70),
            "right": rng.uniform(0.10, 0.35),
        })
    return normalize(dict(value))


def generate_route_file(
    spec: NetworkSpec,
    output_path: Path,
    total_volume: int,
    vehicle_mix: dict[str, float],
    directional_weights,
    turn_weights,
    duration: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    connections = parse_connections(spec.source_net)
    direction_weights = normalize(resolve_directional_weights(spec, directional_weights, rng))
    turn_weights = resolve_turn_weights(turn_weights, rng)
    vehicle_mix = normalize(vehicle_mix)

    routes = ET.Element("routes")
    add_vehicle_types(routes)

    for direction, source_edge in spec.approaches.items():
        source_volume = total_volume * direction_weights[direction]
        valid_targets = connections[source_edge]

        for target_edge, turn_name, turn_share in movement_shares(valid_targets, turn_weights):
            for vehicle_type, vehicle_share in vehicle_mix.items():
                vehs_per_hour = round(source_volume * turn_share * vehicle_share, 2)
                if vehs_per_hour <= 0:
                    continue
                flow_id = f"{direction.lower()}_{turn_name}_{vehicle_type}".replace(
                    "DEFAULT_BIKETYPE", "bike"
                )
                ET.SubElement(routes, "flow", {
                    "id": flow_id,
                    "type": vehicle_type,
                    "from": source_edge,
                    "to": target_edge,
                    "begin": "0",
                    "end": str(duration),
                    "vehsPerHour": f"{vehs_per_hour:.2f}",
                    "departLane": "random",
                })

    indent(routes)
    tree = ET.ElementTree(routes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def indent(element: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "    "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spacer + "    "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spacer


def write_detectors(spec: NetworkSpec, output_path: Path) -> None:
    lane_lengths = {
        lane.attrib["id"]: float(lane.attrib["length"])
        for lane in ET.parse(spec.source_net).iterfind(".//lane")
        if "length" in lane.attrib
    }
    additional = ET.Element("additional")
    for direction, lanes in spec.detector_lanes.items():
        for lane in lanes:
            lane_length = lane_lengths[lane]
            detector_length = min(25.0, lane_length)
            ET.SubElement(additional, "laneAreaDetector", {
                "id": f"det_{direction}_{lane.rsplit('_', 1)[-1]}",
                "lane": lane,
                # SUMO E2 positions are measured from the lane start.  This
                # covers the final detector_length metres ending at the TLS.
                "pos": f"{lane_length - detector_length:.2f}",
                "length": f"{detector_length:.2f}",
                "file": "detector.xml",
            })
    indent(additional)
    ET.ElementTree(additional).write(output_path, encoding="UTF-8", xml_declaration=True)


def write_sumocfg(output_path: Path) -> None:
    output_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes/balanced.rou.xml"/>
        <additional-files value="detectors.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
    </processing>
</configuration>
""",
        encoding="utf-8",
    )


def prepare_network(spec: NetworkSpec) -> None:
    target_dir = NETWORKS / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec.source_net, target_dir / "network.net.xml")
    write_detectors(spec, target_dir / "detectors.add.xml")
    write_sumocfg(target_dir / "scenario.sumocfg")


def generate_all(duration: int) -> None:
    for spec in NETWORK_SPECS.values():
        prepare_network(spec)

        for route_dir_name, patterns in ROUTE_SETS.items():
            route_names = list(patterns)
            if spec.name == "3_way" and "ew_heavy" in route_names:
                route_names.remove("ew_heavy")

            for route_name in route_names:
                pattern = patterns[route_name]
                generate_route_file(
                    spec=spec,
                    output_path=NETWORKS / spec.name / route_dir_name / f"{route_name}.rou.xml",
                    total_volume=pattern["total_volume"],
                    vehicle_mix=pattern["vehicle_mix"],
                    directional_weights=pattern["directional_weights"],
                    turn_weights=pattern["turn_weights"],
                    duration=duration,
                    seed=pattern["seed"],
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SUMO route scenarios.")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(duration=args.duration)
