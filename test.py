"""
Generate emergency-route variants for every SUMO route scenario
in both the 3-way and 4-way networks.

Run from the project root (sumo_rl_traffic):
    python generate_all_emergency_routes.py

The script:
- Reads the route folders defined by config/settings.py
- Skips files that already contain "_emergency"
- Adds an ambulance vType if missing
- Adds exactly one randomized ambulance flow per scenario
- Uses valid routes for the 3-way and 4-way networks
- Uses deterministic seeds so results are reproducible
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from config.settings import SCENARIO_GROUPS


AMBULANCE_VTYPE = {
    "id": "ambulance",
    "accel": "3.0",
    "decel": "5.0",
    "sigma": "0",
    "length": "5.0",
    "minGap": "2.0",
    "maxSpeed": "25.0",
    "color": "1,0,0",
}

EMERGENCY_ROUTES = {
    "4_way": {
        "north": {"from": "-E2", "to": "E1"},
        "south": {"from": "-E1", "to": "E2"},
        "east": {"from": "-E3", "to": "-E0"},
        "west": {"from": "E0", "to": "E3"},
    },
    "3_way": {
        "north": {"from": "-E2", "to": "E1"},
        "south": {"from": "-E1", "to": "E2"},
        "east": {"from": "E0", "to": "E1"},
    },
}


def ensure_ambulance_vtype(root: ET.Element) -> None:
    for elem in root.findall("vType"):
        if elem.get("id") == "ambulance":
            return

    ambulance_type = ET.Element("vType", AMBULANCE_VTYPE)

    insert_at = 0
    for i, child in enumerate(list(root)):
        if child.tag == "vType":
            insert_at = i + 1
        else:
            break

    root.insert(insert_at, ambulance_type)


def remove_previous_generated_ambulance(root: ET.Element) -> None:
    for child in list(root):
        if child.get("id") == "ambulance_0":
            root.remove(child)


def generate_emergency_route(
    source_route_file: Path,
    network_type: str,
    seed: int,
    min_depart: int = 120,
    max_depart: int = 600,
) -> dict:
    if network_type not in EMERGENCY_ROUTES:
        raise ValueError(f"Unsupported network type: {network_type}")

    rng = random.Random(seed)
    direction = rng.choice(list(EMERGENCY_ROUTES[network_type].keys()))
    movement = EMERGENCY_ROUTES[network_type][direction]
    depart_time = rng.randint(min_depart, max_depart)

    tree = ET.parse(source_route_file)
    root = tree.getroot()

    ensure_ambulance_vtype(root)
    remove_previous_generated_ambulance(root)

    ambulance_flow = ET.Element(
        "flow",
        {
            "id": "ambulance_0",
            "type": "ambulance",
            "from": movement["from"],
            "to": movement["to"],
            "begin": str(depart_time),
            "end": str(depart_time + 1),
            "number": "1",
            "departLane": "best",
        },
    )

    root.append(ambulance_flow)

    output_file = (
        source_route_file.parent
        / f"{source_route_file.name.removesuffix('.rou.xml')}_emergency.rou.xml"
    )

    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

    return {
        "network": network_type,
        "source": source_route_file,
        "output": output_file,
        "direction": direction,
        "from": movement["from"],
        "to": movement["to"],
        "depart": depart_time,
        "seed": seed,
    }


def route_directories(base_dir: Path) -> list[Path]:
    dirs = []
    for name in ("routes", "test_routes"):
        path = base_dir / name
        if path.exists():
            dirs.append(path)
    return dirs


def generate_for_network(
    network_type: str,
    base_seed: int,
    min_depart: int = 120,
    max_depart: int = 600,
) -> list[dict]:
    group = SCENARIO_GROUPS[network_type]
    base_dir = Path(group["base_dir"])

    results = []
    scenario_index = 0

    for routes_dir in route_directories(base_dir):
        source_files = sorted(
            p for p in routes_dir.glob("*.rou.xml") if "_emergency" not in p.stem
        )

        for source in source_files:
            seed = base_seed + scenario_index
            result = generate_emergency_route(
                source_route_file=source,
                network_type=network_type,
                seed=seed,
                min_depart=min_depart,
                max_depart=max_depart,
            )
            results.append(result)
            scenario_index += 1

    return results


def main() -> None:
    print("\n" + "=" * 72)
    print(" GENERATING EMERGENCY ROUTE SCENARIOS")
    print("=" * 72)

    all_results = []
    all_results.extend(generate_for_network("4_way", base_seed=42))
    all_results.extend(generate_for_network("3_way", base_seed=1000))

    if not all_results:
        print("\nNo route files were found.")
        return

    print()
    for result in all_results:
        print(
            f"[OK] {result['network']:5s} | "
            f"{result['source'].name:24s} -> "
            f"{result['output'].name:34s} | "
            f"t={result['depart']:4d}s | "
            f"{result['direction']:5s} | "
            f"{result['from']} -> {result['to']}"
        )

    print("\n" + "-" * 72)
    print(f"Generated {len(all_results)} emergency route files.")
    print("Emergency copies use the suffix: _emergency.rou.xml")
    print("-" * 72 + "\n")


if __name__ == "__main__":
    main()
