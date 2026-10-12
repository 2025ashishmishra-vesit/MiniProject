import random
import xml.etree.ElementTree as ET
from pathlib import Path


def generate_emergency_route(
    source_route_file,
    output_route_file=None,
    seed=42,
    min_depart=120,
    max_depart=600,
    network_type="4_way",
):
    rng = random.Random(seed)

    source_route_file = Path(source_route_file)

    if output_route_file is None:
        output_route_file = (
            source_route_file.parent
            / f"{source_route_file.stem}_emergency.rou.xml"
        )

    output_route_file = Path(output_route_file)

    depart_time = rng.randint(min_depart, max_depart)

    if network_type == "4_way":
        directions = {
            "north": {"from": "-E2", "to": "E1"},
            "south": {"from": "-E1", "to": "E2"},
            "east":  {"from": "-E3", "to": "-E0"},
            "west":  {"from": "E0", "to": "E3"},
        }

    elif network_type == "3_way":
        directions = {
            # replace these with your actual valid 3-way routes
            "north": {"from": "-E2", "to": "E1"},
            "south": {"from": "-E1", "to": "E2"},
            "west":  {"from": "E0", "to": "E1"},
        }

    else:
        raise ValueError(
            f"Unsupported network type: {network_type}"
        )

    direction = rng.choice(list(directions.keys()))
    route = directions[direction]

    tree = ET.parse(source_route_file)
    root = tree.getroot()

    # Remove any older generated ambulance if present
    for element in list(root):
        if element.get("id") == "ambulance_0":
            root.remove(element)

    ambulance_flow = ET.Element(
        "flow",
        {
            "id": "ambulance_0",
            "type": "ambulance",
            "from": route["from"],
            "to": route["to"],
            "begin": str(depart_time),
            "end": str(depart_time + 1),
            "number": "1",
            "departLane": "best",
        },
    )

    root.append(ambulance_flow)

    tree.write(
        output_route_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print(
        f"[EMERGENCY GENERATED] "
        f"{source_route_file.name} -> "
        f"{output_route_file.name} | "
        f"time={depart_time}s | "
        f"direction={direction}"
    )

    return {
        "route_file": str(output_route_file),
        "depart_time": depart_time,
        "direction": direction,
    }



def generate_all_emergency_routes(
    routes_directory,
    network_type,
    seed=42,
):
    routes_directory = Path(routes_directory)

    route_files = [
        path
        for path in routes_directory.glob("*.rou.xml")
        if "_emergency" not in path.stem
    ]

    generated = []

    for index, route_file in enumerate(route_files):

        result = generate_emergency_route(
            source_route_file=route_file,
            seed=seed + index,
            network_type=network_type,
        )

        generated.append(result)

    return generated


generate_all_emergency_routes(
    routes_directory=
        "sumo_net/networks/4_way/routes",
    network_type="4_way",
    seed=42,
)