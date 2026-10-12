"""Summarize RL switch behavior from an evaluate_compare.py output directory.

Example:
    python analyze_switch_trace.py --dir logs/config02_switch_trace
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report applied-switch timing and evaluation performance."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing rl_switch_events.csv and comparison.csv.",
    )
    parser.add_argument(
        "--min-hold",
        type=float,
        default=None,
        help="Earliest legal logged green hold (for example 11). Defaults to the smallest observed hold.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_switches(label: str, events: list[dict[str, str]], min_hold: float) -> None:
    holds = [float(event["green_time"]) for event in events]
    at_min = sum(abs(hold - min_hold) < 1e-9 for hold in holds)
    print(
        f"{label:24s} switches={len(events):4d} | "
        f"at {min_hold:g}s={at_min:4d} ({100.0 * at_min / len(events):5.1f}%) | "
        f"avg green={mean(holds):5.2f}s"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.dir)
    events = read_csv(output_dir / "rl_switch_events.csv")
    comparisons = read_csv(output_dir / "comparison.csv")
    if not events:
        print("No applied RL switches were recorded.")
        return

    min_hold = args.min_hold if args.min_hold is not None else min(
        float(event["green_time"]) for event in events
    )
    print(f"Trace directory: {output_dir}")
    print(f"Earliest legal logged hold: {min_hold:g}s\n")

    print("Switch behavior")
    summarize_switches("All scenarios", events, min_hold)

    by_network: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        by_network[event["scenario"].split("/", 1)[0]].append(event)
    for network, group in sorted(by_network.items()):
        summarize_switches(network, group, min_hold)

    print("\nPer scenario")
    for scenario, group in sorted(
        ((name, group) for name, group in _group_by(events, "scenario").items()),
    ):
        summarize_switches(scenario, group, min_hold)

    waiting_reductions = [float(row["waiting_reduction_pct"]) for row in comparisons]
    queue_reductions = [float(row["queue_reduction_pct"]) for row in comparisons]
    throughput_improvements = [float(row["throughput_improvement_pct"]) for row in comparisons]
    print("\nPerformance versus fixed time")
    print(f"Mean waiting-time reduction: {mean(waiting_reductions):.2f}%")
    print(f"Mean queue reduction:        {mean(queue_reductions):.2f}%")
    print(f"Mean throughput improvement: {mean(throughput_improvements):.2f}%")
    print(f"Scenarios with lower waiting: {sum(value > 0 for value in waiting_reductions)}/{len(waiting_reductions)}")


def _group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


if __name__ == "__main__":
    main()
