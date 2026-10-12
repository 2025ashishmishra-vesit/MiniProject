"""
Run controlled reward-tuning experiments.

Each experiment trains a fresh model with one reward setting, saves it in a
clearly named folder, evaluates it on held-out test routes, and writes a master
summary.

Default command:
    python reward_tuning_experiments.py

Useful shorter trial:
    python reward_tuning_experiments.py --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class RewardSetting:
    throughput_weight: float
    switch_penalty: float
    min_green_time: int

    @property
    def name(self) -> str:
        return (
            f"tw_{safe_number(self.throughput_weight)}"
            f"__sp_{safe_number(self.switch_penalty)}"
            f"__mg_{self.min_green_time}"
        )


DEFAULT_SETTINGS = [
    # RewardSetting(throughput_weight=1.0, switch_penalty=0.2, min_green_time=10),
    RewardSetting(throughput_weight=0.5, switch_penalty=4.0, min_green_time=10),
    # RewardSetting(throughput_weight=3.0, switch_penalty=0.5, min_green_time=15),
    #RewardSetting(throughput_weight=5.0, switch_penalty=0.7, min_green_time=15),
]


def safe_number(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate multiple reward settings.")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per setting.")
    parser.add_argument("--network", choices=["3_way", "4_way", "all"], default="all")
    parser.add_argument("--scenario-selection", choices=["cycle", "random"], default="random")
    parser.add_argument("--episodes-per-scenario", type=int, default=1, help="Evaluation repeats.")
    parser.add_argument("--control-mode", choices=["rl", "hybrid"], default="rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate during training every N episodes and save models/best/best.pt; 0 uses final.pt as best.pt.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/reward_tuning",
        help="Root folder for all experiment runs.",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help=(
            "Optional semicolon-separated settings: "
            "throughput,switch_penalty,min_green;throughput,switch_penalty,min_green"
        ),
    )
    return parser.parse_args()


def parse_settings(raw: str | None) -> list[RewardSetting]:
    if not raw:
        return DEFAULT_SETTINGS

    settings = []
    for chunk in raw.split(";"):
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("Each custom setting must be throughput,switch_penalty,min_green.")
        settings.append(RewardSetting(
            throughput_weight=float(parts[0]),
            switch_penalty=float(parts[1]),
            min_green_time=int(parts[2]),
        ))
    return settings


def run_command(cmd: list[str], cwd: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def read_comparison_summary(comparison_csv: Path) -> dict:
    rows = []
    with open(comparison_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found in {comparison_csv}")

    def avg(key: str) -> float:
        return mean(float(row[key]) for row in rows)

    throughput = avg("throughput_improvement_pct")
    waiting = avg("waiting_reduction_pct")
    queue = avg("queue_reduction_pct")
    switch_rate_pct = 100.0 * avg("rl_switch_rate")

    return {
        "mean_throughput_improvement_pct": throughput,
        "mean_waiting_reduction_pct": waiting,
        "mean_queue_reduction_pct": queue,
        "mean_reward_delta_rl_minus_fixed": avg("reward_delta_rl_minus_fixed"),
        "mean_rl_switch_rate_pct": switch_rate_pct,
        "composite_score": throughput + 0.25 * waiting + 0.15 * queue - 0.10 * switch_rate_pct,
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / args.output_root / run_id
    settings = parse_settings(args.settings)

    all_results = []
    print(f"[experiment] Output folder: {run_dir}")
    print(f"[experiment] Episodes per setting: {args.episodes}")
    print(f"[experiment] Settings: {len(settings)}\n")

    for index, setting in enumerate(settings, start=1):
        config_id = f"{index:02d}__{setting.name}__ep_{args.episodes}"
        config_dir = run_dir / config_id
        model_path = config_dir / "models" / "final.pt"
        best_dir = config_dir / "models" / "best"
        checkpoint_dir = config_dir / "models" / "checkpoints"
        train_log_dir = config_dir / "logs" / "train"
        eval_dir = config_dir / "logs" / "evaluation_test"

        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "reward_config.json", "w", encoding="utf-8") as f:
            json.dump({
                **asdict(setting),
                "episodes": args.episodes,
                "network": args.network,
                "scenario_selection": args.scenario_selection,
                "control_mode": args.control_mode,
                "seed": args.seed,
            }, f, indent=2)

        print(f"\n[experiment] Training {config_id}")
        train_cmd = [
            sys.executable,
            "train.py",
            "--network", args.network,
            "--scenario-selection", args.scenario_selection,
            "--control-mode", args.control_mode,
            "--episodes", str(args.episodes),
            "--save-every", str(args.save_every),
            "--eval-every", str(args.eval_every),
            "--throughput-reward-weight", str(setting.throughput_weight),
            "--switch-penalty", str(setting.switch_penalty),
            "--min-green-time", str(setting.min_green_time),
            "--checkpoint-dir", str(checkpoint_dir),
            "--best-model-dir", str(best_dir),
            "--log-dir", str(train_log_dir),
            "--final-model", str(model_path),
        ]
        run_command(train_cmd, cwd=project_dir, log_file=config_dir / "train_console.log")

        print(f"\n[experiment] Evaluating {config_id}")
        eval_cmd = [
            sys.executable,
            "evaluate_compare.py",
            "--network", args.network,
            "--route-split", "test",
            "--model", str(model_path),
            "--episodes-per-scenario", str(args.episodes_per_scenario),
            "--rl-control-mode", args.control_mode,
            "--throughput-reward-weight", str(setting.throughput_weight),
            "--switch-penalty", str(setting.switch_penalty),
            "--min-green-time", str(setting.min_green_time),
            "--output-dir", str(eval_dir),
        ]
        run_command(eval_cmd, cwd=project_dir, log_file=config_dir / "eval_console.log")

        summary = read_comparison_summary(eval_dir / "comparison.csv")
        best_model_path = best_dir / "best.pt"
        result = {
            "config_id": config_id,
            "model_path": str(model_path),
            "best_model_path": str(best_model_path) if best_model_path.exists() else "",
            "evaluation_dir": str(eval_dir),
            "episodes": args.episodes,
            **asdict(setting),
            **summary,
        }
        all_results.append(result)

        print(
            f"[experiment] {config_id} summary | "
            f"throughput={summary['mean_throughput_improvement_pct']:.2f}% | "
            f"waiting={summary['mean_waiting_reduction_pct']:.2f}% | "
            f"queue={summary['mean_queue_reduction_pct']:.2f}% | "
            f"switch={summary['mean_rl_switch_rate_pct']:.2f}%"
        )

    all_results.sort(key=lambda row: row["composite_score"], reverse=True)
    write_summary(run_dir / "summary.csv", all_results)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    best = all_results[0]
    print("\n[experiment] Best setting by composite score:")
    print(f"  {best['config_id']}")
    print(f"  model: {best['model_path']}")
    print(f"  throughput improvement: {best['mean_throughput_improvement_pct']:.2f}%")
    print(f"  waiting reduction: {best['mean_waiting_reduction_pct']:.2f}%")
    print(f"  queue reduction: {best['mean_queue_reduction_pct']:.2f}%")
    print(f"  RL switch rate: {best['mean_rl_switch_rate_pct']:.2f}%")
    print(f"\n[experiment] Master summary saved to {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
