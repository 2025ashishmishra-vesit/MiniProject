"""
Training utilities: logger and episode metric tracker.
"""

import csv
import json
import os
from datetime import datetime


class TrainingLogger:
    """Writes per-episode metrics to CSV and JSON."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"train_{ts}.csv")
        self.json_path = os.path.join(log_dir, f"train_{ts}.json")
        self._header_written = False

    def log(self, data: dict):
        if not self._header_written:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                writer.writeheader()
            self._header_written = True

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()), extrasaction="ignore")
            writer.writerow(data)

    def save_history(self, history: list):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"[Logger] History saved -> {self.json_path}")
