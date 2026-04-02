"""
Trace loading utilities for Phase 6.

Responsibilities:
- load raw trace CSV files from disk
- load previously materialized train/val/test episode JSON files
- keep disk I/O separate from trace preprocessing / episode generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class TraceLoader:
    """Loads raw trace inputs or saved episode splits from disk."""

    def __init__(self, trace_dir: Optional[str] = None):
        self.trace_dir = Path(trace_dir) if trace_dir else Path("data/traces")

    def load_trace_frames(self, pattern: str = "*.csv") -> List[pd.DataFrame]:
        """Load raw trace CSV files from the configured directory."""
        if not self.trace_dir.exists():
            return []

        frames: List[pd.DataFrame] = []
        for trace_file in sorted(self.trace_dir.glob(pattern)):
            try:
                frames.append(pd.read_csv(trace_file))
            except Exception:
                continue
        return frames

    def saved_episode_paths(self) -> Dict[str, Path]:
        """Return canonical train/val/test episode JSON paths."""
        return {
            "train": self.trace_dir / "train_episodes.json",
            "val": self.trace_dir / "val_episodes.json",
            "test": self.trace_dir / "test_episodes.json",
        }

    def has_saved_episode_splits(self) -> bool:
        """Whether all canonical episode split files already exist."""
        return all(path.exists() for path in self.saved_episode_paths().values())

    def load_saved_episode_splits(self) -> Tuple[list, list, list]:
        """Load canonical train/val/test episode JSON files as TraceEpisode objects."""
        paths = self.saved_episode_paths()
        return (
            self._load_episode_file(paths["train"]),
            self._load_episode_file(paths["val"]),
            self._load_episode_file(paths["test"]),
        )

    def _load_episode_file(self, path: Path) -> list:
        if not path.exists():
            return []

        from src.core.trace_processor import TraceEpisode, TraceTask

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        episodes = []
        for episode_data in payload.get("episodes", []):
            tasks = [
                TraceTask(
                    task_id=int(task["task_id"]),
                    device_id=int(task["device_id"]),
                    arrival_time=float(task["arrival_time"]),
                    deadline=float(task["deadline"]),
                    data_size=int(task["data_size"]),
                    cpu_cycles=int(task["cpu_cycles"]),
                    priority=int(task["priority"]),
                    location=tuple(task["location"]),
                )
                for task in episode_data.get("tasks", [])
            ]
            episodes.append(
                TraceEpisode(
                    episode_id=int(episode_data["episode_id"]),
                    tasks=tasks,
                    trace_name=episode_data.get("trace_name", path.stem),
                    device_density=int(episode_data.get("device_density", len({t.device_id for t in tasks}))),
                )
            )
        return episodes
