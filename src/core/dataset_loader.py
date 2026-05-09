"""
Dataset/source loading utilities for Phase 6 and Phase 6R.

This module intentionally keeps two different responsibilities together:

1. `DebugDatasetFactory`
   - lightweight debug/synthetic helpers
   - schema/sample generation for quick local validation

2. `RealDataLoader`
   - structured readers for downloaded real datasets
   - lightweight inventory/profiling helpers

Materialized episode split loading stays in `trace_loader.py` because it
operates on processed train/val/test trace episodes rather than raw sources.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RealDatasetSummary:
    name: str
    files: List[str]
    row_counts: Dict[str, int]
    columns: Dict[str, List[str]]


class DebugDatasetFactory:
    """
    Debug/synthetic dataset helpers.

    These utilities are kept for quick smoke tests and schema validation, but
    they are no longer the primary path for real-data experiments.
    """

    @staticmethod
    def load_google_cluster_trace(filepath: str | None = None, num_tasks: int = 1000) -> pd.DataFrame:
        """
        Load task attributes from a Google Cluster Trace style CSV.

        If `filepath` is missing, generate a lightweight synthetic debug sample.
        """
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath)

            if "timestamp" in df.columns:
                df["submit_time"] = df["timestamp"] - df["timestamp"].min()

            if "cpu_request" in df.columns:
                df["cpu_request"] = df["cpu_request"] * 1e9

            return df

        data = {
            "task_id": range(num_tasks),
            "submit_time": np.sort(np.random.exponential(scale=10.0, size=num_tasks)),
            "cpu_request": np.random.pareto(a=2.0, size=num_tasks) * 1e9,
            "ram_request": np.random.uniform(128, 4096, size=num_tasks),
            "task_type": np.random.choice(
                ["AI_INFERENCE", "VIDEO_TRANSCODE", "IOT_SENSING", "CRITICAL_HEALTH"],
                size=num_tasks,
                p=[0.2, 0.3, 0.4, 0.1],
            ),
        }
        return pd.DataFrame(data)

    @staticmethod
    def load_didi_gaia_mobility(
        filepath: str | None = None,
        num_users: int = 20,
        duration: int = 1000,
    ) -> Dict[int, List[tuple[float, float]]]:
        """
        Load mobility traces from a Didi Gaia style CSV.

        If `filepath` is missing, generate a lightweight synthetic debug sample.
        """
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath)

            mobility_traces: Dict[int, List[tuple[float, float]]] = {}
            id_column = "user_id" if "user_id" in df.columns else "vehicle_id"
            time_column = "timestamp" if "timestamp" in df.columns else "time"
            lat_column = "latitude"
            lon_column = "longitude"

            for user_id in df[id_column].dropna().unique()[:num_users]:
                user_data = df[df[id_column] == user_id].sort_values(time_column)

                lat_min, lat_max = user_data[lat_column].min(), user_data[lat_column].max()
                lon_min, lon_max = user_data[lon_column].min(), user_data[lon_column].max()
                lat_span = max(lat_max - lat_min, 1e-9)
                lon_span = max(lon_max - lon_min, 1e-9)

                x = ((user_data[lat_column] - lat_min) / lat_span * 1000).tolist()
                y = ((user_data[lon_column] - lon_min) / lon_span * 1000).tolist()
                mobility_traces[int(user_id)] = list(zip(x, y))

            return mobility_traces

        mobility_traces: Dict[int, List[tuple[float, float]]] = {}
        for user_id in range(num_users):
            x, y = random.uniform(0, 1000), random.uniform(0, 1000)
            path: List[tuple[float, float]] = []
            velocity_x = random.uniform(-10, 10)
            velocity_y = random.uniform(-10, 10)

            for _ in range(duration):
                x += velocity_x
                y += velocity_y

                if x < 0 or x > 1000:
                    velocity_x *= -1
                if y < 0 or y > 1000:
                    velocity_y *= -1

                if random.random() < 0.05:
                    velocity_x = random.uniform(-10, 10)
                    velocity_y = random.uniform(-10, 10)

                path.append((x, y))

            mobility_traces[user_id] = path

        return mobility_traces

    @staticmethod
    def sample_csv_payloads() -> Dict[str, pd.DataFrame]:
        """Return sample CSV payloads for quick schema checks without writing files."""
        sample_tasks = pd.DataFrame(
            {
                "timestamp": [0, 10, 20, 30],
                "task_id": [1, 2, 3, 4],
                "cpu_request": [0.5, 1.0, 0.3, 2.0],
                "memory_request": [512, 1024, 256, 2048],
                "task_type": ["AI_INFERENCE", "VIDEO_TRANSCODE", "IOT_SENSING", "CRITICAL_HEALTH"],
            }
        )
        sample_mobility = pd.DataFrame(
            {
                "user_id": [0, 0, 0, 1, 1, 1],
                "timestamp": [0, 1, 2, 0, 1, 2],
                "latitude": [39.9042, 39.9043, 39.9044, 40.7128, 40.7129, 40.7130],
                "longitude": [116.4074, 116.4075, 116.4076, -74.0060, -74.0061, -74.0062],
            }
        )
        return {
            "sample_google_trace.csv": sample_tasks,
            "sample_didi_mobility.csv": sample_mobility,
        }


class RealDataLoader:
    """Structured readers for raw real datasets stored under `data/raw_real_datasets/`."""

    def __init__(self, data_root: str | Path = "data/raw_real_datasets"):
        self.data_root = Path(data_root)
        self.large_file_threshold_bytes = 100 * 1024 * 1024

    def load_glasgow_mec(self) -> Dict[str, pd.DataFrame]:
        root = self.data_root / "glasgow_mec"
        return {
            "consecutive": pd.read_csv(root / "consecutiveTimeWDPaper3.csv"),
            "random_based": pd.read_csv(root / "RandomBasedDataset.csv"),
        }

    def load_uci_execution_times(self) -> Dict[str, pd.DataFrame]:
        root = self.data_root / "uci_mec_execution_times"
        frames: Dict[str, pd.DataFrame] = {}
        for csv_path in sorted(root.glob("*.csv")):
            frames[csv_path.stem] = pd.read_csv(csv_path)
        return frames

    def load_alibaba_subset(self) -> Dict[str, pd.DataFrame]:
        root = self.data_root / "alibaba_cluster_trace_2018"
        machine_meta = pd.read_csv(
            root / "machine_meta.csv",
            header=None,
            names=[
                "machine_id",
                "time_stamp",
                "failure_domain_1",
                "failure_domain_2",
                "cpu_num",
                "mem_size",
                "status",
            ],
        )
        batch_task = pd.read_csv(
            root / "batch_task.csv",
            header=None,
            names=[
                "task_name",
                "instance_num",
                "job_name",
                "task_type",
                "status",
                "start_time",
                "end_time",
                "plan_cpu",
                "plan_mem",
            ],
        )
        machine_usage = pd.read_csv(
            root / "machine_usage.csv",
            header=None,
            names=[
                "machine_id",
                "time_stamp",
                "cpu_util_percent",
                "mem_util_percent",
                "mem_gps",
                "mpki",
                "net_in",
                "net_out",
                "disk_io_percent",
            ],
        )
        return {
            "machine_meta": machine_meta,
            "batch_task": batch_task,
            "machine_usage": machine_usage,
        }

    def load_google_subset(self) -> Dict[str, pd.DataFrame]:
        root = self.data_root / "google_cluster_trace"
        frames: Dict[str, pd.DataFrame] = {}
        for csv_path in sorted(root.glob("*.csv")):
            frames[csv_path.stem] = pd.read_csv(csv_path, header=None)
        return frames

    def load_didi_sample(self) -> Dict[str, pd.DataFrame]:
        root = self.data_root / "didi_gaia"
        frames: Dict[str, pd.DataFrame] = {}
        for csv_path in sorted(root.glob("*.csv")):
            frames[csv_path.stem] = pd.read_csv(csv_path)
        return frames

    def summarize(self) -> List[RealDatasetSummary]:
        summaries: List[RealDatasetSummary] = []
        dataset_loaders = {
            "glasgow_mec": self.load_glasgow_mec,
            "uci_mec_execution_times": self.load_uci_execution_times,
            "alibaba_cluster_trace_2018": self.load_alibaba_subset,
            "google_cluster_trace": self.load_google_subset,
            "didi_gaia": self.load_didi_sample,
        }

        for dataset_name, loader in dataset_loaders.items():
            root = self.data_root / dataset_name
            if not root.exists():
                continue

            frames = loader()
            summaries.append(
                RealDatasetSummary(
                    name=dataset_name,
                    files=[path.name for path in sorted(root.glob("*")) if path.is_file()],
                    row_counts={name: int(len(frame)) for name, frame in frames.items()},
                    columns={name: [str(col) for col in frame.columns.tolist()] for name, frame in frames.items()},
                )
            )

        return summaries

    def summarize_light(self) -> List[RealDatasetSummary]:
        summaries: List[RealDatasetSummary] = []

        for dataset_root in sorted(self.data_root.iterdir()):
            if not dataset_root.is_dir():
                continue

            files = sorted(path for path in dataset_root.glob("*.csv") if path.is_file())
            row_counts: Dict[str, int] = {}
            columns: Dict[str, List[str]] = {}

            for file_path in files:
                if dataset_root.name == "alibaba_cluster_trace_2018":
                    known_columns = {
                        "machine_meta": [
                            "machine_id",
                            "time_stamp",
                            "failure_domain_1",
                            "failure_domain_2",
                            "cpu_num",
                            "mem_size",
                            "status",
                        ],
                        "batch_task": [
                            "task_name",
                            "instance_num",
                            "job_name",
                            "task_type",
                            "status",
                            "start_time",
                            "end_time",
                            "plan_cpu",
                            "plan_mem",
                        ],
                        "machine_usage": [
                            "machine_id",
                            "time_stamp",
                            "cpu_util_percent",
                            "mem_util_percent",
                            "mem_gps",
                            "mpki",
                            "net_in",
                            "net_out",
                            "disk_io_percent",
                        ],
                    }
                    columns[file_path.stem] = known_columns.get(file_path.stem, [])
                elif dataset_root.name == "google_cluster_trace":
                    sample = pd.read_csv(file_path, nrows=5, header=None)
                    columns[file_path.stem] = [f"col_{idx}" for idx in range(sample.shape[1])]
                else:
                    sample = pd.read_csv(file_path, nrows=5, header=0)
                    columns[file_path.stem] = [str(col) for col in sample.columns.tolist()]

                if file_path.stat().st_size <= self.large_file_threshold_bytes:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                        row_counts[file_path.stem] = max(sum(1 for _ in handle) - 1, 0)
                else:
                    row_counts[file_path.stem] = -1

            summaries.append(
                RealDatasetSummary(
                    name=dataset_root.name,
                    files=[path.name for path in files],
                    row_counts=row_counts,
                    columns=columns,
                )
            )

        return summaries


# Backward-compatible aliases for internal refactor safety.
DataLoader = DebugDatasetFactory
RealTraceLoader = RealDataLoader
