from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.dataset_loader import RealDataLoader
from src.core.trace_processor import TraceEpisode, TraceTask


@dataclass
class RealCompositeBuildConfig:
    max_tasks: int = 5000
    tasks_per_episode: int = 50
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


class RealCompositeTraceBuilder:
    def __init__(self, data_root: str | Path = "data/raw_real_datasets", seed: int = 42):
        self.data_root = Path(data_root)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.loader = RealDataLoader(data_root=self.data_root)

    def _mobility_pool(self) -> pd.DataFrame:
        glasgow = self.loader.load_glasgow_mec()
        consecutive = glasgow["consecutive"].copy()
        random_based = glasgow["random_based"].copy()

        consecutive_pool = pd.DataFrame(
            {
                "lat": consecutive["lat"],
                "long": consecutive["long"],
                "machine_name": consecutive["machine_name"],
                "server_id": consecutive["serverId"],
                "server_cpu_utilization": consecutive["cpu_utilization"],
                "server_mem_utilization": consecutive["mem_utilization"],
                "mobility_delay_hint": np.nan,
            }
        )

        random_pool = pd.DataFrame(
            {
                "lat": random_based["lat"],
                "long": random_based["long"],
                "machine_name": random_based["MachineName"],
                "server_id": np.nan,
                "server_cpu_utilization": np.nan,
                "server_mem_utilization": np.nan,
                "mobility_delay_hint": random_based["TotalDelay"],
            }
        )

        pooled = pd.concat([consecutive_pool, random_pool], ignore_index=True)
        pooled = pooled.dropna(subset=["lat", "long", "machine_name"]).copy()
        pooled["device_id"] = pd.factorize(pooled["machine_name"])[0] + 1
        pooled["server_id"] = pooled["server_id"].fillna(0).astype(int)
        pooled["server_cpu_utilization"] = pooled["server_cpu_utilization"].fillna(
            consecutive_pool["server_cpu_utilization"].median()
        )
        pooled["server_mem_utilization"] = pooled["server_mem_utilization"].fillna(
            consecutive_pool["server_mem_utilization"].median()
        )
        pooled = pooled[(pooled["device_id"] > 0) & pooled["lat"].notna() & pooled["long"].notna()]
        return pooled.reset_index(drop=True)

    def _execution_time_pool(self) -> np.ndarray:
        uci_frames = self.loader.load_uci_execution_times()
        values: List[float] = []
        for frame in uci_frames.values():
            values.extend(frame["Execution Time"].astype(float).tolist())
        execution_times = np.asarray(values, dtype=float)
        return execution_times[execution_times > 0.0]

    def _alibaba_task_sample(self, max_tasks: int) -> pd.DataFrame:
        csv_path = self.data_root / "alibaba_cluster_trace_2018" / "batch_task.csv"
        columns = [
            "task_name",
            "instance_num",
            "job_name",
            "task_type",
            "status",
            "start_time",
            "end_time",
            "plan_cpu",
            "plan_mem",
        ]
        chunks = []
        remaining = max_tasks
        for chunk in pd.read_csv(csv_path, header=None, names=columns, chunksize=200_000):
            valid = chunk[
                (chunk["status"] == "Terminated")
                & (chunk["end_time"] > chunk["start_time"])
                & (chunk["plan_cpu"] > 0)
                & (chunk["plan_mem"] > 0)
            ]
            if valid.empty:
                continue
            sample_n = min(len(valid), remaining)
            sampled = valid.sample(n=sample_n, random_state=self.seed)
            chunks.append(sampled)
            remaining -= sample_n
            if remaining <= 0:
                break
        if not chunks:
            raise RuntimeError("No valid Alibaba batch_task rows were found for composite real-data build.")
        return pd.concat(chunks, ignore_index=True).reset_index(drop=True)

    def build_task_records(self, config: RealCompositeBuildConfig) -> pd.DataFrame:
        mobility = self._mobility_pool()
        execution_times = self._execution_time_pool()
        alibaba_tasks = self._alibaba_task_sample(config.max_tasks)
        mobility_indices = self.rng.integers(0, len(mobility), size=len(alibaba_tasks))
        execution_indices = self.rng.integers(0, len(execution_times), size=len(alibaba_tasks))

        records: List[dict] = []
        for idx, task_row in alibaba_tasks.iterrows():
            mobility_row = mobility.iloc[int(mobility_indices[idx])]
            execution_time = float(execution_times[int(execution_indices[idx])])
            start_time = float(task_row["start_time"])
            end_time = float(task_row["end_time"])
            duration = max(end_time - start_time, 1.0)

            plan_cpu = float(task_row["plan_cpu"])
            plan_mem = float(task_row["plan_mem"])

            # Transparent proxy mappings for MEC-specific fields missing in raw traces.
            cpu_cycles_proxy = int(max(1e8, plan_cpu * 1e9))
            data_size_proxy_kb = int(max(128, plan_mem * 1024))
            deadline_window_proxy = float(min(10.0, max(0.2, execution_time * 4.0 + 0.5)))
            deadline = start_time + deadline_window_proxy

            priority_score = (
                3 if execution_time <= np.quantile(execution_times, 0.25)
                else 2 if execution_time <= np.quantile(execution_times, 0.5)
                else 1 if execution_time <= np.quantile(execution_times, 0.75)
                else 0
            )

            records.append(
                {
                    "task_id": idx,
                    "device_id": int(mobility_row["device_id"]),
                    "arrival_time": start_time,
                    "deadline": deadline,
                    "data_size": data_size_proxy_kb,
                    "cpu_cycles": cpu_cycles_proxy,
                    "priority": int(priority_score),
                    "location_x": float(mobility_row["lat"]),
                    "location_y": float(mobility_row["long"]),
                    "server_id": int(mobility_row["server_id"]),
                    "server_cpu_utilization": float(mobility_row["server_cpu_utilization"]),
                    "server_mem_utilization": float(mobility_row["server_mem_utilization"]),
                    "execution_time_s": execution_time,
                    "alibaba_duration_s": duration,
                    "task_name": str(task_row["task_name"]),
                    "job_name": str(task_row["job_name"]),
                    "task_type_raw": str(task_row["task_type"]),
                    "field_source_arrival_time": "alibaba_batch_task.start_time",
                    "field_source_location": "glasgow_mec.mobility",
                    "field_source_execution_time": "uci_execution_times",
                    "field_source_server_context": "glasgow_mec.server_context",
                    "field_source_workload": "alibaba_batch_task",
                    "field_proxy_deadline": True,
                    "field_proxy_data_size": True,
                    "field_proxy_cpu_cycles": True,
                    "field_proxy_priority": True,
                }
            )

        records_df = pd.DataFrame.from_records(records)
        return records_df.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)

    def build_episode_splits(self, records_df: pd.DataFrame, config: RealCompositeBuildConfig) -> Dict[str, List[TraceEpisode]]:
        episodes: List[TraceEpisode] = []
        for episode_id, start in enumerate(range(0, len(records_df), config.tasks_per_episode)):
            chunk = records_df.iloc[start:start + config.tasks_per_episode]
            if len(chunk) < config.tasks_per_episode:
                break
            tasks = [
                TraceTask(
                    task_id=int(row.task_id),
                    device_id=int(row.device_id),
                    arrival_time=float(row.arrival_time),
                    deadline=float(row.deadline),
                    data_size=int(row.data_size),
                    cpu_cycles=int(row.cpu_cycles),
                    priority=int(row.priority),
                    location=(float(row.location_x), float(row.location_y)),
                )
                for row in chunk.itertuples(index=False)
            ]
            episodes.append(
                TraceEpisode(
                    episode_id=episode_id,
                    tasks=tasks,
                    trace_name=f"real_composite_ep{episode_id}",
                    device_density=len({task.device_id for task in tasks}),
                )
            )

        indices = self.rng.permutation(len(episodes))
        n_train = int(len(episodes) * config.train_ratio)
        n_val = int(len(episodes) * config.val_ratio)

        train = [episodes[i] for i in indices[:n_train]]
        val = [episodes[i] for i in indices[n_train:n_train + n_val]]
        test = [episodes[i] for i in indices[n_train + n_val:]]
        return {"train": train, "val": val, "test": test}

    @staticmethod
    def save_episode_split(episodes: List[TraceEpisode], output_path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "tasks": [task.to_dict() for task in episode.tasks],
                    "trace_name": episode.trace_name,
                    "device_density": episode.device_density,
                }
                for episode in episodes
            ],
            "metadata": metadata or {},
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def save_records(records_df: pd.DataFrame, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records_df.to_csv(path, index=False)
