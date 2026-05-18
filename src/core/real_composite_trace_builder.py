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
    reference_datarate_bps: float = 25e6
    max_server_context_rows: int = 50_000


class RealCompositeTraceBuilder:
    LOCAL_CPU_HZ = 1e9
    EDGE_CPU_HZ = 2e9
    CLOUD_CPU_HZ = 5e9
    CLOUD_FIXED_LATENCY_S = 0.1

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
            }
        )
        random_pool = pd.DataFrame(
            {
                "lat": random_based["lat"],
                "long": random_based["long"],
                "machine_name": random_based["MachineName"],
            }
        )

        pooled = pd.concat([consecutive_pool, random_pool], ignore_index=True)
        pooled = pooled.dropna(subset=["lat", "long", "machine_name"]).copy()
        pooled["device_id"] = pd.factorize(pooled["machine_name"])[0] + 1
        pooled = pooled[(pooled["device_id"] > 0) & pooled["lat"].notna() & pooled["long"].notna()]
        pooled = pooled.reset_index(drop=True)
        pooled["raw_lat"] = pooled["lat"].astype(float)
        pooled["raw_long"] = pooled["long"].astype(float)
        pooled[["lat", "long"]] = self._project_geo_to_env_plane(
            pooled["raw_lat"],
            pooled["raw_long"],
        )
        return pooled

    @staticmethod
    def _project_geo_to_env_plane(latitudes: pd.Series, longitudes: pd.Series) -> pd.DataFrame:
        lat_min, lat_max = float(latitudes.min()), float(latitudes.max())
        lon_min, lon_max = float(longitudes.min()), float(longitudes.max())
        lat_span = max(lat_max - lat_min, 1e-9)
        lon_span = max(lon_max - lon_min, 1e-9)

        env_x = 50.0 + 900.0 * ((latitudes.astype(float) - lat_min) / lat_span)
        env_y = 50.0 + 900.0 * ((longitudes.astype(float) - lon_min) / lon_span)
        return pd.DataFrame({"lat": env_x.clip(50.0, 950.0), "long": env_y.clip(50.0, 950.0)})

    def _server_context_pool(self, max_rows: int) -> pd.DataFrame:
        machine_meta_path = self.data_root / "alibaba_cluster_trace_2018" / "machine_meta.csv"
        machine_usage_path = self.data_root / "alibaba_cluster_trace_2018" / "machine_usage.csv"

        machine_meta = pd.read_csv(
            machine_meta_path,
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
        valid_machine_ids = set(
            machine_meta.loc[machine_meta["status"] == "USING", "machine_id"].astype(str).tolist()
        )

        chunks = []
        remaining = max_rows
        for chunk in pd.read_csv(
            machine_usage_path,
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
            chunksize=200_000,
        ):
            valid = chunk[
                chunk["machine_id"].astype(str).isin(valid_machine_ids)
                & chunk["cpu_util_percent"].between(0, 100)
                & chunk["mem_util_percent"].between(0, 100)
            ][["machine_id", "cpu_util_percent", "mem_util_percent"]]
            if valid.empty:
                continue
            sample_n = min(len(valid), remaining)
            sampled = valid.sample(n=sample_n, random_state=self.seed)
            chunks.append(sampled)
            remaining -= sample_n
            if remaining <= 0:
                break

        if not chunks:
            raise RuntimeError("No valid Alibaba machine_usage rows were found for server-context calibration.")

        pooled = pd.concat(chunks, ignore_index=True).reset_index(drop=True)
        pooled["server_id"] = pd.factorize(pooled["machine_id"])[0] + 1
        return pooled

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

    @staticmethod
    def _rank_score(values: pd.Series) -> pd.Series:
        return values.rank(method="average", pct=True).clip(lower=0.0, upper=1.0)

    @classmethod
    def _reference_best_case_delay(
        cls,
        cpu_cycles: float,
        size_bits: float,
        reference_datarate_bps: float,
    ) -> float:
        tx_delay = size_bits / max(reference_datarate_bps, 1.0)
        local_delay = cpu_cycles / cls.LOCAL_CPU_HZ
        edge_delay = tx_delay + (cpu_cycles / cls.EDGE_CPU_HZ)
        cloud_delay = tx_delay + cls.CLOUD_FIXED_LATENCY_S + (cpu_cycles / cls.CLOUD_CPU_HZ)
        partial_delay = max(
            0.25 * cpu_cycles / cls.LOCAL_CPU_HZ,
            0.75 * ((0.75 * size_bits) / max(reference_datarate_bps, 1.0) + (0.75 * cpu_cycles / cls.EDGE_CPU_HZ)),
        )
        return float(min(local_delay, edge_delay, cloud_delay, partial_delay))

    def _calibrate_task_fields(
        self,
        task_row: pd.Series,
        execution_time_s: float,
        reference_datarate_bps: float,
    ) -> dict:
        difficulty = float(task_row["difficulty_score"])
        plan_mem_score = float(task_row["plan_mem_score"])

        edge_execution_anchor_s = float(np.clip(execution_time_s, 0.08, 1.5))
        target_edge_service_s = edge_execution_anchor_s * (1.35 + 1.7 * difficulty)
        cpu_cycles = int(np.clip(target_edge_service_s * self.EDGE_CPU_HZ, 1.5e8, 9.5e9))

        data_size_kb = int(np.clip(128 + (plan_mem_score ** 0.9) * (1280 - 128), 128, 1280))
        size_bits = float(data_size_kb * 8 * 1024)

        reference_best_case_delay_s = self._reference_best_case_delay(
            cpu_cycles,
            size_bits,
            reference_datarate_bps=reference_datarate_bps,
        )
        deadline_slack = 0.72 + 0.35 * difficulty + 0.55 * float(self.rng.random())
        deadline_window_s = max(0.15, reference_best_case_delay_s * deadline_slack)

        urgency = reference_best_case_delay_s / max(deadline_window_s, 1e-6)
        if urgency >= 0.92:
            priority = 3
        elif urgency >= 0.82:
            priority = 2
        elif urgency >= 0.72:
            priority = 1
        else:
            priority = 0

        return {
            "cpu_cycles": cpu_cycles,
            "data_size_kb": data_size_kb,
            "size_bits": size_bits,
            "reference_best_case_delay_s": reference_best_case_delay_s,
            "deadline_window_s": deadline_window_s,
            "priority": priority,
            "difficulty_score": difficulty,
            "edge_execution_anchor_s": edge_execution_anchor_s,
            "target_edge_service_s": target_edge_service_s,
        }

    @staticmethod
    def _log_quantile_hint(series: pd.Series) -> tuple[pd.Series, dict]:
        q05 = float(series.quantile(0.05))
        q95 = float(series.quantile(0.95))
        q95 = max(q95, q05 + 1.0)
        log_q05 = float(np.log1p(q05))
        log_q95 = float(np.log1p(q95))
        denom = max(log_q95 - log_q05, 1e-9)
        hint = ((np.log1p(series.astype(float)) - log_q05) / denom).clip(lower=0.0, upper=1.0)
        return hint.astype(float), {
            "p05": q05,
            "p95": q95,
        }

    def build_task_records(self, config: RealCompositeBuildConfig) -> pd.DataFrame:
        mobility = self._mobility_pool()
        server_context = self._server_context_pool(config.max_server_context_rows)
        execution_times = self._execution_time_pool()
        alibaba_tasks = self._alibaba_task_sample(config.max_tasks)

        alibaba_tasks["duration_s"] = (alibaba_tasks["end_time"] - alibaba_tasks["start_time"]).clip(lower=1.0)
        alibaba_tasks["plan_cpu_score"] = self._rank_score(np.log1p(alibaba_tasks["plan_cpu"].astype(float)))
        alibaba_tasks["plan_mem_score"] = self._rank_score(np.log1p(alibaba_tasks["plan_mem"].astype(float)))
        alibaba_tasks["duration_score"] = self._rank_score(np.log1p(alibaba_tasks["duration_s"].astype(float)))
        alibaba_tasks["difficulty_score"] = (
            0.65 * alibaba_tasks["plan_cpu_score"]
            + 0.20 * alibaba_tasks["plan_mem_score"]
            + 0.15 * alibaba_tasks["duration_score"]
        ).clip(lower=0.0, upper=1.0)

        mobility_indices = self.rng.integers(0, len(mobility), size=len(alibaba_tasks))
        server_indices = self.rng.integers(0, len(server_context), size=len(alibaba_tasks))
        execution_indices = self.rng.integers(0, len(execution_times), size=len(alibaba_tasks))

        records: List[dict] = []
        for idx, task_row in alibaba_tasks.iterrows():
            mobility_row = mobility.iloc[int(mobility_indices[idx])]
            server_row = server_context.iloc[int(server_indices[idx])]
            execution_time = float(execution_times[int(execution_indices[idx])])
            start_time = float(task_row["start_time"])
            end_time = float(task_row["end_time"])
            duration = max(end_time - start_time, 1.0)

            calibrated = self._calibrate_task_fields(
                task_row,
                execution_time_s=execution_time,
                reference_datarate_bps=config.reference_datarate_bps,
            )
            deadline = start_time + calibrated["deadline_window_s"]

            records.append(
                {
                    "task_id": idx,
                    "device_id": int(mobility_row["device_id"]),
                    "arrival_time": start_time,
                    "deadline": deadline,
                    "data_size": int(calibrated["data_size_kb"]),
                    "cpu_cycles": int(calibrated["cpu_cycles"]),
                    "priority": int(calibrated["priority"]),
                    "location_x": float(mobility_row["lat"]),
                    "location_y": float(mobility_row["long"]),
                    "raw_location_x": float(mobility_row["raw_lat"]),
                    "raw_location_y": float(mobility_row["raw_long"]),
                    "server_id": int(server_row["server_id"]),
                    "server_cpu_utilization": float(server_row["cpu_util_percent"]),
                    "server_mem_utilization": float(server_row["mem_util_percent"]),
                    "execution_time_s": execution_time,
                    "alibaba_duration_s": duration,
                    "difficulty_score": float(calibrated["difficulty_score"]),
                    "reference_best_case_delay_s": float(calibrated["reference_best_case_delay_s"]),
                    "deadline_window_proxy_s": float(calibrated["deadline_window_s"]),
                    "edge_execution_anchor_s": float(calibrated["edge_execution_anchor_s"]),
                    "target_edge_service_s": float(calibrated["target_edge_service_s"]),
                    "task_name": str(task_row["task_name"]),
                    "job_name": str(task_row["job_name"]),
                    "task_type_raw": str(task_row["task_type"]),
                    "field_source_arrival_time": "alibaba_batch_task.start_time",
                    "field_source_location": "glasgow_mec.mobility",
                    "field_source_execution_time": "uci_execution_times",
                    "field_source_server_context": "alibaba_machine_usage",
                    "field_source_workload": "alibaba_batch_task",
                    "field_proxy_deadline": True,
                    "field_proxy_data_size": True,
                    "field_proxy_cpu_cycles": True,
                    "field_proxy_priority": True,
                }
            )

        records_df = pd.DataFrame.from_records(records)
        records_df = records_df.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)
        records_df["size_bits"] = records_df["data_size"].astype(float) * 8 * 1024
        records_df["cpu_norm_hint"], cpu_norm_meta = self._log_quantile_hint(records_df["cpu_cycles"])
        records_df["size_norm_hint"], size_norm_meta = self._log_quantile_hint(records_df["size_bits"])
        records_df.attrs["state_normalization"] = {
            "cpu_cycles": cpu_norm_meta,
            "size_bits": size_norm_meta,
        }
        return records_df

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
                    server_id=int(row.server_id),
                    server_cpu_utilization=float(row.server_cpu_utilization),
                    server_mem_utilization=float(row.server_mem_utilization),
                    cpu_norm_hint=float(row.cpu_norm_hint),
                    size_norm_hint=float(row.size_norm_hint),
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
