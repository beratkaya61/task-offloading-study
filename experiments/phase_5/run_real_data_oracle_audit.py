#!/usr/bin/env python3
"""Phase 5R oracle/feasibility audit for the real composite trace benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator
from src.core.config_adapters import build_trace_training_config
from src.core.experiment_artifacts import load_yaml
from src.core.trace_processor import TraceEpisode


ACTION_NAMES = {
    0: "local",
    1: "edge_25",
    2: "edge_50",
    3: "edge_75",
    4: "edge_100",
    5: "cloud",
}


def _flatten_tasks(episodes):
    tasks = []
    for episode in episodes:
        tasks.extend(episode.tasks)
    return tasks


def _summary(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _fmt_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def run_real_data_oracle_audit(
    config_path: str = "configs/phase_5/real_data_rl_training.yaml",
    split: str = "test",
    seed: int = 42,
    max_tasks: int | None = None,
    csv_path: str = "results/phase_5/metrics/real_data/oracle/real_data_oracle_action_audit.csv",
    report_path: str = "v2_docs/phase_5/real_data_oracle_audit.md",
    trace_dir: str | None = None,
) -> dict[str, object]:
    base_config = load_yaml(config_path)
    if trace_dir:
        base_config.setdefault("trace", {})["trace_dir"] = trace_dir
    run_config = build_trace_training_config(
        base_config,
        algorithm="ppo",
        seed=int(seed),
        log_dir="results/tmp/real_data_oracle_audit",
        checkpoint_dir="models/tmp/real_data_oracle_audit",
        report_path=report_path,
        overrides=None,
    )
    orchestrator = TraceTrainingOrchestrator(
        config_path=config_path,
        seed=int(seed),
        config_dict=run_config,
    )
    train_eps, val_eps, test_eps = orchestrator.prepare_traces()
    split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
    tasks = _flatten_tasks(split_map[split])
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    detail_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    action_success = defaultdict(int)
    action_delay = defaultdict(list)

    for index, task in enumerate(tasks):
        single_episode = TraceEpisode(
            episode_id=index,
            tasks=[task],
            trace_name=f"oracle_task_{getattr(task, 'task_id', index)}",
            device_density=1,
        )
        env = orchestrator._build_trace_env([single_episode])
        action_rows = []

        for action in range(6):
            obs, _ = env.reset()
            _next_obs, _reward, _terminated, _truncated, info = env.step(action)
            deadline_window = float(getattr(task, "deadline", 0.0) - getattr(task, "arrival_time", 0.0))
            row = {
                "task_id": int(getattr(task, "task_id", index)),
                "action": action,
                "action_name": ACTION_NAMES[action],
                "success": bool(info.get("task_success", False)),
                "delay_s": float(info.get("delay", 0.0)),
                "energy_j": float(info.get("energy", 0.0)),
                "deadline_window_s": deadline_window,
                "deadline_tightness": float(info.get("delay", 0.0)) / max(deadline_window, 1e-9),
                "cpu_cycles": int(getattr(task, "cpu_cycles", 0)),
                "data_size_kb": int(getattr(task, "data_size", 0)),
                "priority": int(getattr(task, "priority", 0)),
                "queue_delay_s": float(info.get("queue_delay", 0.0)),
            }
            detail_rows.append(row)
            action_rows.append(row)
            action_success[action] += int(row["success"])
            action_delay[action].append(float(row["delay_s"]))

        feasible_rows = [row for row in action_rows if row["success"]]
        if feasible_rows:
            best = min(feasible_rows, key=lambda row: (row["delay_s"], row["energy_j"]))
            oracle_success = True
        else:
            best = min(action_rows, key=lambda row: row["delay_s"])
            oracle_success = False

        oracle_rows.append(
            {
                "task_id": int(getattr(task, "task_id", index)),
                "oracle_success": oracle_success,
                "oracle_action": int(best["action"]),
                "oracle_action_name": str(best["action_name"]),
                "oracle_delay_s": float(best["delay_s"]),
                "oracle_energy_j": float(best["energy_j"]),
                "deadline_window_s": float(best["deadline_window_s"]),
                "deadline_tightness": float(best["deadline_tightness"]),
                "feasible_action_count": int(sum(1 for row in action_rows if row["success"])),
                "cpu_cycles": int(getattr(task, "cpu_cycles", 0)),
                "data_size_kb": int(getattr(task, "data_size", 0)),
                "priority": int(getattr(task, "priority", 0)),
            }
        )

    detail_path = Path(csv_path)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        with open(detail_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    oracle_frame = pd.DataFrame(oracle_rows)
    action_frame = pd.DataFrame(detail_rows)
    total_tasks = max(1, len(oracle_frame))
    oracle_success_rate = float(oracle_frame["oracle_success"].mean()) if not oracle_frame.empty else 0.0
    impossible_task_ratio = 1.0 - oracle_success_rate
    action_summary = []
    for action in range(6):
        action_subset = action_frame[action_frame["action"] == action]
        action_summary.append(
            {
                "action": action,
                "action_name": ACTION_NAMES[action],
                "feasible_rate": float(action_subset["success"].mean()) if not action_subset.empty else 0.0,
                "avg_delay_s": float(action_subset["delay_s"].mean()) if not action_subset.empty else 0.0,
                "p95_delay_s": float(action_subset["delay_s"].quantile(0.95)) if not action_subset.empty else 0.0,
            }
        )

    dominant_oracle_action = (
        oracle_frame["oracle_action_name"].value_counts().idxmax() if not oracle_frame.empty else "n/a"
    )
    best_action_share = (
        float(oracle_frame["oracle_action_name"].value_counts(normalize=True).max())
        if not oracle_frame.empty
        else 0.0
    )
    status = "usable"
    if oracle_success_rate < 0.55:
        status = "too_hard_or_miscalibrated"
    elif oracle_success_rate > 0.95:
        status = "possibly_too_easy"

    audit = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_name": "real-world trace-driven hybrid benchmark",
        "config_path": config_path,
        "split": split,
        "seed": int(seed),
        "task_count": int(total_tasks),
        "oracle_success_rate": oracle_success_rate,
        "impossible_task_ratio": impossible_task_ratio,
        "dominant_oracle_action": dominant_oracle_action,
        "dominant_oracle_action_share": best_action_share,
        "status": status,
        "action_summary": action_summary,
        "deadline_window_summary": _summary(oracle_frame["deadline_window_s"] if not oracle_frame.empty else pd.Series(dtype=float)),
        "deadline_tightness_summary": _summary(oracle_frame["deadline_tightness"] if not oracle_frame.empty else pd.Series(dtype=float)),
        "cpu_cycles_summary": _summary(oracle_frame["cpu_cycles"] if not oracle_frame.empty else pd.Series(dtype=float)),
        "data_size_kb_summary": _summary(oracle_frame["data_size_kb"] if not oracle_frame.empty else pd.Series(dtype=float)),
        "detail_csv": str(detail_path),
        "trace_dir": trace_dir or base_config.get("trace", {}).get("trace_dir", ""),
    }

    write_oracle_report(audit, Path(report_path))
    json_path = Path(report_path).with_suffix(".json")
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def write_oracle_report(audit: dict[str, object], report_path: Path) -> None:
    lines = [
        "# Phase 5R Real-Data Oracle Audit",
        "",
        "Bu rapor, mevcut veri setini `real-world trace-driven hybrid benchmark` olarak okur; tam gercek MEC logu iddiasi tasimaz.",
        "Amac, algoritma sonuclarindan once benchmark'in fiziksel olarak cozulur olup olmadigini gormektir.",
        "",
        "## Gate Result",
        "",
        f"- split: `{audit['split']}`",
        f"- task count: `{audit['task_count']}`",
        f"- oracle ceiling / success rate: `{_fmt_pct(float(audit['oracle_success_rate']))}`",
        f"- impossible task ratio: `{_fmt_pct(float(audit['impossible_task_ratio']))}`",
        f"- dominant oracle action: `{audit['dominant_oracle_action']}` (`{_fmt_pct(float(audit['dominant_oracle_action_share']))}`)",
        f"- status: `{audit['status']}`",
        "",
    ]

    if audit["status"] == "too_hard_or_miscalibrated":
        lines.append("Karar: Oracle bile dusuk oldugu icin once benchmark mapping/deadline kalibrasyonu incelenmelidir.")
    elif audit["status"] == "possibly_too_easy":
        lines.append("Karar: Oracle cok yuksek; benchmark cozulur, fakat deadline baskisi fazla yumusamis olabilir.")
    else:
        lines.append("Karar: Benchmark, Faz 5R policy ve ablation kosulari icin makul bir cozulur-zor rejimde gorunuyor.")

    lines.extend(
        [
            "",
            "## Per-Action Feasibility",
            "",
            "| Action | Name | Feasible Rate | Avg Delay (s) | P95 Delay (s) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in audit["action_summary"]:
        lines.append(
            f"| {row['action']} | {row['action_name']} | {_fmt_pct(float(row['feasible_rate']))} | "
            f"{_fmt_float(float(row['avg_delay_s']))} | {_fmt_float(float(row['p95_delay_s']))} |"
        )

    lines.extend(
        [
            "",
            "## Distribution Summaries",
            "",
            "| Field | Mean | Std | Min | P50 | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("deadline_window_summary", "deadline_window_s"),
        ("deadline_tightness_summary", "oracle_deadline_tightness"),
        ("cpu_cycles_summary", "cpu_cycles"),
        ("data_size_kb_summary", "data_size_kb"),
    ]:
        summary = audit[key]
        lines.append(
            f"| {label} | {_fmt_float(summary['mean'])} | {_fmt_float(summary['std'])} | "
            f"{_fmt_float(summary['min'])} | {_fmt_float(summary['p50'])} | "
            f"{_fmt_float(summary['p95'])} | {_fmt_float(summary['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Oracle ceiling dusukse: dusuk PPO/heuristic sonucu algoritma basarisizligi olarak yorumlanmaz.",
            "- Oracle ceiling yuksek ama modeller dusukse: reward, semantic prior, state veya action-collapse sorunu aranir.",
            "- Tek aksiyona cok yogunlasan politika final bilimsel iddia olarak kullanilmadan once action-collapse diagnostiginden gecmelidir.",
            "",
            f"Detailed action CSV: `{audit['detail_csv']}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5R real-data oracle/feasibility audit")
    parser.add_argument("--config", default="configs/phase_5/real_data_rl_training.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--csv-path", default="results/phase_5/metrics/real_data/oracle/real_data_oracle_action_audit.csv")
    parser.add_argument("--report-path", default="v2_docs/phase_5/real_data_oracle_audit.md")
    parser.add_argument("--trace-dir", default=None)
    args = parser.parse_args()
    run_real_data_oracle_audit(
        config_path=args.config,
        split=args.split,
        seed=args.seed,
        max_tasks=args.max_tasks,
        csv_path=args.csv_path,
        report_path=args.report_path,
        trace_dir=args.trace_dir,
    )
