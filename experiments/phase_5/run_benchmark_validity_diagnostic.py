#!/usr/bin/env python3
"""Benchmark validity diagnostics for Phase 5R real-data fusion experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIAGNOSTIC_COLUMNS = [
    "cpu_cycles",
    "data_size",
    "deadline_window_s",
    "difficulty_score",
    "execution_time_s",
    "server_cpu_utilization",
    "server_mem_utilization",
    "server_load_score",
    "location_x",
    "location_y",
    "arrival_rank",
    "mobility_rank",
]


def _fmt(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}"


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _load_records(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path)
    if "deadline_window_s" not in records.columns:
        records["deadline_window_s"] = records["deadline"] - records["arrival_time"]
    if "server_load_score" not in records.columns:
        records["server_load_score"] = (
            0.65 * records.get("server_cpu_utilization", 0.0).astype(float)
            + 0.35 * records.get("server_mem_utilization", 0.0).astype(float)
        ) / 100.0
    if "arrival_rank" not in records.columns:
        records = records.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)
        records["arrival_rank"] = records.index / max(len(records) - 1, 1)
    return records


def _correlation_payload(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    available = [col for col in DIAGNOSTIC_COLUMNS if col in records.columns]
    corr = records[available].corr(numeric_only=True).fillna(0.0)
    return {
        row: {col: round(float(corr.loc[row, col]), 4) for col in corr.columns}
        for row in corr.index
    }


def _key_correlations(corr: dict[str, dict[str, float]]) -> dict[str, float]:
    def get(a, b):
        return float(corr.get(a, {}).get(b, 0.0))

    return {
        "difficulty_vs_data_size": get("difficulty_score", "data_size"),
        "difficulty_vs_deadline": get("difficulty_score", "deadline_window_s"),
        "difficulty_vs_execution_time": get("difficulty_score", "execution_time_s"),
        "difficulty_vs_server_load": get("difficulty_score", "server_load_score"),
        "arrival_vs_mobility_rank": get("arrival_rank", "mobility_rank"),
        "arrival_vs_server_load": get("arrival_rank", "server_load_score"),
        "location_x_vs_server_load": get("location_x", "server_load_score"),
    }


def _oracle_summary(path: Path | None) -> dict[str, object]:
    if not path or not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"available": False}

    task_count = int(frame["task_id"].nunique())
    action_summary = (
        frame.groupby(["action", "action_name"], as_index=False)
        .agg(feasible_rate=("success", "mean"), avg_delay_s=("delay_s", "mean"))
        .sort_values("action")
    )
    feasible = frame[frame["success"].astype(bool)].copy()
    if feasible.empty:
        oracle_success_rate = 0.0
        dominant_action = "n/a"
        dominant_share = 0.0
        oracle_actions = pd.Series(dtype=str)
    else:
        idx = feasible.sort_values(["task_id", "delay_s", "energy_j"]).groupby("task_id").head(1).index
        best = feasible.loc[idx]
        oracle_success_rate = len(best) / max(task_count, 1)
        oracle_actions = best["action_name"]
        dominant_action = str(oracle_actions.value_counts().idxmax())
        dominant_share = float(oracle_actions.value_counts(normalize=True).max())

    return {
        "available": True,
        "task_count": task_count,
        "oracle_success_rate": float(oracle_success_rate),
        "impossible_task_ratio": float(1.0 - oracle_success_rate),
        "dominant_action": dominant_action,
        "dominant_action_share": dominant_share,
        "action_summary": action_summary.to_dict(orient="records"),
        "action_diversity": int(oracle_actions.nunique()) if not feasible.empty else 0,
    }


def _policy_summary(path: Path | None) -> dict[str, object]:
    if not path or not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"available": False}
    grouped = (
        frame.groupby("config_model_type", as_index=False)
        .agg(
            success_mean=("metric_success_rate", "mean"),
            success_std=("metric_success_rate", "std"),
            p95_latency_mean=("metric_p95_latency", "mean"),
            partial_ratio=("metric_partial_offload_ratio", "mean"),
            action_5_rate=("metric_action_5_rate", "mean"),
        )
        .sort_values("success_mean", ascending=False)
    )
    return {"available": True, "rows": grouped.fillna(0.0).to_dict(orient="records")}


def _gate_status(key_corr: dict[str, float], oracle: dict[str, object]) -> dict[str, object]:
    warnings = []
    if abs(key_corr["difficulty_vs_server_load"]) < 0.05:
        warnings.append("workload difficulty ile server load neredeyse bagimsiz")
    if abs(key_corr["arrival_vs_mobility_rank"]) < 0.20:
        warnings.append("arrival sirasi ile mobility rank zayif eslesiyor")
    if oracle.get("available"):
        oracle_success = float(oracle.get("oracle_success_rate", 0.0))
        cloud_share = float(oracle.get("dominant_action_share", 0.0)) if oracle.get("dominant_action") == "cloud" else 0.0
        if oracle_success < 0.70:
            warnings.append("oracle ceiling hedef bandin altinda")
        if oracle_success > 0.95:
            warnings.append("oracle ceiling cok yuksek; benchmark fazla yumusak olabilir")
        if cloud_share > 0.70:
            warnings.append("oracle cloud dominance %70 ustunde")
        action_rates = {str(row.get("action_name")): float(row.get("feasible_rate", 0.0)) for row in oracle.get("action_summary", [])}
        if action_rates.get("local", 0.0) < 0.05:
            warnings.append("local action feasible coverage %5 altinda")
        if max(action_rates.get("edge_25", 0.0), action_rates.get("edge_50", 0.0)) < 0.10:
            warnings.append("dusuk/orta partial edge coverage zayif")
        if action_rates.get("edge_100", 0.0) < 0.10:
            warnings.append("full-edge coverage %10 altinda")
    status = "pass" if not warnings else "needs_revision"
    return {"status": status, "warnings": warnings}


def run_diagnostic(
    records_path: str,
    report_path: str,
    oracle_csv: str | None = None,
    policy_csv: str | None = None,
    benchmark_label: str = "real_composite_trace",
) -> dict[str, object]:
    records = _load_records(Path(records_path))
    corr = _correlation_payload(records)
    key_corr = _key_correlations(corr)
    oracle = _oracle_summary(Path(oracle_csv) if oracle_csv else None)
    policy = _policy_summary(Path(policy_csv) if policy_csv else None)
    gate = _gate_status(key_corr, oracle)

    priority_summary = (
        records.groupby("priority", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            cpu_cycles_mean=("cpu_cycles", "mean"),
            data_size_mean=("data_size", "mean"),
            deadline_window_mean=("deadline_window_s", "mean"),
            difficulty_mean=("difficulty_score", "mean"),
            server_load_mean=("server_load_score", "mean"),
        )
        .round(4)
        .to_dict(orient="records")
    )

    payload = {
        "benchmark_label": benchmark_label,
        "records_path": records_path,
        "record_count": int(len(records)),
        "fusion_mode": str(records.get("fusion_mode", pd.Series(["unknown"])).iloc[0]),
        "key_correlations": key_corr,
        "correlation_matrix": corr,
        "priority_summary": priority_summary,
        "oracle": oracle,
        "policy": policy,
        "gate": gate,
    }

    write_report(payload, Path(report_path))
    Path(report_path).with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_report(payload: dict[str, object], report_path: Path) -> None:
    lines = [
        f"# Benchmark Validity Diagnostic - {payload['benchmark_label']}",
        "",
        "Bu rapor, hibrit trace-driven benchmark'in dogal korelasyonlari ne kadar korudugunu ve policy sonuclarini yorumlamak icin yeterince dengeli olup olmadigini denetler.",
        "",
        "## Gate Result",
        "",
        f"- fusion mode: `{payload['fusion_mode']}`",
        f"- record count: `{payload['record_count']}`",
        f"- status: `{payload['gate']['status']}`",
    ]
    for warning in payload["gate"]["warnings"]:
        lines.append(f"- warning: {warning}")
    if not payload["gate"]["warnings"]:
        lines.append("- warning: yok")

    lines.extend(["", "## Key Correlations", "", "| Relation | Correlation |", "|---|---:|"])
    for key, value in payload["key_correlations"].items():
        lines.append(f"| {key} | {_fmt(float(value))} |")

    oracle = payload["oracle"]
    lines.extend(["", "## Oracle / Action Feasibility", ""])
    if oracle.get("available"):
        lines.extend(
            [
                f"- oracle success ceiling: `{_pct(float(oracle['oracle_success_rate']))}`",
                f"- impossible task ratio: `{_pct(float(oracle['impossible_task_ratio']))}`",
                f"- dominant oracle action: `{oracle['dominant_action']}` (`{_pct(float(oracle['dominant_action_share']))}`)",
                f"- oracle action diversity: `{oracle['action_diversity']}`",
                "",
                "| Action | Feasible Rate | Avg Delay (s) |",
                "|---|---:|---:|",
            ]
        )
        for row in oracle["action_summary"]:
            lines.append(f"| {row['action_name']} | {_pct(_safe_float(row['feasible_rate']))} | {_fmt(_safe_float(row['avg_delay_s']))} |")
    else:
        lines.append("Oracle CSV bulunmadigi icin action feasibility okunmadi.")

    lines.extend(["", "## Priority / Proxy Summary", "", "| Priority | Tasks | CPU Mean | Size Mean | Deadline Mean | Difficulty Mean | Server Load Mean |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload["priority_summary"]:
        lines.append(
            f"| {row['priority']} | {row['tasks']} | {_fmt(_safe_float(row['cpu_cycles_mean']))} | {_fmt(_safe_float(row['data_size_mean']))} | "
            f"{_fmt(_safe_float(row['deadline_window_mean']))} | {_fmt(_safe_float(row['difficulty_mean']))} | {_fmt(_safe_float(row['server_load_mean']))} |"
        )

    policy = payload["policy"]
    lines.extend(["", "## Policy Snapshot", ""])
    if policy.get("available"):
        lines.extend(["| Model | Success Mean | P95 Mean | Partial Ratio | Cloud Rate |", "|---|---:|---:|---:|---:|"])
        for row in policy["rows"]:
            lines.append(
                f"| {row['config_model_type']} | {_pct(_safe_float(row['success_mean']))} | {_fmt(_safe_float(row['p95_latency_mean']))} | "
                f"{_pct(_safe_float(row['partial_ratio']))} | {_pct(_safe_float(row['action_5_rate']))} |"
            )
    else:
        lines.append("Policy evaluation CSV bulunmadigi icin model snapshot okunmadi.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `needs_revision` sonucu model basarisizligi degil, benchmark-policy hizalama kapisinin gecilmedigi anlamina gelir.",
            "- Cloud dominance yuksekse benchmark zengin offloading karar probleminden cok cloud-agirlikli probleme donusmus olabilir.",
            "- Workload-server-location korelasyonlari zayifsa hibrit pairing dogal korelasyonlari korumuyor demektir.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark validity diagnostics")
    parser.add_argument("--records", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--oracle-csv", default=None)
    parser.add_argument("--policy-csv", default=None)
    parser.add_argument("--label", default="real_composite_trace")
    args = parser.parse_args()
    payload = run_diagnostic(
        records_path=args.records,
        report_path=args.report,
        oracle_csv=args.oracle_csv,
        policy_csv=args.policy_csv,
        benchmark_label=args.label,
    )
    print(json.dumps({"status": payload["gate"]["status"], "warnings": payload["gate"]["warnings"]}, indent=2))
    print(f"[INFO] Diagnostic report written to {args.report}")


if __name__ == "__main__":
    main()
