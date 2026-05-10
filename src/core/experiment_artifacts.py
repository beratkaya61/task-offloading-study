from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def summarize_eval(metrics: Dict) -> Dict[str, float]:
    success_rates = metrics.get("success_rates", [])
    delays = metrics.get("avg_delays", [])
    energies = metrics.get("avg_energies", [])
    return {
        "success_rate_mean": mean(success_rates),
        "success_rate_std": std(success_rates),
        "avg_delay_mean": mean(delays),
        "avg_delay_std": std(delays),
        "avg_energy_mean": mean(energies),
        "avg_energy_std": std(energies),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_last_train_success(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.0
    with open(metrics_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    last_row = lines[-1].split(",")
    return float(last_row[1]) if len(last_row) >= 2 else 0.0


def _read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_source_file"] = path.name
    return rows


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def refresh_real_data_phase5_report(
    output_path: str | Path = "v2_docs/phase_5/real_data_phase_5_report.md",
    metrics_root: str | Path = "results/phase_5/metrics/real_data",
) -> None:
    output_path = Path(output_path)
    metrics_root = Path(metrics_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retraining_rows: List[Dict] = []
    canonical_retraining_csv = metrics_root / "rl_retraining" / "real_data_rl_retraining.csv"
    if canonical_retraining_csv.exists():
        retraining_rows.extend(_read_csv_rows(canonical_retraining_csv))
    else:
        for csv_path in sorted((metrics_root / "rl_retraining").glob("real_data_rl_retraining_*.csv")):
            retraining_rows.extend(_read_csv_rows(csv_path))

    policy_rows = _read_csv_rows(metrics_root / "policy_evaluation" / "real_data_policy_evaluation.csv")

    ablation_rows: List[Dict] = []
    for csv_path in sorted((metrics_root / "ablation").glob("real_data_ablation_*.csv")):
        ablation_rows.extend(_read_csv_rows(csv_path))

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Faz 5 - Real Data Report",
        "",
        "Bu dosya, Faz 5'in `real_data` kolu icin tek kanonik rapordur.",
        "Sentetik Faz 5 sonuclari ayri olarak `v2_docs/phase_5/synthetic_phase_5_report.md` dosyasinda tutulur.",
        "",
        "## Artefakt Haritasi",
        "",
        "- `results/phase_5/metrics/real_data/rl_retraining/`",
        "- `results/phase_5/metrics/real_data/policy_evaluation/`",
        "- `results/phase_5/metrics/real_data/ablation/`",
        "- `results/phase_5/figures/real_data/ablation/`",
        "",
    ]

    if retraining_rows:
        lines.extend(["## RL Retraining", ""])
        if "config_model_type" in retraining_rows[0]:
            lines.extend(
                [
                    "| Algorithm | Seed | Success Rate | Avg Reward | P95 Latency | Avg Energy | QoE | Dominant Action |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in retraining_rows:
                lines.append(
                    f"| {row['config_model_type']} | {row['config_seed']} | {_fmt_pct(float(row['metric_success_rate']) * 100.0)} | "
                    f"{_fmt_float(float(row['metric_avg_reward']), 2)} | {_fmt_float(float(row['metric_p95_latency']))} | "
                    f"{_fmt_float(float(row['metric_avg_energy']))} | {_fmt_float(float(row['metric_qoe']), 2)} | {row['metric_dominant_action']} |"
                )
            lines.append("")
        else:
            lines.extend(
                [
                    "| Algorithm | Seed | Train Success Last | Val Success Mean | Test Success Mean | Test Avg Delay | Test Avg Energy |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in retraining_rows:
                lines.append(
                    f"| {row['algorithm'].upper()} | {row['seed']} | {_fmt_pct(float(row['train_success_last']))} | "
                    f"{_fmt_pct(float(row['val_success_rate_mean']))} | {_fmt_pct(float(row['test_success_rate_mean']))} | "
                    f"{_fmt_float(float(row['test_avg_delay_mean']))} | {_fmt_float(float(row['test_avg_energy_mean']), 6)} |"
                )
            lines.append("")
    else:
        lines.extend(["## RL Retraining", "", "Henuz retraining sonucu uretilmedi.", ""])

    if policy_rows:
        lines.extend(
            [
                "## Policy Evaluation",
                "",
                "| Policy | Seed | Success Rate | P95 Latency | Avg Energy | QoE | Dominant Action |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in policy_rows:
            lines.append(
                f"| {row['config_model_type']} | {row['config_seed']} | {_fmt_pct(float(row['metric_success_rate']) * 100.0)} | "
                f"{_fmt_float(float(row['metric_p95_latency']))} | {_fmt_float(float(row['metric_avg_energy']))} | "
                f"{_fmt_float(float(row['metric_qoe']), 2)} | {row['metric_dominant_action']} |"
            )
        lines.append("")
    else:
        lines.extend(["## Policy Evaluation", "", "Henuz policy evaluation sonucu uretilmedi.", ""])

    if ablation_rows:
        lines.extend(["## Ablation", ""])
        if "config_model_type" in ablation_rows[0]:
            grouped: Dict[str, List[Dict]] = {}
            for row in ablation_rows:
                grouped.setdefault(row["_source_file"], []).append(row)

            for source_file, rows in grouped.items():
                label = source_file.replace("real_data_ablation_", "").replace(".csv", "")
                lines.extend(
                    [
                        f"### {label}",
                        "",
                        "| Variant | Seed | Success Rate | Avg Reward | P95 Latency | Avg Energy | QoE | Dominant Action |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for row in rows:
                    lines.append(
                        f"| {row['config_model_type']} | {row['config_seed']} | {_fmt_pct(float(row['metric_success_rate']) * 100.0)} | "
                        f"{_fmt_float(float(row['metric_avg_reward']), 2)} | {_fmt_float(float(row['metric_p95_latency']))} | "
                        f"{_fmt_float(float(row['metric_avg_energy']))} | {_fmt_float(float(row['metric_qoe']), 2)} | {row['metric_dominant_action']} |"
                    )
                lines.append("")
        else:
            grouped: Dict[str, List[Dict]] = {}
            for row in ablation_rows:
                grouped.setdefault(row["algorithm"], []).append(row)

            for algorithm, rows in grouped.items():
                lines.extend(
                    [
                        f"### {algorithm.upper()}",
                        "",
                        "| Variant | Seed | Train Success Last | Val Success Mean | Test Success Mean | Test Avg Delay | Test Avg Energy |",
                        "|---|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for row in rows:
                    lines.append(
                        f"| {row['variant']} | {row['seed']} | {_fmt_pct(float(row['train_success_last']))} | "
                        f"{_fmt_pct(float(row['val_success_rate_mean']))} | {_fmt_pct(float(row['test_success_rate_mean']))} | "
                        f"{_fmt_float(float(row['test_avg_delay_mean']))} | {_fmt_float(float(row['test_avg_energy_mean']), 6)} |"
                    )
                lines.append("")
    else:
        lines.extend(["## Ablation", "", "Henuz ablation sonucu uretilmedi.", ""])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
