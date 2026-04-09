from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Checkpoint compatibility: some saved SB3 objects may reference numpy._core.*
try:
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.trace.train_ppo import TraceTrainingOrchestrator

DEFAULT_CONFIG = REPO_ROOT / "configs" / "trace" / "holdout_evaluation.yaml"
CSV_COLUMNS = [
    "timestamp",
    "batch_id",
    "split",
    "checkpoint_path",
    "num_episodes",
    "metric_success_rate_mean",
    "metric_success_rate_std",
    "metric_success_rate_min",
    "metric_success_rate_max",
    "metric_avg_delay_mean",
    "metric_avg_energy_mean",
    "status",
]


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def summarize(eval_metrics: dict) -> dict:
    success_rates = np.array(eval_metrics.get("success_rates", []), dtype=float)
    avg_delays = np.array(eval_metrics.get("avg_delays", []), dtype=float)
    avg_energies = np.array(eval_metrics.get("avg_energies", []), dtype=float)

    return {
        "metric_success_rate_mean": round(float(success_rates.mean()) if success_rates.size else 0.0, 2),
        "metric_success_rate_std": round(float(success_rates.std()) if success_rates.size else 0.0, 2),
        "metric_success_rate_min": round(float(success_rates.min()) if success_rates.size else 0.0, 2),
        "metric_success_rate_max": round(float(success_rates.max()) if success_rates.size else 0.0, 2),
        "metric_avg_delay_mean": round(float(avg_delays.mean()) if avg_delays.size else 0.0, 4),
        "metric_avg_energy_mean": round(float(avg_energies.mean()) if avg_energies.size else 0.0, 4),
    }


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def find_row(rows: list[dict], split: str) -> dict | None:
    for row in rows:
        if row.get("split") == split:
            return row
    return None


def write_report(report_path: Path, rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    train_row = find_row(rows, "train")
    val_row = find_row(rows, "val")
    test_row = find_row(rows, "test")

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Trace Hold-Out Test Report",
        "",
        "Bu rapor, Faz 6 kapanisindan once ayni trace checkpoint'in `train`, `val` ve `test` splitlerinde nasil davrandigini birlikte ozetler.",
        "Amac, validation sonucunun test splitini ne kadar temsil ettigini ve belirgin bir overfitting izi olup olmadigini gormektir.",
        "",
        "## Split Karsilastirmasi",
        "",
        "| Split | Episode Count | Mean Success | Std | Min | Max | Mean Delay | Mean Energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['split']} | {row['num_episodes']} | {row['metric_success_rate_mean']:.2f}% | {row['metric_success_rate_std']:.2f} | {row['metric_success_rate_min']:.2f}% | {row['metric_success_rate_max']:.2f}% | {row['metric_avg_delay_mean']:.4f} s | {row['metric_avg_energy_mean']:.4f} |"
        )

    if train_row and val_row and test_row:
        train_val_gap = round(train_row['metric_success_rate_mean'] - val_row['metric_success_rate_mean'], 2)
        val_test_gap = round(val_row['metric_success_rate_mean'] - test_row['metric_success_rate_mean'], 2)
        train_test_gap = round(train_row['metric_success_rate_mean'] - test_row['metric_success_rate_mean'], 2)
        lines.extend([
            "",
            "## Gap Analizi",
            "",
            f"- Train - Val success gap: `{train_val_gap:+.2f}` puan",
            f"- Val - Test success gap: `{val_test_gap:+.2f}` puan",
            f"- Train - Test success gap: `{train_test_gap:+.2f}` puan",
            "",
            "## Yorum",
            "",
        ])
        if abs(val_test_gap) <= 1.0:
            lines.append("Validation ve test sonuclari birbirine cok yakin. Bu, Faz 6 kapanisinda belirgin bir split-mismatch veya validation yanilgisi olmadigini destekler.")
        else:
            lines.append("Validation ve test arasinda anlamli bir fark var. Bu durum, Faz 6 kapanisinda ek dikkat gerektirir.")

        if abs(train_test_gap) <= 1.0:
            lines.append("Train ve test arasindaki fark da sinirli; belirgin bir overfitting izi gorulmuyor.")
        else:
            lines.append("Train ve test arasindaki fark belirgin; modelin trace splitleri arasinda fazla uyum gostermis olma ihtimali not edilmelidir.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    training_config = REPO_ROOT / config.get("training_config", "configs/trace/ppo_training.yaml")
    checkpoint_path = REPO_ROOT / config.get("checkpoint_path", "models/ppo/trace_training/ppo_v3_trace_best.zip")
    csv_path = REPO_ROOT / config.get("output", {}).get("csv_path", "results/raw/trace/holdout/trace_holdout_evaluation.csv")
    report_path = REPO_ROOT / config.get("output", {}).get("report_path", "v2_docs/phase_6/trace_holdout_test_report.md")
    seed = int(config.get("evaluation", {}).get("seed", 42))
    splits = config.get("evaluation", {}).get("splits", ["test"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    orchestrator = TraceTrainingOrchestrator(config_path=str(training_config), seed=seed)
    train_eps, val_eps, test_eps = orchestrator.prepare_traces()
    split_map = {
        "train": train_eps,
        "val": val_eps,
        "test": test_eps,
    }

    rows: list[dict] = []
    batch_id = datetime.now().strftime("trace_holdout_%Y%m%d_%H%M%S")
    for split in splits:
        episodes = split_map[split]
        eval_metrics = orchestrator.evaluate_model(episodes, str(checkpoint_path))
        summary = summarize(eval_metrics)
        rows.append({
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "split": split,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "num_episodes": len(episodes),
            "status": "completed",
            **summary,
        })

    write_csv(csv_path, rows)
    write_report(report_path, rows)
    print(f"[INFO] Hold-out CSV: {csv_path}")
    print(f"[INFO] Hold-out report: {report_path}")
    for row in rows:
        print(f"[INFO] {row['split']} success: {row['metric_success_rate_mean']:.2f}%")


if __name__ == "__main__":
    main()

