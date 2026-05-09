from pathlib import Path
import argparse
import csv
import math
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pretrain_graph_policy import run_graph_supervised_pretraining


def _mean(values):
    return sum(values) / max(1, len(values))


def _sample_std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _ci95(values):
    if len(values) <= 1:
        return 0.0
    return 1.96 * _sample_std(values) / math.sqrt(len(values))


def _aggregate_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["fusion"], []).append(row)

    summary_rows = []
    for fusion, fusion_rows in grouped.items():
        test_acc = [float(row["test_accuracy"]) for row in fusion_rows]
        diversity = [float(row["test_prediction_diversity"]) for row in fusion_rows]
        best_val = [float(row["best_val_accuracy"]) for row in fusion_rows]
        summary_rows.append(
            {
                "fusion": fusion,
                "num_seeds": len(fusion_rows),
                "seeds": ",".join(str(row["seed"]) for row in fusion_rows),
                "best_val_accuracy_mean": _mean(best_val),
                "best_val_accuracy_std": _sample_std(best_val),
                "test_accuracy_mean": _mean(test_acc),
                "test_accuracy_std": _sample_std(test_acc),
                "test_accuracy_ci95": _ci95(test_acc),
                "test_prediction_diversity_mean": _mean(diversity),
                "test_prediction_diversity_std": _sample_std(diversity),
                "test_prediction_diversity_ci95": _ci95(diversity),
            }
        )
    return sorted(summary_rows, key=lambda row: row["fusion"])


def _write_comparison_report(report_path: Path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = _aggregate_rows(rows)
    full_protocol = bool(summary_rows) and all(row["num_seeds"] >= 5 for row in summary_rows) and all(
        int(row["executed_epochs"]) >= 12 for row in rows
    )
    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Phase 8 Semantic Prior Fusion Comparison",
        "",
        "Bu rapor Faz 8.3 kapsaminda graph-aware policy icin semantic prior fusion varyantlarini ayni teacher ve ayni synthetic graph warm-start protokolu altinda karsilastirir.",
        "",
    ]

    if not full_protocol:
        lines.extend(
            [
                "## Durum",
                "",
                "Henuz final bilimsel fusion karsilastirmasi calistirilmadi.",
                "Bu ana rapor, tek seed veya kisa epoch smoke kosularini final sonuc gibi gostermeyecek.",
                "Smoke/diagnostic kosular yalnizca hat dogrulama icindir; Faz 8 kapanis yorumuna alinmayacaktir.",
                "",
                "## Final Protokol",
                "",
                "Faz 8 fusion sonucu icin calistirilmesi gereken minimum komut:",
                "",
                "```powershell",
                "python experiments\\synthetic\\run_graph_fusion_comparison.py --config configs\\synthetic\\graph_supervised_pretraining.yaml --fusions none late --seeds 42 43 44 45 46",
                "```",
                "",
                "Bu protokol su kosullari saglamalidir:",
                "",
                "- en az 5 seed",
                "- full config",
                "- minimum 12 epoch before early stopping",
                "- mean/std/95% CI raporu",
                "- tek konsolide rapor dosyasi",
                "",
                "Full protokol tamamlandiginda bu dosya gercek seed-aggregated tablo ile guncellenecektir.",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
        "## Seed Aggregated Summary",
        "",
        "| Fusion | Seeds | Best Val Acc Mean | Test Acc Mean | Test Acc 95% CI | Diversity Mean | Diversity 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {fusion} | {num_seeds} | {best_val_accuracy_mean:.2%} | {test_accuracy_mean:.2%} | +/- {test_accuracy_ci95:.2%} | {test_prediction_diversity_mean:.4f} | +/- {test_prediction_diversity_ci95:.4f} |".format(
                **row
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Per-Run Details",
            "",
            "| Fusion | Seed | Samples | Epochs | Best Val Acc | Test Acc | Test Prediction Diversity | Checkpoint |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {fusion} | {seed} | {num_samples} | {executed_epochs} | {best_val_accuracy:.2%} | {test_accuracy:.2%} | {test_prediction_diversity:.4f} | `{checkpoint_path}` |".format(
                **row
            )
        )
    lines.append("")
    if summary_rows:
        best = max(summary_rows, key=lambda item: (item["test_accuracy_mean"], item["test_prediction_diversity_mean"]))
        lines.append(f"Bu protokolde ortalama test accuracy + diversity sirasina gore en iyi varyant: `{best['fusion']}`.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Phase 8 graph semantic prior fusion modes")
    parser.add_argument(
        "--config",
        default="configs/phase_8/graph_supervised_pretraining.yaml",
        help="Graph supervised pretraining config path",
    )
    parser.add_argument(
        "--fusions",
        nargs="+",
        default=["none", "late"],
        choices=["none", "input", "late", "input_late"],
        help="Fusion modes to compare",
    )
    parser.add_argument(
        "--csv",
        default="results/phase_8/metrics/synthetic_debug/graph_fusion_comparison.csv",
        help="Optional comparison CSV output path, used only with --write_csv",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional comparison report output path. Leave empty to avoid extra docs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seed list for multi-seed comparison",
    )
    parser.add_argument("--write_csv", action="store_true", help="Write optional comparison metrics CSV files")
    args = parser.parse_args()

    rows = []
    for fusion in args.fusions:
        for seed in args.seeds:
            print(f"[INFO] Running graph supervised warm-start with fusion={fusion} seed={seed}")
            result = run_graph_supervised_pretraining(
                args.config,
                semantic_prior_fusion=fusion,
                seed_override=seed,
                output_suffix=f"seed{seed}",
                write_report=False,
                write_metrics=False,
            )
            rows.append(
                {
                    "fusion": fusion,
                    "seed": int(result["seed"]),
                    "teacher_policy": result["teacher_policy"],
                    "num_samples": int(result["num_samples"]),
                    "executed_epochs": int(result["executed_epochs"]),
                    "best_epoch": int(result["best_epoch"]),
                    "best_val_accuracy": float(result["best_val_accuracy"]),
                    "test_accuracy": float(result["final_test"]["accuracy"]),
                    "test_prediction_diversity": float(result["final_test"]["prediction_diversity"]),
                    "metrics_csv": result["metrics_csv"],
                    "report_path": args.report,
                    "checkpoint_path": result["checkpoint_path"],
                }
            )

    csv_path = Path(args.csv)
    summary_rows = _aggregate_rows(rows)
    if args.write_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_path = csv_path.with_name(f"{csv_path.stem}_summary{csv_path.suffix}")
        with open(summary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if args.report:
        _write_comparison_report(Path(args.report), rows)
    if args.write_csv:
        print(f"[INFO] Comparison CSV: {csv_path}")
        print(f"[INFO] Summary CSV: {summary_path}")
    if args.report:
        print(f"[INFO] Comparison report: {args.report}")
    else:
        print("[INFO] Comparison report not written; summarize results in phase_8_explaination_of_studies.md")

