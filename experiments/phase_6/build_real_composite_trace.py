from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.real_composite_trace_builder import RealCompositeBuildConfig, RealCompositeTraceBuilder


def _default_output_dir(fusion_mode: str, decision_balance: str = "standard", service_class_mode: str = "none") -> Path:
    if fusion_mode == "random" and decision_balance == "standard" and service_class_mode == "none":
        return REPO_ROOT / "data" / "real_composite_trace"
    suffix_parts = [fusion_mode]
    if decision_balance != "standard":
        suffix_parts.append(decision_balance)
    if service_class_mode != "none":
        suffix_parts.append(service_class_mode)
    suffix = "_".join(suffix_parts)
    return REPO_ROOT / "data" / f"real_composite_trace_{suffix}"


def build_real_composite_trace(
    fusion_mode: str = "random",
    output_dir: str | Path | None = None,
    max_tasks: int = 5000,
    tasks_per_episode: int = 50,
    seed: int = 42,
    decision_balance: str = "standard",
    service_class_mode: str = "none",
) -> dict:
    real_composite_trace_dir = Path(output_dir) if output_dir else _default_output_dir(fusion_mode, decision_balance, service_class_mode)
    processed_dir = real_composite_trace_dir / "records"
    split_dir = real_composite_trace_dir / "splits"
    report_parts = [fusion_mode]
    if decision_balance != "standard":
        report_parts.append(decision_balance)
    if service_class_mode != "none":
        report_parts.append(service_class_mode)
    report_suffix = "_".join(report_parts)
    report_path = REPO_ROOT / "v2_docs" / "phase_6" / f"real_composite_build_report_{report_suffix}.md"
    if fusion_mode == "random" and decision_balance == "standard":
        report_path = REPO_ROOT / "v2_docs" / "phase_6" / "real_composite_build_report.md"

    config = RealCompositeBuildConfig(
        max_tasks=max_tasks,
        tasks_per_episode=tasks_per_episode,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=seed,
        fusion_mode=fusion_mode,
        decision_balance=decision_balance,
        service_class_mode=service_class_mode,
    )
    builder = RealCompositeTraceBuilder(data_root=REPO_ROOT / "data" / "raw_real_datasets", seed=config.seed)
    records_df = builder.build_task_records(config)
    splits = builder.build_episode_splits(records_df, config)
    normalization_meta = records_df.attrs.get("state_normalization", {})

    builder.save_records(records_df, processed_dir / "composite_task_records.csv")
    (processed_dir / "calibration_metadata.json").write_text(
        json.dumps(normalization_meta, indent=2),
        encoding="utf-8",
    )
    for split_name, episodes in splits.items():
        builder.save_episode_split(
            episodes,
            split_dir / f"{split_name}_episodes.json",
            metadata={
                "source_label": f"real_composite_trace_{fusion_mode}",
                "tasks_per_episode": config.tasks_per_episode,
                "seed": config.seed,
                "split": split_name,
                "field_policy": "mixed_direct_and_proxy_documented",
                "calibration_policy": "difficulty_rank_to_mec_scale_v2",
                "fusion_mode": fusion_mode,
        "decision_balance": decision_balance,
        "service_class_mode": service_class_mode,
                "decision_balance": decision_balance,
        "service_class_mode": service_class_mode,
                "service_class_mode": service_class_mode,
            },
        )

    deadline_window = records_df["deadline"] - records_df["arrival_time"]
    summary = {
        "fusion_mode": fusion_mode,
        "decision_balance": decision_balance,
        "service_class_mode": service_class_mode,
        "records": int(len(records_df)),
        "episodes_total": int(sum(len(episodes) for episodes in splits.values())),
        "episodes_train": int(len(splits["train"])),
        "episodes_val": int(len(splits["val"])),
        "episodes_test": int(len(splits["test"])),
        "device_count": int(records_df["device_id"].nunique()),
        "server_count": int(records_df["server_id"].nunique()),
        "priority_distribution": {
            str(key): int(value) for key, value in records_df["priority"].value_counts().sort_index().items()
        },
        "cpu_cycles_median": float(records_df["cpu_cycles"].median()),
        "deadline_window_median_s": float(deadline_window.median()),
        "reference_best_case_delay_median_s": float(records_df["reference_best_case_delay_s"].median()),
        "records_path": str(processed_dir / "composite_task_records.csv"),
        "split_dir": str(split_dir),
    }

    report_lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        f"# Real Composite Build Report ({fusion_mode})",
        "",
        "Bu rapor, lokal gercek veri kaynaklarindan olusturulan kompozit MEC task kayitlarini ve episode splitlerini ozetler.",
        "",
        "## Build Summary",
        "",
        f"- fusion mode: `{summary['fusion_mode']}`",
        f"- decision balance: `{summary['decision_balance']}`",
        f"- service class mode: `{summary['service_class_mode']}`",
        f"- total task record: `{summary['records']}`",
        f"- train episode: `{summary['episodes_train']}`",
        f"- val episode: `{summary['episodes_val']}`",
        f"- test episode: `{summary['episodes_test']}`",
        f"- device count: `{summary['device_count']}`",
        f"- server count: `{summary['server_count']}`",
        "",
        "## Source Composition",
        "",
        "- `arrival_time`, workload identity ve difficulty ranking: Alibaba `batch_task.csv`",
        "- `location`, `device_id`: Glasgow MEC dataset",
        "- `server context`: Alibaba `machine_usage.csv` + `machine_meta.csv`",
        "- `execution_time_s`: UCI MEC execution-time dataset",
        "- `Google Cluster Trace` ve `Didi Gaia`: secondary validation / cross-check havuzunda tutulur",
        "",
        "## Fusion Policy",
        "",
    ]
    if fusion_mode == "random":
        report_lines.append("- Bu build `independent random fusion baseline` olarak okunmalidir; kaynaklar arasi dogal korelasyonlari koruma iddiasi tasimaz.")
    else:
        report_lines.append("- Bu build `correlation-aware fusion` olarak okunur; Alibaba arrival/difficulty rank, Glasgow mobility rank, server-load rank ve UCI execution-time rank birlikte eslenir.")
    if decision_balance == "edge_balanced":
        report_lines.append("- `edge_balanced` karar dengesi, payload bandini genisletip deadline slack'ini daraltarak cloud-dominant oracle davranisini azaltmayi hedefler.")
    if service_class_mode == "mec_mixed":
        report_lines.append("- `mec_mixed` service-class kalibrasyonu local-sensing, balanced-partial, urgent-edge ve cloud-batch task aileleri uretir.")

    report_lines.extend(
        [
            "",
            "## Calibrated Proxy Fields",
            "",
            "- `cpu_cycles`: Alibaba difficulty ranking + UCI MEC execution-time olcegi ile MEC kapasitesine kalibre edilmis proxy",
            "- `data_size`: Alibaba `plan_mem` ranking'i ile MEC payload bandina map edilmis proxy",
            "- `deadline`: task-specific best-case lower bound uzerinden kurulan compute-proportional proxy",
            "- `priority`: deadline tightness / urgency proxy",
            "",
            "## Calibration Summary",
            "",
            f"- median cpu_cycles: `{summary['cpu_cycles_median']:.0f}`",
            f"- median deadline window: `{summary['deadline_window_median_s']:.3f} s`",
            f"- median reference best-case delay: `{summary['reference_best_case_delay_median_s']:.3f} s`",
            f"- state normalization metadata: `{(processed_dir / 'calibration_metadata.json').as_posix()}`",
            "",
            "## Priority Distribution",
            "",
        ]
    )
    for priority, count in summary["priority_distribution"].items():
        report_lines.append(f"- priority `{priority}`: `{count}` task")

    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- records csv: `{(processed_dir / 'composite_task_records.csv').as_posix()}`",
            f"- calibration metadata: `{(processed_dir / 'calibration_metadata.json').as_posix()}`",
            f"- train split: `{(split_dir / 'train_episodes.json').as_posix()}`",
            f"- val split: `{(split_dir / 'val_episodes.json').as_posix()}`",
            f"- test split: `{(split_dir / 'test_episodes.json').as_posix()}`",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Composite report written to {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real composite trace splits")
    parser.add_argument("--fusion-mode", choices=["random", "correlation_aware"], default="random")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tasks", type=int, default=5000)
    parser.add_argument("--tasks-per-episode", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-balance", choices=["standard", "edge_balanced"], default="standard")
    parser.add_argument("--service-class-mode", choices=["none", "mec_mixed"], default="none")
    args = parser.parse_args()
    build_real_composite_trace(
        fusion_mode=args.fusion_mode,
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
        tasks_per_episode=args.tasks_per_episode,
        seed=args.seed,
        decision_balance=args.decision_balance,
        service_class_mode=args.service_class_mode,
    )


if __name__ == "__main__":
    main()
