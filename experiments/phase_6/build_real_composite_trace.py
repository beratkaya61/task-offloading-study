from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.real_composite_trace_builder import RealCompositeBuildConfig, RealCompositeTraceBuilder


REAL_COMPOSITE_TRACE_DIR = REPO_ROOT / "data" / "real_composite_trace"
PROCESSED_DIR = REAL_COMPOSITE_TRACE_DIR / "records"
SPLIT_DIR = REAL_COMPOSITE_TRACE_DIR / "splits"
REPORT_PATH = REPO_ROOT / "v2_docs" / "phase_6" / "real_composite_build_report.md"


def main() -> None:
    config = RealCompositeBuildConfig(
        max_tasks=5000,
        tasks_per_episode=50,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )
    builder = RealCompositeTraceBuilder(data_root=REPO_ROOT / "data" / "raw_real_datasets", seed=config.seed)
    records_df = builder.build_task_records(config)
    splits = builder.build_episode_splits(records_df, config)
    normalization_meta = records_df.attrs.get("state_normalization", {})

    builder.save_records(records_df, PROCESSED_DIR / "composite_task_records.csv")
    (PROCESSED_DIR / "calibration_metadata.json").write_text(
        json.dumps(normalization_meta, indent=2),
        encoding="utf-8",
    )
    for split_name, episodes in splits.items():
        builder.save_episode_split(
            episodes,
            SPLIT_DIR / f"{split_name}_episodes.json",
            metadata={
                "source_label": "real_composite_trace",
                "tasks_per_episode": config.tasks_per_episode,
                "seed": config.seed,
                "split": split_name,
                "field_policy": "mixed_direct_and_proxy_documented",
                "calibration_policy": "difficulty_rank_to_mec_scale_v2",
            },
        )

    summary = {
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
        "deadline_window_median_s": float((records_df["deadline"] - records_df["arrival_time"]).median()),
        "reference_best_case_delay_median_s": float(records_df["reference_best_case_delay_s"].median()),
    }

    report_lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Real Composite Build Report",
        "",
        "Bu rapor, Faz 6R.4 kapsaminda lokal gercek veri kaynaklarindan olusturulan guncel kompozit MEC task kayitlarini ve episode splitlerini ozetler.",
        "",
        "## Build Summary",
        "",
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
        "- `Google Cluster Trace` ve `Didi Gaia`: bu buildde cekirdek split icin zorunlu degil; secondary validation / cross-check havuzunda tutuluyor",
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
        f"- state normalization metadata: `{(PROCESSED_DIR / 'calibration_metadata.json').as_posix()}`",
        "",
        "## Priority Distribution",
        "",
    ]
    for priority, count in summary["priority_distribution"].items():
        report_lines.append(f"- priority `{priority}`: `{count}` task")

    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- records csv: `{(PROCESSED_DIR / 'composite_task_records.csv').as_posix()}`",
            f"- calibration metadata: `{(PROCESSED_DIR / 'calibration_metadata.json').as_posix()}`",
            f"- train split: `{(SPLIT_DIR / 'train_episodes.json').as_posix()}`",
            f"- val split: `{(SPLIT_DIR / 'val_episodes.json').as_posix()}`",
            f"- test split: `{(SPLIT_DIR / 'test_episodes.json').as_posix()}`",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Composite report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
