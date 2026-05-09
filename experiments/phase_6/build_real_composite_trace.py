from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
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

    builder.save_records(records_df, PROCESSED_DIR / "composite_task_records.csv")
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
    }

    report_lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Real Composite Build Report",
        "",
        "Bu rapor, Faz 6R.4 kapsaminda lokal gercek veri kaynaklarindan olusturulan ilk kompozit MEC task kayitlarini ve episode splitlerini ozetler.",
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
        "- `arrival_time`, workload identity ve raw duration: Alibaba `batch_task.csv`",
        "- `location`, `device_id`, `server context`: Glasgow MEC dataset",
        "- `execution_time_s`: UCI MEC execution-time dataset",
        "- `Google Cluster Trace` ve `Didi Gaia`: bu ilk buildde cekirdek split icin zorunlu degil; secondary validation / cross-check havuzunda tutuluyor",
        "",
        "## Proxy Fields",
        "",
        "- `deadline`: UCI execution time tabanli design proxy",
        "- `data_size`: Alibaba `plan_mem` tabanli proxy",
        "- `cpu_cycles`: Alibaba `plan_cpu` tabanli proxy",
        "- `priority`: execution-time quantile tabanli proxy",
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
