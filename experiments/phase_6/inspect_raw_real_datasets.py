from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.dataset_loader import RealDataLoader
from src.core.real_data_manifest import (
    assert_real_data_ready,
    dataset_inventory,
    load_real_data_manifest,
)


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "phase_6" / "raw_real_data_manifest.yaml"
DEFAULT_REPORT = REPO_ROOT / "v2_docs" / "phase_6" / "real_data_inventory_report.md"


def _build_inventory_report(data_root: Path, output_path: Path) -> None:
    loader = RealDataLoader(data_root=data_root)
    summaries = loader.summarize_light()

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Real Data Inventory Report",
        "",
        "Bu rapor, Faz 6R icin lokal olarak mevcut gercek veri kaynaklarinin dosya, satir sayisi ve kolon yapisini ozetler.",
        "",
    ]

    for summary in summaries:
        lines.extend([f"## {summary.name}", "", "**Files**", ""])
        for file_name in summary.files:
            lines.append(f"- `{file_name}`")
        lines.extend(["", "**Tables**", ""])
        for table_name, row_count in summary.row_counts.items():
            row_text = "row_count_skipped_large_file" if row_count < 0 else f"{row_count} rows"
            lines.append(f"- `{table_name}`: {row_text}")
            lines.append(f"  columns: `{', '.join(summary.columns[table_name])}`")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([summary.__dict__ for summary in summaries], indent=2))
    print(f"[INFO] Inventory report written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect real-data manifest and local dataset inventory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    manifest = load_real_data_manifest(args.manifest)
    rows = dataset_inventory(manifest, repo_root=REPO_ROOT)

    print("Real-data inventory")
    print("-------------------")
    for row in rows:
        required = "required" if row["required"] else "optional"
        print(
            f"{row['dataset']}: status={row['status']}, role={required}, "
            f"path_exists={row['path_exists']}, matched_files={row['matched_files']}, "
            f"local_root={row['local_root']}"
        )

    print("")
    print("Readiness check")
    print("---------------")
    try:
        assert_real_data_ready(manifest_path=args.manifest, repo_root=REPO_ROOT)
    except Exception as exc:
        print(f"NOT READY: {exc}")
        if not args.skip_report:
            _build_inventory_report(REPO_ROOT / "data" / "raw_real_datasets", args.report)
        return 1

    print("READY: required real-data sources are present and match manifest expectations.")

    if not args.skip_report:
        _build_inventory_report(REPO_ROOT / "data" / "raw_real_datasets", args.report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
