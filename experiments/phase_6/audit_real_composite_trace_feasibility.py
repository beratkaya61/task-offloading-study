from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.dataset_loader import RealDataLoader
from src.env.simulation_env import WirelessChannel


RECORDS_PATH = REPO_ROOT / "data" / "real_composite_trace" / "records" / "composite_task_records.csv"
REPORT_PATH = REPO_ROOT / "v2_docs" / "phase_6" / "real_composite_feasibility_audit.md"

LOCAL_CPU_HZ = 1e9
EDGE_CPU_HZ = 2e9
CLOUD_CPU_HZ = 5e9
CLOUD_FIXED_LATENCY_S = 0.1
REFERENCE_DATARATE_BPS = 25e6
STATE_CPU_NORM_CAP = 1e10
STATE_SIZE_BITS_CAP = 1e7

CPU_DIVISOR_SCENARIOS = [1, 2, 5, 10, 20]
DEADLINE_MULTIPLIER_SCENARIOS = [0.8, 1.0, 1.2, 1.5, 2.0]


def _series_summary(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(series.max()),
    }


def _compute_reference_best_case_delay(cpu_cycles: pd.Series, size_bits: pd.Series) -> pd.Series:
    tx_delay = size_bits / REFERENCE_DATARATE_BPS
    local_delay = cpu_cycles / LOCAL_CPU_HZ
    edge_delay = tx_delay + (cpu_cycles / EDGE_CPU_HZ)
    cloud_delay = tx_delay + CLOUD_FIXED_LATENCY_S + (cpu_cycles / CLOUD_CPU_HZ)
    partial_delay = np.maximum(
        0.25 * cpu_cycles / LOCAL_CPU_HZ,
        (0.75 * size_bits) / REFERENCE_DATARATE_BPS + (0.75 * cpu_cycles / EDGE_CPU_HZ),
    )
    return np.minimum.reduce([local_delay, edge_delay, cloud_delay, partial_delay])


def _derive_edge_locations(records: pd.DataFrame, num_edge_servers: int = 3) -> list[tuple[float, float]]:
    coords = records[["location_x", "location_y"]].dropna().to_numpy(dtype=float)
    if len(coords) == 0:
        return [
            (200.0, 200.0),
            (800.0, 200.0),
            (500.0, 800.0),
        ][:num_edge_servers]

    order = np.argsort(coords[:, 0])
    sorted_coords = coords[order]
    chunks = np.array_split(sorted_coords, num_edge_servers)
    locations = []
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            locations.append((150.0 + index * 300.0, 500.0))
        else:
            centroid = chunk.mean(axis=0)
            locations.append((float(centroid[0]), float(centroid[1])))
    return locations


def _compute_env_best_case_delay(records: pd.DataFrame) -> pd.Series:
    channel = WirelessChannel()
    edge_locations = _derive_edge_locations(records, num_edge_servers=3)
    best_delays: list[float] = []

    random.seed(42)
    np.random.seed(42)
    for row in records.itertuples(index=False):
        device = SimpleNamespace(location=[float(row.location_x), float(row.location_y)])
        datarates = []
        for edge_location in edge_locations:
            edge = SimpleNamespace(location=edge_location)
            datarate, _distance = channel.calculate_datarate(device, edge)
            datarates.append(max(float(datarate), 1e-6))
        best_datarate = max(datarates)

        size_bits = float(row.size_bits)
        cpu_cycles = float(row.cpu_cycles)

        local_delay = cpu_cycles / LOCAL_CPU_HZ
        edge_delay = size_bits / best_datarate + (cpu_cycles / EDGE_CPU_HZ)
        cloud_delay = size_bits / best_datarate + CLOUD_FIXED_LATENCY_S + (cpu_cycles / CLOUD_CPU_HZ)
        partial_delay = max(
            0.25 * cpu_cycles / LOCAL_CPU_HZ,
            (0.75 * size_bits) / best_datarate + (0.75 * cpu_cycles / EDGE_CPU_HZ),
        )
        best_delays.append(min(local_delay, edge_delay, cloud_delay, partial_delay))

    return pd.Series(best_delays, index=records.index, dtype=float)


def _format_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def load_raw_source_context() -> dict[str, object]:
    loader = RealDataLoader(data_root=REPO_ROOT / "data" / "raw_real_datasets")
    glasgow = loader.load_glasgow_mec()
    uci = loader.load_uci_execution_times()

    alibaba = pd.read_csv(
        REPO_ROOT / "data" / "raw_real_datasets" / "alibaba_cluster_trace_2018" / "batch_task.csv",
        header=None,
        names=[
            "task_name",
            "instance_num",
            "job_name",
            "task_type",
            "status",
            "start_time",
            "end_time",
            "plan_cpu",
            "plan_mem",
        ],
        nrows=200_000,
    )
    valid_alibaba = alibaba[
        (alibaba["status"] == "Terminated")
        & (alibaba["end_time"] > alibaba["start_time"])
        & (alibaba["plan_cpu"] > 0)
        & (alibaba["plan_mem"] > 0)
    ].copy()
    valid_alibaba["duration_s"] = valid_alibaba["end_time"] - valid_alibaba["start_time"]

    uci_stats = {}
    for name, frame in uci.items():
        execution_times = frame["Execution Time"].astype(float)
        uci_stats[name] = _series_summary(execution_times)

    return {
        "glasgow_consecutive_rows": int(len(glasgow["consecutive"])),
        "glasgow_random_rows": int(len(glasgow["random_based"])),
        "glasgow_server_ids": sorted(
            {int(server_id) for server_id in glasgow["consecutive"]["serverId"].dropna().unique().tolist()}
        ),
        "glasgow_machine_names": int(glasgow["consecutive"]["machine_name"].nunique()),
        "glasgow_cpu_util_median": float(glasgow["consecutive"]["cpu_utilization"].median()),
        "glasgow_mem_util_median": float(glasgow["consecutive"]["mem_utilization"].median()),
        "alibaba_plan_cpu_counts": {
            str(key): int(value) for key, value in valid_alibaba["plan_cpu"].value_counts().head(10).items()
        },
        "alibaba_duration_summary": _series_summary(valid_alibaba["duration_s"]),
        "uci_execution_summaries": uci_stats,
    }


def build_audit() -> dict[str, object]:
    records = pd.read_csv(RECORDS_PATH)
    records["deadline_window_s"] = records["deadline"] - records["arrival_time"]
    records["size_bits"] = records["data_size"] * 8 * 1024
    records["reference_best_case_delay_s"] = _compute_reference_best_case_delay(
        records["cpu_cycles"],
        records["size_bits"],
    )
    records["env_best_case_delay_s"] = _compute_env_best_case_delay(records)
    records["reference_lower_bound_feasible"] = (
        records["reference_best_case_delay_s"] <= records["deadline_window_s"]
    )
    records["env_lower_bound_feasible"] = records["env_best_case_delay_s"] <= records["deadline_window_s"]

    audit = {
        "record_count": int(len(records)),
        "deadline_window_summary": _series_summary(records["deadline_window_s"]),
        "cpu_cycles_summary": _series_summary(records["cpu_cycles"]),
        "size_bits_summary": _series_summary(records["size_bits"]),
        "reference_best_case_delay_summary": _series_summary(records["reference_best_case_delay_s"]),
        "env_best_case_delay_summary": _series_summary(records["env_best_case_delay_s"]),
        "reference_lower_bound_feasible_rate": float(records["reference_lower_bound_feasible"].mean()),
        "env_lower_bound_feasible_rate": float(records["env_lower_bound_feasible"].mean()),
        "deadline_shorter_than_cloud_lower_rate": float(
            (
                records["deadline_window_s"]
                < (
                    records["size_bits"] / REFERENCE_DATARATE_BPS
                    + CLOUD_FIXED_LATENCY_S
                    + records["cpu_cycles"] / CLOUD_CPU_HZ
                )
            ).mean()
        ),
        "cpu_norm_saturation_rate": float((records["cpu_cycles"] >= STATE_CPU_NORM_CAP).mean()),
        "size_norm_saturation_rate": float((records["size_bits"] >= STATE_SIZE_BITS_CAP).mean()),
        "priority_feasibility": (
            pd.crosstab(records["priority"], records["env_lower_bound_feasible"], normalize="index")
            .rename(columns={False: "impossible_share", True: "feasible_share"})
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "impossible_examples": records.loc[
            ~records["env_lower_bound_feasible"],
            [
                "task_id",
                "cpu_cycles",
                "size_bits",
                "deadline_window_s",
                "reference_best_case_delay_s",
                "env_best_case_delay_s",
                "execution_time_s",
                "data_size",
                "priority",
            ],
        ]
        .head(10)
        .to_dict(orient="records"),
        "cpu_divisor_scenarios": [],
        "deadline_multiplier_scenarios": [],
        "raw_source_context": load_raw_source_context(),
    }

    for divisor in CPU_DIVISOR_SCENARIOS:
        scaled_cpu = records["cpu_cycles"] / divisor
        best_case_delay = _compute_reference_best_case_delay(scaled_cpu, records["size_bits"])
        audit["cpu_divisor_scenarios"].append(
            {
                "cpu_divisor": divisor,
                "feasible_rate": float((best_case_delay <= records["deadline_window_s"]).mean()),
                "cpu_norm_saturation_rate": float((scaled_cpu >= STATE_CPU_NORM_CAP).mean()),
                "cloud_delay_median_s": float(
                    (
                        records["size_bits"] / REFERENCE_DATARATE_BPS
                        + CLOUD_FIXED_LATENCY_S
                        + scaled_cpu / CLOUD_CPU_HZ
                    ).median()
                ),
            }
        )

    for multiplier in DEADLINE_MULTIPLIER_SCENARIOS:
        scaled_deadline = records["deadline_window_s"] * multiplier
        audit["deadline_multiplier_scenarios"].append(
            {
                "deadline_multiplier": multiplier,
                "feasible_rate": float((records["env_best_case_delay_s"] <= scaled_deadline).mean()),
                "deadline_median_s": float(scaled_deadline.median()),
            }
        )

    return audit


def write_report(audit: dict[str, object]) -> None:
    raw = audit["raw_source_context"]
    feasible_rate = audit["env_lower_bound_feasible_rate"]
    reference_feasible_rate = audit["reference_lower_bound_feasible_rate"]
    cpu_sat = audit["cpu_norm_saturation_rate"]
    size_sat = audit["size_norm_saturation_rate"]

    if feasible_rate < 0.4:
        status_line = "Env-faithful audit, benchmark'in henuz asiri sert veya env ile hizasiz oldugunu gosteriyor."
    elif feasible_rate < 0.75:
        status_line = "Env-faithful audit, benchmark'in artik cozulur ama halen sert bir rejimde oldugunu gosteriyor."
    elif feasible_rate < 0.95:
        status_line = "Env-faithful audit, benchmark'in anlamli bir MEC rejimine girdigini ve deadline baskisini korudugunu gosteriyor."
    else:
        status_line = "Env-faithful audit, benchmark'in fazla yumusaklasmis olabilecegini dusunduruyor."

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Real Composite Feasibility Audit",
        "",
        "Bu rapor, guncel `real_composite_trace` benchmark'inin MEC task offloading problemi icin fiziksel olarak tutarli olup olmadigini denetler.",
        "Bu surum artik iki ayri alt-sinir okur:",
        "- builder referans datarate varsayimi",
        "- env-faithful wireless/topology varsayimi",
        "",
        "## Ana Hukum",
        "",
        f"- builder-referans alt sinir feasibility: `{_format_pct(reference_feasible_rate)}`",
        f"- env-faithful alt sinir feasibility: `{_format_pct(feasible_rate)}`",
        f"- deadline penceresi cloud alt sinirindan daha kisa olan task orani: `{_format_pct(audit['deadline_shorter_than_cloud_lower_rate'])}`",
        f"- state tarafinda `cpu_norm` saturasyon orani: `{_format_pct(cpu_sat)}`",
        f"- state tarafinda `size_norm` saturasyon orani: `{_format_pct(size_sat)}`",
        "",
        "Ilk okuma:",
        status_line,
        "",
        "## Kompozit Kayit Ozetleri",
        "",
        f"- task sayisi: `{audit['record_count']}`",
        f"- deadline pencere medyani: `{_format_float(audit['deadline_window_summary']['p50'], 3)} s`",
        f"- cpu_cycles medyani: `{audit['cpu_cycles_summary']['p50']:.0f}`",
        f"- size_bits medyani: `{audit['size_bits_summary']['p50']:.0f}`",
        f"- reference best-case delay medyani: `{_format_float(audit['reference_best_case_delay_summary']['p50'], 3)} s`",
        f"- env-faithful best-case delay medyani: `{_format_float(audit['env_best_case_delay_summary']['p50'], 3)} s`",
        "",
        "## Ham Kaynaklarin Karakteri",
        "",
        f"- Glasgow consecutive satir sayisi: `{raw['glasgow_consecutive_rows']}`",
        f"- Glasgow random satir sayisi: `{raw['glasgow_random_rows']}`",
        f"- Glasgow consecutive icindeki serverId kumesi: `{raw['glasgow_server_ids']}`",
        f"- Glasgow machine_name sayisi: `{raw['glasgow_machine_names']}`",
        f"- Glasgow median cpu_utilization: `{_format_float(raw['glasgow_cpu_util_median'], 2)}`",
        f"- Glasgow median mem_utilization: `{_format_float(raw['glasgow_mem_util_median'], 2)}`",
        "",
        "Alibaba ilk 200k satir icinden gecerli alt-kume gozlemi:",
    ]
    for key, value in raw["alibaba_plan_cpu_counts"].items():
        lines.append(f"- plan_cpu `{key}` gorulme sayisi: `{value}`")
    lines.extend(
        [
            f"- Alibaba duration medyani: `{_format_float(raw['alibaba_duration_summary']['p50'], 2)} s`",
            f"- Alibaba duration p95: `{_format_float(raw['alibaba_duration_summary']['p95'], 2)} s`",
            "",
            "UCI MEC execution-time kaynaklari:",
        ]
    )
    for name, summary in raw["uci_execution_summaries"].items():
        lines.append(
            f"- `{name}` median: `{_format_float(summary['p50'], 4)} s`, p95: `{_format_float(summary['p95'], 4)} s`, max: `{_format_float(summary['max'], 4)} s`"
        )

    lines.extend(
        [
            "",
            "## Priority ve Env-Feasibility",
            "",
            "| Priority | Impossible Share | Feasible Share |",
            "|---|---:|---:|",
        ]
    )
    for row in audit["priority_feasibility"]:
        lines.append(
            f"| {row['priority']} | {_format_pct(row['impossible_share'])} | {_format_pct(row['feasible_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Kalibrasyon Senaryolari",
            "",
            "### CPU Divisor Senaryolari",
            "",
            "| cpu_cycles boleni | Feasible Lower-Bound | CPU Norm Saturation | Median Cloud Delay |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in audit["cpu_divisor_scenarios"]:
        lines.append(
            f"| {row['cpu_divisor']} | {_format_pct(row['feasible_rate'])} | {_format_pct(row['cpu_norm_saturation_rate'])} | {_format_float(row['cloud_delay_median_s'], 3)} s |"
        )

    lines.extend(
        [
            "",
            "### Deadline Multiplier Senaryolari",
            "",
            "| deadline carpani | Env-Feasible Lower-Bound | Median Deadline |",
            "|---:|---:|---:|",
        ]
    )
    for row in audit["deadline_multiplier_scenarios"]:
        lines.append(
            f"| {row['deadline_multiplier']} | {_format_pct(row['feasible_rate'])} | {_format_float(row['deadline_median_s'], 3)} s |"
        )

    lines.extend(
        [
            "",
            "## Profesyonel Sonuc",
            "",
            "Guncel `real_composite_trace` benchmark'i, onceki kalibrasyonsuz surume gore belirgin bicimde toparlanmistir.",
            "Ancak builder-referans feasibility ile env-faithful feasibility birlikte okunmadan benchmark tam saglam ilan edilmemelidir.",
            "Bundan sonraki yorumlarda env-faithful audit ana referans kabul edilmelidir.",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    write_report(audit)
    print(json.dumps(audit, indent=2))
    print(f"[INFO] Feasibility audit written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
