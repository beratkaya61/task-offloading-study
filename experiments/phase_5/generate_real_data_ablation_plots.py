#!/usr/bin/env python3
"""Generate Phase 5 real-data ablation plots from canonical CSV artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.experiment_artifacts import refresh_real_data_phase5_report


ABLATION_MODELS = [
    "full_model",
    "w_o_semantics",
    "w_o_reward_shaping",
    "w_o_semantic_prior",
    "w_o_confidence",
    "w_o_partial_offloading",
    "w_o_battery_awareness",
    "w_o_queue_awareness",
    "w_o_mobility_features",
]


def resolve_output_png(algorithm: str, scope: str) -> Path:
    return Path(f"results/phase_5/figures/real_data/ablation/real_data_ablation_{algorithm}_{scope}_success_rate.png")


def load_latest_ablation_batch(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["config_model_type"].isin(ABLATION_MODELS)].copy()
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "config_batch_id" in df.columns:
        batch_rows = df[
            df["config_eval_group"].astype(str).isin(
                [
                    "real_data_ablation_retraining",
                    "real_data_ablation_evaluation",
                ]
            )
        ].copy()
        if not batch_rows.empty:
            latest_batch_id = batch_rows.sort_values("timestamp")["config_batch_id"].iloc[-1]
            return batch_rows[batch_rows["config_batch_id"] == latest_batch_id].copy()

    return df.sort_values("timestamp").groupby("config_model_type", as_index=False).tail(1).copy()


def generate_plot(csv_path: Path, algorithm: str, output_path: Path) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = load_latest_ablation_batch(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    grouped = (
        df.groupby("config_model_type")[["metric_success_rate", "metric_avg_energy", "metric_p95_latency", "metric_qoe"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    grouped = grouped.sort_values("metric_success_rate_mean", ascending=False)

    baseline_row = grouped[grouped["config_model_type"] == "full_model"]
    baseline_success = float(baseline_row["metric_success_rate_mean"].iloc[0]) if not baseline_row.empty else 0.0

    names = grouped["config_model_type"].tolist()
    means = (grouped["metric_success_rate_mean"] * 100.0).tolist()
    stds = (grouped["metric_success_rate_std"].fillna(0.0) * 100.0).tolist()

    colors = []
    for name, value in zip(names, means):
        if name == "full_model":
            colors.append("royalblue")
        elif value > baseline_success * 100.0:
            colors.append("seagreen")
        else:
            colors.append("indianred")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(baseline_success * 100.0, color="royalblue", linestyle="--", linewidth=2, label="Baseline (Full Model)")
    ax.set_title(f"Real-Data Ablation Impact on Success Rate ({algorithm.upper()})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, max(means + stds) + 5 if means else 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    refresh_real_data_phase5_report()
    return output_path


def list_ablation_csvs(algorithm: str | None = None, scope: str | None = None) -> list[Path]:
    base_dir = Path("results/phase_5/metrics/real_data/ablation")
    if not base_dir.exists():
        return []
    pattern = "real_data_ablation_*.csv" if scope is None else f"real_data_ablation_*_{scope}.csv"
    candidates = []
    for csv_path in sorted(base_dir.glob(pattern)):
        if algorithm and f"real_data_ablation_{algorithm}_" not in csv_path.name:
            continue
        candidates.append(csv_path)
    return candidates


def parse_csv_name(csv_path: Path) -> tuple[str, str]:
    stem = csv_path.stem.replace("real_data_ablation_", "")
    parts = stem.split("_")
    algorithm = parts[0]
    scope = "_".join(parts[1:])
    return algorithm, scope


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-data ablation figures")
    parser.add_argument("--csv", default=None, help="Explicit ablation CSV path")
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "a2c"], default=None)
    parser.add_argument(
        "--scope",
        choices=["single_seed_evaluation", "multi_seed_evaluation", "single_seed_retraining", "multi_seed_retraining"],
        default=None,
    )
    parser.add_argument("--all", action="store_true", help="Generate figures for all matching real-data ablation CSV files")
    args = parser.parse_args()

    if args.all:
        csv_paths = list_ablation_csvs(algorithm=args.algorithm, scope=args.scope)
        if not csv_paths:
            raise SystemExit("No real-data ablation CSV files matched the requested filters.")
        for csv_path in csv_paths:
            algorithm, scope = parse_csv_name(csv_path)
            output_path = resolve_output_png(algorithm, scope)
            generate_plot(csv_path, algorithm, output_path)
            print(f"[INFO] Plot written: {output_path}")
        return

    if not args.algorithm and not args.csv:
        raise SystemExit("--algorithm gerekir; tumunu uretmek icin --all kullan")

    if args.csv:
        csv_path = Path(args.csv)
        algorithm, scope = parse_csv_name(csv_path)
    else:
        if not args.scope:
            raise SystemExit("--scope gerekir; ornek: --scope multi_seed_retraining")
        csv_path = Path(f"results/phase_5/metrics/real_data/ablation/real_data_ablation_{args.algorithm}_{args.scope}.csv")
        algorithm, scope = args.algorithm, args.scope

    output_path = resolve_output_png(algorithm, scope)
    generate_plot(csv_path, algorithm, output_path)
    print(f"[INFO] Plot written: {output_path}")


if __name__ == "__main__":
    main()
