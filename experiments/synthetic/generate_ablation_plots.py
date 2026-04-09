#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from src.core.evaluation import summarize_logs
except ModuleNotFoundError:
    from core.evaluation import summarize_logs


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


def load_ablation_config(config_path="configs/synthetic/ablation.yaml"):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_scope_label(seeds, mode):
    seed_scope = "single_seed" if len(seeds) == 1 else "multi_seed"
    return f"{seed_scope}_{mode}"


def resolve_ablation_paths(config):
    experiment_cfg = config.get("experiment", {})
    retraining_cfg = config.get("retraining", {})
    output_cfg = config.get("output", {})

    mode = experiment_cfg.get("mode", "evaluation")
    algorithm = (
        retraining_cfg.get("algorithm", experiment_cfg.get("algorithm", "ppo"))
        if mode == "retrain"
        else experiment_cfg.get("algorithm", "ppo")
    )
    seeds = retraining_cfg.get("seeds", [42, 43, 44]) if mode == "retrain" else experiment_cfg.get("seeds", [42, 43, 44])
    scope = resolve_scope_label(seeds, "retraining" if mode == "retrain" else "evaluation")

    csv_template = output_cfg.get(
        "csv_path_template",
        "results/raw/synthetic_ablation_{algorithm}_{scope}.csv",
    )
    plot_template = output_cfg.get(
        "plot_path_template",
        "results/figures/synthetic_ablation_{algorithm}_{scope}_success_rate.png",
    )

    return (
        csv_template.format(algorithm=algorithm, scope=scope),
        plot_template.format(algorithm=algorithm, scope=scope),
        algorithm,
        scope,
    )


def resolve_latest_ablation_csv(default_csv_path, algorithm=None, scope=None):
    search_root = os.path.dirname(os.path.dirname(default_csv_path)) or "results/raw"
    if not os.path.isdir(search_root):
        return default_csv_path

    candidates = []
    for root, _, files in os.walk(search_root):
        for file_name in files:
            if not file_name.startswith("synthetic_ablation_") or not file_name.endswith(".csv"):
                continue
            if algorithm and f"synthetic_ablation_{algorithm}_" not in file_name:
                continue
            if scope and not file_name.endswith(f"_{scope}.csv"):
                continue
            file_path = os.path.join(root, file_name)
            candidates.append(file_path)

    if not candidates:
        return default_csv_path

    return max(candidates, key=os.path.getmtime)


def describe_ablation_csv(csv_path, fallback_algorithm, fallback_scope):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    parts = base_name.split("_")
    if len(parts) >= 5:
        algorithm = parts[2]
        scope = "_".join(parts[3:])
        return algorithm, scope
    return fallback_algorithm, fallback_scope


def load_latest_ablation_batch(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["config_model_type"].isin(ABLATION_MODELS)].copy()
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "config_batch_id" in df.columns:
        retrain_rows = df[
            df["config_eval_group"].astype(str).isin(
                [
                    "ablation_retraining_multiseed",
                    "phase5_ablation_retraining",
                    "synthetic_ablation_retraining",
                ]
            )
        ].copy()
        if not retrain_rows.empty:
            latest_batch_id = retrain_rows.sort_values("timestamp")["config_batch_id"].iloc[-1]
            return retrain_rows[retrain_rows["config_batch_id"] == latest_batch_id].copy()

        batch_rows = df[
            df["config_batch_id"].astype(str).str.startswith("ablation_")
            | df["config_batch_id"].astype(str).str.startswith("synthetic_ablation_")
        ].copy()
        if not batch_rows.empty:
            latest_batch_id = batch_rows.sort_values("timestamp")["config_batch_id"].iloc[-1]
            return batch_rows[batch_rows["config_batch_id"] == latest_batch_id].copy()

    return (
        df.sort_values("timestamp")
        .groupby("config_model_type", as_index=False)
        .tail(1)
        .copy()
    )


def generate_ablation_visuals(config_path="configs/synthetic/ablation.yaml", csv_path=None, algorithm=None, scope=None):
    config = load_ablation_config(config_path)
    default_csv_path, _, default_algorithm, default_scope = resolve_ablation_paths(config)
    csv_path = csv_path or resolve_latest_ablation_csv(
        default_csv_path,
        algorithm=algorithm,
        scope=scope,
    )
    algorithm = algorithm or default_algorithm
    scope = scope or default_scope
    algorithm, scope = describe_ablation_csv(csv_path, algorithm, scope)
    plot_template = config.get("output", {}).get(
        "plot_path_template",
        "results/figures/synthetic/ablation/synthetic_ablation_{algorithm}_{scope}_success_rate.png",
    )
    plot_path = plot_template.format(algorithm=algorithm, scope=scope)
    if not os.path.exists(csv_path):
        print("[ERROR] CSV file not found:", csv_path)
        return

    df_ablation = load_latest_ablation_batch(csv_path)
    if df_ablation.empty:
        print("[ERROR] No ablation data found in the CSV!")
        return

    grouped = (
        df_ablation.groupby("config_model_type")[
            ["metric_success_rate", "metric_avg_energy", "metric_p95_latency", "metric_qoe"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    grouped = grouped.sort_values("metric_success_rate_mean", ascending=False)

    baseline_row = grouped[grouped["config_model_type"] == "full_model"]
    baseline_success_mean = (
        float(baseline_row["metric_success_rate_mean"].iloc[0]) if not baseline_row.empty else 0.0
    )

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    models_plot = grouped["config_model_type"].tolist()
    means_plot = grouped["metric_success_rate_mean"].tolist()
    stds_plot = grouped["metric_success_rate_std"].fillna(0.0).tolist()
    means_pct = [value * 100.0 for value in means_plot]
    stds_pct = [value * 100.0 for value in stds_plot]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for model, value in zip(models_plot, means_pct):
        if model == "full_model":
            colors.append("royalblue")
        elif value > baseline_success_mean * 100.0:
            colors.append("seagreen")
        else:
            colors.append("indianred")

    bars = ax.bar(models_plot, means_pct, yerr=stds_pct, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(baseline_success_mean * 100.0, color="royalblue", linestyle="--", linewidth=2, label="Baseline (Full Model)")
    ax.set_title(
        f"Synthetic Ablation Impact on Success Rate ({algorithm.upper()}, {scope})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, max(means_pct + stds_pct) + 10 if means_pct else 100)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot generated at {plot_path}")
    summarize_logs(
        results_dir="results/raw",
        output_table="v2_docs/phase_5/offloading_experiment_report.md",
        figure_path=plot_path,
    )
    print("[INFO] Canonical report refreshed: v2_docs/phase_5/offloading_experiment_report.md")
    return plot_path


def list_ablation_csvs(raw_dir="results/raw", algorithm=None, scope=None):
    if not os.path.isdir(raw_dir):
        return []

    candidates = []
    for root, _, files in os.walk(raw_dir):
        for file_name in files:
            if not file_name.startswith("synthetic_ablation_") or not file_name.endswith(".csv"):
                continue
            if algorithm and f"synthetic_ablation_{algorithm}_" not in file_name:
                continue
            if scope and not file_name.endswith(f"_{scope}.csv"):
                continue
            candidates.append(os.path.join(root, file_name))
    return sorted(candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ablation figures")
    parser.add_argument("--config", default="configs/synthetic/ablation.yaml", help="Ablation config path")
    parser.add_argument("--csv", default=None, help="Explicit ablation CSV path")
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "a2c"], default=None, help="Filter ablation CSVs by algorithm")
    parser.add_argument(
        "--scope",
        choices=["single_seed_evaluation", "multi_seed_evaluation", "single_seed_retraining", "multi_seed_retraining"],
        default=None,
        help="Filter ablation CSVs by scope",
    )
    parser.add_argument("--all", action="store_true", help="Generate figures for all matching ablation CSV files")
    args = parser.parse_args()

    if args.all:
        csv_paths = list_ablation_csvs(algorithm=args.algorithm, scope=args.scope)
        if not csv_paths:
            print("[ERROR] No ablation CSV files matched the requested filters.")
        for csv_path in csv_paths:
            generate_ablation_visuals(
                config_path=args.config,
                csv_path=csv_path,
                algorithm=args.algorithm,
                scope=args.scope,
            )
    else:
        generate_ablation_visuals(
            config_path=args.config,
            csv_path=args.csv,
            algorithm=args.algorithm,
            scope=args.scope,
        )

