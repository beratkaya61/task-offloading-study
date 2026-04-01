#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                ["ablation_retraining_multiseed", "phase5_ablation_retraining"]
            )
        ].copy()
        if not retrain_rows.empty:
            latest_batch_id = retrain_rows.sort_values("timestamp")["config_batch_id"].iloc[-1]
            return retrain_rows[retrain_rows["config_batch_id"] == latest_batch_id].copy()

        batch_rows = df[df["config_batch_id"].astype(str).str.startswith("ablation_")].copy()
        if not batch_rows.empty:
            latest_batch_id = batch_rows.sort_values("timestamp")["config_batch_id"].iloc[-1]
            return batch_rows[batch_rows["config_batch_id"] == latest_batch_id].copy()

    return (
        df.sort_values("timestamp")
        .groupby("config_model_type", as_index=False)
        .tail(1)
        .copy()
    )


def generate_ablation_visuals():
    csv_path = "results/raw/master_experiments.csv"
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

    os.makedirs("results/figures", exist_ok=True)
    plot_path = "results/figures/ablation_impact.png"
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
    ax.set_title("Ablation Study Impact on Success Rate", fontsize=14, fontweight="bold")
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
    summarize_logs(results_dir="results/raw", output_table="results/tables/offloading_experiment_report.md")
    print("[INFO] Canonical report refreshed: results/tables/offloading_experiment_report.md")


if __name__ == "__main__":
    generate_ablation_visuals()
