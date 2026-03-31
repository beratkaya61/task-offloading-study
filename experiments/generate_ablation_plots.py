#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_ablation_visuals_and_table():
    csv_path = "results/raw/master_experiments.csv"
    if not os.path.exists(csv_path):
        print("[ERROR] CSV file not found:", csv_path)
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Identify Phase 5 Ablation Models
    ablation_models = [
        "full_model", "w_o_semantics", "w_o_reward_shaping", "w_o_semantic_prior",
        "w_o_confidence", "w_o_partial_offloading", "w_o_battery_awareness",
        "w_o_queue_awareness", "w_o_mobility_features"
    ]
    
    # Filter the dataframe
    df_ablation = df[df['config_model_type'].isin(ablation_models)].copy()
    
    if df_ablation.empty:
        print("[ERROR] No ablation data found in the CSV!")
        return

    # To group by model type and get mean and std for specific numeric columns
    # We will pick the latest N episodes. But to be safe, we group by model and calculate statistical values.
    
    metrics_of_interest = {
        'metric_success_rate': 'Success Rate (%)',
        'metric_avg_energy': 'Avg Energy (J)',
        'metric_p95_latency': 'p95 Latency (s)',
        'metric_qoe': 'QoE Score'
    }
    
    # Replace any missing metrics with 0.0 temporarily if they weren't logged properly before
    for col in metrics_of_interest.keys():
        if col not in df_ablation.columns:
            df_ablation[col] = 0.0
            
    # Calculate Mean and Std Deviation
    grouped = df_ablation.groupby('config_model_type')[list(metrics_of_interest.keys())].agg(['mean', 'std']).reset_index()
    
    # Clean up column names horizontally
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Extract baseline
    baseline_row = grouped[grouped['config_model_type'] == 'full_model']
    if baseline_row.empty:
        baseline_success_mean = 0.0
    else:
        baseline_success_mean = baseline_row['metric_success_rate_mean'].values[0]
        
    # Prepare sorting (descending success rate)
    grouped = grouped.sort_values(by='metric_success_rate_mean', ascending=False)
    
    # 1. Generate Extended Markdown Table
    os.makedirs("results/tables", exist_ok=True)
    table_path = "results/tables/ablation_extended_comparison.md"
    
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Faz 5: Kapsamlı Ablation Analizi (Genişletilmiş Veri Seti)\n\n")
        f.write("| Ablation Model | Success Rate (± StdDev) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        for _, row in grouped.iterrows():
            model = row['config_model_type']
            sr_mean = row['metric_success_rate_mean'] * 100
            sr_std = row['metric_success_rate_std'] * 100 if not np.isnan(row['metric_success_rate_std']) else 0.0
            
            e_mean = row['metric_avg_energy_mean']
            p95_mean = row['metric_p95_latency_mean']
            qoe_mean = row['metric_qoe_mean']
            
            delta = sr_mean - (baseline_success_mean * 100)
            delta_str = f"{delta:+.2f}%" if model != 'full_model' else "0.00% (Baseline)"
            
            f.write(f"| {model} | **{sr_mean:.2f}%** (±{sr_std:.2f}) | {e_mean:.3f} | {p95_mean:.3f} | {qoe_mean:.2f} | {delta_str} |\n")
            
        f.write("\n_Not: Tüm veriler 25 bağımsız bölüm (episode) üzerinden varyans hesaplanarak türetilmiştir._\n")
    print(f"[INFO] Extended table written to {table_path}")

    # 2. Generate Matplotlib Plot with Error Bars
    os.makedirs("results/figures", exist_ok=True)
    plot_path = "results/figures/ablation_impact.png"
    
    # Sort for plot based on ablation models order given in config or by success rate
    models_plot = grouped['config_model_type'].tolist()
    means_plot = [row['metric_success_rate_mean'] * 100 for _, row in grouped.iterrows()]
    stds_plot = [row['metric_success_rate_std'] * 100 if not np.isnan(row['metric_success_rate_std']) else 0 for _, row in grouped.iterrows()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create colors: Baseline as blue, positive delta as green, negative delta as orange/red
    colors = []
    for m, val in zip(models_plot, means_plot):
        if m == 'full_model':
            colors.append('royalblue')
        elif val > (baseline_success_mean * 100):
            colors.append('seagreen') # Anomalies / improvements
        else:
            colors.append('indianred') # Expected drop
            
    bars = ax.bar(models_plot, means_plot, yerr=stds_plot, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    
    # Add horizontal line for baseline
    ax.axhline(baseline_success_mean * 100, color='royalblue', linestyle='--', linewidth=2, label='Baseline (Full Model)')
    
    ax.set_title('Ablation Study Impact on Success Rate (with 95% CI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, max(means_plot + stds_plot) + 10 if means_plot else 100)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Plot generated at {plot_path}")

if __name__ == "__main__":
    generate_ablation_visuals_and_table()
