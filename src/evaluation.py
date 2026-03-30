import os
import pandas as pd

def summarize_logs(results_dir="results/raw", output_table="results/tables/summary.md"):
    """
    Reads the master experiment CSV and creates a summary markdown table comparing different runs.
    Fulfills Phase 1 requirements: 'evaluation.py ile bu logları tek tabloda birleştir.'
    """
    csv_path = os.path.join(results_dir, "master_experiments.csv")
    if not os.path.exists(csv_path):
        print(f"[Evaluation] Kayıtlı log dosyası bulunamadı: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
        
        # Standard columns we want to show as requested in the Phase 1 Task List
        desired_columns = [
            "run_id", "timestamp", "config_seed", "config_config_hash", "config_model_type", 
            "config_semantic_mode", "config_total_tasks", "metric_success_rate", "metric_avg_latency", 
            "metric_p95_latency", "metric_avg_energy", "metric_fairness", "metric_jitter", 
            "metric_qoe", "metric_decision_overhead"
        ]
        
        # Filter to only the columns that actually exist in the dataframe to avoid KeyErrors
        cols_to_show = [c for c in desired_columns if c in df.columns]
        
        # If there are any other columns that are not in desired_columns but exist in the CSV, append them
        extra_cols = [c for c in df.columns if c not in desired_columns]
        cols_to_show.extend(extra_cols)
        
        summary_df = df[cols_to_show].fillna("N/A")
        
        os.makedirs(os.path.dirname(output_table), exist_ok=True)
        # Create markdown representation
        markdown_str = summary_df.to_markdown(index=False)
        
        with open(output_table, "w", encoding="utf-8") as f:
            f.write("# Experiment Evaluation Summary\n\n")
            f.write("Aşağıdaki tablo `results/raw/master_experiments.csv` loglarından otomatik olarak türetilmiştir.\n\n")
            f.write(markdown_str)
            
        print(f"✅ Evaluation summary created successfully at: {output_table}")
        
    except Exception as e:
        print(f"[Evaluation] Logları değerlendirme esnasında hata oluştu: {e}")

if __name__ == "__main__":
    # Test execution
    summarize_logs()
