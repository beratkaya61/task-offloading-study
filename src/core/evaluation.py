import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import uuid

def evaluate_policy(env, model, num_episodes=5, run_name="Baseline", semantic_mode="None"):
    """
    Belirli bir modeli (baseline veya RL) Gymnasium ortamında test eder.
    Metrikleri toplar ve CSV'ye kaydeder.
    """
    print(f"[EVAL] Değerlendirme Başlatıldı: {run_name} ({num_episodes} bölüm)")
    
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        ep_latencies = []
        ep_energies = []
        ep_successes = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Not: rl_env.py içerisinde delay/energy/success gibi ek metrikler
            # info sözlüğüne veya self.current_task üzerinden çekilmelidir.
            # Şimdilik reward üzerinden basit bir takip:
            if reward > 50: ep_successes += 1
            
        results.append({
            "reward": episode_reward,
            "steps": step_count,
            "success_rate": ep_successes / max(1, step_count)
        })

    # Ortalamaları hesapla
    avg_reward = np.mean([r["reward"] for r in results])
    avg_success = np.mean([r["success_rate"] for r in results])
    
    # Log kaydı
    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/master_experiments.csv"
    
    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": 42,
        "config_model_type": run_name,
        "config_semantic_mode": semantic_mode,
        "config_total_tasks": num_episodes * 50, # max_steps=50 varsayıldı
        "metric_success_rate": round(avg_success, 4),
        "metric_avg_reward": round(avg_reward, 2)
    }
    
    df = pd.DataFrame([log_entry])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
    print(f"✅ {run_name} değerlendirildi. Ortalama Başarı: {avg_success:.2%}")

def summarize_logs(results_dir="results/raw", output_table="results/tables/summary.md"):
    """
    Master CSV'den özet tablo üretir.
    """
    csv_path = os.path.join(results_dir, "master_experiments.csv")
    if not os.path.exists(csv_path):
        return
        
    try:
        df = pd.read_csv(csv_path)
        os.makedirs(os.path.dirname(output_table), exist_ok=True)
        markdown_str = df.to_markdown(index=False)
        
        with open(output_table, "w", encoding="utf-8") as f:
            f.write("# Experiment Evaluation Summary\n\n")
            f.write(markdown_str)
            
    except Exception as e:
        print(f"[Evaluation] Hata: {e}")

if __name__ == "__main__":
    summarize_logs()
