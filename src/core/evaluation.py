import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import uuid


def _is_sb3_model(model):
    """
    Stable Baselines3 modeli olup olmadığını kontrol et.
    SB3 modellerinin 'policy' ve 'learn' attribute'ları vardır.
    Özel baseline'lar basit sınıflardır.
    """
    return hasattr(model, "policy") and hasattr(model, "learn")


def evaluate_policy(
    env, model, num_episodes=5, run_name="Baseline", semantic_mode="None"
):
    """
    Belirli bir modeli (baseline veya RL) Gymnasium ortamında test eder.
    Metrikleri toplar ve CSV'ye kaydeder.

    SB3 ve özel baseline modellerini her ikisini de destekler:
    - SB3 modelleri: batch formatında (1, obs_dim) gözlem beklerler
    - Özel baseline'lar: düz vektör (obs_dim,) ile çalışırlar
    """
    print(f"[EVAL] Değerlendirme Başlatıldı: {run_name} ({num_episodes} bölüm)")

    is_sb3 = _is_sb3_model(model)
    if is_sb3:
        print(f"[EVAL] ✓ SB3 Model (PPO/DQN/A2C) tespit edildi: {run_name}")
    else:
        print(f"[EVAL] ✓ Özel Baseline Model: {run_name}")

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
            # SB3 modelleri batch formatında (1, obs_dim) gözlem beklerler
            if is_sb3:
                obs_batch = obs[np.newaxis, :]  # (11,) -> (1, 11)
                action, _ = model.predict(obs_batch, deterministic=True)
                # SB3 action'ı numpy array olarak döner, int'e çevir
                action = int(action)
            else:
                # Özel baseline'lar düz vektör ile çalışırlar
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1

            # Başarı takibi: reward > 50 ise başarılı sayıl
            if reward > 50:
                ep_successes += 1

        results.append(
            {
                "reward": episode_reward,
                "steps": step_count,
                "success_rate": ep_successes / max(1, step_count),
            }
        )

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
        "config_total_tasks": num_episodes * 50,  # max_steps=50 varsayıldı
        "metric_success_rate": round(avg_success, 4),
        "metric_avg_reward": round(avg_reward, 2),
    }

    df = pd.DataFrame([log_entry])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"✅ {run_name} değerlendirildi. Ortalama Başarı: {avg_success:.2%}")


def summarize_logs(results_dir="results/raw", output_table="results/tables/summary.md"):
    """
    Master CSV'den özet tablo üretir ve Markdown formatında kaydeder.

    Tüm baselines'ların (GA, PPO, Greedy vb.) performans kıyaslaması
    bu dosyada görüntülenir.
    """
    csv_path = os.path.join(results_dir, "master_experiments.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV dosyası bulunamadı: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        os.makedirs(os.path.dirname(output_table), exist_ok=True)
        markdown_str = df.to_markdown(index=False)

        with open(output_table, "w", encoding="utf-8") as f:
            f.write("# 📊 Experiment Evaluation Summary\n\n")
            f.write("## Tüm Baseline Modellerinin Performans Karşılaştırması\n\n")
            f.write(markdown_str)
            f.write("\n\n---\n")
            f.write(f"*Güncellenme Tarihi: {datetime.now().isoformat()}*\n")
            f.write("*Faz 4: Baseline Ailesinin Genişletilmesi*\n")

        print(f"✅ Özet tablo oluşturuldu: {output_table}")

    except Exception as e:
        print(f"[ERROR] Özet tablo oluşturması başarısız: {e}")


if __name__ == "__main__":
    summarize_logs()
