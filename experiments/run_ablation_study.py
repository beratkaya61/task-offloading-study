#!/usr/bin/env python3
"""
Faz 5: Sistematik Ablation Study
9 farklı ablation senaryosu üzerinde PPO_v2 politikasını değerlendir.
Hangi semantic bileşenlerin toplam başarıya katkı sağladığını ölçmek hedef.
"""

import os
import yaml
import simpy
import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

from env.rl_env import OffloadingEnv
from env.simulation_env import WirelessChannel, EdgeServer, CloudServer, IoTDevice
from core.evaluation import evaluate_policy, summarize_logs
from core.metrics import OffloadingMetrics, compute_episode_metrics
from stable_baselines3 import PPO

def load_ablation_config(config_path='configs/ablation.yaml'):
    """Load ablation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_env_with_ablation(ablation_spec):
    """Create OffloadingEnv with specific ablation flags."""
    # Simulation environment'ı ayarla (setup)
    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(env_sim, 1, (200, 200), 2.5e9),
        EdgeServer(env_sim, 2, (800, 200), 2.0e9),
        EdgeServer(env_sim, 3, (500, 800), 2.2e9)
    ]
    
    # IoT cihazları oluştur
    devices = [
        IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, 
                  cloud_server=cloud, battery_capacity=10000.0)
        for i in range(5)
    ]
    
    # Ablation spec'inden flagları çıkar
    flags = ablation_spec.get('enabled_features', {})
    disable_flags = {
        'disable_semantics': not flags.get('semantics', True),
        'disable_reward_shaping': not flags.get('reward_shaping', True),
        'disable_semantic_prior': not flags.get('semantic_prior', True),
        'disable_confidence_weighting': not flags.get('confidence_weighting', True),
        'disable_partial_offloading': not flags.get('partial_offloading', True),
        'disable_battery_awareness': not flags.get('battery_awareness', True),
        'disable_queue_awareness': not flags.get('queue_awareness', True),
        'disable_mobility_features': not flags.get('mobility_features', True),
    }
    
    # Ablation flagları ile environment oluştur
    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        **disable_flags
    )
    
    return env

def run_ablation_study():
    """Run full ablation study with 9 configurations."""
    print("=" * 80)
    print("FAZ 5: SİSTEMATİK ABLATION STUDY")
    print("=" * 80)
    print(f"Başlama zamanı: {datetime.now().isoformat()}")
    print()
    
    # Konfigurasyonu yükle
    config = load_ablation_config()
    ablation_specs = config['ablation_studies']
    num_episodes = config['experiment']['num_episodes_per_ablation']
    
    print(f"[INFO] {len(ablation_specs)} ablation senaryosu çalıştırılacak")
    print(f"[INFO] Her ablation: {num_episodes} episode")
    print(f"[INFO] Toplam run: {len(ablation_specs) * num_episodes}")
    print()
    
    # PPO Policy Modelı yükle
    ppo_model_path = "models/ppo_offloading_agent_v2.zip"
    if not os.path.exists(ppo_model_path):
        print(f"[ERROR] PPO modeli bulunamadı: {ppo_model_path}")
        return
    
    print(f"[INFO] PPO modeli yükleniyor: {ppo_model_path}")
    
    results_summary = []
    
    # Run each ablation study
    for idx, (ablation_name, ablation_spec) in enumerate(ablation_specs.items(), 1):
        print()
        print(f"[{idx}/{len(ablation_specs)}] Ablation başlatılıyor: {ablation_name}")
        print(f"    Tanım: {ablation_spec['description']}")
        print(f"    Beklenen düşüş: {ablation_spec.get('expected_drop', 'Baseline')}")
        
        try:
            # Create environment with ablation flags
            eval_env = create_env_with_ablation(ablation_spec)
            
            # Load model for this environment
            policy = PPO.load(ppo_model_path, env=eval_env)
            
            # Evaluate policy
            print(f"    → {num_episodes} episode üzerinde değerlendiriliyor...")
            evaluate_policy(eval_env, policy, num_episodes=num_episodes, 
                          run_name=f"{ablation_name}")
            
            print(f"    ✅ {ablation_name} tamamlandı")
            
        except Exception as e:
            print(f"    ❌ HATA: {ablation_name} değerlendirilirken: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    print()
    print("=" * 80)
    print("ABLATION STUDY TAMAMLANDI")
    print("=" * 80)
    summarize_logs(results_dir="results/raw", output_table="results/tables/summary.md")
    
    # Generate ablation comparison table
    print("[INFO] Ablation karşılaştırma tablosu oluşturuluyor...")
    generate_ablation_comparison_table()
    
    print()
    print(f"Bitiş zamanı: {datetime.now().isoformat()}")
    print(f"Sonuçlar: results/tables/summary.md")
    print(f"Ablation karşılaştırması: results/tables/ablation_comparison.md")

def generate_ablation_comparison_table():
    """
    Generate comparison table from master_experiments.csv.
    Ablation impact'ları hesapla ve tablo oluştur.
    """
    csv_path = "results/raw/master_experiments.csv"
    if not os.path.exists(csv_path):
        print("[WARN] CSV dosyası bulunamadı")
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter only Faz 5 ablation results (latest runs)
        # Sort by timestamp and get last run of each model
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_ablations = df.groupby('config_model_type').tail(1).sort_values('metric_success_rate', ascending=False)
        
        os.makedirs("results/tables", exist_ok=True)
        
        # Create markdown table
        with open("results/tables/ablation_comparison.md", "w", encoding="utf-8") as f:
            f.write("# 📊 Faz 5: Ablation Study Karşılaştırması\n\n")
            f.write("## Semantic Bileşenlerin Bireysel Katkısı\n\n")
            f.write("**Hedef:** Her ablation senaryosu, Full Model (%62.67) ile kıyaslanarak, \n")
            f.write("bileşenlerin toplam başarıya katkısını ölçmek.\n\n")
            f.write(latest_ablations[['config_model_type', 'metric_success_rate', 'metric_avg_reward']].to_markdown(index=False))
            f.write("\n\n---\n\n")
            f.write("## İstatistiksel Analiz\n\n")
            
            # Calculate deltas from full model
            full_success = latest_ablations[latest_ablations['config_model_type'] == 'full_model']['metric_success_rate'].values
            if len(full_success) > 0:
                full_success = full_success[0]
                f.write(f"**Baseline (Full Model):** {full_success*100:.2f}%\n\n")
                f.write("| Ablation | Başarı % | Delta | Katkı |\n")
                f.write("|----------|---------|-------|-------|\n")
                for _, row in latest_ablations.iterrows():
                    model_name = row['config_model_type']
                    success = row['metric_success_rate']
                    delta = (success - full_success) * 100
                    contribution = -delta if model_name != 'full_model' else 0
                    f.write(f"| {model_name} | {success*100:.2f}% | {delta:+.2f}% | {contribution:.2f}% |\n")
            
            f.write("\n---\n")
            f.write(f"*Güncellenme: {datetime.now().isoformat()}*\n")
        
        print("[INFO] Ablation karşılaştırma tablosu oluşturuldu: results/tables/ablation_comparison.md")
        
    except Exception as e:
        print(f"[ERROR] Tablo oluşturmada hata: {e}")

if __name__ == "__main__":
    run_ablation_study()
