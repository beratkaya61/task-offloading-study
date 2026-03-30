import gymnasium as gym
import simpy
import os
import torch
from stable_baselines3 import PPO, DQN, A2C

import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

from env.rl_env import OffloadingEnv
from env.simulation_env import WirelessChannel, EdgeServer, CloudServer, IoTDevice
from core.evaluation import evaluate_policy, summarize_logs
from agents.baselines import (
    LocalOnlyPolicy, 
    EdgeOnlyPolicy, 
    CloudOnlyPolicy, 
    RandomPolicy, 
    GreedyLatencyPolicy, 
    GeneticAlgorithmPolicy
)

def run_all_baselines():
    print("--------------------------------------------------")
    print("   IOT TASK OFFLOADING BASELINE EVALUATION        ")
    print("--------------------------------------------------")
    
    # 1. Ortam Kurulumu (Hizalanmış modda)
    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(env_sim, 1, (200, 200), 2.5e9),
        EdgeServer(env_sim, 2, (800, 200), 2.0e9),
        EdgeServer(env_sim, 3, (500, 800), 2.2e9)
    ]
    
    # Test cihazları
    devices = [IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud, battery_capacity=10000.0) for i in range(5)]
    
    # Gymnasium Ortamı
    eval_env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel
    )
    
    # 2. Rakipleri Listele
    policies = {
        "LocalOnly": LocalOnlyPolicy(),
        "EdgeOnly": EdgeOnlyPolicy(),
        "CloudOnly": CloudOnlyPolicy(),
        "Random": RandomPolicy(),
        "GreedyLatency": GreedyLatencyPolicy(),
        "GeneticAlgorithm": GeneticAlgorithmPolicy(population_size=10, generations=5),
    }
    
    # 3. PPO v2 modelini yükle
    ppo_v2_path = "models/ppo_offloading_agent_v2.zip"
    ppo_v1_path = "models/ppo_offloading_agent.zip"
    
    ppo_path = ppo_v2_path if os.path.exists(ppo_v2_path) else ppo_v1_path
    ppo_label = "PPO_v2" if os.path.exists(ppo_v2_path) else "PPO_v1"
    
    if os.path.exists(ppo_path):
        print(f"[INIT] PPO Modeli yuklendi: {ppo_path}")
        try:
            policies[ppo_label] = PPO.load(ppo_path, env=eval_env)
            print(f"[INIT] {ppo_label} hazir, obs_space={eval_env.observation_space.shape}")
        except Exception as e:
            print(f"[WARN] PPO yuklenemedi: {e}")
    
    # 4. Klasik RL Baselines - Faz 5+ icin eklenecek
    # policies["DQN"] = DQN("MlpPolicy", eval_env)
    # policies["A2C"] = A2C("MlpPolicy", eval_env)


    # 5. Her Birini Değerlendir
    for name, policy in policies.items():
        try:
            evaluate_policy(eval_env, policy, num_episodes=3, run_name=name)
        except Exception as e:
            print(f"[ERROR] {name} değerlendirilirken hata oluştu: {e}")

    # 6. Sonuçları Özetle
    summarize_logs()
    print("\n[FINISH] Tüm baselinelar test edildi. Sonuçlar results/tables/summary.md dosyasında.")

if __name__ == "__main__":
    run_all_baselines()
