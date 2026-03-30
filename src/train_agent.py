import os
import random
import time
import uuid
import simpy
from stable_baselines3 import PPO
import torch
import pandas as pd
from datetime import datetime

from rl_env import OffloadingEnv
from simulation_env import WirelessChannel, EdgeServer, CloudServer, IoTDevice, Task, TaskType, LLM_ANALYZER
from config import load_config
from utils.reproducibility import set_seed

def train():
    print("[TRAIN] Initializing Training Environment...")
    
    # Load config
    cfg = load_config()
    exp_cfg = cfg.get('experiment', {})
    ppo_cfg = cfg.get('ppo', {})
    env_cfg = cfg.get('env', {})
    
    # 1. Setup Simulation Components (Mock for training)
    env = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env) # CloudServer only takes 'env'
    num_edge_servers = env_cfg.get('num_edge_servers', 3)
    edge_servers = [EdgeServer(env, i+1, (random.uniform(0, 1000), random.uniform(0, 1000)), 2e9) for i in range(num_edge_servers)]
    
    # Create the custom Gym environment
    train_env = OffloadingEnv(
        devices=[], # Will be populated per step/reset
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel
    )
    
    # Apply Seed
    seed = exp_cfg.get('seed', 42)
    set_seed(seed, env=train_env)

    # 2. Setup PPO Agent
    print("[TRAIN] Instantiating PPO Agent (MLP Policy)...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=ppo_cfg.get('learning_rate', 3e-4),
        n_steps=ppo_cfg.get('n_steps', 2048),
        batch_size=ppo_cfg.get('batch_size', 64),
        n_epochs=ppo_cfg.get('n_epochs', 10),
        gamma=ppo_cfg.get('gamma', 0.99),
        device=ppo_cfg.get('device', "cpu")
    )
    
    # 3. Training Loop
    total_timesteps = exp_cfg.get('total_timesteps', 100000)
    print(f"[TRAIN] Starting Training ({total_timesteps} steps)...")
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - start_time

    # 4. Save the model
    os.makedirs("src/models", exist_ok=True)
    model.save("src/models/ppo_offloading_agent")
    print("[TRAIN] Model saved to src/models/ppo_offloading_agent.zip")
    
    # 5. Experiment Logging to CSV
    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/master_experiments.csv"
    
    # Create dummy metrics for now (actual metrics should come from eval loop)
    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": seed,
        "config_config_hash": "N/A",
        "config_model_type": exp_cfg.get('model_type', 'PPO'),
        "config_semantic_mode": exp_cfg.get('semantic_mode', 'None'),
        "config_total_tasks": total_timesteps,
        "metric_success_rate": 0.0, # Placeholder
        "metric_avg_latency": 0.0,  # Placeholder
        "metric_p95_latency": 0.0,  # Placeholder
        "metric_avg_energy": 0.0,   # Placeholder
        "metric_fairness": 0.0,     # Placeholder
        "metric_jitter": 0.0,       # Placeholder
        "metric_qoe": 0.0,          # Placeholder
        "metric_decision_overhead": 0.0, # Placeholder
        "training_time_sec": round(training_time, 2)
    }
    
    df = pd.DataFrame([log_entry])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
    print(f"[TRAIN] Logged experiment run to {csv_path}")

if __name__ == "__main__":
    train()
