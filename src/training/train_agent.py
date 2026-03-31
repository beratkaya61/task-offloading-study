import os
import random
import time
import uuid
import simpy
import argparse
from stable_baselines3 import PPO, DQN, A2C
import torch
import pandas as pd
from datetime import datetime

from src.env.rl_env import OffloadingEnv
from src.env.simulation_env import WirelessChannel, EdgeServer, CloudServer, IoTDevice
from src.core.config import load_config
from src.utils.reproducibility import set_seed

def train():
    parser = argparse.ArgumentParser(description="Train Baseline RL Agents")
    parser.add_argument("--model", type=str, default="PPO", choices=["PPO", "DQN", "A2C"])
    args = parser.parse_args()
    
    print(f"[TRAIN] Initializing Training Environment for {args.model}...")
    
    # Load config
    cfg = load_config()
    exp_cfg = cfg.get('experiment', {})
    ppo_cfg = cfg.get('ppo', {})
    env_cfg = cfg.get('env', {})
    
    # 1. Setup Simulation Components (Mock for training)
    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim) 
    num_edge_servers = env_cfg.get('num_edge_servers', 3)
    edge_servers = [EdgeServer(env_sim, i+1, (random.uniform(0, 1000), random.uniform(0, 1000)), 2e9) for i in range(num_edge_servers)]
    
    num_devices_for_training = 5
    devices = [IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud, battery_capacity=10000.0) for i in range(num_devices_for_training)]
    
    # Create the custom Gym environment
    train_env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel
    )
    
    # Apply Seed
    seed = exp_cfg.get('seed', 42)
    set_seed(seed, env=train_env)

    # 2. Setup RL Agent
    print(f"[TRAIN] Instantiating {args.model} Agent (MLP Policy)...")
    if args.model == "PPO":
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
    elif args.model == "DQN":
        model = DQN(
            "MlpPolicy", 
            train_env, 
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            device=ppo_cfg.get('device', "cpu")
        )
    elif args.model == "A2C":
        model = A2C(
            "MlpPolicy", 
            train_env, 
            verbose=1,
            learning_rate=7e-4,
            n_steps=2048,
            gamma=0.99,
            device=ppo_cfg.get('device', "cpu")
        )
    
    # 3. Training Loop
    # Hızlı test ve uyumluluk için 20 bin step (v2 başlangıcı için)
    total_timesteps = 20000 
    print(f"[TRAIN] Starting Training v2 ({total_timesteps} steps)...")
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - start_time

    # 4. Save the model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.model.lower()}_offloading_agent_v2")
    print(f"[TRAIN] Model saved to models/{args.model.lower()}_offloading_agent_v2.zip")
    
    # 5. Experiment Logging to CSV
    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/master_experiments.csv"
    
    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": seed,
        "config_model_type": f"{args.model}_v2_Retrained",
        "config_semantic_mode": "action_prior",
        "config_total_tasks": total_timesteps,
        "training_time_sec": round(training_time, 2)
    }
    
    df = pd.DataFrame([log_entry])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
    print(f"[TRAIN] Logged training run to {csv_path}")

if __name__ == "__main__":
    train()

