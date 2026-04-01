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

from stable_baselines3.common.callbacks import BaseCallback

class SimpleLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(SimpleLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"[PROGRESS] Training: Step {self.num_timesteps} / {self.locals['total_timesteps']}")
        return True

def train():
    parser = argparse.ArgumentParser(description="Train Baseline RL Agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "a2c"],
        help="RL algorithm to train"
    )
    parser.add_argument("--steps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--save_path", type=str, default=None, help="Custom path to save model")
    args = parser.parse_args()
    algorithm = args.algorithm.lower()
    
    print(f"[TRAIN] Initializing Training Environment...")
    print(f"[TRAIN] Algorithm: {algorithm.upper()}")
    print(f"[TRAIN] Total Timesteps: {args.steps}")
    
    # ... (Config and Env setup remains same) ...
    cfg = load_config()
    exp_cfg = cfg.get('experiment', {})
    ppo_cfg = cfg.get('ppo', {})
    env_cfg = cfg.get('env', {})
    
    # (Env setup block)
    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim) 
    edge_servers = [EdgeServer(env_sim, i+1, (random.uniform(0, 1000), random.uniform(0, 1000)), 2e9) for i in range(3)]
    devices = [IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud, battery_capacity=10000.0) for i in range(5)]

    
    train_env = OffloadingEnv(devices=devices, edge_servers=edge_servers, cloud_server=cloud, channel=channel)
    set_seed(42, env=train_env)

    # 2. Setup RL Agent
    if algorithm == "ppo":
        print(f"[TRAIN] Instantiating PPO Agent (MLP, n_steps=512)...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cpu"
        )
        default_save_path = "models/ppo_offloading_agent_v2"
        run_name = "PPO_v2_Retrained"
    elif algorithm == "dqn":
        print(f"[TRAIN] Instantiating DQN Agent...")
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500,
            device="cpu"
        )
        default_save_path = "models/dqn_offloading_agent_v2"
        run_name = "DQN_v2"
    else:
        print(f"[TRAIN] Instantiating A2C Agent...")
        model = A2C(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            device="cpu"
        )
        default_save_path = "models/a2c_offloading_agent_v2"
        run_name = "A2C_v2"
    
    # 3. Training Loop with Logging Callback
    print(f"[TRAIN] Starting Training ({args.steps} steps)...")
    logging_callback = SimpleLoggingCallback(check_freq=5000)
    
    start_time = time.time()
    model.learn(total_timesteps=args.steps, callback=logging_callback, progress_bar=False) 
    training_time = time.time() - start_time

    # 4. Save and Log
    os.makedirs("models", exist_ok=True)
    save_path = args.save_path if args.save_path else default_save_path
    model.save(save_path)
    print(f"[TRAIN] Model saved to {save_path}.zip")


    
    # 5. Experiment Logging to CSV
    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/master_experiments.csv"
    
    # 5. Experiment Evaluation (Automatic Performance Logging)
    print(f"[TRAIN] Running final evaluation for the new model...")
    from src.core.evaluation import evaluate_policy
    evaluate_policy(
        train_env, 
        model, 
        num_episodes=10, 
        run_name=run_name,
        semantic_mode="action_prior"
    )
    
    print(f"[TRAIN] Training and Evaluation completed successfully.")


if __name__ == "__main__":
    train()

