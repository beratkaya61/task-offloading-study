import os
import random
import simpy
from stable_baselines3 import PPO
import torch
from rl_env import OffloadingEnv
from simulation_env import WirelessChannel, EdgeServer, CloudServer, IoTDevice, Task, TaskType, LLM_ANALYZER

def train():
    print("[TRAIN] Initializing Training Environment...")
    
    # 1. Setup Simulation Components (Mock for training)
    env = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env) # CloudServer only takes 'env'
    edge_servers = [
        EdgeServer(env, 1, (200, 200), 2e9),
        EdgeServer(env, 2, (800, 200), 2e9),
        EdgeServer(env, 3, (500, 800), 2e9)
    ]
    
    # Create the custom Gym environment
    train_env = OffloadingEnv(
        devices=[], # Will be populated per step/reset
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel
    )



    # Bir Çatışmaya Dikkat!
    # Burada ilginç bir durum var:

    # LLM (TinyLlama): GPU'da (cuda) çalışması çok daha iyidir (yoksa çok yavaş analiz yapar).

    # RL Agent (PPO): SB3'e göre CPU'da çalışması daha verimlidir.

    # 2. Setup PPO Agent
    print("[TRAIN] Instantiating PPO Agent (MLP Policy)...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device= "cpu"
    )
    
    # 3. Training Loop (Single call for better PPO stability)
    print("[TRAIN] Starting Training (100,000 steps)...")
    
    # We use a large total_timesteps for better convergence
    model.learn(total_timesteps=100000, progress_bar=True)

    # 4. Save the model (Save inside src/models for unified pathing)
    os.makedirs("src/models", exist_ok=True)
    model.save("src/models/ppo_offloading_agent")
    print("[TRAIN] Model saved to src/models/ppo_offloading_agent.zip")

if __name__ == "__main__":
    train()
