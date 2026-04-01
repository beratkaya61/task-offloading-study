import os
import random
import sys
from datetime import datetime

import numpy as np
import simpy
import yaml
from stable_baselines3 import A2C, DQN, PPO

sys.path.append(os.path.join(os.getcwd(), "src"))

from agents.baselines import (
    CloudOnlyPolicy,
    EdgeOnlyPolicy,
    GeneticAlgorithmPolicy,
    GreedyLatencyPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
)
from core.evaluation import evaluate_policy, summarize_logs
from env.rl_env import OffloadingEnv
from env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel


def load_baseline_config(config_path="configs/baselines.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_env(seed, max_steps=50):
    random.seed(seed)
    np.random.seed(seed)

    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(env_sim, 1, (200, 200), 2.5e9),
        EdgeServer(env_sim, 2, (800, 200), 2.0e9),
        EdgeServer(env_sim, 3, (500, 800), 2.2e9),
    ]
    devices = [
        IoTDevice(
            env_sim,
            id=i,
            channel=channel,
            edge_servers=edge_servers,
            cloud_server=cloud,
            battery_capacity=10000.0,
        )
        for i in range(5)
    ]

    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        max_steps=max_steps,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def load_policy(policy_name, eval_env):
    heuristic_policies = {
        "LocalOnly": LocalOnlyPolicy,
        "EdgeOnly": EdgeOnlyPolicy,
        "CloudOnly": CloudOnlyPolicy,
        "Random": RandomPolicy,
        "GreedyLatency": GreedyLatencyPolicy,
        "GeneticAlgorithm": lambda: GeneticAlgorithmPolicy(population_size=10, generations=5),
    }
    rl_models = {
        "PPO_v2": ("models/ppo_offloading_agent_v2.zip", PPO),
        "DQN_v2": ("models/dqn_offloading_agent_v2.zip", DQN),
        "A2C_v2": ("models/a2c_offloading_agent_v2.zip", A2C),
    }

    if policy_name in heuristic_policies:
        ctor = heuristic_policies[policy_name]
        return ctor() if callable(ctor) else ctor

    if policy_name in rl_models:
        path, model_class = rl_models[policy_name]
        if os.path.exists(path):
            print(f"[INIT] {policy_name} modeli yukleniyor: {path}")
            return model_class.load(path, env=eval_env)
        raise FileNotFoundError(path)

    raise KeyError(policy_name)


def run_all_baselines():
    print("--------------------------------------------------")
    print("   IOT TASK OFFLOADING BASELINE EVALUATION        ")
    print("--------------------------------------------------")

    baseline_cfg = load_baseline_config()
    evaluation_cfg = baseline_cfg.get("evaluation", {})
    num_episodes = evaluation_cfg.get("num_episodes", 10)
    seeds = evaluation_cfg.get("seeds", [42, 43, 44])
    max_steps = evaluation_cfg.get("max_steps", 50)
    batch_id = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")

    policy_names = [
        "LocalOnly",
        "EdgeOnly",
        "CloudOnly",
        "Random",
        "GreedyLatency",
        "GeneticAlgorithm",
        "PPO_v2",
        "DQN_v2",
        "A2C_v2",
    ]

    for seed in seeds:
        print(f"\n[SEED] Evaluating seed={seed}")
        for policy_name in policy_names:
            try:
                eval_env = make_env(seed=seed, max_steps=max_steps)
                policy = load_policy(policy_name, eval_env)
                result = evaluate_policy(
                    eval_env,
                    policy,
                    num_episodes=num_episodes,
                    run_name=policy_name,
                    semantic_mode="action_prior",
                    config_seed=seed,
                    extra_fields={
                        "config_batch_id": batch_id,
                        "config_eval_group": "baseline_multiseed",
                    },
                )
            except Exception as e:
                print(f"[ERROR] {policy_name} degerlendirilirken hata olustu (seed={seed}): {e}")

    summarize_logs()
    print("\n[FINISH] Tum baselines multi-seed olarak degerlendirildi.")
    print("[INFO] Ana rapor: results/tables/offloading_experiment_report.md")


if __name__ == "__main__":
    run_all_baselines()
