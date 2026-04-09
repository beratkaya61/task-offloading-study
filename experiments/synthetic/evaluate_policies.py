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


RL_MODEL_CLASSES = {
    "PPO_v2": PPO,
    "DQN_v2": DQN,
    "A2C_v2": A2C,
}


def load_policy_evaluation_config(config_path="configs/synthetic/policy_evaluation.yaml"):
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


def resolve_rl_checkpoint(policy_name, config):
    rl_cfg = config.get("rl_models", {})
    family = rl_cfg.get("checkpoint_family", "synthetic_rl_retraining")
    training_seed = rl_cfg.get("training_seed", 42)
    family_map = rl_cfg.get("checkpoints", {})
    family_paths = family_map.get(family, {})
    if policy_name not in family_paths:
        raise KeyError(f"{policy_name} not configured under checkpoint family: {family}")
    return family_paths[policy_name].format(seed=training_seed)


def load_policy(policy_name, eval_env, config):
    heuristic_policies = {
        "LocalOnly": LocalOnlyPolicy,
        "EdgeOnly": EdgeOnlyPolicy,
        "CloudOnly": CloudOnlyPolicy,
        "Random": RandomPolicy,
        "GreedyLatency": GreedyLatencyPolicy,
        "GeneticAlgorithm": lambda: GeneticAlgorithmPolicy(population_size=10, generations=5),
    }

    if policy_name in heuristic_policies:
        ctor = heuristic_policies[policy_name]
        return ctor() if callable(ctor) else ctor

    if policy_name in RL_MODEL_CLASSES:
        path = resolve_rl_checkpoint(policy_name, config)
        model_class = RL_MODEL_CLASSES[policy_name]
        if os.path.exists(path):
            print(f"[INIT] {policy_name} modeli yukleniyor: {path}")
            return model_class.load(path, env=eval_env)
        raise FileNotFoundError(path)

    raise KeyError(policy_name)


def run_policy_evaluation():
    print("--------------------------------------------------")
    print("   IOT TASK OFFLOADING SYNTHETIC POLICY EVALUATION")
    print("--------------------------------------------------")

    evaluation_cfg_file = load_policy_evaluation_config()
    evaluation_cfg = evaluation_cfg_file.get("evaluation", {})
    output_cfg = evaluation_cfg_file.get("output", {})
    num_episodes = evaluation_cfg.get("num_episodes", 10)
    seeds = evaluation_cfg.get("seeds", [42, 43, 44])
    max_steps = evaluation_cfg.get("max_steps", 50)
    csv_path = output_cfg.get("csv_path", "results/raw/synthetic_policy_evaluation.csv")
    batch_id = datetime.now().strftime("policy_eval_%Y%m%d_%H%M%S")
    policy_names = evaluation_cfg_file.get("policies", {}).get("heuristic", []) + evaluation_cfg_file.get("policies", {}).get("rl", [])

    for seed in seeds:
        print(f"\n[SEED] Evaluating seed={seed}")
        for policy_name in policy_names:
            try:
                eval_env = make_env(seed=seed, max_steps=max_steps)
                policy = load_policy(policy_name, eval_env, evaluation_cfg_file)
                evaluate_policy(
                    eval_env,
                    policy,
                    num_episodes=num_episodes,
                    run_name=policy_name,
                    semantic_mode="action_prior",
                    config_seed=seed,
                    csv_path=csv_path,
                    extra_fields={
                        "config_batch_id": batch_id,
                        "config_eval_group": "synthetic_policy_evaluation",
                    },
                )
            except Exception as e:
                print(f"[ERROR] {policy_name} degerlendirilirken hata olustu (seed={seed}): {e}")

    summarize_logs()
    print("\n[FINISH] Tum synthetic policies multi-seed olarak degerlendirildi.")
    print("[INFO] Ana rapor: v2_docs/phase_5/offloading_experiment_report.md")


if __name__ == "__main__":
    run_policy_evaluation()

