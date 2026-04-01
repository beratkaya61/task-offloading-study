#!/usr/bin/env python3
"""
Faz 5: Sistematik Ablation Study
9 farkli ablation senaryosu uzerinde PPO_v2 politikasini coklu seed ile degerlendir.
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import simpy
import yaml
from stable_baselines3 import PPO

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.append(os.path.join(os.getcwd(), "src"))

from core.evaluation import evaluate_policy, summarize_logs
from env.rl_env import OffloadingEnv
from env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel


ABLATION_MODELS = [
    "full_model",
    "w_o_semantics",
    "w_o_reward_shaping",
    "w_o_semantic_prior",
    "w_o_confidence",
    "w_o_partial_offloading",
    "w_o_battery_awareness",
    "w_o_queue_awareness",
    "w_o_mobility_features",
]


def load_ablation_config(config_path="configs/ablation.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_env_with_ablation(ablation_spec, seed, max_steps=50):
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

    flags = ablation_spec.get("enabled_features", {})
    disable_flags = {
        "disable_semantics": not flags.get("semantics", True),
        "disable_reward_shaping": not flags.get("reward_shaping", True),
        "disable_semantic_prior": not flags.get("semantic_prior", True),
        "disable_confidence_weighting": not flags.get("confidence_weighting", True),
        "disable_partial_offloading": not flags.get("partial_offloading", True),
        "disable_battery_awareness": not flags.get("battery_awareness", True),
        "disable_queue_awareness": not flags.get("queue_awareness", True),
        "disable_mobility_features": not flags.get("mobility_features", True),
    }

    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        max_steps=max_steps,
        **disable_flags,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def run_ablation_study():
    print("=" * 80)
    print("FAZ 5: SISTEMATIK ABLATION STUDY")
    print("=" * 80)
    print(f"Baslama zamani: {datetime.now().isoformat()}")
    print()

    config = load_ablation_config()
    ablation_specs = config["ablation_studies"]
    experiment_cfg = config["experiment"]
    num_episodes = experiment_cfg["num_episodes_per_ablation"]
    seeds = experiment_cfg.get("seeds", [42, 43, 44])
    max_steps = experiment_cfg.get("max_steps", 50)
    batch_id = datetime.now().strftime("ablation_%Y%m%d_%H%M%S")

    print(f"[INFO] {len(ablation_specs)} ablation senaryosu calistirilacak")
    print(f"[INFO] Her ablation: {num_episodes} episode")
    print(f"[INFO] Seedler: {seeds}")
    print(f"[INFO] Toplam run: {len(ablation_specs) * len(seeds)}")
    print()

    ppo_model_path = "models/ppo_offloading_agent_v2.zip"
    if not os.path.exists(ppo_model_path):
        print(f"[ERROR] PPO modeli bulunamadi: {ppo_model_path}")
        return

    print(f"[INFO] PPO modeli yukleniyor: {ppo_model_path}")

    for idx, (ablation_name, ablation_spec) in enumerate(ablation_specs.items(), 1):
        print()
        print(f"[{idx}/{len(ablation_specs)}] Ablation baslatiliyor: {ablation_name}")
        print(f"    Tanim: {ablation_spec['description']}")
        print(f"    Beklenen dusus: {ablation_spec.get('expected_drop', 'Baseline')}")

        for seed in seeds:
            try:
                eval_env = create_env_with_ablation(ablation_spec, seed=seed, max_steps=max_steps)
                policy = PPO.load(ppo_model_path, env=eval_env)
                print(f"    -> seed={seed}, {num_episodes} episode")
                evaluate_policy(
                    eval_env,
                    policy,
                    num_episodes=num_episodes,
                    run_name=ablation_name,
                    semantic_mode="action_prior",
                    config_seed=seed,
                    extra_fields={
                        "config_batch_id": batch_id,
                        "config_eval_group": "ablation_multiseed",
                    },
                )
            except Exception as e:
                print(f"    [ERROR] {ablation_name} degerlendirilirken (seed={seed}): {e}")
                import traceback

                traceback.print_exc()

    print()
    print("=" * 80)
    print("ABLATION STUDY TAMAMLANDI")
    print("=" * 80)
    summarize_logs(results_dir="results/raw", output_table="results/tables/offloading_experiment_report.md")
    print()
    print(f"Bitis zamani: {datetime.now().isoformat()}")
    print("Sonuclar: results/tables/offloading_experiment_report.md")


if __name__ == "__main__":
    run_ablation_study()
