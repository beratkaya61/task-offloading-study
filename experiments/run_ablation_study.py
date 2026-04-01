#!/usr/bin/env python3
"""
Phase 5 ablation entrypoint.

Supports two modes:
- evaluation: evaluate an existing PPO checkpoint family on ablated environments
- retrain: train each ablation variant from scratch across multiple seeds
"""

import argparse
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
from training.train_agent import train_single_agent


def load_ablation_config(config_path="configs/ablation.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_disable_flags(enabled_features):
    flags = enabled_features or {}
    return {
        "disable_semantics": not flags.get("semantics", True),
        "disable_reward_shaping": not flags.get("reward_shaping", True),
        "disable_semantic_prior": not flags.get("semantic_prior", True),
        "disable_confidence_weighting": not flags.get("confidence_weighting", True),
        "disable_partial_offloading": not flags.get("partial_offloading", True),
        "disable_battery_awareness": not flags.get("battery_awareness", True),
        "disable_queue_awareness": not flags.get("queue_awareness", True),
        "disable_mobility_features": not flags.get("mobility_features", True),
    }


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

    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        max_steps=max_steps,
        **to_disable_flags(ablation_spec.get("enabled_features", {})),
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def run_evaluation_mode(config):
    ablation_specs = config["ablation_studies"]
    experiment_cfg = config["experiment"]
    num_episodes = experiment_cfg["num_episodes_per_ablation"]
    seeds = experiment_cfg.get("seeds", [42, 43, 44])
    max_steps = experiment_cfg.get("max_steps", 50)
    model_path_template = experiment_cfg.get(
        "evaluation_model_path_template",
        "models/ppo/phase5_baseline_retraining/seed{seed}.zip",
    )
    batch_id = datetime.now().strftime("ablation_%Y%m%d_%H%M%S")

    print("=" * 80)
    print("PHASE 5 ABLATION EVALUATION")
    print("=" * 80)
    print(f"[INFO] Variants: {len(ablation_specs)}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Episodes per variant: {num_episodes}")
    print(f"[INFO] Model source: {model_path_template}")
    print()

    sample_model_path = (
        model_path_template.format(seed=seeds[0]) if "{seed}" in model_path_template else model_path_template
    )
    if not os.path.exists(sample_model_path):
        raise FileNotFoundError(sample_model_path)

    for ablation_name, ablation_spec in ablation_specs.items():
        print(f"[VARIANT] {ablation_name}")
        for seed in seeds:
            model_path = model_path_template.format(seed=seed) if "{seed}" in model_path_template else model_path_template
            eval_env = create_env_with_ablation(ablation_spec, seed=seed, max_steps=max_steps)
            policy = PPO.load(model_path, env=eval_env)
            evaluate_policy(
                eval_env,
                policy,
                num_episodes=num_episodes,
                run_name=ablation_name,
                semantic_mode="action_prior",
                config_seed=seed,
                extra_fields={
                    "config_batch_id": batch_id,
                    "config_eval_group": "ablation_evaluation",
                },
            )

    summarize_logs(results_dir="results/raw", output_table="results/tables/offloading_experiment_report.md")
    print("[INFO] Canonical report refreshed: results/tables/offloading_experiment_report.md")


def run_retraining_mode(config):
    ablation_specs = config["ablation_studies"]
    retrain_cfg = config.get("retraining", {})
    seeds = retrain_cfg.get("seeds", [42, 43, 44])
    total_timesteps = retrain_cfg.get("total_timesteps", 30000)
    eval_episodes = retrain_cfg.get("eval_episodes", 10)
    base_config = retrain_cfg.get("base_config", "configs/synthetic_rl_training.yaml")
    model_root = retrain_cfg.get("model_root", "models/ppo/phase5_ablation_retraining")
    batch_id = datetime.now().strftime("ablation_retrain_%Y%m%d_%H%M%S")

    print("=" * 80)
    print("PHASE 5 ABLATION RETRAINING")
    print("=" * 80)
    print(f"[INFO] Variants: {len(ablation_specs)}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Total runs: {len(ablation_specs) * len(seeds)}")
    print(f"[INFO] Model root: {model_root}")
    print()

    for ablation_name, ablation_spec in ablation_specs.items():
        env_kwargs = to_disable_flags(ablation_spec.get("enabled_features", {}))
        variant_dir = os.path.join(model_root, ablation_name)
        os.makedirs(variant_dir, exist_ok=True)
        for seed in seeds:
            save_path = os.path.join(variant_dir, f"seed{seed}")
            print(f"[RUN] variant={ablation_name} seed={seed}")
            train_single_agent(
                algorithm="ppo",
                total_timesteps=total_timesteps,
                seed=seed,
                save_path=save_path,
                run_name=ablation_name,
                config_path=base_config,
                eval_episodes=eval_episodes,
                env_kwargs=env_kwargs,
                extra_eval_fields={
                    "config_batch_id": batch_id,
                    "config_eval_group": "phase5_ablation_retraining",
                },
            )

    summarize_logs(results_dir="results/raw", output_table="results/tables/offloading_experiment_report.md")
    print("[INFO] Canonical report refreshed: results/tables/offloading_experiment_report.md")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 ablation workflows")
    parser.add_argument(
        "--mode",
        choices=["evaluation", "retrain"],
        default=None,
        help="evaluation: test an existing PPO family, retrain: train each ablation from scratch",
    )
    parser.add_argument("--config", default="configs/ablation.yaml", help="Ablation config path")
    args = parser.parse_args()

    config = load_ablation_config(args.config)
    mode = args.mode or config.get("experiment", {}).get("mode", "evaluation")

    if mode == "retrain":
        run_retraining_mode(config)
    else:
        run_evaluation_mode(config)


if __name__ == "__main__":
    main()
