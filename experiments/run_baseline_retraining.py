#!/usr/bin/env python3
"""
Phase 5 baseline retraining on the synthetic environment.

Each baseline RL algorithm is retrained from scratch for multiple seeds.
Checkpoints are stored under agent-specific folders in `models/`.
"""

import os
from datetime import datetime

import yaml

from src.core.evaluation import summarize_logs
from src.training.train_agent import train_single_agent


RUN_LABELS = {
    "ppo": "PPO_v2",
    "dqn": "DQN_v2",
    "a2c": "A2C_v2",
}


def load_config(config_path="configs/phase5_baseline_retraining.yaml"):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_baseline_retraining(config_path="configs/phase5_baseline_retraining.yaml"):
    config = load_config(config_path)
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    algorithms = training_cfg.get("algorithms", ["ppo", "dqn", "a2c"])
    seeds = training_cfg.get("seeds", [42, 43, 44])
    total_timesteps = training_cfg.get("total_timesteps", 30000)
    eval_episodes = training_cfg.get("eval_episodes", 10)
    train_config_path = training_cfg.get("base_config", "configs/synthetic_rl_training.yaml")
    model_root = output_cfg.get("model_root", "models")
    report_path = output_cfg.get("report_path", "results/tables/offloading_experiment_report.md")
    batch_id = datetime.now().strftime("baseline_retrain_%Y%m%d_%H%M%S")

    print("=" * 80)
    print("PHASE 5 BASELINE RETRAINING")
    print("=" * 80)
    print(f"[INFO] Algorithms: {algorithms}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Total runs: {len(algorithms) * len(seeds)}")
    print(f"[INFO] Model root: {model_root}")
    print()

    for algorithm in algorithms:
        algorithm_dir = os.path.join(model_root, algorithm, "phase5_baseline_retraining")
        os.makedirs(algorithm_dir, exist_ok=True)
        for seed in seeds:
            save_path = os.path.join(algorithm_dir, f"seed{seed}")
            print(f"[RUN] algorithm={algorithm} seed={seed}")
            train_single_agent(
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                save_path=save_path,
                run_name=RUN_LABELS.get(algorithm, algorithm.upper()),
                config_path=train_config_path,
                eval_episodes=eval_episodes,
                extra_eval_fields={
                    "config_batch_id": batch_id,
                    "config_eval_group": "phase5_baseline_retraining",
                },
            )

    summarize_logs(results_dir="results/raw", output_table=report_path)

    print()
    print("=" * 80)
    print("PHASE 5 BASELINE RETRAINING COMPLETE")
    print("=" * 80)
    print(f"[INFO] Canonical report: {report_path}")


if __name__ == "__main__":
    run_baseline_retraining()
