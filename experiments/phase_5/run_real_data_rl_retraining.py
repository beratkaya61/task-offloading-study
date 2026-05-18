#!/usr/bin/env python3
"""Real-data counterpart of the synthetic Phase 5 RL retraining experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import A2C, DQN, PPO

from core.evaluation import evaluate_policy
from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator
from src.core.config_adapters import build_trace_training_config
from src.core.experiment_artifacts import load_yaml, refresh_real_data_phase5_report


RL_MODEL_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

RUN_LABELS = {
    "ppo": "PPO",
    "dqn": "DQN",
    "a2c": "A2C",
}


def run_real_data_rl_retraining(
    config_path: str = "configs/phase_5/real_data_rl_retraining.yaml",
    algorithm_override: str | None = None,
) -> list[dict]:
    config = load_yaml(config_path)
    training_cfg = config["training"]
    base_config_path = training_cfg["base_config"]
    base_config = load_yaml(base_config_path)

    algorithms = training_cfg.get("algorithms", ["ppo", "dqn", "a2c"])
    if algorithm_override:
        algorithms = [algorithm_override]
    seeds = training_cfg.get("seeds", [42, 43, 44])
    force_retrain = bool(training_cfg.get("force_retrain", False))
    overrides = {
        "max_episodes": int(training_cfg.get("max_episodes", base_config.get("experiment", {}).get("max_episodes", 500))),
        "episodes_per_eval": int(training_cfg.get("eval_episodes", base_config.get("experiment", {}).get("episodes_per_eval", 10))),
        "n_steps": int(training_cfg.get("n_steps", 1024)),
        "batch_size": int(training_cfg.get("batch_size", 64)),
        "n_epochs": int(training_cfg.get("n_epochs", 10)),
    }

    output_cfg = config.get("output", {})
    model_root = output_cfg.get("model_root", "models")
    csv_path = Path(output_cfg.get("csv_path", "results/phase_5/metrics/real_data/rl_retraining/real_data_rl_retraining.csv"))
    report_path = Path(output_cfg["report_path"])
    batch_id = datetime.now().strftime("real_data_rl_retraining_%Y%m%d_%H%M%S")

    if csv_path.exists():
        csv_path.unlink()

    all_rows: list[dict] = []
    for algorithm in algorithms:
        warmup_config = build_trace_training_config(
            base_config,
            algorithm=algorithm,
            seed=int(seeds[0]),
            log_dir="results/tmp/warmup",
            checkpoint_dir="models/tmp/warmup",
            report_path=str(report_path),
            overrides=overrides,
        )
        warmup = TraceTrainingOrchestrator(config_path=base_config_path, seed=int(seeds[0]), config_dict=warmup_config)
        train_eps, _, test_eps = warmup.prepare_traces()

        model_class = RL_MODEL_CLASSES[algorithm]
        for seed in seeds:
            checkpoint_dir = f"{model_root}/{algorithm}/real_data_rl_retraining/seed{seed}"
            log_dir = f"{checkpoint_dir}/training"
            run_config = build_trace_training_config(
                base_config,
                algorithm=algorithm,
                seed=int(seed),
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                report_path=str(report_path),
                overrides=overrides,
            )
            orchestrator = TraceTrainingOrchestrator(config_path=base_config_path, seed=int(seed), config_dict=run_config)

            checkpoint_name = f"{algorithm}_retraining_best.zip"
            checkpoint_path = Path(orchestrator.checkpoint_dir) / checkpoint_name
            metrics_name = "training_metrics.csv"

            if force_retrain or not checkpoint_path.exists():
                orchestrator.train_model(
                    train_eps,
                    checkpoint_name=checkpoint_name,
                    metrics_name=metrics_name,
                )

            test_env = orchestrator._build_trace_env(test_eps)
            model = model_class.load(str(checkpoint_path), env=test_env)
            row = evaluate_policy(
                test_env,
                model,
                num_episodes=int(training_cfg.get("eval_episodes", 10)),
                run_name=RUN_LABELS.get(algorithm, algorithm.upper()),
                semantic_mode="action_prior",
                config_seed=int(seed),
                csv_path=str(csv_path),
                extra_fields={
                    "config_batch_id": batch_id,
                    "config_eval_group": "real_data_rl_retraining",
                },
            )
            all_rows.append(row)

    refresh_real_data_phase5_report(report_path)
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 real-data RL retraining")
    parser.add_argument("--config", default="configs/phase_5/real_data_rl_retraining.yaml")
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "a2c"], default=None)
    args = parser.parse_args()
    run_real_data_rl_retraining(config_path=args.config, algorithm_override=args.algorithm)
