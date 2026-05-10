#!/usr/bin/env python3
"""Faz 5'in sentetik policy evaluation deneylerinin real-data esdegeri."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.baselines import (
    CloudOnlyPolicy,
    EdgeOnlyPolicy,
    GeneticAlgorithmPolicy,
    GreedyLatencyPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
)
from core.evaluation import evaluate_policy
from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator
from src.core.config_adapters import build_trace_training_config
from src.core.experiment_artifacts import load_yaml, refresh_real_data_phase5_report


RL_MODEL_CLASSES = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}


def load_policy(policy_name: str, checkpoint_path: str | None, env):
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

    model_class = RL_MODEL_CLASSES[policy_name]
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(checkpoint_path or f"Missing checkpoint for {policy_name}")
    return model_class.load(checkpoint_path, env=env)


def resolve_rl_checkpoint(policy_name: str, config: dict) -> str | None:
    rl_cfg = config.get("rl_models", {})
    family = rl_cfg.get("checkpoint_family", "real_data_rl_retraining")
    training_seed = int(rl_cfg.get("training_seed", 42))
    family_map = rl_cfg.get("checkpoints", {})
    family_paths = family_map.get(family, {})
    template = family_paths.get(policy_name)
    return template.format(seed=training_seed) if template else None
def run_real_data_policy_evaluation(
    config_path: str = "configs/phase_5/real_data_policy_evaluation.yaml",
) -> list[dict]:
    config = load_yaml(config_path)
    eval_cfg = config.get("evaluation", {})
    output_cfg = config.get("output", {})
    backbone_cfg = config.get("trace_backbone", {})
    rl_model_cfg = config.get("rl_models", {})

    num_episodes = int(eval_cfg.get("num_episodes", 10))
    seeds = eval_cfg.get("seeds", [42, 43, 44])
    csv_path = Path(output_cfg["csv_path"])
    report_path = Path(output_cfg["report_path"])
    batch_id = datetime.now().strftime("real_data_policy_eval_%Y%m%d_%H%M%S")
    split_name = backbone_cfg.get("evaluation_split", "test")
    training_config_path = backbone_cfg.get("training_config", "configs/phase_5/real_data_rl_training.yaml")
    training_config = load_yaml(training_config_path)

    policy_names = config.get("policies", {}).get("heuristic", []) + config.get("policies", {}).get("rl", [])
    if csv_path.exists():
        csv_path.unlink()

    rows: list[dict] = []
    for seed in seeds:
        orchestrator_config = build_trace_training_config(
            training_config,
            algorithm="ppo",
            seed=int(seed),
            log_dir="results/tmp/policy_eval",
            checkpoint_dir="models/tmp/policy_eval",
            report_path=str(report_path),
            overrides=None,
        )
        orchestrator = TraceTrainingOrchestrator(
            config_path=training_config_path,
            seed=int(seed),
            config_dict=orchestrator_config,
        )
        train_eps, val_eps, test_eps = orchestrator.prepare_traces()
        split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
        env = orchestrator._build_trace_env(split_map[split_name])

        for policy_name in policy_names:
            checkpoint_path = resolve_rl_checkpoint(policy_name, config)
            policy = load_policy(policy_name, checkpoint_path, env)
            row = evaluate_policy(
                env,
                policy,
                num_episodes=num_episodes,
                run_name=policy_name,
                semantic_mode="action_prior",
                config_seed=seed,
                csv_path=str(csv_path),
                extra_fields={
                    "config_batch_id": batch_id,
                    "config_eval_group": "real_data_policy_evaluation",
                },
            )
            rows.append(row)

    refresh_real_data_phase5_report(report_path)
    return rows


if __name__ == "__main__":
    run_real_data_policy_evaluation()
