#!/usr/bin/env python3
"""Faz 5 real-data ablation study entrypoint."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config_adapters import build_trace_training_config
from src.core.experiment_artifacts import (
    refresh_real_data_phase5_report,
    load_yaml,
    read_last_train_success,
    write_csv,
)
from src.core.evaluation import EXPERIMENT_LOG_COLUMNS
from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator


def resolve_scope_label(seeds, mode):
    seed_scope = "single_seed" if len(seeds) == 1 else "multi_seed"
    return f"{seed_scope}_{mode}"


def resolve_ablation_csv_path(config, algorithm, scope):
    template = config["output"].get(
        "csv_path_template",
        "results/phase_5/metrics/real_data/ablation/real_data_ablation_{algorithm}_{scope}.csv",
    )
    return Path(template.format(algorithm=algorithm, scope=scope))


def resolve_ablation_checkpoint_path(config: dict, algorithm: str, variant_name: str, seed: int) -> Path:
    experiment_cfg = config.get("experiment", {})
    retrain_cfg = config.get("retraining", {})
    template = experiment_cfg.get(
        "evaluation_model_path_template",
        retrain_cfg.get(
            "evaluation_model_path_template",
            "models/{algorithm}/real_data_ablation/{variant}/seed{seed}/{algorithm}_ablation_best.zip",
        ),
    )
    return Path(template.format(algorithm=algorithm, variant=variant_name, seed=seed))


def evaluate_trace_checkpoint_phase5(
    orchestrator: TraceTrainingOrchestrator,
    episodes,
    checkpoint_path: str,
    run_name: str,
    config_seed: int,
    feature_overrides: dict | None,
    batch_id: str,
    eval_group: str,
) -> dict:
    env = orchestrator._build_trace_env(episodes, feature_overrides=feature_overrides)
    model = orchestrator._load_model(checkpoint_path, env=env)

    results = []
    action_counts = {index: 0 for index in range(6)}

    for trace_ep in episodes:
        obs, _ = env.reset(options={"episode_tasks": trace_ep.tasks} if False else None)
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_latencies = []
        episode_energies = []
        episode_successes = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if hasattr(action, "item"):
                action = int(action.item())
            else:
                action = int(action)
            action_counts[action] = action_counts.get(action, 0) + 1

            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward
            step_count += 1
            episode_latencies.append(info.get("delay", 0.0))
            episode_energies.append(info.get("energy", 0.0))
            if info.get("task_success", False):
                episode_successes += 1

        p95_latency = float(pd.Series(episode_latencies).quantile(0.95)) if episode_latencies else 0.0
        avg_energy = float(sum(episode_energies) / len(episode_energies)) if episode_energies else 0.0
        success_rate = episode_successes / max(1, step_count)
        qoe = 100.0 * success_rate - (p95_latency * 5.0)

        results.append(
            {
                "reward": float(episode_reward),
                "steps": step_count,
                "success_rate": success_rate,
                "p95_latency": p95_latency,
                "avg_energy": avg_energy,
                "qoe": float(qoe),
            }
        )

    avg_reward = float(sum(row["reward"] for row in results) / len(results)) if results else 0.0
    avg_success = float(sum(row["success_rate"] for row in results) / len(results)) if results else 0.0
    avg_p95_latency = float(sum(row["p95_latency"] for row in results) / len(results)) if results else 0.0
    avg_energy = float(sum(row["avg_energy"] for row in results) / len(results)) if results else 0.0
    avg_qoe = float(sum(row["qoe"] for row in results) / len(results)) if results else 0.0
    total_tasks = int(sum(row["steps"] for row in results))
    total_actions = max(1, sum(action_counts.values()))
    action_rates = {
        f"metric_action_{index}_rate": round(action_counts.get(index, 0) / total_actions, 4)
        for index in range(6)
    }
    unique_actions = sum(1 for count in action_counts.values() if count > 0)
    dominant_action = max(action_counts, key=action_counts.get) if action_counts else -1

    log_values = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": int(config_seed),
        "config_model_type": run_name,
        "config_semantic_mode": "action_prior",
        "config_total_tasks": total_tasks,
        "metric_success_rate": round(avg_success, 4),
        "metric_avg_reward": round(avg_reward, 2),
        "metric_p95_latency": round(avg_p95_latency, 4),
        "metric_avg_energy": round(avg_energy, 4),
        "metric_qoe": round(avg_qoe, 2),
        "metric_unique_actions": unique_actions,
        "metric_dominant_action": dominant_action,
        "config_batch_id": batch_id,
        "config_eval_group": eval_group,
    }
    log_values.update(action_rates)
    return {column: log_values[column] for column in EXPERIMENT_LOG_COLUMNS}


def _prepare_trace_splits(base_config_path: str, base_config: dict, algorithm: str, seed: int, report_path: Path, overrides: dict):
    warmup_config = build_trace_training_config(
        base_config,
        algorithm=algorithm,
        seed=int(seed),
        log_dir="results/tmp/warmup",
        checkpoint_dir="models/tmp/warmup",
        report_path=str(report_path),
        overrides=overrides,
    )
    warmup = TraceTrainingOrchestrator(config_path=base_config_path, seed=int(seed), config_dict=warmup_config)
    return warmup.prepare_traces()


def run_retraining_mode(config: dict) -> list[dict]:
    retrain_cfg = config["retraining"]
    base_config_path = retrain_cfg["base_config"]
    base_config = load_yaml(base_config_path)
    algorithms = retrain_cfg.get("algorithms", ["ppo", "dqn", "a2c"])
    seeds = retrain_cfg.get("seeds", [42, 43, 44])
    scope = resolve_scope_label(seeds, "retraining")
    overrides = {
        "max_episodes": int(retrain_cfg.get("max_episodes", base_config.get("experiment", {}).get("max_episodes", 500))),
        "episodes_per_eval": int(retrain_cfg.get("eval_episodes", base_config.get("experiment", {}).get("episodes_per_eval", 10))),
        "n_steps": int(retrain_cfg.get("n_steps", 1024)),
        "batch_size": int(retrain_cfg.get("batch_size", 64)),
        "n_epochs": int(retrain_cfg.get("n_epochs", 10)),
    }
    ablation_specs = config.get("ablation_studies", {})
    model_root_template = retrain_cfg.get("model_root", "models/{algorithm}/real_data_ablation")
    report_path = Path(config["output"]["table_path"])
    batch_id_template = datetime.now().strftime(f"real_data_ablation_{{algorithm}}_retrain_%Y%m%d_%H%M%S")

    all_rows: list[dict] = []
    for algorithm in algorithms:
        batch_id = batch_id_template.format(algorithm=algorithm)
        train_eps, val_eps, test_eps = _prepare_trace_splits(
            base_config_path, base_config, algorithm, int(seeds[0]), report_path, overrides
        )

        algorithm_rows: list[dict] = []
        for variant_name, variant_cfg in ablation_specs.items():
            feature_flags = variant_cfg.get("enabled_features", {})
            for seed in seeds:
                checkpoint_dir = model_root_template.format(algorithm=algorithm) + f"/{variant_name}/seed{seed}"
                log_dir = checkpoint_dir + "/training"
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

                checkpoint_name = f"{algorithm}_ablation_best.zip"
                checkpoint_path = Path(orchestrator.checkpoint_dir) / checkpoint_name
                metrics_name = "training_metrics.csv"
                metrics_path = Path(orchestrator.log_dir) / metrics_name

                if checkpoint_path.exists():
                    training_history = {"success_rate": [read_last_train_success(metrics_path)]}
                else:
                    training_history = orchestrator.train_model(
                        train_eps,
                        feature_overrides=feature_flags,
                        checkpoint_name=checkpoint_name,
                        metrics_name=metrics_name,
                    )

                orchestrator.evaluate_model(
                    val_eps,
                    str(checkpoint_path),
                    feature_overrides=feature_flags,
                )
                test_entry = evaluate_trace_checkpoint_phase5(
                    orchestrator,
                    test_eps,
                    str(checkpoint_path),
                    run_name=variant_name,
                    config_seed=int(seed),
                    feature_overrides=feature_flags,
                    batch_id=batch_id,
                    eval_group="real_data_ablation_retraining",
                )
                algorithm_rows.append(test_entry)
                all_rows.append(test_entry)

        csv_path = resolve_ablation_csv_path(config, algorithm, scope)
        write_csv(csv_path, algorithm_rows)

    refresh_real_data_phase5_report(report_path)
    return all_rows


def run_evaluation_mode(config: dict) -> list[dict]:
    experiment_cfg = config.get("experiment", {})
    retrain_cfg = config["retraining"]
    base_config_path = retrain_cfg["base_config"]
    base_config = load_yaml(base_config_path)
    algorithms = retrain_cfg.get("algorithms", ["ppo", "dqn", "a2c"])
    seeds = experiment_cfg.get("seeds", [42, 43, 44])
    scope = resolve_scope_label(seeds, "evaluation")
    split_name = experiment_cfg.get("evaluation_split", "test")
    overrides = {
        "max_episodes": int(retrain_cfg.get("max_episodes", base_config.get("experiment", {}).get("max_episodes", 500))),
        "episodes_per_eval": int(retrain_cfg.get("eval_episodes", base_config.get("experiment", {}).get("episodes_per_eval", 10))),
        "n_steps": int(retrain_cfg.get("n_steps", 1024)),
        "batch_size": int(retrain_cfg.get("batch_size", 64)),
        "n_epochs": int(retrain_cfg.get("n_epochs", 10)),
    }
    ablation_specs = config.get("ablation_studies", {})
    report_path = Path(config["output"]["table_path"])
    batch_id_template = datetime.now().strftime(f"real_data_ablation_{{algorithm}}_eval_%Y%m%d_%H%M%S")

    all_rows: list[dict] = []
    for algorithm in algorithms:
        batch_id = batch_id_template.format(algorithm=algorithm)
        train_eps, val_eps, test_eps = _prepare_trace_splits(
            base_config_path, base_config, algorithm, int(seeds[0]), report_path, overrides
        )
        split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
        target_episodes = split_map[split_name]

        algorithm_rows: list[dict] = []
        for variant_name, variant_cfg in ablation_specs.items():
            feature_flags = variant_cfg.get("enabled_features", {})
            for seed in seeds:
                run_config = build_trace_training_config(
                    base_config,
                    algorithm=algorithm,
                    seed=int(seed),
                    log_dir="results/tmp/real_data_ablation_eval",
                    checkpoint_dir="models/tmp/real_data_ablation_eval",
                    report_path=str(report_path),
                    overrides=overrides,
                )
                orchestrator = TraceTrainingOrchestrator(config_path=base_config_path, seed=int(seed), config_dict=run_config)
                checkpoint_path = resolve_ablation_checkpoint_path(
                    config,
                    algorithm=algorithm,
                    variant_name=variant_name,
                    seed=int(seed),
                )
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Ablation checkpoint not found: {checkpoint_path}")
                row = evaluate_trace_checkpoint_phase5(
                    orchestrator,
                    target_episodes,
                    str(checkpoint_path),
                    run_name=variant_name,
                    config_seed=int(seed),
                    feature_overrides=feature_flags,
                    batch_id=batch_id,
                    eval_group="real_data_ablation_evaluation",
                )
                algorithm_rows.append(row)
                all_rows.append(row)

        csv_path = resolve_ablation_csv_path(config, algorithm, scope)
        write_csv(csv_path, algorithm_rows)

    refresh_real_data_phase5_report(report_path)
    return all_rows


def run_real_data_ablation_study(
    config_path: str = "configs/phase_5/real_data_ablation.yaml",
    mode: str | None = None,
    algorithm_override: str | None = None,
) -> list[dict]:
    config = load_yaml(config_path)
    if algorithm_override:
        config.setdefault("experiment", {})["algorithm"] = algorithm_override
        config.setdefault("retraining", {})["algorithms"] = [algorithm_override]
    selected_mode = mode or config.get("experiment", {}).get("mode", "evaluation")
    if selected_mode == "retrain":
        return run_retraining_mode(config)
    return run_evaluation_mode(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 real-data ablation workflows")
    parser.add_argument("--config", default="configs/phase_5/real_data_ablation.yaml")
    parser.add_argument("--mode", choices=["evaluation", "retrain"], default=None)
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "a2c"], default=None)
    args = parser.parse_args()
    run_real_data_ablation_study(
        config_path=args.config,
        mode=args.mode,
        algorithm_override=args.algorithm,
    )
