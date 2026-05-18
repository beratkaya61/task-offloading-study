from __future__ import annotations

from typing import Dict


def build_trace_training_config(
    phase5_training_config: Dict,
    algorithm: str,
    seed: int,
    log_dir: str,
    checkpoint_dir: str,
    report_path: str,
    overrides: Dict | None = None,
) -> Dict:
    exp_cfg = phase5_training_config.get("experiment", {})
    env_cfg = phase5_training_config.get("env", {})
    trace_cfg = phase5_training_config.get("trace", {})
    algo_cfg = phase5_training_config.get(algorithm.lower(), {})
    device_cfg = phase5_training_config.get("device", {})

    trace_training = {
        "training": {
            "algorithm": algorithm.upper(),
            "version": f"phase5_{algorithm.lower()}",
            "max_episodes": int(exp_cfg.get("max_episodes", 500)),
            "episodes_per_eval": int(exp_cfg.get("episodes_per_eval", 10)),
            "learning_rate": algo_cfg.get("learning_rate", 3e-4),
            "gamma": algo_cfg.get("gamma", 0.99),
            "batch_size": algo_cfg.get("batch_size", 64),
            "n_steps": algo_cfg.get("n_steps", 512),
            "n_epochs": algo_cfg.get("n_epochs", 10),
            "buffer_size": algo_cfg.get("buffer_size", 10000),
            "learning_starts": algo_cfg.get("learning_starts", 1000),
            "train_freq": algo_cfg.get("train_freq", 4),
            "target_update_interval": algo_cfg.get("target_update_interval", 500),
            "ent_coef": algo_cfg.get("ent_coef", 0.0),
        },
        "environment": {
            "trace_source": trace_cfg.get("trace_source", "real_data"),
            "n_devices": int(env_cfg.get("num_devices", 20)),
            "n_edge_servers": int(env_cfg.get("num_edge_servers", 3)),
            "n_tasks_per_episode": int(env_cfg.get("tasks_per_episode", 50)),
            "use_reward_shaping": bool(env_cfg.get("use_reward_shaping", True)),
            "use_partial_offloading": bool(env_cfg.get("use_partial_offloading", True)),
            "use_semantic_features": bool(env_cfg.get("use_semantic_features", True)),
            "use_confidence_weighting": bool(env_cfg.get("use_confidence_weighting", False)),
            "use_battery_awareness": bool(env_cfg.get("use_battery_awareness", True)),
            "use_mobility_features": bool(env_cfg.get("use_mobility_features", True)),
            "use_queue_awareness": bool(env_cfg.get("use_queue_awareness", False)),
            "use_deadline_features": bool(env_cfg.get("use_deadline_features", False)),
            "use_success_bonus": bool(env_cfg.get("use_success_bonus", True)),
            "success_bonus": float(env_cfg.get("success_bonus", 100.0)),
            "cloud_fixed_latency": float(env_cfg.get("cloud_fixed_latency", 0.1)),
        },
        "data": {
            "real_data_mode": bool(trace_cfg.get("real_data_mode", True)),
            "real_data_manifest": trace_cfg.get(
                "real_data_manifest", "configs/phase_6/raw_real_data_manifest.yaml"
            ),
            "trace_dir": trace_cfg.get("trace_dir", "data/real_composite_trace/splits"),
            "train_episodes": int(trace_cfg.get("train_episodes", 80)),
            "val_episodes": int(trace_cfg.get("val_episodes", 10)),
            "test_episodes": int(trace_cfg.get("test_episodes", 10)),
            "normalize_features": bool(trace_cfg.get("normalize_features", True)),
            "remove_outliers": bool(trace_cfg.get("remove_outliers", True)),
            "seed": int(seed),
        },
        "logging": {
            "log_dir": log_dir,
            "tensorboard": True,
            "checkpoint_dir": checkpoint_dir,
            "report_path": report_path,
        },
        "device": {
            "cuda": bool(device_cfg.get("cuda", True)),
            "device_id": int(device_cfg.get("device_id", 0)),
        },
        "seed": int(seed),
    }

    for key, value in (overrides or {}).items():
        trace_training["training"][key] = value
    return trace_training
