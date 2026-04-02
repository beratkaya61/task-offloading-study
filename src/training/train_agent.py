import argparse
import os
import random
import time

import simpy
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.core.config import load_config
from src.core.evaluation import evaluate_policy
from src.env.rl_env import OffloadingEnv
from src.env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel
from src.utils.reproducibility import set_seed


ALGORITHM_DEFAULTS = {
    "ppo": {
        "model_class": PPO,
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "device": "cpu",
        "default_save_path": "models/ppo/single_run_synthetic/ppo_offloading_agent_v2",
        "default_run_name": "PPO_v2_Retrained",
    },
    "dqn": {
        "model_class": DQN,
        "learning_rate": 1e-4,
        "buffer_size": 10000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 500,
        "device": "cpu",
        "default_save_path": "models/dqn/single_run_synthetic/dqn_offloading_agent_v2",
        "default_run_name": "DQN_v2",
    },
    "a2c": {
        "model_class": A2C,
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "device": "cpu",
        "default_save_path": "models/a2c/single_run_synthetic/a2c_offloading_agent_v2",
        "default_run_name": "A2C_v2",
    },
}


class SimpleLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"[PROGRESS] Training: Step {self.num_timesteps} / {self.locals['total_timesteps']}")
        return True


def build_training_env(seed, max_steps=50, num_edge_servers=3, num_devices=5, env_kwargs=None):
    random.seed(seed)
    env_kwargs = env_kwargs or {}

    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(
            env_sim,
            i + 1,
            (random.uniform(0, 1000), random.uniform(0, 1000)),
            2e9,
        )
        for i in range(num_edge_servers)
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
        for i in range(num_devices)
    ]

    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        max_steps=max_steps,
        **env_kwargs,
    )
    set_seed(seed, env=env)
    return env


def _resolve_hyperparameters(algorithm, cfg):
    params = dict(ALGORITHM_DEFAULTS[algorithm])
    algo_cfg = cfg.get(algorithm, {}) if cfg else {}
    env_cfg = cfg.get("env", {}) if cfg else {}

    for key in list(params.keys()):
        if key in {"model_class", "default_save_path", "default_run_name"}:
            continue
        if key in algo_cfg:
            params[key] = algo_cfg[key]

    params["max_steps"] = env_cfg.get("max_steps", 50)
    params["num_edge_servers"] = env_cfg.get("num_edge_servers", 3)
    params["num_devices"] = env_cfg.get("num_devices", 5)
    return params


def create_model(algorithm, env, hyperparams):
    model_class = ALGORITHM_DEFAULTS[algorithm]["model_class"]
    model_kwargs = {
        key: value
        for key, value in hyperparams.items()
        if key
        not in {
            "model_class",
            "default_save_path",
            "default_run_name",
            "max_steps",
            "num_edge_servers",
            "num_devices",
        }
    }
    return model_class("MlpPolicy", env, verbose=0, **model_kwargs)


def train_single_agent(
    algorithm,
    total_timesteps=30000,
    seed=42,
    save_path=None,
    run_name=None,
    config_path="configs/synthetic/rl_training.yaml",
    eval_episodes=10,
    eval_csv_path="results/raw/synthetic_rl_retraining.csv",
    extra_eval_fields=None,
    env_overrides=None,
    env_kwargs=None,
):
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHM_DEFAULTS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    cfg = load_config(config_path)
    hyperparams = _resolve_hyperparameters(algorithm, cfg)
    env_overrides = env_overrides or {}
    for key in ("max_steps", "num_edge_servers", "num_devices"):
        if key in env_overrides:
            hyperparams[key] = env_overrides[key]
    env = build_training_env(
        seed=seed,
        max_steps=hyperparams["max_steps"],
        num_edge_servers=hyperparams["num_edge_servers"],
        num_devices=hyperparams["num_devices"],
        env_kwargs=env_kwargs,
    )

    resolved_run_name = run_name or ALGORITHM_DEFAULTS[algorithm]["default_run_name"]
    resolved_save_path = save_path or ALGORITHM_DEFAULTS[algorithm]["default_save_path"]

    print("[TRAIN] Initializing Training Environment...")
    print(f"[TRAIN] Algorithm: {algorithm.upper()}")
    print(f"[TRAIN] Seed: {seed}")
    print(f"[TRAIN] Total Timesteps: {total_timesteps}")

    model = create_model(algorithm, env, hyperparams)

    print(f"[TRAIN] Starting Training ({total_timesteps} steps)...")
    logging_callback = SimpleLoggingCallback(check_freq=5000)
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=logging_callback, progress_bar=False)
    training_time = time.time() - start_time

    save_dir = os.path.dirname(resolved_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(resolved_save_path)
    print(f"[TRAIN] Model saved to {resolved_save_path}.zip")

    print("[TRAIN] Running final evaluation for the new model...")
    eval_result = evaluate_policy(
        env,
        model,
        num_episodes=eval_episodes,
        run_name=resolved_run_name,
        semantic_mode="action_prior",
        config_seed=seed,
        csv_path=eval_csv_path,
        extra_fields=extra_eval_fields,
    )

    print("[TRAIN] Training and Evaluation completed successfully.")
    return {
        "algorithm": algorithm,
        "seed": seed,
        "save_path": f"{resolved_save_path}.zip",
        "training_time_sec": training_time,
        "evaluation": eval_result,
    }


def train():
    parser = argparse.ArgumentParser(description="Train Baseline RL Agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "a2c"],
        help="RL algorithm to train",
    )
    parser.add_argument("--steps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--save_path", type=str, default=None, help="Custom path to save model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic/rl_training.yaml",
        help="Training config path",
    )
    parser.add_argument("--eval_episodes", type=int, default=10, help="Final evaluation episodes")
    args = parser.parse_args()

    train_single_agent(
        algorithm=args.algorithm,
        total_timesteps=args.steps,
        seed=args.seed,
        save_path=args.save_path,
        config_path=args.config,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    train()
