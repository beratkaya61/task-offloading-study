import argparse
import csv
import os
import random
import time
from collections import Counter
from datetime import datetime

import numpy as np
import simpy
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.core.config import load_config
from src.core.evaluation import evaluate_policy
from src.env.rl_env import OffloadingEnv
from src.env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel
from src.utils.reproducibility import set_seed


STATE_FEATURE_COLUMNS = [
    "state_snr_norm",
    "state_task_size_norm",
    "state_cpu_cycles_norm",
    "state_battery_norm",
    "state_edge_load_norm",
    "state_edge_energy_norm",
    "prior_local",
    "prior_edge_25",
    "prior_edge_50",
    "prior_edge_75",
    "prior_edge_full",
    "prior_cloud",
]


def _normalize_teacher_policy_name(name):
    mapping = {
        "teacher_latency_greedy": "teacher_latency_greedy",
        "latency_oracle": "teacher_latency_greedy",
        "teacher_energy_greedy": "teacher_energy_greedy",
        "energy_oracle": "teacher_energy_greedy",
        "teacher_balanced_semantic": "teacher_balanced_semantic",
        "weighted_objective_oracle": "teacher_balanced_semantic",
        "teacher_reward_aligned": "teacher_reward_aligned",
        "reward_aligned_oracle": "teacher_reward_aligned",
        "teacher_contextual_reward_aligned": "teacher_contextual_reward_aligned",
    }
    return mapping.get(name, name)


def _anchor_row_teacher_policy(row):
    return _normalize_teacher_policy_name(str(row.get("teacher_policy") or row.get("objective") or ""))


def _anchor_row_action_id(row):
    raw_value = row.get("selected_action_id", row.get("oracle_action", -1))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _anchor_row_state_vector(row):
    if all(column in row for column in STATE_FEATURE_COLUMNS):
        return [float(row[column]) for column in STATE_FEATURE_COLUMNS]
    return [float(row[f"obs_{i}"]) for i in range(len(STATE_FEATURE_COLUMNS))]


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


def _load_anchor_dataloader(dataset_path, teacher_policy, min_margin=0.0, batch_size=128, allowed_actions=None, allowed_splits=None, balance_actions=False, balance_power=1.0, samples_per_epoch=None):
    requested_teacher_policy = _normalize_teacher_policy_name(teacher_policy)
    allowed_action_set = None if allowed_actions is None else {int(action) for action in allowed_actions}
    allowed_splits = set(allowed_splits or ["train"])
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if _anchor_row_teacher_policy(row) != requested_teacher_policy:
                continue
            if row.get("split", "train") not in allowed_splits:
                continue
            try:
                margin = float(row.get("teacher_margin", row.get("oracle_margin", 0.0)))
            except (TypeError, ValueError):
                margin = 0.0
            if margin < float(min_margin):
                continue
            try:
                oracle_action = _anchor_row_action_id(row)
            except (TypeError, ValueError):
                oracle_action = -1
            if allowed_action_set is not None and oracle_action not in allowed_action_set:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No anchor samples found for teacher_policy={teacher_policy!r} min_margin={min_margin}")

    observations = torch.tensor(
        [_anchor_row_state_vector(row) for row in rows],
        dtype=torch.float32,
    )
    labels = torch.tensor([_anchor_row_action_id(row) for row in rows], dtype=torch.long)
    dataset = TensorDataset(observations, labels)
    if not balance_actions:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    label_list = labels.cpu().tolist()
    label_counts = Counter(label_list)
    weights = []
    for label in label_list:
        count = max(1, int(label_counts.get(label, 1)))
        weights.append(1.0 / (count ** float(balance_power)))
    num_samples = int(samples_per_epoch or len(label_list))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


class PolicyAnchoringCallback(BaseCallback):
    def __init__(self, anchor_loader, interval_steps, until_step, batches_per_round, loss_weight=1.0, verbose=0):
        super().__init__(verbose)
        self.anchor_loader = anchor_loader
        self.interval_steps = int(interval_steps)
        self.until_step = int(until_step)
        self.batches_per_round = int(batches_per_round)
        self.loss_weight = float(loss_weight)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._loader_iter = iter(anchor_loader)

    def _next_batch(self):
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self.anchor_loader)
            return next(self._loader_iter)

    def _on_step(self) -> bool:
        if self.interval_steps <= 0 or self.batches_per_round <= 0:
            return True
        if self.num_timesteps > self.until_step:
            return True
        if self.num_timesteps % self.interval_steps != 0:
            return True

        self.model.policy.train()
        losses = []
        for _ in range(self.batches_per_round):
            observations, labels = self._next_batch()
            observations = observations.to(self.model.device)
            labels = labels.to(self.model.device)
            distribution = self.model.policy.get_distribution(observations)
            logits = distribution.distribution.logits
            loss = self._criterion(logits, labels) * self.loss_weight
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), max_norm=0.5)
            self.model.policy.optimizer.step()
            losses.append(float(loss.item()))

        if self.verbose:
            avg_loss = sum(losses) / max(1, len(losses))
            print(f"[ANCHOR] Step {self.num_timesteps}: auxiliary BC loss {avg_loss:.4f}")
        return True


def _evaluate_model_no_log(env, model, num_episodes=5):
    results = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_latencies = []
        episode_energies = []
        episode_successes = 0

        while not done:
            obs_batch = obs[np.newaxis, :]
            action, _ = model.predict(obs_batch, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward
            step_count += 1
            episode_latencies.append(info.get("delay", 0.0))
            episode_energies.append(info.get("energy", 0.0))
            if info.get("task_success", False):
                episode_successes += 1

        p95_latency = np.percentile(episode_latencies, 95) if episode_latencies else 0.0
        avg_energy = float(np.mean(episode_energies)) if episode_energies else 0.0
        success_rate = episode_successes / max(1, step_count)
        qoe = 100.0 * success_rate - (p95_latency * 5.0)
        results.append(
            {
                "reward": float(episode_reward),
                "success_rate": float(success_rate),
                "p95_latency": float(p95_latency),
                "avg_energy": float(avg_energy),
                "qoe": float(qoe),
            }
        )

    return {
        "avg_reward": float(np.mean([row["reward"] for row in results])) if results else 0.0,
        "success_rate": float(np.mean([row["success_rate"] for row in results])) if results else 0.0,
        "p95_latency": float(np.mean([row["p95_latency"] for row in results])) if results else 0.0,
        "avg_energy": float(np.mean([row["avg_energy"] for row in results])) if results else 0.0,
        "qoe": float(np.mean([row["qoe"] for row in results])) if results else 0.0,
    }


class PeriodicEvaluationCallback(BaseCallback):
    def __init__(self, eval_env, eval_interval_steps, eval_episodes, progress_csv_path, run_name, init_mode, seed, extra_fields=None, progress_step_offset=0, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval_steps = eval_interval_steps
        self.eval_episodes = eval_episodes
        self.progress_csv_path = progress_csv_path
        self.run_name = run_name
        self.init_mode = init_mode
        self.seed = seed
        self.extra_fields = extra_fields or {}
        self.progress_step_offset = int(progress_step_offset)

    def _on_step(self) -> bool:
        if self.eval_interval_steps <= 0:
            return True
        if self.num_timesteps % self.eval_interval_steps != 0:
            return True

        metrics = _evaluate_model_no_log(self.eval_env, self.model, num_episodes=self.eval_episodes)
        row = {
            "timestamp": datetime.now().isoformat(),
            "training_step": self.num_timesteps + self.progress_step_offset,
            "phase_step": self.num_timesteps,
            "run_name": self.run_name,
            "init_mode": self.init_mode,
            "seed": self.seed,
            "metric_success_rate": round(metrics["success_rate"], 4),
            "metric_avg_reward": round(metrics["avg_reward"], 2),
            "metric_p95_latency": round(metrics["p95_latency"], 4),
            "metric_avg_energy": round(metrics["avg_energy"], 4),
            "metric_qoe": round(metrics["qoe"], 2),
        }
        row.update(self.extra_fields)

        os.makedirs(os.path.dirname(self.progress_csv_path) or ".", exist_ok=True)
        write_header = not os.path.exists(self.progress_csv_path)
        with open(self.progress_csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return True


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
    hyperparam_overrides=None,
    init_model_path=None,
    init_mode="scratch",
    progress_csv_path=None,
    eval_interval_steps=0,
    progress_eval_episodes=5,
    progress_extra_fields=None,
    anchor_config=None,
    skip_final_evaluation=False,
    progress_step_offset=0,
    progress_init_mode=None,
):
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHM_DEFAULTS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    cfg = load_config(config_path)
    hyperparams = _resolve_hyperparameters(algorithm, cfg)
    env_overrides = env_overrides or {}
    hyperparam_overrides = hyperparam_overrides or {}
    anchor_config = anchor_config or {}
    for key in ("max_steps", "num_edge_servers", "num_devices"):
        if key in env_overrides:
            hyperparams[key] = env_overrides[key]
    for key, value in hyperparam_overrides.items():
        if key in {"model_class", "default_save_path", "default_run_name"}:
            continue
        hyperparams[key] = value
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
    print(f"[TRAIN] Init Mode: {init_mode}")
    print(f"[TRAIN] Total Timesteps: {total_timesteps}")
    if hyperparam_overrides:
        print(f"[TRAIN] Hyperparameter overrides: {hyperparam_overrides}")

    if init_model_path:
        if algorithm != "ppo":
            raise ValueError("Pretrained initialization is currently supported only for PPO")
        loaded_model = PPO.load(init_model_path, device=hyperparams.get("device", "cpu"))
        model = create_model(algorithm, env, hyperparams)
        model.set_parameters(loaded_model.get_parameters(), exact_match=True)
        print(f"[TRAIN] Loaded pretrained checkpoint weights: {init_model_path}")
    else:
        model = create_model(algorithm, env, hyperparams)

    callbacks = [SimpleLoggingCallback(check_freq=5000)]
    if algorithm == "ppo" and init_model_path and anchor_config.get("enabled", False):
        anchor_loader = _load_anchor_dataloader(
            dataset_path=anchor_config["dataset_path"],
            teacher_policy=anchor_config.get("teacher_policy", anchor_config.get("objective", "teacher_reward_aligned")),
            min_margin=float(anchor_config.get("min_margin", 0.0)),
            batch_size=int(anchor_config.get("batch_size", 128)),
            allowed_actions=anchor_config.get("allowed_actions"),
            allowed_splits=anchor_config.get("allowed_splits", ["train"]),
            balance_actions=anchor_config.get("balance_actions", False),
            balance_power=float(anchor_config.get("balance_power", 1.0)),
            samples_per_epoch=anchor_config.get("samples_per_epoch"),
        )
        callbacks.append(
            PolicyAnchoringCallback(
                anchor_loader=anchor_loader,
                interval_steps=int(anchor_config.get("interval_steps", 2500)),
                until_step=int(anchor_config.get("until_step", 15000)),
                batches_per_round=int(anchor_config.get("batches_per_round", 4)),
                loss_weight=float(anchor_config.get("loss_weight", 0.5)),
                verbose=1,
            )
        )
    if progress_csv_path and eval_interval_steps > 0:
        eval_env = build_training_env(
            seed=seed + 1000,
            max_steps=hyperparams["max_steps"],
            num_edge_servers=hyperparams["num_edge_servers"],
            num_devices=hyperparams["num_devices"],
            env_kwargs=env_kwargs,
        )
        callbacks.append(
            PeriodicEvaluationCallback(
                eval_env=eval_env,
                eval_interval_steps=eval_interval_steps,
                eval_episodes=progress_eval_episodes,
                progress_csv_path=progress_csv_path,
                run_name=resolved_run_name,
                init_mode=progress_init_mode or init_mode,
                seed=seed,
                extra_fields=progress_extra_fields,
                progress_step_offset=progress_step_offset,
            )
        )

    callback = CallbackList(callbacks)

    print(f"[TRAIN] Starting Training ({total_timesteps} steps)...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    training_time = time.time() - start_time

    save_dir = os.path.dirname(resolved_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(resolved_save_path)
    print(f"[TRAIN] Model saved to {resolved_save_path}.zip")

    if skip_final_evaluation:
        print("[TRAIN] Final evaluation skipped for this stage.")
        return {
            "algorithm": algorithm,
            "seed": seed,
            "init_mode": init_mode,
            "save_path": f"{resolved_save_path}.zip",
            "training_time_sec": training_time,
            "evaluation": None,
        }

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
        "init_mode": init_mode,
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


