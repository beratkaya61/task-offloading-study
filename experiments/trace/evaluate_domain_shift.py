from datetime import datetime
from pathlib import Path
import csv

import numpy as np
import simpy
import yaml
from stable_baselines3 import PPO

from src.core.evaluation import evaluate_policy
from src.core.trace_loader import TraceLoader
from src.core.trace_processor import TraceProcessor
from src.env.rl_env import OffloadingEnv, OffloadingEnv_v2
from src.env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "trace" / "domain_shift_evaluation.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_synthetic_env(seed, max_steps=50):
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
        IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud, battery_capacity=10000.0)
        for i in range(5)
    ]
    env = OffloadingEnv(devices=devices, edge_servers=edge_servers, cloud_server=cloud, channel=channel)
    env.max_steps = max_steps
    env.reset(seed=seed)
    return env


def make_trace_env(config, seed):
    trace_cfg = config.get("data", {})
    loader = TraceLoader(trace_dir=trace_cfg.get("trace_dir", "data/traces"))
    processor = TraceProcessor(trace_dir=trace_cfg.get("trace_dir", "data/traces"), seed=seed)

    if loader.has_saved_episode_splits():
        _, val_eps, test_eps = loader.load_saved_episode_splits()
        episodes = val_eps or test_eps
    else:
        traces = loader.load_trace_frames()
        if not traces:
            traces = processor.load_traces()
        processed = processor.preprocess_traces(traces)
        processor.generate_episodes(
            processed,
            tasks_per_episode=config.get("trace_env", {}).get("tasks_per_episode", 50),
            n_episodes=config.get("trace_env", {}).get("episodes", 20),
        )
        _, val_eps, test_eps = processor.split_episodes(train_ratio=0.7, val_ratio=0.15)
        episodes = val_eps or test_eps

    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(env_sim, i + 1, (np.random.uniform(0, 1000), np.random.uniform(0, 1000)), 2e9)
        for i in range(config.get("trace_env", {}).get("num_edge_servers", 3))
    ]
    devices = [
        IoTDevice(env_sim, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud, battery_capacity=10000.0)
        for i in range(config.get("trace_env", {}).get("num_devices", 20))
    ]
    env = OffloadingEnv_v2(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        success_bonus=float(config.get("trace_env", {}).get("success_bonus", 100.0)),
    )
    env.max_steps = config.get("trace_env", {}).get("max_steps", 50)
    env._domain_shift_episodes = episodes
    env.reset(seed=seed, episode_tasks=episodes[0].tasks if episodes else None)
    return env


def evaluate_trace_policy(env, model, num_episodes=5):
    episodes = getattr(env, "_domain_shift_episodes", [])
    action_counts = {idx: 0 for idx in range(6)}
    success_rates = []
    p95_values = []
    energy_values = []
    total_steps = 0

    for episode in episodes[:num_episodes]:
        obs, _ = env.reset(episode_tasks=episode.tasks)
        done = False
        latencies = []
        energies = []
        successes = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            action_counts[action] += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            latencies.append(info.get("delay", 0.0))
            energies.append(info.get("energy", 0.0))
            successes += int(info.get("task_success", False))
            steps += 1
        total_steps += steps
        success_rates.append(successes / max(steps, 1))
        p95_values.append(float(np.percentile(latencies, 95)) if latencies else 0.0)
        energy_values.append(float(np.mean(energies)) if energies else 0.0)

    dominant_action = max(action_counts, key=action_counts.get) if action_counts else -1
    unique_actions = sum(1 for count in action_counts.values() if count > 0)
    return {
        "metric_success_rate": round(float(np.mean(success_rates)) if success_rates else 0.0, 4),
        "metric_p95_latency": round(float(np.mean(p95_values)) if p95_values else 0.0, 4),
        "metric_avg_energy": round(float(np.mean(energy_values)) if energy_values else 0.0, 4),
        "metric_unique_actions": unique_actions,
        "metric_dominant_action": dominant_action,
        "config_total_tasks": int(total_steps),
    }


def resolve_trace_checkpoint(config):
    for candidate in config.get("models", {}).get("trace_candidates", []):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    return None


def append_csv_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_report(rows, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Trace Domain-Shift Report",
        "",
        "Bu rapor, Faz 6 kapsaminda domain shift kavramini olcmek icin uretilir.",
        "Domain shift, bir veri dagiliminda egitilen modelin farkli bir veri dagiliminda ne kadar bozuldugunu gosterir.",
        "- synthetic train -> trace test",
        "- trace train -> synthetic test",
        "",
        "## Son Durum",
        "",
        "| Train Domain | Test Domain | Model | Success Rate | P95 Latency | Avg Energy | Dominant Action | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['train_domain']} | {row['test_domain']} | {row['model_name']} | {float(row['metric_success_rate']) * 100:.2f}% | {float(row['metric_p95_latency']):.4f} | {float(row['metric_avg_energy']):.4f} | {row['metric_dominant_action']} | {row['status']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    csv_path = REPO_ROOT / config.get("output", {}).get("csv_path", "results/raw/trace/domain_shift/trace_domain_shift_evaluation.csv")
    report_path = REPO_ROOT / config.get("output", {}).get("report_path", "results/tables/trace_domain_shift_report.md")
    seed = int(config.get("evaluation", {}).get("seed", 42))
    num_episodes = int(config.get("evaluation", {}).get("num_episodes", 5))
    batch_id = datetime.now().strftime("trace_domain_shift_%Y%m%d_%H%M%S")
    rows = []

    synthetic_ckpt = REPO_ROOT / config.get("models", {}).get("synthetic_ppo", "models/ppo/synthetic_rl_retraining/seed42.zip")
    if synthetic_ckpt.exists():
        trace_env = make_trace_env(config, seed)
        synthetic_model = PPO.load(str(synthetic_ckpt), env=trace_env)
        metrics = evaluate_trace_policy(trace_env, synthetic_model, num_episodes=num_episodes)
        row = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "train_domain": "synthetic",
            "test_domain": "trace",
            "model_name": "PPO",
            "checkpoint_path": str(synthetic_ckpt.relative_to(REPO_ROOT)),
            "status": "completed",
            **metrics,
        }
        append_csv_row(csv_path, row)
        rows.append(row)

    trace_ckpt = resolve_trace_checkpoint(config)
    if trace_ckpt and trace_ckpt.exists():
        synthetic_env = make_synthetic_env(seed=seed, max_steps=config.get("synthetic_env", {}).get("max_steps", 50))
        metrics = evaluate_policy(
            synthetic_env,
            PPO.load(str(trace_ckpt), env=synthetic_env),
            num_episodes=num_episodes,
            run_name="trace_trained_PPO_on_synthetic",
            semantic_mode="trace_domain_shift",
            config_seed=seed,
            csv_path=str(csv_path),
            extra_fields={
                "config_batch_id": batch_id,
                "config_eval_group": "trace_domain_shift",
            },
        )
        rows.append({
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "train_domain": "trace",
            "test_domain": "synthetic",
            "model_name": "PPO",
            "checkpoint_path": str(trace_ckpt.relative_to(REPO_ROOT)),
            "status": "completed",
            "metric_success_rate": metrics.get("metric_success_rate", 0.0),
            "metric_p95_latency": metrics.get("metric_p95_latency", 0.0),
            "metric_avg_energy": metrics.get("metric_avg_energy", 0.0),
            "metric_unique_actions": metrics.get("metric_unique_actions", 0),
            "metric_dominant_action": metrics.get("metric_dominant_action", -1),
            "config_total_tasks": metrics.get("config_total_tasks", 0),
        })
    else:
        rows.append({
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "train_domain": "trace",
            "test_domain": "synthetic",
            "model_name": "PPO",
            "checkpoint_path": "missing",
            "status": "waiting_for_trace_training_checkpoint",
            "metric_success_rate": 0.0,
            "metric_p95_latency": 0.0,
            "metric_avg_energy": 0.0,
            "metric_unique_actions": 0,
            "metric_dominant_action": -1,
            "config_total_tasks": 0,
        })

    write_report(rows, report_path)
    for row in rows:
        if row.get("status") != "completed":
            append_csv_row(csv_path, row)


if __name__ == "__main__":
    main()
