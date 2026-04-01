import csv
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd


EXPERIMENT_LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "config_seed",
    "config_model_type",
    "config_semantic_mode",
    "config_total_tasks",
    "metric_success_rate",
    "metric_avg_reward",
    "metric_p95_latency",
    "metric_avg_energy",
    "metric_qoe",
    "config_batch_id",
    "config_eval_group",
]


def _is_sb3_model(model):
    return hasattr(model, "policy") and hasattr(model, "learn")


def normalize_experiment_csv(csv_path="results/raw/master_experiments.csv"):
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return

        for raw in reader:
            if not raw:
                continue
            if len(raw) < len(EXPERIMENT_LOG_COLUMNS):
                raw = raw + [""] * (len(EXPERIMENT_LOG_COLUMNS) - len(raw))
            elif len(raw) > len(EXPERIMENT_LOG_COLUMNS):
                raw = raw[: len(EXPERIMENT_LOG_COLUMNS)]
            rows.append(raw)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPERIMENT_LOG_COLUMNS)
        writer.writerows(rows)


def evaluate_policy(
    env,
    model,
    num_episodes=5,
    run_name="Baseline",
    semantic_mode="None",
    config_seed=42,
    extra_fields=None,
):
    print(f"[EVAL] Starting evaluation: {run_name} ({num_episodes} episodes)")

    is_sb3 = _is_sb3_model(model)
    if is_sb3:
        print(f"[EVAL] SB3 model detected: {run_name}")
    else:
        print(f"[EVAL] Custom baseline model detected: {run_name}")

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
            if is_sb3:
                obs_batch = obs[np.newaxis, :]
                action, _ = model.predict(obs_batch, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
            else:
                action, _ = model.predict(obs, deterministic=True)

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
                "steps": step_count,
                "success_rate": success_rate,
                "p95_latency": float(p95_latency),
                "avg_energy": avg_energy,
                "qoe": float(qoe),
            }
        )

    avg_reward = float(np.mean([row["reward"] for row in results]))
    avg_success = float(np.mean([row["success_rate"] for row in results]))
    avg_p95_latency = float(np.mean([row["p95_latency"] for row in results]))
    avg_energy = float(np.mean([row["avg_energy"] for row in results]))
    avg_qoe = float(np.mean([row["qoe"] for row in results]))
    total_tasks = int(sum(row["steps"] for row in results))

    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/master_experiments.csv"
    normalize_experiment_csv(csv_path)

    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": config_seed,
        "config_model_type": run_name,
        "config_semantic_mode": semantic_mode,
        "config_total_tasks": total_tasks,
        "metric_success_rate": round(avg_success, 4),
        "metric_avg_reward": round(avg_reward, 2),
        "metric_p95_latency": round(avg_p95_latency, 4),
        "metric_avg_energy": round(avg_energy, 4),
        "metric_qoe": round(avg_qoe, 2),
        "config_batch_id": "",
        "config_eval_group": "",
    }
    if extra_fields:
        log_entry.update(extra_fields)

    df = pd.DataFrame([log_entry], columns=EXPERIMENT_LOG_COLUMNS)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)

    print(
        f"[EVAL] {run_name} evaluated. Average success: {avg_success:.2%}, "
        f"P95 latency: {avg_p95_latency:.3f}s"
    )
    return log_entry


def summarize_logs(results_dir="results/raw", output_table="results/tables/offloading_experiment_report.md"):
    csv_path = os.path.join(results_dir, "master_experiments.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return

    try:
        normalize_experiment_csv(csv_path)
        try:
            from src.core.reporting import write_experiment_report
        except ModuleNotFoundError:
            from core.reporting import write_experiment_report

        write_experiment_report(csv_path=csv_path, output_path=output_table)
        print(f"[INFO] Canonical experiment report written: {output_table}")
    except Exception as exc:
        print(f"[ERROR] Failed to summarize logs: {exc}")


if __name__ == "__main__":
    summarize_logs()
