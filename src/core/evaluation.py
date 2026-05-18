import csv
import os
import time
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
    "metric_deadline_miss_ratio",
    "metric_avg_reward",
    "metric_avg_latency",
    "metric_p95_latency",
    "metric_p99_latency",
    "metric_avg_energy",
    "metric_energy_per_success",
    "metric_jitter",
    "metric_avg_queue_delay",
    "metric_battery_depletion_rate",
    "metric_partial_offload_ratio",
    "metric_decision_overhead_ms",
    "metric_qoe",
    "metric_unique_actions",
    "metric_action_0_rate",
    "metric_action_1_rate",
    "metric_action_2_rate",
    "metric_action_3_rate",
    "metric_action_4_rate",
    "metric_action_5_rate",
    "metric_dominant_action",
    "config_batch_id",
    "config_eval_group",
]


def _is_sb3_model(model):
    return hasattr(model, "policy") and hasattr(model, "learn")


def _percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def summarize_step_logs(step_logs, total_reward, action_counts, reward_denominator=None):
    total_tasks = len(step_logs)
    successes = [row["success"] for row in step_logs]
    latencies = [row["delay"] for row in step_logs]
    energies = [row["energy"] for row in step_logs]
    queue_delays = [row["queue_delay"] for row in step_logs]
    decision_overheads = [row["decision_overhead_ms"] for row in step_logs]
    battery_empty = [row["battery_empty"] for row in step_logs]
    partial_flags = [row["partial_offload"] for row in step_logs]

    success_count = int(sum(successes))
    success_rate = success_count / max(1, total_tasks)
    avg_latency = _mean(latencies)
    p95_latency = _percentile(latencies, 95)
    p99_latency = _percentile(latencies, 99)
    avg_energy = _mean(energies)
    energy_per_success = float(sum(energies) / success_count) if success_count > 0 else 0.0
    jitter = float(np.std(latencies)) if len(latencies) > 1 else 0.0
    deadline_miss_ratio = 1.0 - success_rate if total_tasks > 0 else 0.0
    avg_queue_delay = _mean(queue_delays)
    battery_depletion_rate = _mean(battery_empty)
    partial_offload_ratio = _mean(partial_flags)
    decision_overhead_ms = _mean(decision_overheads)
    avg_reward = float(total_reward / max(1, reward_denominator if reward_denominator is not None else total_tasks))
    qoe = 100.0 * success_rate - (p95_latency * 5.0)

    total_actions = max(1, sum(action_counts.values()))
    action_rates = {
        f"metric_action_{index}_rate": round(action_counts.get(index, 0) / total_actions, 4)
        for index in range(6)
    }
    unique_actions = sum(1 for count in action_counts.values() if count > 0)
    dominant_action = max(action_counts, key=action_counts.get) if action_counts else -1

    summary = {
        "config_total_tasks": int(total_tasks),
        "metric_success_rate": round(success_rate, 4),
        "metric_deadline_miss_ratio": round(deadline_miss_ratio, 4),
        "metric_avg_reward": round(avg_reward, 4),
        "metric_avg_latency": round(avg_latency, 4),
        "metric_p95_latency": round(p95_latency, 4),
        "metric_p99_latency": round(p99_latency, 4),
        "metric_avg_energy": round(avg_energy, 6),
        "metric_energy_per_success": round(energy_per_success, 6),
        "metric_jitter": round(jitter, 4),
        "metric_avg_queue_delay": round(avg_queue_delay, 4),
        "metric_battery_depletion_rate": round(battery_depletion_rate, 4),
        "metric_partial_offload_ratio": round(partial_offload_ratio, 4),
        "metric_decision_overhead_ms": round(decision_overhead_ms, 4),
        "metric_qoe": round(qoe, 4),
        "metric_unique_actions": unique_actions,
        "metric_dominant_action": dominant_action,
    }
    summary.update(action_rates)
    return summary


def build_experiment_log_entry(
    *,
    run_name,
    semantic_mode,
    config_seed,
    total_reward,
    action_counts,
    step_logs,
    reward_denominator=None,
    extra_fields=None,
):
    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "config_seed": config_seed,
        "config_model_type": run_name,
        "config_semantic_mode": semantic_mode,
    }
    log_entry.update(summarize_step_logs(step_logs, total_reward, action_counts, reward_denominator=reward_denominator))
    log_entry.update(
        {
            "config_batch_id": "",
            "config_eval_group": "",
        }
    )
    if extra_fields:
        log_entry.update(extra_fields)
    return {column: log_entry.get(column, "") for column in EXPERIMENT_LOG_COLUMNS}


def normalize_experiment_csv(csv_path="results/phase_5/metrics/synthetic/policy_evaluation/experiment_results.csv"):
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
    csv_path="results/phase_5/metrics/synthetic/policy_evaluation/experiment_results.csv",
):
    print(f"[EVAL] Starting evaluation: {run_name} ({num_episodes} episodes)")

    is_sb3 = _is_sb3_model(model)
    if is_sb3:
        print(f"[EVAL] SB3 model detected: {run_name}")
    else:
        print(f"[EVAL] Custom baseline model detected: {run_name}")

    results = []
    action_counts = {index: 0 for index in range(6)}
    all_step_logs = []
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_latencies = []
        episode_energies = []
        episode_successes = 0

        while not done:
            start_time = time.perf_counter()
            if is_sb3:
                obs_batch = obs[np.newaxis, :]
                action, _ = model.predict(obs_batch, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
            elif hasattr(model, "predict_with_env"):
                action, _ = model.predict_with_env(obs, env, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            decision_overhead_ms = (time.perf_counter() - start_time) * 1000.0
            action_counts[int(action)] = action_counts.get(int(action), 0) + 1

            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward
            total_reward += reward
            step_count += 1
            episode_latencies.append(info.get("delay", 0.0))
            episode_energies.append(info.get("energy", 0.0))
            if info.get("task_success", False):
                episode_successes += 1
            all_step_logs.append(
                {
                    "success": bool(info.get("task_success", False)),
                    "delay": float(info.get("delay", 0.0)),
                    "energy": float(info.get("energy", 0.0)),
                    "queue_delay": float(info.get("queue_delay", 0.0)),
                    "battery_empty": bool(info.get("battery_empty", False)),
                    "partial_offload": bool(info.get("partial_offload", 1 <= int(action) <= 3)),
                    "decision_overhead_ms": float(decision_overhead_ms),
                }
            )

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

    log_entry = build_experiment_log_entry(
        run_name=run_name,
        semantic_mode=semantic_mode,
        config_seed=config_seed,
        total_reward=total_reward,
        action_counts=action_counts,
        step_logs=all_step_logs,
        reward_denominator=len(results),
        extra_fields=extra_fields,
    )

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        normalize_experiment_csv(csv_path)
        df = pd.DataFrame([log_entry], columns=EXPERIMENT_LOG_COLUMNS)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    print(
        f"[EVAL] {run_name} evaluated. Average success: {log_entry['metric_success_rate']:.2%}, "
        f"P95 latency: {log_entry['metric_p95_latency']:.3f}s, dominant action: {log_entry['metric_dominant_action']}"
    )
    return log_entry


def summarize_logs(
    results_dir="results/phase_5/metrics",
    output_table="v2_docs/phase_5/synthetic_phase_5_report.md",
    figure_path="results/phase_5/figures/ablation_impact.png",
):
    if not os.path.exists(results_dir):
        print(f"[WARN] Results directory not found: {results_dir}")
        return

    try:
        try:
            from src.core.reporting import write_experiment_report
        except ModuleNotFoundError:
            from core.reporting import write_experiment_report

        write_experiment_report(csv_path=results_dir, output_path=output_table, figure_path=figure_path)
        print(f"[INFO] Canonical experiment report written: {output_table}")
    except Exception as exc:
        print(f"[ERROR] Failed to summarize logs: {exc}")


if __name__ == "__main__":
    summarize_logs()

