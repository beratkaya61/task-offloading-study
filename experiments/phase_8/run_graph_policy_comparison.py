from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import simpy
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.graph_policy_evaluator import GraphPolicyEnvAdapter, load_graph_policy_checkpoint
from src.core.evaluation import evaluate_policy
from src.env.rl_env import OffloadingEnv
from src.env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel
from src.training.pretrain_graph_policy import run_graph_supervised_pretraining
from src.utils.reproducibility import set_seed


def make_env(seed: int, max_steps: int = 50) -> OffloadingEnv:
    env_sim = simpy.Environment()
    channel = WirelessChannel()
    cloud = CloudServer(env_sim)
    edge_servers = [
        EdgeServer(env_sim, 1, (200, 200), 2.5e9),
        EdgeServer(env_sim, 2, (800, 200), 2.0e9),
        EdgeServer(env_sim, 3, (500, 800), 2.2e9),
    ]
    devices = [
        IoTDevice(
            env_sim,
            id=index,
            channel=channel,
            edge_servers=edge_servers,
            cloud_server=cloud,
            battery_capacity=10000.0,
        )
        for index in range(5)
    ]
    env = OffloadingEnv(
        devices=devices,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=channel,
        max_steps=max_steps,
    )
    set_seed(seed, env=env)
    return env


def _mean(values):
    return sum(values) / max(1, len(values))


def _sample_std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _ci95(values):
    if len(values) <= 1:
        return 0.0
    return 1.96 * _sample_std(values) / math.sqrt(len(values))


def _action_label(action_id: int) -> str:
    names = ["local", "edge_25", "edge_50", "edge_75", "edge_100", "cloud"]
    return names[action_id] if 0 <= int(action_id) < len(names) else str(action_id)


def _graph_run_name(fusion: str) -> str:
    return f"GraphPolicy_{fusion}"


def _evaluate_checkpoint(run_name: str, seed: int, model, num_episodes: int, max_steps: int):
    env = make_env(seed=seed, max_steps=max_steps)
    return evaluate_policy(
        env,
        model,
        num_episodes=num_episodes,
        run_name=run_name,
        semantic_mode="action_prior",
        config_seed=seed,
        csv_path="",
        extra_fields={"config_eval_group": "phase_8_policy_comparison"},
    )


def _load_scratch_ppo(seed: int):
    checkpoint = REPO_ROOT / f"models/ppo/synthetic_rl_retraining/seed{seed}.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Scratch PPO checkpoint not found: {checkpoint}")
    return PPO.load(str(checkpoint), device="cpu"), str(checkpoint.relative_to(REPO_ROOT))


def _load_pretrained_ppo(seed: int):
    checkpoint = REPO_ROOT / f"models/ppo/teacher_policy_sensitivity/contextual_reward_aligned/pretrained/seed{seed}/refinement.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained PPO checkpoint not found: {checkpoint}")
    return PPO.load(str(checkpoint), device="cpu"), str(checkpoint.relative_to(REPO_ROOT))


def _build_graph_model(seed: int, fusion: str, graph_config: str):
    result = run_graph_supervised_pretraining(
        config_path=graph_config,
        semantic_prior_fusion=fusion,
        seed_override=seed,
        output_suffix=f"phase8_seed{seed}",
        write_report=False,
        write_metrics=False,
    )
    checkpoint_path = result["checkpoint_path"]
    model = load_graph_policy_checkpoint(checkpoint_path, device="cpu")
    return model, checkpoint_path


def _aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["run_name"], []).append(row)

    summary = []
    for run_name, run_rows in grouped.items():
        success_values = [float(row["metric_success_rate"]) for row in run_rows]
        qoe_values = [float(row["metric_qoe"]) for row in run_rows]
        latency_values = [float(row["metric_p95_latency"]) for row in run_rows]
        energy_values = [float(row["metric_avg_energy"]) for row in run_rows]
        action_rate_means = {
            action_id: _mean([float(row[f"metric_action_{action_id}_rate"]) for row in run_rows])
            for action_id in range(6)
        }
        summary.append(
            {
                "run_name": run_name,
                "num_seeds": len(run_rows),
                "success_mean": _mean(success_values),
                "success_std": _sample_std(success_values),
                "success_ci95": _ci95(success_values),
                "qoe_mean": _mean(qoe_values),
                "latency_mean": _mean(latency_values),
                "energy_mean": _mean(energy_values),
                "dominant_action": max(action_rate_means, key=action_rate_means.get),
            }
        )
    return sorted(summary, key=lambda item: item["run_name"])


def _write_report(report_path: Path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = _aggregate(rows)
    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Phase 8 Policy Comparison",
        "",
        "Bu rapor Faz 8.4 kapsaminda ayni evaluator mantigi altinda vector-state PPO ve graph-aware policy karsilastirmasini toplar.",
        "",
        "## Seed Aggregated Summary",
        "",
        "| Model | Seeds | Success Mean | Success 95% CI | P95 Latency Mean | Avg Energy Mean | QoE Mean | Dominant Action |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_name']} | {row['num_seeds']} | {row['success_mean']:.2%} | "
            f"+/- {row['success_ci95']:.2%} | {row['latency_mean']:.3f} | {row['energy_mean']:.4f} | "
            f"{row['qoe_mean']:.2f} | {_action_label(int(row['dominant_action']))} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Details",
            "",
            "| Model | Seed | Success | P95 Latency | Avg Energy | QoE | Dominant Action |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['seed']} | {float(row['metric_success_rate']):.2%} | "
            f"{float(row['metric_p95_latency']):.3f} | {float(row['metric_avg_energy']):.4f} | "
            f"{float(row['metric_qoe']):.2f} | {_action_label(int(row['metric_dominant_action']))} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 8 MLP-vs-graph policy comparison")
    parser.add_argument("--graph_config", default="configs/phase_8/graph_supervised_pretraining.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        scratch_model, scratch_path = _load_scratch_ppo(seed)
        scratch_metrics = _evaluate_checkpoint("MLP-PPO", seed, scratch_model, args.eval_episodes, args.max_steps)
        rows.append({"run_name": "MLP-PPO", "seed": seed, "checkpoint_path": scratch_path, **scratch_metrics})

        pretrained_model, pretrained_path = _load_pretrained_ppo(seed)
        pretrained_metrics = _evaluate_checkpoint("Pretrained MLP-PPO", seed, pretrained_model, args.eval_episodes, args.max_steps)
        rows.append({"run_name": "Pretrained MLP-PPO", "seed": seed, "checkpoint_path": pretrained_path, **pretrained_metrics})

        for fusion in ("none", "late"):
            graph_model, graph_checkpoint = _build_graph_model(seed, fusion, args.graph_config)
            graph_env = make_env(seed=seed, max_steps=args.max_steps)
            graph_adapter = GraphPolicyEnvAdapter(graph_model, graph_env)
            graph_metrics = evaluate_policy(
                graph_env,
                graph_adapter,
                num_episodes=args.eval_episodes,
                run_name=_graph_run_name(fusion),
                semantic_mode="action_prior",
                config_seed=seed,
                csv_path="",
                extra_fields={"config_eval_group": "phase_8_policy_comparison"},
            )
            rows.append(
                {
                    "run_name": _graph_run_name(fusion),
                    "seed": seed,
                    "checkpoint_path": graph_checkpoint,
                    **graph_metrics,
                }
            )

    for row in _aggregate(rows):
        print(
            f"[SUMMARY] {row['run_name']}: success={row['success_mean']:.2%} "
            f"+/- {row['success_ci95']:.2%}, p95={row['latency_mean']:.3f}, "
            f"energy={row['energy_mean']:.4f}, qoe={row['qoe_mean']:.2f}, "
            f"dominant={_action_label(int(row['dominant_action']))}"
        )

    if args.report:
        _write_report(Path(args.report), rows)
        print(f"[INFO] Phase 8 comparison report: {args.report}")


if __name__ == "__main__":
    main()

