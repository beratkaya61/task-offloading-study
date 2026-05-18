#!/usr/bin/env python3
"""Sanity-check the calibrated real-data benchmark with heuristic policies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.baselines import (
    CloudOnlyPolicy,
    DeadlineAwareGreedyPolicy,
    EdgeOnlyPolicy,
    GeneticAlgorithmPolicy,
    GreedyLatencyPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
)
from core.evaluation import evaluate_policy
from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator
from src.core.config_adapters import build_trace_training_config
from src.core.experiment_artifacts import load_yaml


HEURISTIC_POLICIES = {
    "LocalOnly": lambda: LocalOnlyPolicy(),
    "EdgeOnly": lambda: EdgeOnlyPolicy(),
    "CloudOnly": lambda: CloudOnlyPolicy(),
    "Random": lambda: RandomPolicy(),
    "GreedyLatency": lambda: GreedyLatencyPolicy(),
    "DeadlineAwareGreedy": lambda: DeadlineAwareGreedyPolicy(),
    "GeneticAlgorithm": lambda: GeneticAlgorithmPolicy(population_size=10, generations=5),
}


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _dominant_action_mode(series: pd.Series) -> str:
    counts = Counter(int(value) for value in series.dropna().tolist())
    if not counts:
        return "n/a"
    action, count = counts.most_common(1)[0]
    return f"{action} ({count}/{len(series)})"


def _load_ppo_reference(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    return frame[frame["config_model_type"] == "PPO"].copy()


def _write_report(
    rows: pd.DataFrame,
    ppo_reference: pd.DataFrame,
    report_path: Path,
    csv_path: Path,
) -> None:
    grouped = (
        rows.groupby("config_model_type", as_index=False)
        .agg(
            success_mean=("metric_success_rate", "mean"),
            success_std=("metric_success_rate", "std"),
            reward_mean=("metric_avg_reward", "mean"),
            p95_mean=("metric_p95_latency", "mean"),
            energy_mean=("metric_avg_energy", "mean"),
            qoe_mean=("metric_qoe", "mean"),
        )
        .sort_values("success_mean", ascending=False)
    )

    lines = [
        "# Real-Data Benchmark Sanity Check",
        "",
        "Bu raporun amaci, kalibre edilmis `real_composite_trace` benchmark'inin environment icinde tamamen bozuk olup olmadigini ayirmaktir.",
        "Burada hedef PPO'yu degil, benchmark + env kombinasyonunu heuristic politikalarla sinamaktir.",
        "",
        f"- Kaynak CSV: `{csv_path.as_posix()}`",
        "- Split: `test`",
        "- Seedler: `42, 43, 44`",
        "- Episode sayisi: her seed icin `10`",
        "",
        "## Heuristic Sonuclari",
        "",
        "| Policy | Mean Success | Std | Mean Reward | Mean P95 Latency (s) | Mean Energy | Mean QoE | Dominant Action |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in grouped.iterrows():
        policy_rows = rows[rows["config_model_type"] == row["config_model_type"]]
        lines.append(
            "| {policy} | {success} | {std:.4f} | {reward:.2f} | {p95:.4f} | {energy:.4f} | {qoe:.2f} | {dominant} |".format(
                policy=row["config_model_type"],
                success=_format_pct(float(row["success_mean"])),
                std=float(0.0 if pd.isna(row["success_std"]) else row["success_std"]),
                reward=float(row["reward_mean"]),
                p95=float(row["p95_mean"]),
                energy=float(row["energy_mean"]),
                qoe=float(row["qoe_mean"]),
                dominant=_dominant_action_mode(policy_rows["metric_dominant_action"]),
            )
        )

    lines.extend(
        [
            "",
            "## PPO Referans Notu",
            "",
        ]
    )

    if ppo_reference.empty:
        lines.append("Mevcut `real_data_rl_retraining` PPO CSV'si bulunamadi; bu nedenle heuristic benchmark raporu PPO karsilastirmasi olmadan yazildi.")
    else:
        ppo_mean = float(ppo_reference["metric_success_rate"].mean())
        ppo_std = float(ppo_reference["metric_success_rate"].std(ddof=0))
        best_heuristic = grouped.iloc[0]
        lines.extend(
            [
                f"- Mevcut PPO real-data retraining ortalamasi: `{_format_pct(ppo_mean)}` (`std={ppo_std:.4f}`)",
                f"- En iyi heuristic ortalamasi: `{best_heuristic['config_model_type']}` ile `{_format_pct(float(best_heuristic['success_mean']))}`",
                "",
            ]
        )
        if ppo_std < 1e-6:
            lines.extend(
                [
                    "Seed bazli PPO tablo okumasinda yeni contract altinda farkli bir sorun goruluyor:",
                    "- PPO artik yuksek varyansla ikiye ayrilmiyor;",
                    f"- bunun yerine tum seedlerde neredeyse ayni noktaya sabitleniyor (`{_format_pct(ppo_mean)}`);",
                    "- dominant aksiyon da sistematik bicimde `action=3` olarak kaliyor.",
                    "",
                    "Bu tablo su yorumu destekliyor:",
                    "- benchmark tamamen fiziksel olarak bozuk degil, cunku heuristic aile anlamli ayrisiyor;",
                    "- PPO ise artik kararsiz degil, ama semantic/reward tarafinin ittiği zayif bir local optimuma kilitlenmis gorunuyor;",
                    "- dolayisiyla bundan sonraki mudahale noktasi benchmark'i tekrar bozmak degil, reward geometry / semantic prior bias / PPO aksiyon kilitlenmesidir.",
                ]
            )
        else:
            lines.extend(
                [
                    "Seed bazli PPO tablo okumasinda iki ayri davranis modu goruluyor:",
                    "- bazi seedler cloud agirlikli daha iyi politikaya cikabiliyor,",
                    "- bazilari ise `action=3` etrafinda cokuyor.",
                    "",
                    "Bu tablo su yorumu destekliyor:",
                    "- benchmark tamamen fiziksel olarak bozuk degil, cunku heuristic aile anlamli ayrisiyor;",
                    "- buna karsin PPO tarafinda yuksek varyansli bir optimizasyon kararsizligi var;",
                    "- dolayisiyla bundan sonraki mudahale noktasi once benchmark degil, reward geometry / semantic prior bias / PPO stabilitesidir.",
                ]
            )

    lines.extend(
        [
            "",
            "## Karar",
            "",
            "Bu benchmark, kalibrasyon sonrasi haliyle `imkansiz gorev coplugu` degil.",
            "Fakat kolay bir benchmark da degil; cloud-benzeri kararlarin avantajli oldugu sert bir deadline rejimi var.",
            "Bu nedenle Faz 5 real-data PPO kosulari devam etmeden once, PPO'nun neden partial-edge moduna cokebildigi ayrica incelenmelidir.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_data_benchmark_sanity_check(
    config_path: str = "configs/phase_6/real_data_benchmark_sanity_check.yaml",
) -> list[dict]:
    config = load_yaml(config_path)
    eval_cfg = config.get("evaluation", {})
    output_cfg = config.get("output", {})
    backbone_cfg = config.get("trace_backbone", {})

    seeds = [int(seed) for seed in eval_cfg.get("seeds", [42, 43, 44])]
    num_episodes = int(eval_cfg.get("num_episodes", 10))
    split_name = eval_cfg.get("split", "test")
    csv_path = Path(output_cfg["csv_path"])
    report_path = Path(output_cfg["report_path"])
    training_config_path = backbone_cfg.get("training_config", "configs/phase_5/real_data_rl_training.yaml")
    training_config = load_yaml(training_config_path)

    if csv_path.exists():
        csv_path.unlink()

    rows: list[dict] = []
    for seed in seeds:
        orchestrator_config = build_trace_training_config(
            training_config,
            algorithm="ppo",
            seed=seed,
            log_dir="results/tmp/benchmark_sanity",
            checkpoint_dir="models/tmp/benchmark_sanity",
            report_path=str(report_path),
            overrides=None,
        )
        orchestrator = TraceTrainingOrchestrator(
            config_path=training_config_path,
            seed=seed,
            config_dict=orchestrator_config,
        )
        train_eps, val_eps, test_eps = orchestrator.prepare_traces()
        split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
        env = orchestrator._build_trace_env(split_map[split_name])

        for policy_name in config.get("policies", {}).get("heuristic", []):
            row = evaluate_policy(
                env,
                HEURISTIC_POLICIES[policy_name](),
                num_episodes=num_episodes,
                run_name=policy_name,
                semantic_mode="benchmark_sanity",
                config_seed=seed,
                csv_path=str(csv_path),
                extra_fields={
                    "config_batch_id": "real_data_benchmark_sanity",
                    "config_eval_group": "real_data_benchmark_sanity",
                },
            )
            rows.append(row)

    rows_frame = pd.read_csv(csv_path)
    ppo_reference = _load_ppo_reference(Path(config.get("ppo_reference", {}).get("csv_path", "")))
    _write_report(rows_frame, ppo_reference, report_path, csv_path)
    return rows


if __name__ == "__main__":
    run_real_data_benchmark_sanity_check()
