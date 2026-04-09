#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.synthetic.run_staged_training_comparison import (
    _aggregate_action_profile,
    _aggregate_convergence,
    _aggregate_final,
    _prepare_final_df,
    run_staged_training_comparison,
)
from src.training.pretrain_policy import run_supervised_pretraining

ACTION_ORDER = ["local", "edge_25", "edge_50", "edge_75", "edge_100", "cloud"]
TEACHER_POLICIES = [
    "teacher_latency_greedy",
    "teacher_energy_greedy",
    "teacher_balanced_semantic",
    "teacher_contextual_reward_aligned",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _teacher_slug(name: str) -> str:
    return name.replace("teacher_", "")


def _coverage_rows(dataset_csv: Path, teacher_policy: str) -> Dict[str, int]:
    counts = {f"coverage_total_{action}": 0 for action in ACTION_ORDER}
    counts.update({f"coverage_train_{action}": 0 for action in ACTION_ORDER})
    total = 0
    train_total = 0
    with dataset_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("teacher_policy") != teacher_policy:
                continue
            action = row.get("selected_action_name", "")
            split = row.get("split", "")
            total += 1
            key = f"coverage_total_{action}"
            if key in counts:
                counts[key] += 1
            if split == "train":
                train_total += 1
                train_key = f"coverage_train_{action}"
                if train_key in counts:
                    counts[train_key] += 1
    counts["coverage_total_samples"] = total
    counts["coverage_train_samples"] = train_total
    counts["coverage_val_samples"] = max(0, (total - train_total) // 2)
    counts["coverage_test_samples"] = max(0, total - train_total - counts["coverage_val_samples"])
    return counts


def _extract_model_row(final_summary: pd.DataFrame, action_profile: pd.DataFrame, convergence_summary: pd.DataFrame, model_type: str, init_mode: str) -> Dict[str, float]:
    model_row = final_summary[final_summary["config_model_type"] == model_type].iloc[0]
    action_row = action_profile[action_profile["config_model_type"] == model_type].iloc[0]
    conv_row = convergence_summary[convergence_summary["init_mode"] == init_mode].iloc[0]
    payload = {
        "success_mean": float(model_row["success_mean"]),
        "success_std": float(model_row["success_std"] if pd.notna(model_row["success_std"]) else 0.0),
        "success_ci95": float(model_row["success_ci95"]),
        "reward_mean": float(model_row["reward_mean"]),
        "p95_mean": float(model_row["p95_mean"]),
        "energy_mean": float(model_row["energy_mean"]),
        "qoe_mean": float(model_row["qoe_mean"]),
        "deadline_miss_mean": float(model_row["dmr_mean"]),
        "energy_per_success_mean": float(model_row["eps_mean"]),
        "step_to_75_mean": float(conv_row["step_to_threshold_mean"]) if pd.notna(conv_row["step_to_threshold_mean"]) else float("nan"),
        "best_success_mean": float(conv_row["best_success_mean"]),
        "success_auc_mean": float(conv_row["success_auc_mean"]),
        "dominant_action_profile": str(action_row["dominant_action_profile"]),
    }
    for column in [
        "metric_action_local_rate",
        "metric_action_edge_25_rate",
        "metric_action_edge_50_rate",
        "metric_action_edge_75_rate",
        "metric_action_edge_full_rate",
        "metric_action_cloud_rate",
    ]:
        payload[column] = float(action_row[column]) if column in action_row.index and pd.notna(action_row[column]) else 0.0
    return payload


def _latest_batch_value(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns or df.empty:
        return None
    values = [str(value) for value in df[column].dropna().tolist()]
    return values[-1] if values else None


def _aggregate_staged_outputs(final_csv: Path, progress_csv: Path) -> Dict[str, dict]:
    final_df = _prepare_final_df(pd.read_csv(final_csv))
    progress_df = pd.read_csv(progress_csv)
    latest_batch = _latest_batch_value(final_df, 'config_batch_id')
    if latest_batch is not None and 'config_batch_id' in final_df.columns:
        final_df = final_df[final_df['config_batch_id'].astype(str) == latest_batch].copy()
    latest_progress_batch = _latest_batch_value(progress_df, 'batch_id')
    if latest_progress_batch is not None and 'batch_id' in progress_df.columns:
        progress_df = progress_df[progress_df['batch_id'].astype(str) == latest_progress_batch].copy()
    final_summary = _aggregate_final(final_df)
    convergence_summary, _ = _aggregate_convergence(progress_df)
    action_profile = _aggregate_action_profile(final_df)
    return {
        "scratch": _extract_model_row(final_summary, action_profile, convergence_summary, "PPO_from_scratch", "scratch"),
        "pretrained": _extract_model_row(final_summary, action_profile, convergence_summary, "PPO_pretrained_finetuned", "pretrained"),
    }


def _summary_lookup(result_root: Path, teacher_policy: str) -> Dict[str, object] | None:
    summary_csv = result_root / "teacher_policy_sensitivity.csv"
    if not summary_csv.exists():
        return None
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return None
    rows = df[df["teacher_policy"] == teacher_policy]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _teacher_run_complete(result_root: Path, doc_root: Path, slug: str) -> bool:
    final_csv = result_root / slug / 'staged_training_comparison.csv'
    progress_csv = result_root / slug / 'staged_training_progress.csv'
    pretrain_csv = result_root / slug / 'supervised_pretraining_metrics.csv'
    if not (final_csv.exists() and progress_csv.exists() and pretrain_csv.exists()):
        return False
    try:
        final_df = pd.read_csv(final_csv)
        progress_df = pd.read_csv(progress_csv)
    except Exception:
        return False
    latest_final_batch = _latest_batch_value(final_df, 'config_batch_id')
    if latest_final_batch is not None and 'config_batch_id' in final_df.columns:
        final_df = final_df[final_df['config_batch_id'].astype(str) == latest_final_batch].copy()
    latest_progress_batch = _latest_batch_value(progress_df, 'batch_id')
    if latest_progress_batch is not None and 'batch_id' in progress_df.columns:
        progress_df = progress_df[progress_df['batch_id'].astype(str) == latest_progress_batch].copy()
    final_ok = len(final_df) >= 6 and {'PPO_from_scratch', 'PPO_pretrained_finetuned'}.issubset(set(final_df.get('config_model_type', [])))
    progress_ok = len(progress_df) >= 30 and {'scratch', 'pretrained'}.issubset(set(progress_df.get('init_mode', [])))
    return bool(final_ok and progress_ok)


def _write_markdown(report_path: Path, rows: List[Dict[str, object]]) -> None:
    def _pct(value: object) -> str:
        try:
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return "n/a"

    def _scalar(value: object, precision: int = 4) -> str:
        try:
            value = float(value)
            if value != value:
                return "n/a"
            return f"{value:.{precision}f}"
        except Exception:
            return "n/a"

    def _step(value: object) -> str:
        try:
            value = float(value)
            if value != value:
                return "n/a"
            return f"{value:.0f}"
        except Exception:
            return "n/a"

    def _teacher_title(name: str) -> str:
        return name.replace('teacher_', '').replace('_', ' ').title()

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Teacher Policy Sensitivity Report",
        "",
        "Bu rapor, Faz 7 kapsaminda her teacher policy icin uretilen supervised pretraining ve staged PPO fine-tuning sonuclarini tek yerde toplar.",
        "Ayrica ayri teacher markdown dosyalari tutulmaz; Faz 7 teacher bazli tum ozet burada korunur.",
        "",
        "## Kanonik Secim",
        "",
        "Kanonik Faz 7 teacher policy olarak `teacher_contextual_reward_aligned` ile devam edilmistir.",
        "Secim gerekcesi sadece en yuksek success degil, ayni zamanda daha savunulabilir decision structure elde edilmesidir.",
        "`teacher_latency_greedy` daha yuksek success delta verse de final politika `Full Cloud` tarafina kaymistir.",
        "`teacher_contextual_reward_aligned` ise `Pretrained + PPO > Scratch PPO` sonucunu korurken dominant davranisi `Edge %75` ekseninde tutmustur.",
        "",
        "## Final Karsilastirma",
        "",
        "| Teacher Policy | Coverage (train) | Pretrain Val Acc | Pretrain Test Acc | Scratch Success | Pretrained Success | Delta Success | Scratch Dominant | Pretrained Dominant | Delta Step-to-75 | Delta AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        coverage_train = "/".join(str(row[f"coverage_train_{action}"]) for action in ACTION_ORDER)
        lines.append(
            f"| {row['teacher_policy']} | {coverage_train} | "
            f"{float(row['pretrain_best_val_accuracy']):.2f}% | {float(row['pretrain_test_accuracy']):.2f}% | "
            f"{_pct(row['scratch_success_mean'])} | {_pct(row['pretrained_success_mean'])} | "
            f"{float(row['delta_success_points']):+.2f} puan | {row['scratch_dominant_action_profile']} | {row['pretrained_dominant_action_profile']} | "
            f"{_step(row['delta_step_to_75'])} | {float(row['delta_success_auc']):+.4f} |"
        )

    lines.extend([
        "",
        "Coverage (train) sirasi: `local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`.",
        "Delta kolonlari `Pretrained + PPO - Scratch PPO` olarak hesaplanmistir.",
        "",
        "## Teacher Bazli Ozetler",
        "",
    ])

    for row in rows:
        slug = row['teacher_slug']
        coverage_total = "/".join(str(row[f"coverage_total_{action}"]) for action in ACTION_ORDER)
        coverage_train = "/".join(str(row[f"coverage_train_{action}"]) for action in ACTION_ORDER)
        lines.extend([
            f"### {_teacher_title(str(row['teacher_policy']))}",
            "",
            f"- Teacher policy: `{row['teacher_policy']}`",
            f"- Teacher config seti: `configs/synthetic/teacher_policy_configs/supervised_pretraining_{slug}.yaml` ve `configs/synthetic/teacher_policy_configs/staged_training_{slug}.yaml`",
            f"- Pretrained checkpoint: `{row['pretrain_checkpoint_path']}`",
            f"- Pretraining metrics CSV: `{row['pretrain_metrics_csv']}`",
            f"- Staged final CSV: `{row['staged_final_csv']}`",
            f"- Staged progress CSV: `{row['staged_progress_csv']}`",
            f"- Coverage total (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `{coverage_total}`",
            f"- Coverage train (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `{coverage_train}`",
            f"- Supervised pretraining: best epoch `{int(row['pretrain_best_epoch'])}`, val acc `{float(row['pretrain_best_val_accuracy']):.2f}%`, test acc `{float(row['pretrain_test_accuracy']):.2f}%`",
            f"- Scratch PPO success: `{_pct(row['scratch_success_mean'])}`",
            f"- Pretrained + PPO success: `{_pct(row['pretrained_success_mean'])}`",
            f"- Delta success: `{float(row['delta_success_points']):+.2f} puan`",
            f"- Scratch dominant action: `{row['scratch_dominant_action_profile']}`",
            f"- Pretrained dominant action: `{row['pretrained_dominant_action_profile']}`",
            f"- Delta p95 latency: `{_scalar(row['delta_p95'])}`",
            f"- Delta energy: `{_scalar(row['delta_energy'])}`",
            f"- Delta QoE: `{_scalar(row['delta_qoe'], 2)}`",
            f"- Delta deadline miss: `{_scalar(row['delta_deadline_miss_points'], 2)} puan`",
            f"- Delta energy per success: `{_scalar(row['delta_energy_per_success'])}`",
            f"- Delta success AUC: `{_scalar(row['delta_success_auc'])}`",
            "",
        ])

        teacher_name = str(row['teacher_policy'])
        if teacher_name == 'teacher_contextual_reward_aligned':
            lines.extend([
                "Yorum:",
                "Bu teacher, Faz 7 icin kanonik secim olarak sabitlendi. Success artisi, dominant action'in `Full Cloud`a kaymadan korunmasi ve davranissal olarak daha savunulabilir bir warm-start saglamasi nedeniyle tercih edildi.",
                "",
            ])
        elif teacher_name == 'teacher_latency_greedy':
            lines.extend([
                "Yorum:",
                "Bu teacher en yuksek final success artisini verdi, ancak final politika `Full Cloud` agirlikli hale geldi. Bu nedenle Faz 8 oncesi davranissal risk tasidigi kabul edildi.",
                "",
            ])
        elif teacher_name == 'teacher_energy_greedy':
            lines.extend([
                "Yorum:",
                "Bu teacher success artisi saglasa da final karar yapisi yine `Full Cloud` tarafina kaydi. Energy perspektifi yararli kaldi, ancak Faz 7'nin context-sensitive behavior hedefi icin yeterli gorulmedi.",
                "",
            ])
        else:
            lines.extend([
                "Yorum:",
                "Bu teacher dengeli bir ara nokta sundu, ancak final politika yine `Full Cloud` baskinligina yaklasti. Bu nedenle kanonik secim olarak alinmadi.",
                "",
            ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_teacher_policy_sensitivity(
    pretrain_config_path: str = "configs/synthetic/supervised_pretraining.yaml",
    staged_config_path: str = "configs/synthetic/staged_training_comparison.yaml",
    dataset_csv_path: str = "results/raw/synthetic/pretraining/oracle_label_dataset.csv",
) -> Dict[str, str]:
    pretrain_base = _load_yaml(Path(pretrain_config_path))
    staged_base = _load_yaml(Path(staged_config_path))
    dataset_csv = Path(dataset_csv_path)

    result_root = Path("results/raw/synthetic/teacher_policy_sensitivity")
    doc_root = Path("v2_docs/phase_7")
    config_root = Path("configs/synthetic/teacher_policy_configs")
    config_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for teacher_policy in TEACHER_POLICIES:
        slug = _teacher_slug(teacher_policy)
        coverage = _coverage_rows(dataset_csv, teacher_policy)

        if _teacher_run_complete(result_root, doc_root, slug):
            pretrain_cfg = _load_yaml(config_root / f"supervised_pretraining_{slug}.yaml")
            staged_cfg = _load_yaml(config_root / f"staged_training_{slug}.yaml")
            pretrain_result = {
                "checkpoint_path": pretrain_cfg["output"]["checkpoint_path"] + '.zip',
                "metrics_csv": pretrain_cfg["output"]["metrics_csv"],
                "report_path": pretrain_cfg["output"]["report_path"],
                "best_epoch": 0,
                "best_val_accuracy": 0.0,
                "test_accuracy": 0.0,
            }
            metrics_df = pd.read_csv(pretrain_cfg["output"]["metrics_csv"])
            if not metrics_df.empty:
                best_row = metrics_df.loc[metrics_df['val_accuracy'].idxmax()]
                pretrain_result["best_epoch"] = int(best_row['epoch'])
                pretrain_result["best_val_accuracy"] = float(best_row['val_accuracy']) * 100.0
                pretrain_result["test_accuracy"] = 0.0
            cached_summary = _summary_lookup(result_root, teacher_policy)
            if cached_summary is not None:
                pretrain_result["test_accuracy"] = float(cached_summary.get("pretrain_test_accuracy", 0.0))
        else:
            pretrain_cfg = yaml.safe_load(yaml.safe_dump(pretrain_base))
            pretrain_cfg.setdefault("dataset", {})["teacher_policy"] = teacher_policy
            pretrain_cfg.setdefault("output", {})["checkpoint_path"] = f"models/ppo/teacher_policy_pretrained/{slug}/ppo_pretrained"
            pretrain_cfg["output"]["metrics_csv"] = f"results/raw/synthetic/teacher_policy_sensitivity/{slug}/supervised_pretraining_metrics.csv"
            pretrain_cfg["output"]["report_path"] = None
            pretrain_tmp = config_root / f"supervised_pretraining_{slug}.yaml"
            _write_yaml(pretrain_tmp, pretrain_cfg)
            pretrain_result = run_supervised_pretraining(str(pretrain_tmp))

            staged_cfg = yaml.safe_load(yaml.safe_dump(staged_base))
            staged_cfg.setdefault("pretraining", {})["checkpoint_path"] = pretrain_result["checkpoint_path"]
            schedule = staged_cfg.setdefault("training", {}).setdefault("pretrained_schedule", {})
            for phase in schedule.get("phases", []):
                anchor_cfg = phase.get("anchor_config", {})
                if anchor_cfg.get("enabled", False):
                    anchor_cfg["teacher_policy"] = teacher_policy
                    anchor_cfg["dataset_path"] = dataset_csv.as_posix()
            staged_cfg.setdefault("output", {})["model_root"] = f"models/ppo/teacher_policy_sensitivity/{slug}"
            staged_cfg["output"]["final_csv"] = f"results/raw/synthetic/teacher_policy_sensitivity/{slug}/staged_training_comparison.csv"
            staged_cfg["output"]["progress_csv"] = f"results/raw/synthetic/teacher_policy_sensitivity/{slug}/staged_training_progress.csv"
            staged_cfg["output"]["report_path"] = None
            staged_tmp = config_root / f"staged_training_{slug}.yaml"
            _write_yaml(staged_tmp, staged_cfg)
            run_staged_training_comparison(str(staged_tmp), report_only=False)

        staged_summary = _aggregate_staged_outputs(
            Path(staged_cfg["output"]["final_csv"]),
            Path(staged_cfg["output"]["progress_csv"]),
        )
        scratch = staged_summary["scratch"]
        pretrained = staged_summary["pretrained"]

        row: Dict[str, object] = {
            "teacher_policy": teacher_policy,
            "teacher_slug": slug,
            "pretrain_checkpoint_path": pretrain_result["checkpoint_path"],
            "pretrain_metrics_csv": pretrain_result["metrics_csv"],
            "pretrain_summary_report_path": "v2_docs/phase_7/teacher_policy_sensitivity_report.md",
            "pretrain_best_epoch": int(pretrain_result["best_epoch"]),
            "pretrain_best_val_accuracy": float(pretrain_result["best_val_accuracy"]),
            "pretrain_test_accuracy": float(pretrain_result["test_accuracy"]),
            "staged_final_csv": staged_cfg["output"]["final_csv"],
            "staged_progress_csv": staged_cfg["output"]["progress_csv"],
            "staged_summary_report_path": "v2_docs/phase_7/teacher_policy_sensitivity_report.md",
        }
        row.update(coverage)
        for key, value in scratch.items():
            row[f"scratch_{key}"] = value
        for key, value in pretrained.items():
            row[f"pretrained_{key}"] = value
        row["delta_success_points"] = (row["pretrained_success_mean"] - row["scratch_success_mean"]) * 100.0
        row["delta_qoe"] = row["pretrained_qoe_mean"] - row["scratch_qoe_mean"]
        row["delta_p95"] = row["pretrained_p95_mean"] - row["scratch_p95_mean"]
        row["delta_energy"] = row["pretrained_energy_mean"] - row["scratch_energy_mean"]
        row["delta_deadline_miss_points"] = (row["pretrained_deadline_miss_mean"] - row["scratch_deadline_miss_mean"]) * 100.0
        row["delta_energy_per_success"] = row["pretrained_energy_per_success_mean"] - row["scratch_energy_per_success_mean"]
        row["delta_step_to_75"] = row["pretrained_step_to_75_mean"] - row["scratch_step_to_75_mean"]
        row["delta_success_auc"] = row["pretrained_success_auc_mean"] - row["scratch_success_auc_mean"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["delta_success_points", "delta_success_auc"], ascending=[False, False])
    summary_csv = result_root / "teacher_policy_sensitivity.csv"
    summary_df.to_csv(summary_csv, index=False)
    report_path = doc_root / "teacher_policy_sensitivity_report.md"
    _write_markdown(report_path, summary_df.to_dict(orient="records"))

    return {
        "summary_csv": summary_csv.as_posix(),
        "report_path": report_path.as_posix(),
        "num_teachers": str(len(summary_rows)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Faz 7 teacher policy sensitivity sweep.")
    parser.add_argument("--pretrain-config", default="configs/synthetic/supervised_pretraining.yaml")
    parser.add_argument("--staged-config", default="configs/synthetic/staged_training_comparison.yaml")
    parser.add_argument("--dataset-csv", default="results/raw/synthetic/pretraining/oracle_label_dataset.csv")
    args = parser.parse_args()

    result = run_teacher_policy_sensitivity(
        pretrain_config_path=args.pretrain_config,
        staged_config_path=args.staged_config,
        dataset_csv_path=args.dataset_csv,
    )
    print(f"[INFO] Summary CSV: {result['summary_csv']}")
    print(f"[INFO] Report: {result['report_path']}")
    print(f"[INFO] Teacher count: {result['num_teachers']}")
