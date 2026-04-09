#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.core.evaluation import evaluate_policy
from src.training.train_agent import build_training_env

ACTION_ALIAS_TO_LABEL = {
    'metric_action_0_rate': 'local',
    'metric_action_1_rate': 'edge_25',
    'metric_action_2_rate': 'edge_50',
    'metric_action_3_rate': 'edge_75',
    'metric_action_4_rate': 'edge_100',
    'metric_action_5_rate': 'cloud',
    'metric_action_local_rate': 'local',
    'metric_action_edge_25_rate': 'edge_25',
    'metric_action_edge_50_rate': 'edge_50',
    'metric_action_edge_75_rate': 'edge_75',
    'metric_action_edge_full_rate': 'edge_100',
    'metric_action_cloud_rate': 'cloud',
}
ACTION_ORDER = ['local', 'edge_25', 'edge_50', 'edge_75', 'edge_100', 'cloud']


def _load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _safe_std(series: pd.Series) -> float:
    value = float(series.std()) if len(series) > 1 else 0.0
    if value != value:
        return 0.0
    return value


def _ci95(std_value: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std_value / np.sqrt(n)


def _dominant_action(row: Dict[str, float]) -> str:
    best_action = None
    best_value = -1.0
    for action in ACTION_ORDER:
        value = float(row.get(f'pretrained_only_action_{action}_rate', 0.0) or 0.0)
        if value > best_value:
            best_action = action
            best_value = value
    if best_action is None:
        return 'n/a'
    display = {
        'local': 'Local',
        'edge_25': 'Edge %25',
        'edge_50': 'Edge %50',
        'edge_75': 'Edge %75',
        'edge_100': 'Full Edge',
        'cloud': 'Full Cloud',
    }[best_action]
    return f'{display} ({best_value * 100:.1f}%)'


def _write_markdown(report_path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        'Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md',
        '',
        '# Pretrained Checkpoint Evaluation Report',
        '',
        'Bu rapor, supervised pretraining sonrasi elde edilen checkpointlerin fine-tuning oncesi environment icindeki gercek basarisini olcer.',
        'Amac, `supervised accuracy` ile `fine-tuned RL success` arasindaki farkin teacher bilgisinin unutulmasindan mi yoksa farkli metriklerden mi kaynaklandigini daha net gostermektir.',
        '',
        '| Teacher Policy | Supervised Test Acc | Pretrained-Only Success | Fine-Tuned Success | Scratch Success | Delta (Fine-Tuned - Pretrained-Only) | Dominant Action |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['teacher_policy']} | {float(row['supervised_test_accuracy']):.2f}% | "
            f"{float(row['pretrained_only_success_mean']) * 100:.2f}% | "
            f"{float(row['fine_tuned_success_mean']) * 100:.2f}% | "
            f"{float(row['scratch_success_mean']) * 100:.2f}% | "
            f"{float(row['delta_finetuned_minus_pretrained_only_points']):+.2f} puan | "
            f"{row['pretrained_only_dominant_action']} |"
        )
    lines.extend([
        '',
        'Yorum anahtari:',
        '- `Fine-Tuned Success > Pretrained-Only Success` ise RL fine-tuning faydali okunur.',
        '- `Fine-Tuned Success ~= Pretrained-Only Success` ise RL fine-tuning sinirli katkili okunur.',
        '- `Fine-Tuned Success < Pretrained-Only Success` ise teacher retention problemi guclu bir sinyal verir.',
        '',
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_pretrained_checkpoint_evaluation(
    sensitivity_csv_path: str = 'results/raw/synthetic/teacher_policy_sensitivity/teacher_policy_sensitivity.csv',
    training_config_path: str = 'configs/synthetic/staged_training_comparison.yaml',
    output_csv_path: str = 'results/raw/synthetic/teacher_policy_sensitivity/pretrained_checkpoint_evaluation.csv',
    report_path: str = 'v2_docs/phase_7/pretrained_checkpoint_evaluation_report.md',
) -> Dict[str, str]:
    sensitivity_csv = Path(sensitivity_csv_path)
    training_cfg = _load_yaml(Path(training_config_path))
    rl_cfg = _load_yaml(Path(training_cfg['training']['base_config']))

    env_cfg = rl_cfg.get('env', {})
    seeds = training_cfg.get('training', {}).get('seeds', [42, 43, 44])
    eval_episodes = int(training_cfg.get('training', {}).get('eval_episodes', 10))

    summary_df = pd.read_csv(sensitivity_csv)
    summary_rows: List[Dict[str, object]] = []
    temp_eval_csv = Path('results/raw/synthetic/teacher_policy_sensitivity/_tmp_pretrained_only_eval.csv')
    if temp_eval_csv.exists():
        temp_eval_csv.unlink()

    for _, teacher_row in summary_df.iterrows():
        teacher_policy = str(teacher_row['teacher_policy'])
        checkpoint_path = Path(str(teacher_row['pretrain_checkpoint_path']))
        per_seed_rows = []
        for seed in seeds:
            env = build_training_env(
                seed=seed,
                max_steps=int(env_cfg.get('max_steps', 50)),
                num_edge_servers=int(env_cfg.get('num_edge_servers', 3)),
                num_devices=int(env_cfg.get('num_devices', 5)),
            )
            model = PPO.load(str(checkpoint_path), env=env)
            metrics = evaluate_policy(
                env,
                model,
                num_episodes=eval_episodes,
                run_name=f'{teacher_policy}_pretrained_only',
                semantic_mode='action_prior',
                config_seed=seed,
                csv_path=str(temp_eval_csv),
                extra_fields={
                    'config_batch_id': 'pretrained_only_probe',
                    'config_eval_group': 'teacher_policy_pretrained_only',
                },
            )
            row = {
                'teacher_policy': teacher_policy,
                'seed': seed,
                'metric_success_rate': float(metrics['metric_success_rate']),
                'metric_avg_reward': float(metrics['metric_avg_reward']),
                'metric_p95_latency': float(metrics['metric_p95_latency']),
                'metric_avg_energy': float(metrics['metric_avg_energy']),
                'metric_qoe': float(metrics['metric_qoe']),
            }
            for key, label in ACTION_ALIAS_TO_LABEL.items():
                if key in metrics:
                    row[f'action_{label}_rate'] = float(metrics[key])
            per_seed_rows.append(row)

        per_seed_df = pd.DataFrame(per_seed_rows)
        summary_row: Dict[str, object] = {
            'teacher_policy': teacher_policy,
            'teacher_slug': teacher_row['teacher_slug'],
            'pretrain_checkpoint_path': str(checkpoint_path).replace('\\', '/'),
            'supervised_test_accuracy': float(teacher_row['pretrain_test_accuracy']),
            'scratch_success_mean': float(teacher_row['scratch_success_mean']),
            'fine_tuned_success_mean': float(teacher_row['pretrained_success_mean']),
            'pretrained_only_success_mean': float(per_seed_df['metric_success_rate'].mean()),
            'pretrained_only_success_std': _safe_std(per_seed_df['metric_success_rate']),
            'pretrained_only_success_ci95': _ci95(_safe_std(per_seed_df['metric_success_rate']), len(per_seed_df)),
            'pretrained_only_reward_mean': float(per_seed_df['metric_avg_reward'].mean()),
            'pretrained_only_p95_mean': float(per_seed_df['metric_p95_latency'].mean()),
            'pretrained_only_energy_mean': float(per_seed_df['metric_avg_energy'].mean()),
            'pretrained_only_qoe_mean': float(per_seed_df['metric_qoe'].mean()),
            'delta_finetuned_minus_pretrained_only_points': (float(teacher_row['pretrained_success_mean']) - float(per_seed_df['metric_success_rate'].mean())) * 100.0,
            'delta_pretrained_only_minus_scratch_points': (float(per_seed_df['metric_success_rate'].mean()) - float(teacher_row['scratch_success_mean'])) * 100.0,
        }
        for action in ACTION_ORDER:
            col = f'action_{action}_rate'
            summary_row[f'pretrained_only_action_{action}_rate'] = float(per_seed_df[col].mean()) if col in per_seed_df.columns else 0.0
        summary_row['pretrained_only_dominant_action'] = _dominant_action(summary_row)
        summary_rows.append(summary_row)

    out_df = pd.DataFrame(summary_rows).sort_values('fine_tuned_success_mean', ascending=False)
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    _write_markdown(Path(report_path), out_df.to_dict(orient='records'))
    if temp_eval_csv.exists():
        temp_eval_csv.unlink()
    return {
        'output_csv': str(out_csv).replace('\\', '/'),
        'report_path': str(Path(report_path)).replace('\\', '/'),
        'teacher_count': str(len(out_df)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate supervised-pretrained checkpoints before RL fine-tuning.')
    parser.add_argument('--sensitivity-csv', default='results/raw/synthetic/teacher_policy_sensitivity/teacher_policy_sensitivity.csv')
    parser.add_argument('--training-config', default='configs/synthetic/staged_training_comparison.yaml')
    parser.add_argument('--output-csv', default='results/raw/synthetic/teacher_policy_sensitivity/pretrained_checkpoint_evaluation.csv')
    parser.add_argument('--report-path', default='v2_docs/phase_7/pretrained_checkpoint_evaluation_report.md')
    args = parser.parse_args()
    result = run_pretrained_checkpoint_evaluation(
        sensitivity_csv_path=args.sensitivity_csv,
        training_config_path=args.training_config,
        output_csv_path=args.output_csv,
        report_path=args.report_path,
    )
    print(f"[INFO] Output CSV: {result['output_csv']}")
    print(f"[INFO] Report: {result['report_path']}")
    print(f"[INFO] Teacher count: {result['teacher_count']}")
