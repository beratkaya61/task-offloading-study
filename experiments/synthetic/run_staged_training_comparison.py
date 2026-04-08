#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.train_agent import train_single_agent


ACTION_COLUMN_ALIASES = {
    'metric_action_0_rate': 'metric_action_local_rate',
    'metric_action_1_rate': 'metric_action_edge_25_rate',
    'metric_action_2_rate': 'metric_action_edge_50_rate',
    'metric_action_3_rate': 'metric_action_edge_75_rate',
    'metric_action_4_rate': 'metric_action_edge_full_rate',
    'metric_action_5_rate': 'metric_action_cloud_rate',
}

ACTION_DISPLAY_ORDER = [
    ('metric_action_local_rate', 'Local'),
    ('metric_action_edge_25_rate', 'Edge %25'),
    ('metric_action_edge_50_rate', 'Edge %50'),
    ('metric_action_edge_75_rate', 'Edge %75'),
    ('metric_action_edge_full_rate', 'Full Edge'),
    ('metric_action_cloud_rate', 'Full Cloud'),
]

ACTION_NAME_BY_ID = {
    0: 'local',
    1: 'edge_25',
    2: 'edge_50',
    3: 'edge_75',
    4: 'edge_full',
    5: 'cloud',
}

SUCCESS_THRESHOLD = 0.75


def load_config(config_path='configs/synthetic/staged_training_comparison.yaml'):
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _safe_std(value):
    return 0.0 if pd.isna(value) else float(value)


def _safe_ci95(std_value, n):
    if pd.isna(std_value) or not n or int(n) <= 1:
        return 0.0
    return 1.96 * float(std_value) / np.sqrt(int(n))


def _fmt_pct(mean, std=None):
    if std is None or pd.isna(std):
        return f'{mean * 100:.2f}%'
    return f'{mean * 100:.2f}% +- {_safe_std(std) * 100:.2f}'


def _fmt_scalar(mean, std=None, precision=3):
    if std is None or pd.isna(std):
        return f'{mean:.{precision}f}'
    return f'{mean:.{precision}f} +- {_safe_std(std):.{precision}f}'


def _ensure_action_alias_columns(df):
    df = df.copy()
    for old_name, new_name in ACTION_COLUMN_ALIASES.items():
        if new_name not in df.columns and old_name in df.columns:
            df[new_name] = df[old_name]
    return df


def _persist_enriched_final_csv(final_csv):
    if not os.path.exists(final_csv):
        return
    df = pd.read_csv(final_csv)
    original_columns = list(df.columns)
    df = _ensure_action_alias_columns(df)
    if list(df.columns) != original_columns:
        df.to_csv(final_csv, index=False)


def _prepare_final_df(final_df):
    final_df = _ensure_action_alias_columns(final_df)
    final_df['metric_deadline_miss_ratio'] = 1.0 - pd.to_numeric(final_df['metric_success_rate'], errors='coerce').fillna(0.0)
    final_df['metric_energy_per_success'] = (
        pd.to_numeric(final_df['metric_avg_energy'], errors='coerce').fillna(0.0) /
        pd.to_numeric(final_df['metric_success_rate'], errors='coerce').replace(0, np.nan)
    ).fillna(0.0)
    return final_df


def _aggregate_final(df):
    grouped = (
        df.groupby('config_model_type')
        .agg(
            success_mean=('metric_success_rate', 'mean'),
            success_std=('metric_success_rate', 'std'),
            success_n=('metric_success_rate', 'count'),
            reward_mean=('metric_avg_reward', 'mean'),
            reward_std=('metric_avg_reward', 'std'),
            p95_mean=('metric_p95_latency', 'mean'),
            p95_std=('metric_p95_latency', 'std'),
            energy_mean=('metric_avg_energy', 'mean'),
            energy_std=('metric_avg_energy', 'std'),
            qoe_mean=('metric_qoe', 'mean'),
            qoe_std=('metric_qoe', 'std'),
            dmr_mean=('metric_deadline_miss_ratio', 'mean'),
            dmr_std=('metric_deadline_miss_ratio', 'std'),
            eps_mean=('metric_energy_per_success', 'mean'),
            eps_std=('metric_energy_per_success', 'std'),
        )
        .reset_index()
    )
    grouped['success_ci95'] = grouped.apply(lambda row: _safe_ci95(row['success_std'], row['success_n']), axis=1)
    return grouped


def _aggregate_progress(df):
    grouped = (
        df.groupby(['init_mode', 'training_step'])
        .agg(
            success_mean=('metric_success_rate', 'mean'),
            reward_mean=('metric_avg_reward', 'mean'),
            p95_mean=('metric_p95_latency', 'mean'),
            energy_mean=('metric_avg_energy', 'mean'),
            qoe_mean=('metric_qoe', 'mean'),
        )
        .reset_index()
        .sort_values(['init_mode', 'training_step'])
    )
    return grouped


def _aggregate_convergence(progress_df, threshold=SUCCESS_THRESHOLD):
    rows = []
    for (init_mode, seed), group in progress_df.groupby(['init_mode', 'seed']):
        group = group.sort_values('training_step')
        steps = group['training_step'].to_numpy(dtype=float)
        success_values = group['metric_success_rate'].to_numpy(dtype=float)
        threshold_hits = group[group['metric_success_rate'] >= threshold]
        step_to_threshold = float(threshold_hits['training_step'].iloc[0]) if not threshold_hits.empty else np.nan
        auc = float(np.trapz(success_values, steps) / max(steps.max(), 1.0)) if len(group) >= 2 else float(success_values.mean())
        rows.append({
            'init_mode': init_mode,
            'seed': int(seed),
            'step_to_threshold': step_to_threshold,
            'best_success': float(np.max(success_values)),
            'success_auc': auc,
        })

    seed_level = pd.DataFrame(rows)
    if seed_level.empty:
        return pd.DataFrame(), seed_level

    summary = (
        seed_level.groupby('init_mode')
        .agg(
            step_to_threshold_mean=('step_to_threshold', 'mean'),
            step_to_threshold_std=('step_to_threshold', 'std'),
            best_success_mean=('best_success', 'mean'),
            best_success_std=('best_success', 'std'),
            success_auc_mean=('success_auc', 'mean'),
            success_auc_std=('success_auc', 'std'),
        )
        .reset_index()
        .sort_values('init_mode')
    )
    return summary, seed_level


def _aggregate_action_profile(final_df):
    agg_map = {column: 'mean' for column, _ in ACTION_DISPLAY_ORDER if column in final_df.columns}
    grouped = final_df.groupby('config_model_type').agg(agg_map).reset_index()
    dominant_actions = []
    for _, row in grouped.iterrows():
        if not agg_map:
            dominant_actions.append('-')
            continue
        best_column = max(agg_map.keys(), key=lambda column: float(row.get(column, 0.0) or 0.0))
        best_label = dict(ACTION_DISPLAY_ORDER).get(best_column, best_column)
        dominant_actions.append(f"{best_label} ({float(row.get(best_column, 0.0)) * 100:.1f}%)")
    grouped['dominant_action_profile'] = dominant_actions
    return grouped


def _latest_batch_from_csv(final_csv, eval_group='synthetic_staged_training'):
    if not os.path.exists(final_csv):
        return None
    df = pd.read_csv(final_csv)
    if 'config_eval_group' in df.columns:
        df = df[df['config_eval_group'].astype(str) == str(eval_group)].copy()
    if df.empty or 'config_batch_id' not in df.columns:
        return None
    return str(df['config_batch_id'].dropna().iloc[-1])


def write_report(final_df, final_summary, progress_summary, convergence_summary, action_profile, output_path, batch_id, pretrained_checkpoint):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lines = [
        'Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md',
        '',
        '# Staged Training Comparison Report',
        '',
        f'- Batch ID: `{batch_id}`',
        f'- Pretrained checkpoint: `{pretrained_checkpoint}`',
        '- Comparison: `PPO from scratch` vs `Pretrained + PPO`',
        '- Faz 7.3 ara metrikleri: deadline miss ratio, energy per success, step-to-75% success, best success, success AUC, success 95% CI',
        '',
        '## Action Mapping',
        '',
        '| Action ID | Meaning |',
        '|---:|---|',
        '| 0 | local |',
        '| 1 | edge_25 |',
        '| 2 | edge_50 |',
        '| 3 | edge_75 |',
        '| 4 | edge_full |',
        '| 5 | cloud |',
        '',
        '## Final Karsilastirma',
        '',
        '| Model | Success Rate (mean +- std) | Success 95% CI | Deadline Miss Ratio | Avg Reward | P95 Latency | Avg Energy | Energy per Success | QoE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]

    for _, row in final_summary.iterrows():
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_fmt_pct(row['success_mean'], row['success_std'])} | "
            f"+- {row['success_ci95'] * 100:.2f}% | "
            f"{_fmt_pct(row['dmr_mean'], row['dmr_std'])} | "
            f"{_fmt_scalar(row['reward_mean'], row['reward_std'], 2)} | "
            f"{_fmt_scalar(row['p95_mean'], row['p95_std'], 3)} | "
            f"{_fmt_scalar(row['energy_mean'], row['energy_std'], 4)} | "
            f"{_fmt_scalar(row['eps_mean'], row['eps_std'], 4)} | "
            f"{_fmt_scalar(row['qoe_mean'], row['qoe_std'], 2)} |"
        )

    lines.extend([
        '',
        '## Convergence Ara Metrikleri',
        '',
        '| Init Mode | Step to 75% Success | Best Success During Training | Success Curve AUC |',
        '|---|---:|---:|---:|',
    ])

    for _, row in convergence_summary.iterrows():
        step_text = '-' if pd.isna(row['step_to_threshold_mean']) else _fmt_scalar(row['step_to_threshold_mean'], row['step_to_threshold_std'], 0)
        lines.append(
            f"| {row['init_mode']} | {step_text} | "
            f"{_fmt_pct(row['best_success_mean'], row['best_success_std'])} | "
            f"{_fmt_scalar(row['success_auc_mean'], row['success_auc_std'], 4)} |"
        )

    lines.extend([
        '',
        '## Learning Curve Snapshot',
        '',
        '| Init Mode | Training Step | Success Rate | Avg Reward | P95 Latency | Avg Energy | QoE |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])

    for _, row in progress_summary.iterrows():
        lines.append(
            f"| {row['init_mode']} | {int(row['training_step'])} | "
            f"{row['success_mean'] * 100:.2f}% | {row['reward_mean']:.2f} | {row['p95_mean']:.3f} | {row['energy_mean']:.4f} | {row['qoe_mean']:.2f} |"
        )

    lines.extend([
        '',
        '## Action Profili',
        '',
        '| Model | Dominant Action Profile | Local | Edge %25 | Edge %50 | Edge %75 | Full Edge | Full Cloud |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ])

    display_lookup = dict(ACTION_DISPLAY_ORDER)
    for _, row in action_profile.iterrows():
        values = [f"{float(row.get(column, 0.0) or 0.0) * 100:.2f}%" for column, _ in ACTION_DISPLAY_ORDER]
        lines.append(
            f"| {row['config_model_type']} | {row['dominant_action_profile']} | " + ' | '.join(values) + ' |'
        )

    if len(final_summary) == 2:
        scratch = final_summary[final_summary['config_model_type'] == 'PPO_from_scratch']
        pretrained = final_summary[final_summary['config_model_type'] == 'PPO_pretrained_finetuned']
        scratch_conv = convergence_summary[convergence_summary['init_mode'] == 'scratch']
        pretrained_conv = convergence_summary[convergence_summary['init_mode'] == 'pretrained']
        if not scratch.empty and not pretrained.empty:
            delta_success = (float(pretrained['success_mean'].iloc[0]) - float(scratch['success_mean'].iloc[0])) * 100.0
            delta_qoe = float(pretrained['qoe_mean'].iloc[0]) - float(scratch['qoe_mean'].iloc[0])
            delta_p95 = float(pretrained['p95_mean'].iloc[0]) - float(scratch['p95_mean'].iloc[0])
            delta_dmr = (float(pretrained['dmr_mean'].iloc[0]) - float(scratch['dmr_mean'].iloc[0])) * 100.0
            delta_eps = float(pretrained['eps_mean'].iloc[0]) - float(scratch['eps_mean'].iloc[0])
            lines.extend([
                '',
                '## Kisa Yorum',
                '',
                f"- Success delta (`Pretrained + PPO` - `Scratch PPO`): `{delta_success:+.2f}` puan",
                f"- QoE delta (`Pretrained + PPO` - `Scratch PPO`): `{delta_qoe:+.2f}`",
                f"- P95 latency delta (`Pretrained + PPO` - `Scratch PPO`): `{delta_p95:+.3f}` s",
                f"- Deadline miss ratio delta (`Pretrained + PPO` - `Scratch PPO`): `{delta_dmr:+.2f}` puan",
                f"- Energy per success delta (`Pretrained + PPO` - `Scratch PPO`): `{delta_eps:+.4f}`",
            ])
            if not scratch_conv.empty and not pretrained_conv.empty:
                scratch_step = float(scratch_conv['step_to_threshold_mean'].iloc[0]) if pd.notna(scratch_conv['step_to_threshold_mean'].iloc[0]) else np.nan
                pretrained_step = float(pretrained_conv['step_to_threshold_mean'].iloc[0]) if pd.notna(pretrained_conv['step_to_threshold_mean'].iloc[0]) else np.nan
                scratch_auc = float(scratch_conv['success_auc_mean'].iloc[0])
                pretrained_auc = float(pretrained_conv['success_auc_mean'].iloc[0])
                if not np.isnan(scratch_step) and not np.isnan(pretrained_step):
                    lines.append(f"- `Pretrained + PPO`, `%75` success esigine ortalama `{pretrained_step:.0f}` stepte ulasir; `Scratch PPO` ise `{scratch_step:.0f}` stepte ulasir.")
                lines.append(f"- Success curve AUC karsilastirmasi: scratch `{scratch_auc:.4f}`, pretrained `{pretrained_auc:.4f}`.")
                if pretrained_step < scratch_step:
                    lines.append('- Yorum: pretrained baslangic hizini iyilestiriyor; ancak bu batchte scratch PPO orta-ileri asamada daha guclu final dengeye oturuyor.')
                else:
                    lines.append('- Yorum: pretrained baslangic hizi acisindan belirgin ustunluk kurmuyor; buna ragmen finalde de geride kaliyor.')

    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def run_staged_training_comparison(config_path='configs/synthetic/staged_training_comparison.yaml', report_only=False, batch_id=None):
    config = load_config(config_path)
    training_cfg = config.get('training', {})
    pretraining_cfg = config.get('pretraining', {})
    anchoring_cfg = config.get('anchoring', {})
    output_cfg = config.get('output', {})

    seeds = training_cfg.get('seeds', [42, 43, 44])
    total_timesteps = training_cfg.get('total_timesteps', 30000)
    eval_episodes = training_cfg.get('eval_episodes', 10)
    eval_interval_steps = training_cfg.get('eval_interval_steps', 5000)
    progress_eval_episodes = training_cfg.get('progress_eval_episodes', 5)
    scratch_overrides = training_cfg.get('scratch_overrides', {}) or {}
    pretrained_overrides = training_cfg.get('pretrained_overrides', {}) or {}
    base_config = training_cfg.get('base_config', 'configs/synthetic/rl_training.yaml')
    pretrained_checkpoint = pretraining_cfg.get('checkpoint_path', 'models/ppo/pretrained/ppo_weighted_oracle_pretrained.zip')
    model_root = output_cfg.get('model_root', 'models/ppo/staged_training')
    final_csv = output_cfg.get('final_csv', 'results/raw/synthetic/staged_training/staged_training_comparison.csv')
    progress_csv = output_cfg.get('progress_csv', 'results/raw/synthetic/staged_training/staged_training_progress.csv')
    report_path = output_cfg.get('report_path', 'results/tables/staged_training_comparison_report.md')

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(os.path.dirname(final_csv), exist_ok=True)
    os.makedirs(os.path.dirname(progress_csv), exist_ok=True)

    current_batch_id = batch_id
    if not report_only:
        current_batch_id = datetime.now().strftime('synthetic_staged_training_%Y%m%d_%H%M%S')
        if os.path.exists(progress_csv):
            os.remove(progress_csv)

        for seed in seeds:
            scratch_save = os.path.join(model_root, 'scratch', f'seed{seed}')
            pretrained_save = os.path.join(model_root, 'pretrained', f'seed{seed}')

            train_single_agent(
                algorithm='ppo',
                total_timesteps=total_timesteps,
                seed=seed,
                save_path=scratch_save,
                run_name='PPO_from_scratch',
                config_path=base_config,
                eval_episodes=eval_episodes,
                eval_csv_path=final_csv,
                init_mode='scratch',
                progress_csv_path=progress_csv,
                eval_interval_steps=eval_interval_steps,
                progress_eval_episodes=progress_eval_episodes,
                hyperparam_overrides=scratch_overrides,
                anchor_config=None,
                extra_eval_fields={
                    'config_batch_id': current_batch_id,
                    'config_eval_group': 'synthetic_staged_training',
                },
                progress_extra_fields={
                    'batch_id': current_batch_id,
                    'eval_group': 'synthetic_staged_training',
                },
            )

            train_single_agent(
                algorithm='ppo',
                total_timesteps=total_timesteps,
                seed=seed,
                save_path=pretrained_save,
                run_name='PPO_pretrained_finetuned',
                config_path=base_config,
                eval_episodes=eval_episodes,
                eval_csv_path=final_csv,
                init_model_path=pretrained_checkpoint,
                init_mode='pretrained',
                progress_csv_path=progress_csv,
                eval_interval_steps=eval_interval_steps,
                progress_eval_episodes=progress_eval_episodes,
                hyperparam_overrides=pretrained_overrides,
                anchor_config=anchoring_cfg,
                extra_eval_fields={
                    'config_batch_id': current_batch_id,
                    'config_eval_group': 'synthetic_staged_training',
                },
                progress_extra_fields={
                    'batch_id': current_batch_id,
                    'eval_group': 'synthetic_staged_training',
                },
            )

    _persist_enriched_final_csv(final_csv)
    if current_batch_id is None:
        current_batch_id = _latest_batch_from_csv(final_csv)

    if current_batch_id is None:
        raise RuntimeError('No staged training batch found to report.')

    final_df = pd.read_csv(final_csv)
    final_df = final_df[final_df['config_batch_id'] == current_batch_id].copy()
    final_df = _prepare_final_df(final_df)

    progress_df = pd.read_csv(progress_csv)
    progress_df = progress_df[progress_df['batch_id'] == current_batch_id].copy()

    final_summary = _aggregate_final(final_df)
    progress_summary = _aggregate_progress(progress_df)
    convergence_summary, _ = _aggregate_convergence(progress_df, threshold=SUCCESS_THRESHOLD)
    action_profile = _aggregate_action_profile(final_df)
    write_report(final_df, final_summary, progress_summary, convergence_summary, action_profile, report_path, current_batch_id, pretrained_checkpoint)

    print(f'[INFO] Final CSV: {final_csv}')
    print(f'[INFO] Progress CSV: {progress_csv}')
    print(f'[INFO] Report: {report_path}')
    print(f'[INFO] Batch ID: {current_batch_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run or rebuild the staged PPO training comparison report.')
    parser.add_argument('--config-path', default='configs/synthetic/staged_training_comparison.yaml')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--batch-id', default=None)
    args = parser.parse_args()
    run_staged_training_comparison(config_path=args.config_path, report_only=args.report_only, batch_id=args.batch_id)


