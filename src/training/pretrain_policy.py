from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from stable_baselines3 import PPO

from src.core.config import load_config
from src.core.reward import calculate_reward
from src.training.train_agent import build_training_env

ACTION_LABELS = {
    0: "local",
    1: "edge_25",
    2: "edge_50",
    3: "edge_75",
    4: "edge_100",
    5: "cloud",
}

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

TEACHER_POLICY_LABELS = {
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

TEACHER_POLICY_SCORING_MODE = {
    "teacher_latency_greedy": "latency_oracle",
    "teacher_energy_greedy": "energy_oracle",
    "teacher_balanced_semantic": "weighted_objective_oracle",
    "teacher_reward_aligned": "reward_aligned_oracle",
    "teacher_contextual_reward_aligned": "reward_aligned_oracle",
}


def normalize_teacher_policy_name(name: str) -> str:
    return TEACHER_POLICY_LABELS.get(name, name)


def resolve_teacher_policy_mode(name: str) -> str:
    teacher_policy = normalize_teacher_policy_name(name)
    return TEACHER_POLICY_SCORING_MODE.get(teacher_policy, teacher_policy)


def _row_teacher_policy(row: Dict[str, object]) -> str:
    return normalize_teacher_policy_name(str(row.get("teacher_policy") or row.get("objective") or ""))


def _row_action_id(row: Dict[str, object]) -> int:
    raw_value = row.get("selected_action_id", row.get("oracle_action", -1))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _row_state_vector(row: Dict[str, object]) -> List[float]:
    if all(column in row for column in STATE_FEATURE_COLUMNS):
        return [float(row[column]) for column in STATE_FEATURE_COLUMNS]
    return [float(row[f"obs_{i}"]) for i in range(len(STATE_FEATURE_COLUMNS))]


@dataclass
class OracleDecision:
    action: int
    teacher_policy: str
    score: float
    score_margin: float
    predicted_delay: float
    predicted_energy: float
    predicted_reward: float
    deadline_met: bool
    semantic_target: str
    semantic_match: bool
    edge_energy_cost: float
    edge_energy_ratio: float
    switching_overhead: float


def _predict_action_outcome(env, action: int) -> Dict[str, float]:
    device = env.current_device
    task = env.current_task

    if isinstance(action, np.ndarray):
        action = int(action.item())
    else:
        action = int(action)

    valid_actions = getattr(env, "valid_actions", [0, 1, 2, 3, 4, 5])
    if env.ablation_flags.get("disable_partial_offloading", False):
        action = valid_actions[min(action // 2, len(valid_actions) - 1)]

    closest_edge = None
    edge_energy_cost = 0.0
    edge_energy_ratio = 1.0

    if env.edge_servers:
        closest_edge = min(
            env.edge_servers,
            key=lambda e: math.dist(getattr(device, "location", (0, 0)), e.location),
        )
        datarate, snr = env.channel.calculate_datarate(device, closest_edge)
        link_quality_factor = min(1.0, snr / 20.0)
    else:
        datarate = 10e6
        link_quality_factor = 0.5

    transmission_time_full = task.size_bits / max(datarate, 1e-6)
    tx_energy_pred_full = 0.5 * transmission_time_full
    local_comp_energy_pred_full = 1e-28 * (1e9 ** 2) * task.cpu_cycles

    edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
    ratio = edge_ratios[action]

    overhead = 0.0
    if 1 <= action <= 4:
        size_factor = min(1.0, task.size_bits / 10e6)
        coordination_factor = 1.0 if action in (1, 2, 3) else 0.35
        mobility_penalty = (1.0 - link_quality_factor) * 0.03
        transition_penalty = 0.015 if env.previous_action is not None and env.previous_action != action else 0.0
        overhead = coordination_factor * (0.01 + 0.02 * size_factor + mobility_penalty + transition_penalty)

    if action == 0:
        delay = task.cpu_cycles / 1e9
        energy = local_comp_energy_pred_full
    elif action == 5:
        delay = transmission_time_full + 0.1 + (task.cpu_cycles / 5e9)
        energy = tx_energy_pred_full
    else:
        local_part_lat = ((1 - ratio) * task.cpu_cycles) / 1e9
        local_part_en = (1 - ratio) * local_comp_energy_pred_full
        edge_tx_lat = (ratio * task.size_bits) / max(datarate, 1e-6)
        edge_comp_lat = (ratio * task.cpu_cycles) / 2e9
        edge_tx_en = 0.5 * edge_tx_lat

        delay = max(local_part_lat, edge_tx_lat + edge_comp_lat) + overhead
        energy = local_part_en + edge_tx_en

        if closest_edge is not None:
            edge_energy_cost = 1e-28 * (2e9 ** 2) * (ratio * task.cpu_cycles)
            edge_energy_budget = max(1e-6, float(getattr(closest_edge, "energy_budget", 5000.0)))
            edge_remaining = float(getattr(closest_edge, "remaining_energy", edge_energy_budget))
            edge_energy_ratio = max(0.0, min(1.0, edge_remaining / edge_energy_budget))

    reward = calculate_reward(
        action,
        delay,
        energy,
        task,
        device,
        local_comp_energy_pred_full,
        edge_energy_ratio=edge_energy_ratio,
        edge_energy_cost=edge_energy_cost,
        success_bonus=getattr(env, "success_bonus", 0.0),
    )
    if not env.ablation_flags.get("disable_mobility_features", False) and action != 0:
        reward -= (1.0 - link_quality_factor) * 10.0

    semantic = getattr(task, "semantic_analysis", {}) or {}
    semantic_target = semantic.get("recommended_target", "edge")
    semantic_match = (
        (semantic_target == "local" and action == 0)
        or (semantic_target == "edge" and 1 <= action <= 4)
        or (semantic_target == "cloud" and action == 5)
    )

    return {
        "action": action,
        "delay": float(delay),
        "energy": float(energy),
        "reward": float(reward),
        "deadline": float(getattr(task, "deadline", 1.0)),
        "deadline_met": delay <= max(0.1, getattr(task, "deadline", 1.0)),
        "semantic_target": semantic_target,
        "semantic_match": bool(semantic_match),
        "semantic_confidence": float(semantic.get("confidence", 0.5)),
        "priority_score": float(semantic.get("priority_score", 0.5)),
        "link_quality_factor": float(link_quality_factor),
        "edge_energy_cost": float(edge_energy_cost),
        "edge_energy_ratio": float(edge_energy_ratio),
        "switching_overhead": float(overhead),
        "local_energy_reference": float(local_comp_energy_pred_full),
    }


def _score_outcome(
    outcome: Dict[str, float],
    objective: str,
    battery_ratio: float,
    task,
    scoring_cfg: Dict[str, float] | None = None,
) -> float:
    scoring_cfg = scoring_cfg or {}
    delay = outcome["delay"]
    energy = outcome["energy"]
    deadline = max(0.1, outcome["deadline"])
    deadline_penalty = 0.0 if outcome["deadline_met"] else 2.0
    energy_ref = max(1e-6, outcome["local_energy_reference"])
    delay_norm = delay / deadline
    energy_norm = energy / energy_ref
    cloud_penalty = scoring_cfg.get("cloud_penalty", 0.85) if outcome["action"] == 5 else 0.0
    semantic_bonus = -scoring_cfg.get("semantic_match_bonus", 0.12) if outcome["semantic_match"] else scoring_cfg.get("semantic_mismatch_penalty", 0.08)
    battery_risk = 0.0
    if battery_ratio < 0.25 and outcome["action"] != 0:
        battery_risk = (0.25 - battery_ratio) / 0.25
    edge_risk = 0.0
    if 1 <= outcome["action"] <= 4 and outcome["edge_energy_ratio"] < 0.25:
        edge_risk = (0.25 - outcome["edge_energy_ratio"]) / 0.25

    size_norm = min(1.0, getattr(task, "size_bits", 0.0) / 1e7)
    cpu_norm = min(1.0, getattr(task, "cpu_cycles", 0.0) / 1e10)
    priority_score = float((getattr(task, "semantic_analysis", {}) or {}).get("priority_score", 0.5))
    semantic_target = outcome["semantic_target"]

    partial_bonus = 0.0
    if outcome["action"] in (1, 2, 3):
        partial_bonus = scoring_cfg.get("partial_bonus", 0.22) * (0.35 + 0.35 * size_norm + 0.30 * priority_score)

    edge_bonus = 0.0
    if 1 <= outcome["action"] <= 4 and semantic_target == "edge":
        edge_bonus = scoring_cfg.get("edge_semantic_bonus", 0.18) * (0.4 + 0.6 * priority_score)

    local_bonus = 0.0
    if outcome["action"] == 0 and semantic_target == "local":
        local_bonus = scoring_cfg.get("local_semantic_bonus", 0.12) * (0.4 + 0.6 * (1.0 - size_norm))

    local_penalty = 0.0
    if outcome["action"] == 0:
        local_penalty = scoring_cfg.get("local_large_task_penalty", 0.22) * (0.55 * size_norm + 0.45 * cpu_norm)

    full_edge_penalty = 0.0
    if outcome["action"] == 4:
        full_edge_penalty = scoring_cfg.get("full_edge_penalty", 0.05)

    if objective == "latency_oracle":
        return delay + (deadline_penalty * deadline) + (0.15 * energy_norm)
    if objective == "energy_oracle":
        return energy + (0.20 * delay_norm) + (deadline_penalty * 0.5)
    if objective == "reward_aligned_oracle":
        semantic_confidence = float(outcome.get("semantic_confidence", 0.5))
        link_quality = float(outcome.get("link_quality_factor", 0.5))
        action = int(outcome["action"])

        local_suitability = 0.45 * (1.0 - size_norm) + 0.35 * (1.0 - cpu_norm) + 0.20 * (1.0 - link_quality)
        edge25_suitability = 0.40 * (1.0 - size_norm) + 0.25 * battery_ratio + 0.20 * priority_score + 0.15 * link_quality
        edge50_suitability = 0.45 * (1.0 - abs(size_norm - 0.45)) + 0.20 * (1.0 - abs(cpu_norm - 0.45)) + 0.20 * priority_score + 0.15 * link_quality
        edge75_suitability = 0.40 * size_norm + 0.25 * priority_score + 0.20 * semantic_confidence + 0.15 * link_quality
        edge100_suitability = 0.35 * cpu_norm + 0.25 * size_norm + 0.20 * battery_ratio + 0.20 * link_quality
        cloud_suitability = 0.45 * cpu_norm + 0.30 * size_norm + 0.15 * max(0.0, 1.0 - outcome["edge_energy_ratio"]) + 0.10 * (1.0 - link_quality)

        action_context_bonus = 0.0
        if action == 0:
            action_context_bonus = scoring_cfg.get("reward_aligned_local_bonus", 0.26) * local_suitability
        elif action == 1:
            action_context_bonus = scoring_cfg.get("reward_aligned_edge25_bonus", 0.22) * edge25_suitability
        elif action == 2:
            action_context_bonus = scoring_cfg.get("reward_aligned_edge50_bonus", 0.20) * edge50_suitability
        elif action == 3:
            action_context_bonus = scoring_cfg.get("reward_aligned_edge75_bonus", 0.18) * edge75_suitability
        elif action == 4:
            action_context_bonus = scoring_cfg.get("reward_aligned_edge100_bonus", 0.20) * edge100_suitability
        elif action == 5:
            action_context_bonus = scoring_cfg.get("reward_aligned_cloud_bonus", 0.16) * cloud_suitability

        semantic_alignment_bonus = 0.0
        if semantic_target == "local" and action == 0:
            semantic_alignment_bonus = scoring_cfg.get("reward_aligned_semantic_bonus", 0.22) * (0.5 + 0.5 * semantic_confidence)
        elif semantic_target == "edge" and action in (1, 2, 3, 4):
            semantic_alignment_bonus = scoring_cfg.get("reward_aligned_semantic_bonus", 0.22) * (0.45 + 0.55 * semantic_confidence)
        elif semantic_target == "cloud" and action == 5:
            semantic_alignment_bonus = scoring_cfg.get("reward_aligned_semantic_bonus", 0.22) * (0.4 + 0.6 * semantic_confidence)

        cloud_off_target_penalty = 0.0
        if action == 5 and semantic_target != "cloud":
            cloud_off_target_penalty = scoring_cfg.get("reward_aligned_cloud_penalty", 0.18) * (0.35 + 0.65 * priority_score)

        reward_score = -float(outcome["reward"])
        reward_score += 0.10 * delay_norm
        reward_score += 0.05 * energy_norm
        reward_score += cloud_off_target_penalty
        reward_score -= action_context_bonus
        reward_score -= semantic_alignment_bonus
        return reward_score

    return (
        scoring_cfg.get("delay_weight", 0.46) * delay_norm
        + scoring_cfg.get("energy_weight", 0.14) * energy_norm
        + scoring_cfg.get("deadline_weight", 0.16) * deadline_penalty
        + scoring_cfg.get("cloud_weight", 0.18) * cloud_penalty
        + scoring_cfg.get("battery_weight", 0.04) * battery_risk
        + scoring_cfg.get("edge_risk_weight", 0.02) * edge_risk
        + local_penalty
        + full_edge_penalty
        + semantic_bonus
        - partial_bonus
        - edge_bonus
        - local_bonus
    )


def _contextual_target_action(task, battery_ratio: float, candidates: List[OracleDecision]) -> int:
    semantic = getattr(task, "semantic_analysis", {}) or {}
    semantic_target = semantic.get("recommended_target", "edge")
    size_norm = min(1.0, getattr(task, "size_bits", 0.0) / 1e7)
    cpu_norm = min(1.0, getattr(task, "cpu_cycles", 0.0) / 1e10)
    intensity = 0.55 * size_norm + 0.45 * cpu_norm

    if semantic_target == "local":
        return 0
    if semantic_target == "cloud":
        if intensity < 0.35 and battery_ratio > 0.35:
            return 2
        return 5

    if battery_ratio < 0.18 and intensity < 0.22:
        return 1
    if intensity < 0.18:
        return 1
    if intensity < 0.40:
        return 2
    if intensity < 0.72:
        return 3
    return 4


def _coverage_lookup(mapping: Dict[object, float] | None, action_id: int, default: float) -> float:
    if not mapping:
        return float(default)
    if action_id in mapping:
        return float(mapping[action_id])
    action_name = ACTION_LABELS.get(action_id)
    if action_name in mapping:
        return float(mapping[action_name])
    action_key = str(action_id)
    if action_key in mapping:
        return float(mapping[action_key])
    return float(default)


def _resolve_coverage_selection_cfg(teacher_policy: str, scoring_cfg: Dict[str, float] | None) -> Dict[str, object] | None:
    scoring_cfg = scoring_cfg or {}
    coverage_cfg = (scoring_cfg.get("coverage_aware_selection", {}) or {})
    enabled_teachers = coverage_cfg.get("teacher_policies", [])
    normalized_teachers = {normalize_teacher_policy_name(str(name)) for name in enabled_teachers}
    if teacher_policy not in normalized_teachers:
        return None

    resolved_cfg: Dict[str, object] = {
        key: value for key, value in coverage_cfg.items() if key not in {"teacher_policies", "teacher_overrides"}
    }
    teacher_overrides = (coverage_cfg.get("teacher_overrides", {}) or {}).get(teacher_policy, {})
    if teacher_overrides:
        resolved_cfg.update(teacher_overrides)
    return resolved_cfg


def _select_contextual_candidate(candidates: List[OracleDecision], task, battery_ratio: float, scoring_cfg: Dict[str, float] | None, action_usage_counter: Counter | None = None) -> OracleDecision:
    selection_cfg = scoring_cfg or {}
    ranked = sorted(candidates, key=lambda x: (x.score, x.predicted_delay, x.predicted_energy))
    best = ranked[0]
    preferred_action = _contextual_target_action(task, battery_ratio, ranked)
    coverage_counter = action_usage_counter or Counter()
    total_seen = sum(coverage_counter.values())
    target_ratios = selection_cfg.get(
        "contextual_target_ratios",
        {"local": 0.12, "edge_25": 0.14, "edge_50": 0.18, "edge_75": 0.20, "edge_100": 0.12, "cloud": 0.24},
    )
    per_action_slack = selection_cfg.get(
        "contextual_action_slack",
        {"local": 0.90, "edge_25": 1.00, "edge_50": 0.90, "edge_75": 0.60, "edge_100": 1.10, "cloud": 0.55},
    )
    coverage_weight = float(selection_cfg.get("contextual_coverage_weight", 1.4))
    preference_bonus = float(selection_cfg.get("contextual_preference_bonus", 0.30))
    semantic_bonus = float(selection_cfg.get("contextual_semantic_bonus", 0.18))
    gap_penalty = float(selection_cfg.get("contextual_gap_penalty", 0.85))

    feasible = []
    for cand in ranked:
        allowed_gap = _coverage_lookup(per_action_slack, cand.action, 0.5)
        if cand.score <= best.score + allowed_gap:
            feasible.append(cand)
    if not feasible:
        return best

    bootstrap_min_counts = selection_cfg.get("contextual_bootstrap_min_counts", {})
    bootstrap_action_slack = selection_cfg.get("contextual_bootstrap_action_slack", per_action_slack)
    bootstrap_candidates = []
    for cand in feasible:
        required_count = int(round(_coverage_lookup(bootstrap_min_counts, cand.action, 0.0)))
        current_count = int(coverage_counter.get(cand.action, 0))
        if required_count <= 0 or current_count >= required_count:
            continue
        bootstrap_gap = _coverage_lookup(bootstrap_action_slack, cand.action, 0.0)
        if cand.score <= best.score + bootstrap_gap:
            bootstrap_candidates.append((required_count - current_count, cand.score, cand.predicted_delay, cand))
    if bootstrap_candidates:
        bootstrap_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return bootstrap_candidates[0][3]

    best_candidate = best
    best_utility = -1e9
    for cand in feasible:
        current_ratio = (coverage_counter.get(cand.action, 0) / total_seen) if total_seen else 0.0
        target_ratio = _coverage_lookup(target_ratios, cand.action, 0.0)
        coverage_deficit = max(0.0, target_ratio - current_ratio)
        utility = 0.0
        if cand.action == preferred_action:
            utility += preference_bonus
        if cand.semantic_match:
            utility += semantic_bonus
        utility += coverage_weight * coverage_deficit
        utility -= gap_penalty * max(0.0, cand.score - best.score)
        utility -= 0.02 * cand.predicted_delay
        if utility > best_utility:
            best_utility = utility
            best_candidate = cand
    return best_candidate


def choose_oracle_action(
    env,
    teacher_policy: str = "teacher_balanced_semantic",
    scoring_cfg: Dict[str, float] | None = None,
    action_usage_counter: Counter | None = None,
) -> OracleDecision:
    teacher_policy = normalize_teacher_policy_name(teacher_policy)
    scoring_mode = resolve_teacher_policy_mode(teacher_policy)
    battery_ratio = min(1.0, max(0.0, getattr(env.current_device, "battery", 10000.0) / 10000.0))
    candidates: List[OracleDecision] = []

    for action in getattr(env, "valid_actions", [0, 1, 2, 3, 4, 5]):
        outcome = _predict_action_outcome(env, action)
        score = _score_outcome(outcome, scoring_mode, battery_ratio, env.current_task, scoring_cfg)
        candidates.append(
            OracleDecision(
                action=outcome["action"],
                teacher_policy=teacher_policy,
                score=float(score),
                score_margin=0.0,
                predicted_delay=outcome["delay"],
                predicted_energy=outcome["energy"],
                predicted_reward=outcome["reward"],
                deadline_met=bool(outcome["deadline_met"]),
                semantic_target=outcome["semantic_target"],
                semantic_match=bool(outcome["semantic_match"]),
                edge_energy_cost=outcome["edge_energy_cost"],
                edge_energy_ratio=outcome["edge_energy_ratio"],
                switching_overhead=outcome["switching_overhead"],
            )
        )

    ranked = sorted(candidates, key=lambda x: (x.score, x.predicted_delay, x.predicted_energy))
    selected = ranked[0]
    selection_cfg = _resolve_coverage_selection_cfg(teacher_policy, scoring_cfg)
    if selection_cfg is not None:
        selected = _select_contextual_candidate(ranked, env.current_task, battery_ratio, selection_cfg, action_usage_counter)
    second_best_score = ranked[1].score if len(ranked) > 1 else ranked[0].score
    selected.score_margin = float(second_best_score - selected.score)
    return selected


def _split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def _write_dataset(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rebalance_teacher_train_rows(rows: List[Dict[str, object]], teacher_policy: str, config: Dict[str, object]) -> List[Dict[str, object]]:
    rebalance_cfg = (config.get("dataset", {}) or {}).get("teacher_train_rebalance", {})
    enabled = bool(rebalance_cfg.get("enabled", False))
    normalized_teacher = normalize_teacher_policy_name(teacher_policy)
    target_teachers = rebalance_cfg.get("teacher_policies")
    if target_teachers is None:
        target_teachers = [rebalance_cfg.get("teacher_policy", normalized_teacher)]
    normalized_targets = {normalize_teacher_policy_name(str(name)) for name in target_teachers}
    if not enabled or normalized_teacher not in normalized_targets:
        return rows

    train_rows = [row for row in rows if row.get("split") == "train"]
    other_rows = [row for row in rows if row.get("split") != "train"]
    if not train_rows:
        return rows

    target_ratios = rebalance_cfg.get(
        "target_ratios",
        {"local": 0.12, "edge_25": 0.12, "edge_50": 0.14, "edge_75": 0.22, "edge_100": 0.12, "cloud": 0.28},
    )
    action_order = ["local", "edge_25", "edge_50", "edge_75", "edge_100", "cloud"]
    total_count = len(train_rows)
    rng = random.Random(int(config.get("seed", 42)) + 997)

    target_counts = {}
    running_total = 0
    for action_name in action_order[:-1]:
        count = int(round(total_count * float(target_ratios.get(action_name, 0.0))))
        target_counts[action_name] = count
        running_total += count
    target_counts[action_order[-1]] = max(0, total_count - running_total)

    grouped = defaultdict(list)
    for row in train_rows:
        grouped[str(row.get("selected_action_name", ""))].append(row)

    balanced_rows = []
    for action_name in action_order:
        source_rows = grouped.get(action_name, [])
        target_count = int(target_counts.get(action_name, 0))
        if target_count <= 0 or not source_rows:
            continue
        if len(source_rows) >= target_count:
            balanced_rows.extend(rng.sample(source_rows, target_count))
        else:
            balanced_rows.extend(source_rows)
            extra_needed = target_count - len(source_rows)
            balanced_rows.extend(rng.choices(source_rows, k=extra_needed))

    rng.shuffle(balanced_rows)
    return balanced_rows + other_rows


def _write_summary(report_path: Path, rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_counters = defaultdict(Counter)
    split_counters = Counter()
    for row in rows:
        teacher_counters[row["teacher_policy"]][row["selected_action_name"]] += 1
        split_counters[row["split"]] += 1

    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Synthetic Oracle Label Summary",
        "",
        "Bu rapor, Faz 7 icin uretilen oracle label dataset'inin ilk ozetini verir.",
        "Dataset, supervised pretraining oncesi observation -> action etiketi ureten ogretmen mekanizmanin cikisidir.",
        "",
        "## Uretim Ayarlari",
        "",
        f"- Seed: `{config.get('seed', 42)}`",
        f"- Episode sayisi: `{config.get('dataset', {}).get('n_episodes', 60)}`",
        f"- Teacher policies: `{', '.join(config.get('teacher_policies', config.get('objectives', [])))}`",
        "",
        "## Split Dagilimi",
        "",
        "| Split | Sample Count |",
        "|---|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {split_counters.get(split, 0)} |")

    lines.extend([
        "",
        "## Teacher Policy Bazli Action Dagilimi",
        "",
    ])

    for teacher_policy, counter in teacher_counters.items():
        total = sum(counter.values())
        lines.extend([
            f"### {teacher_policy}",
            "",
            "| Action | Count | Ratio |",
            "|---|---:|---:|",
        ])
        for action_name, count in sorted(counter.items()):
            ratio = (count / total) if total else 0.0
            lines.append(f"| {action_name} | {count} | {ratio:.2%} |")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_oracle_dataset(config_path: str = "configs/synthetic/oracle_labeling.yaml") -> Dict[str, str]:
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    env_cfg = config.get("env", {})
    scoring_cfg = config.get("scoring", {})
    teacher_policies: Iterable[str] = config.get(
        "teacher_policies",
        config.get(
            "objectives",
            [
                "teacher_latency_greedy",
                "teacher_energy_greedy",
                "teacher_balanced_semantic",
                "teacher_reward_aligned",
            ],
        ),
    )

    seed = int(config.get("seed", 42))
    n_episodes = int(dataset_cfg.get("n_episodes", 60))
    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    max_steps = int(env_cfg.get("max_steps", 50))
    num_edge_servers = int(env_cfg.get("num_edge_servers", 3))
    num_devices = int(env_cfg.get("num_devices", 5))

    all_rows: List[Dict[str, object]] = []
    action_usage_counters = defaultdict(Counter)

    for objective_index, teacher_policy_name in enumerate(teacher_policies):
        env = build_training_env(
            seed=seed + objective_index,
            max_steps=max_steps,
            num_edge_servers=num_edge_servers,
            num_devices=num_devices,
        )

        teacher_rows: List[Dict[str, object]] = []

        for episode_idx in range(n_episodes):
            obs, _ = env.reset(seed=seed + objective_index + episode_idx)
            done = False
            step_idx = 0
            split = _split_name(episode_idx, n_episodes, train_ratio, val_ratio)

            while not done:
                teacher_policy = normalize_teacher_policy_name(teacher_policy_name)
                decision = choose_oracle_action(env, teacher_policy, scoring_cfg, action_usage_counter=action_usage_counters[teacher_policy])
                semantic = getattr(env.current_task, "semantic_analysis", {}) or {}
                row = {
                    "teacher_policy": teacher_policy,
                    "split": split,
                    "episode_id": episode_idx,
                    "step_id": step_idx,
                    "selected_action_id": decision.action,
                    "selected_action_name": ACTION_LABELS[decision.action],
                    "teacher_score": round(decision.score, 6),
                    "teacher_margin": round(decision.score_margin, 6),
                    "predicted_delay": round(decision.predicted_delay, 6),
                    "predicted_energy": round(decision.predicted_energy, 6),
                    "predicted_reward": round(decision.predicted_reward, 6),
                    "deadline_met": int(decision.deadline_met),
                    "semantic_target": decision.semantic_target,
                    "semantic_match": int(decision.semantic_match),
                    "priority_score": round(float(semantic.get("priority_score", 0.0)), 6),
                    "semantic_confidence": round(float(semantic.get("confidence", 0.0)), 6),
                }
                state_values = np.asarray(obs, dtype=np.float32).tolist()
                for index, column in enumerate(STATE_FEATURE_COLUMNS):
                    row[column] = round(float(state_values[index]), 6)
                teacher_rows.append(row)
                action_usage_counters[teacher_policy][decision.action] += 1

                obs, _, done, _, _ = env.step(decision.action)
                step_idx += 1

        teacher_rows = _rebalance_teacher_train_rows(teacher_rows, teacher_policy_name, config)
        all_rows.extend(teacher_rows)

    csv_path = Path(config.get("output", {}).get("csv_path", "results/raw/synthetic/pretraining/oracle_label_dataset.csv"))
    report_path = Path(config.get("output", {}).get("report_path", "v2_docs/phase_7/synthetic_oracle_label_summary.md"))
    _write_dataset(csv_path, all_rows)
    _write_summary(report_path, all_rows, config)

    return {
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "num_rows": str(len(all_rows)),
    }



class OracleLabelDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]]):
        self.observations = torch.tensor(
            [_row_state_vector(row) for row in rows],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([_row_action_id(row) for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.observations[index], self.labels[index]


def _read_oracle_rows(csv_path: Path, teacher_policy: str, min_margin: float = 0.0) -> List[Dict[str, object]]:
    requested_teacher = normalize_teacher_policy_name(teacher_policy)
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if _row_teacher_policy(row) == requested_teacher]
    if min_margin > 0.0:
        filtered = []
        for row in rows:
            try:
                margin = float(row.get("teacher_margin", row.get("oracle_margin", 0.0)))
            except (TypeError, ValueError):
                margin = 0.0
            if margin >= min_margin:
                filtered.append(row)
        return filtered
    return rows


def _split_rows(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split", "train"))
        grouped.setdefault(split, []).append(row)
    return grouped


def _build_train_loader(dataset: Dataset, batch_size: int, balance_actions: bool = False, balance_power: float = 1.0, samples_per_epoch: int | None = None) -> DataLoader:
    if len(dataset) == 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if not balance_actions:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels = dataset.labels.cpu().tolist()
    label_counts = Counter(labels)
    weights = []
    for label in labels:
        count = max(1, int(label_counts.get(label, 1)))
        weights.append(1.0 / (count ** float(balance_power)))
    num_samples = int(samples_per_epoch or len(labels))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _build_pretraining_model(config: Dict[str, object]):
    env = build_training_env(
        seed=int(config.get("seed", 42)),
        max_steps=int(config.get("env", {}).get("max_steps", 50)),
        num_edge_servers=int(config.get("env", {}).get("num_edge_servers", 3)),
        num_devices=int(config.get("env", {}).get("num_devices", 5)),
    )
    ppo_cfg = config.get("ppo", {})
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        device=str(ppo_cfg.get("device", "cpu")),
        verbose=0,
    )
    return model, env


def _policy_logits(model, observations: torch.Tensor) -> torch.Tensor:
    distribution = model.policy.get_distribution(observations.to(model.device))
    return distribution.distribution.logits


def _evaluate_supervised(model, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.policy.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for observations, labels in loader:
            labels = labels.to(model.device)
            logits = _policy_logits(model, observations)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.shape[0])
    return {
        "loss": (total_loss / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
    }


def run_supervised_pretraining(config_path: str = "configs/synthetic/supervised_pretraining.yaml") -> Dict[str, str]:
    config = load_config(config_path)
    dataset_path = Path(config.get("dataset", {}).get("csv_path", "results/raw/synthetic/pretraining/oracle_label_dataset.csv"))
    teacher_policy = normalize_teacher_policy_name(
        str(config.get("dataset", {}).get("teacher_policy", config.get("dataset", {}).get("objective", "teacher_balanced_semantic")))
    )
    dataset_cfg = config.get("dataset", {})
    min_margin = float(dataset_cfg.get("min_margin", 0.0))
    batch_size = int(config.get("training", {}).get("batch_size", 128))
    balance_actions = bool(dataset_cfg.get("balance_actions", False))
    balance_power = float(dataset_cfg.get("balance_power", 1.0))
    samples_per_epoch = dataset_cfg.get("samples_per_epoch")
    epochs = int(config.get("training", {}).get("epochs", 15))
    learning_rate = float(config.get("training", {}).get("learning_rate", 1e-3))
    patience = int(config.get("training", {}).get("early_stopping_patience", 5))
    min_delta = float(config.get("training", {}).get("early_stopping_min_delta", 1e-4))

    rows = _read_oracle_rows(dataset_path, teacher_policy, min_margin=min_margin)
    split_rows = _split_rows(rows)
    train_ds = OracleLabelDataset(split_rows.get("train", []))
    val_ds = OracleLabelDataset(split_rows.get("val", []))
    test_ds = OracleLabelDataset(split_rows.get("test", []))

    train_loader = _build_train_loader(train_ds, batch_size=batch_size, balance_actions=balance_actions, balance_power=balance_power, samples_per_epoch=samples_per_epoch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model, env = _build_pretraining_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)

    metrics_rows: List[Dict[str, object]] = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.policy.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for observations, labels in train_loader:
            labels = labels.to(model.device)
            optimizer.zero_grad()
            logits = _policy_logits(model, observations)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * int(labels.shape[0])
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += int(labels.shape[0])

        train_loss = (train_loss_sum / train_total) if train_total else 0.0
        train_acc = (train_correct / train_total) if train_total else 0.0
        val_metrics = _evaluate_supervised(model, val_loader, criterion)

        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )

        if val_metrics["accuracy"] > (best_val_acc + min_delta):
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_path = Path(config.get("output", {}).get("checkpoint_path", "models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained"))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_early = True
            break

    checkpoint_path = Path(config.get("output", {}).get("checkpoint_path", "models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained"))
    best_model = PPO.load(str(checkpoint_path) + ".zip", env=env)
    final_train = _evaluate_supervised(best_model, train_loader, criterion)
    final_val = _evaluate_supervised(best_model, val_loader, criterion)
    final_test = _evaluate_supervised(best_model, test_loader, criterion)

    metrics_csv = Path(config.get("output", {}).get("metrics_csv", "results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/supervised_pretraining_metrics.csv"))
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary_path_value = config.get("output", {}).get("report_path")
    summary_path = Path(summary_path_value) if summary_path_value else None
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Supervised Pretraining Report",
        "",
        f"- Teacher policy: `{teacher_policy}`",
        f"- Min margin filter: `{min_margin}`",
        f"- Action-balanced sampling: `{'yes' if balance_actions else 'no'}`",
        f"- Balance power: `{balance_power}`",
        f"- Configured epoch count: `{epochs}`",
        f"- Executed epoch count: `{len(metrics_rows)}`",
        f"- Early stopping patience: `{patience}`",
        f"- Early stopping triggered: `{'yes' if stopped_early else 'no'}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Best validation accuracy: `{best_val_acc * 100:.2f}%`",
        f"- Final train accuracy: `{final_train['accuracy'] * 100:.2f}%`",
        f"- Final validation accuracy: `{final_val['accuracy'] * 100:.2f}%`",
        f"- Final test accuracy: `{final_test['accuracy'] * 100:.2f}%`",
        f"- Metrics CSV: `{metrics_csv.as_posix()}`",
        f"- Checkpoint: `{str(checkpoint_path.as_posix())}.zip`",
        "",
        "Bu rapor, Faz 7 kapsaminda PPO policy aginin teacher-labeled oracle dataset ile supervised sekilde isitilmasinin sonucunu ozetler.",
    ]
    if summary_path is not None:
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "metrics_csv": str(metrics_csv),
        "report_path": str(summary_path) if summary_path is not None else "",
        "checkpoint_path": str(checkpoint_path) + ".zip",
        "best_epoch": str(best_epoch),
        "best_val_accuracy": f"{best_val_acc * 100:.2f}",
        "test_accuracy": f"{final_test['accuracy'] * 100:.2f}",
        "executed_epochs": str(len(metrics_rows)),
        "early_stopping_triggered": "yes" if stopped_early else "no",
    }



