from __future__ import annotations

import csv
import random
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.agents.graph_policy import GraphPolicyNetwork, build_graph_policy_from_state
from src.core.config import load_config
from src.env.graph_state_builder import ACTION_NAMES, GraphState, build_graph_state
from src.training.pretrain_policy import choose_oracle_action, normalize_teacher_policy_name
from src.training.train_agent import build_training_env
from src.utils.reproducibility import set_seed


@dataclass
class GraphPolicySample:
    graph_state: GraphState
    label: int
    split: str
    teacher_policy: str


class GraphOracleDataset(Dataset):
    def __init__(self, samples: Iterable[GraphPolicySample]):
        self.samples = list(samples)
        self.labels = torch.tensor([sample.label for sample in self.samples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphPolicySample:
        return self.samples[index]


def _collate_graph_samples(batch: List[GraphPolicySample]) -> Tuple[List[GraphState], torch.Tensor]:
    return [sample.graph_state for sample in batch], torch.tensor([sample.label for sample in batch], dtype=torch.long)


def _split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def _graph_state_from_env(env, obs: np.ndarray) -> GraphState:
    trace_context = {"mode": "trace"} if getattr(env, "trace_mode", False) else {}
    return build_graph_state(
        device=env.current_device,
        task=env.current_task,
        edge_servers=env.edge_servers,
        cloud_server=env.cloud_server,
        channel=env.channel,
        previous_action=env.previous_action,
        current_step=env.current_step,
        ablation_flags=env.ablation_flags,
        semantic_analysis=getattr(env.current_task, "semantic_analysis", None),
        semantic_prior=np.asarray(obs[6:12], dtype=np.float32),
        trace_context=trace_context,
        vector_state_reference=np.asarray(obs, dtype=np.float32),
    )


def _generate_graph_oracle_samples(config: Dict[str, object]) -> List[GraphPolicySample]:
    dataset_cfg = config.get("dataset", {}) or {}
    env_cfg = config.get("env", {}) or {}
    scoring_cfg = config.get("scoring", {}) or {}

    teacher_policy = normalize_teacher_policy_name(str(dataset_cfg.get("teacher_policy", "teacher_contextual_reward_aligned")))
    seed = int(config.get("seed", 42))
    n_episodes = int(dataset_cfg.get("n_episodes", 18))
    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    max_steps = int(env_cfg.get("max_steps", 30))
    num_edge_servers = int(env_cfg.get("num_edge_servers", 3))
    num_devices = int(env_cfg.get("num_devices", 5))

    env = build_training_env(
        seed=seed,
        max_steps=max_steps,
        num_edge_servers=num_edge_servers,
        num_devices=num_devices,
    )

    samples: List[GraphPolicySample] = []
    action_usage_counter = Counter()
    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        split = _split_name(episode_idx, n_episodes, train_ratio, val_ratio)

        while not done:
            graph_state = _graph_state_from_env(env, obs)
            decision = choose_oracle_action(
                env,
                teacher_policy=teacher_policy,
                scoring_cfg=scoring_cfg,
                action_usage_counter=action_usage_counter,
            )
            samples.append(
                GraphPolicySample(
                    graph_state=graph_state,
                    label=int(decision.action),
                    split=split,
                    teacher_policy=teacher_policy,
                )
            )
            action_usage_counter[int(decision.action)] += 1
            obs, _, done, truncated, _ = env.step(decision.action)
            done = done or truncated

    return samples


def _split_samples(samples: List[GraphPolicySample]) -> Dict[str, List[GraphPolicySample]]:
    grouped = {"train": [], "val": [], "test": []}
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped


def _build_loader(
    dataset: GraphOracleDataset,
    batch_size: int,
    balance_actions: bool,
    balance_power: float,
    samples_per_epoch: int | None,
    shuffle: bool,
) -> DataLoader:
    if len(dataset) == 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_graph_samples)
    if not balance_actions:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_graph_samples)

    labels = dataset.labels.cpu().tolist()
    label_counts = Counter(labels)
    weights = [1.0 / (max(1, label_counts[label]) ** float(balance_power)) for label in labels]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=int(samples_per_epoch or len(labels)),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=_collate_graph_samples)


def _batch_logits(model: GraphPolicyNetwork, graph_states: List[GraphState]) -> torch.Tensor:
    return torch.stack([model(graph_state).masked_logits for graph_state in graph_states], dim=0)


def _evaluate_graph_policy(model: GraphPolicyNetwork, loader: DataLoader, criterion: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    label_counter = Counter()
    pred_counter = Counter()

    with torch.no_grad():
        for graph_states, labels in loader:
            labels = labels.to(next(model.parameters()).device)
            logits = _batch_logits(model, graph_states)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += float(loss.item()) * int(labels.shape[0])
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
            label_counter.update(int(label) for label in labels.cpu().tolist())
            pred_counter.update(int(pred) for pred in predictions.cpu().tolist())

    return {
        "loss": (total_loss / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "num_samples": total,
        "label_distribution": _named_counter(label_counter),
        "prediction_distribution": _named_counter(pred_counter),
        "prediction_diversity": _normalized_entropy(pred_counter),
    }


def _named_counter(counter: Counter) -> Dict[str, int]:
    return {ACTION_NAMES[action]: int(counter.get(action, 0)) for action in range(len(ACTION_NAMES))}


def _normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = [count / total for count in counter.values() if count > 0]
    entropy = -sum(prob * np.log(prob) for prob in probabilities)
    return float(entropy / np.log(len(ACTION_NAMES)))


def _write_metrics_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_distribution(distribution: Dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in distribution.items())


def _write_report(path: Path, result: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_train = result["final_train"]
    final_val = result["final_val"]
    final_test = result["final_test"]
    lines = [
        "Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md",
        "",
        "# Phase 8 Graph Policy Supervised Warm-Start Report",
        "",
        f"- Seed: `{result['seed']}`",
        f"- Teacher policy: `{result['teacher_policy']}`",
        f"- Semantic prior fusion: `{result['semantic_prior_fusion']}`",
        f"- Semantic prior weight: `{result['semantic_prior_weight']}`",
        f"- Total samples: `{result['num_samples']}`",
        f"- Executed epochs: `{result['executed_epochs']}`",
        f"- Best epoch: `{result['best_epoch']}`",
        f"- Early stopping triggered: `{'yes' if result['early_stopping_triggered'] else 'no'}`",
        f"- Best validation accuracy: `{float(result['best_val_accuracy']) * 100:.2f}%`",
        f"- Final train accuracy: `{float(final_train['accuracy']) * 100:.2f}%`",
        f"- Final validation accuracy: `{float(final_val['accuracy']) * 100:.2f}%`",
        f"- Final test accuracy: `{float(final_test['accuracy']) * 100:.2f}%`",
        f"- Final test prediction diversity: `{float(final_test['prediction_diversity']):.4f}`",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        "",
        "## Final Test Distributions",
        "",
        f"- Teacher labels: `{_format_distribution(final_test['label_distribution'])}`",
        f"- Model predictions: `{_format_distribution(final_test['prediction_distribution'])}`",
        "",
        "Bu rapor Faz 8.3 kapsaminda graph-aware policy icin semantic prior fusion varyantinin supervised teacher-label warm-start etkisini olcer.",
    ]
    if result.get("metrics_csv"):
        lines.insert(-7, f"- Metrics CSV: `{result['metrics_csv']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_output_suffix(path_value: str, suffix: str | None) -> str:
    if not suffix:
        return path_value
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def run_graph_supervised_pretraining(
    config_path: str = "configs/synthetic/graph_supervised_pretraining.yaml",
    semantic_prior_fusion: str | None = None,
    seed_override: int | None = None,
    output_suffix: str | None = None,
    write_report: bool = False,
    write_metrics: bool = False,
) -> Dict[str, object]:
    config = deepcopy(load_config(config_path) or {})
    seed = int(seed_override if seed_override is not None else config.get("seed", 42))
    config["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    graph_cfg = config.get("graph_policy", {}) or {}
    training_cfg = config.get("training", {}) or {}
    dataset_cfg = config.get("dataset", {}) or {}
    output_cfg = config.get("output", {}) or {}

    fusion_mode = semantic_prior_fusion or str(graph_cfg.get("semantic_prior_fusion", "late"))
    using_fusion_override = semantic_prior_fusion is not None
    semantic_prior_weight = float(graph_cfg.get("semantic_prior_weight", 0.35))
    teacher_policy = normalize_teacher_policy_name(str(dataset_cfg.get("teacher_policy", "teacher_contextual_reward_aligned")))

    samples = _generate_graph_oracle_samples(config)
    if not samples:
        raise ValueError("No graph oracle samples were generated")
    grouped = _split_samples(samples)

    train_ds = GraphOracleDataset(grouped.get("train", []))
    val_ds = GraphOracleDataset(grouped.get("val", []))
    test_ds = GraphOracleDataset(grouped.get("test", []))

    batch_size = int(training_cfg.get("batch_size", 32))
    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        balance_actions=bool(dataset_cfg.get("balance_actions", True)),
        balance_power=float(dataset_cfg.get("balance_power", 0.75)),
        samples_per_epoch=dataset_cfg.get("samples_per_epoch"),
        shuffle=True,
    )
    val_loader = _build_loader(val_ds, batch_size=batch_size, balance_actions=False, balance_power=1.0, samples_per_epoch=None, shuffle=False)
    test_loader = _build_loader(test_ds, batch_size=batch_size, balance_actions=False, balance_power=1.0, samples_per_epoch=None, shuffle=False)

    policy = build_graph_policy_from_state(
        samples[0].graph_state,
        hidden_dim=int(graph_cfg.get("hidden_dim", 64)),
        message_passing_steps=int(graph_cfg.get("message_passing_steps", 2)),
        semantic_prior_fusion=fusion_mode,
        semantic_prior_weight=semantic_prior_weight,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.get("learning_rate", 1e-3)))

    metrics_rows: List[Dict[str, object]] = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    patience = int(training_cfg.get("early_stopping_patience", 4))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 1e-4))
    min_epochs_before_stopping = int(training_cfg.get("min_epochs_before_stopping", 1))
    checkpoint_default = _with_output_suffix(f"models/phase_8/graph_policy_{fusion_mode}.pt", output_suffix)
    checkpoint_value = checkpoint_default if using_fusion_override else output_cfg.get("checkpoint_path", checkpoint_default)
    checkpoint_value = _with_output_suffix(str(checkpoint_value), output_suffix) if not using_fusion_override else checkpoint_value
    checkpoint_path = Path(str(checkpoint_value))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(training_cfg.get("epochs", 8)) + 1):
        policy.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for graph_states, labels in train_loader:
            optimizer.zero_grad()
            logits = _batch_logits(policy, graph_states)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(training_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()

            train_loss_sum += float(loss.item()) * int(labels.shape[0])
            predictions = torch.argmax(logits, dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.shape[0])

        train_loss = (train_loss_sum / train_total) if train_total else 0.0
        train_acc = (train_correct / train_total) if train_total else 0.0
        val_metrics = _evaluate_graph_policy(policy, val_loader, criterion)
        metrics_rows.append(
            {
                "epoch": epoch,
                "semantic_prior_fusion": fusion_mode,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                "val_prediction_diversity": round(float(val_metrics["prediction_diversity"]), 6),
            }
        )

        if float(val_metrics["accuracy"]) > (best_val_acc + min_delta):
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "semantic_prior_fusion": fusion_mode,
                    "semantic_prior_weight": semantic_prior_weight,
                    "config": config,
                    "node_feature_dim": samples[0].graph_state.node_features.shape[1],
                    "edge_feature_dim": samples[0].graph_state.edge_features.shape[1],
                    "global_feature_dim": samples[0].graph_state.global_features.shape[0],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epoch >= min_epochs_before_stopping and epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"])
    final_train = _evaluate_graph_policy(policy, train_loader, criterion)
    final_val = _evaluate_graph_policy(policy, val_loader, criterion)
    final_test = _evaluate_graph_policy(policy, test_loader, criterion)

    metrics_default = _with_output_suffix(f"results/raw/synthetic/debug/graph_supervised_{fusion_mode}.csv", output_suffix)
    report_default = ""
    metrics_value = metrics_default if using_fusion_override else output_cfg.get("metrics_csv", metrics_default)
    report_value = report_default if using_fusion_override else output_cfg.get("report_path", report_default)
    if not using_fusion_override:
        metrics_value = _with_output_suffix(str(metrics_value), output_suffix)
        report_value = _with_output_suffix(str(report_value), output_suffix) if report_value else ""
    metrics_csv = Path(str(metrics_value))
    report_path = Path(str(report_value)) if report_value else Path("")
    if write_metrics:
        _write_metrics_csv(metrics_csv, metrics_rows)

    result: Dict[str, object] = {
        "teacher_policy": teacher_policy,
        "semantic_prior_fusion": fusion_mode,
        "semantic_prior_weight": semantic_prior_weight,
        "seed": seed,
        "num_samples": len(samples),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "executed_epochs": len(metrics_rows),
        "early_stopping_triggered": len(metrics_rows) < int(training_cfg.get("epochs", 8)),
        "final_train": final_train,
        "final_val": final_val,
        "final_test": final_test,
        "metrics_csv": metrics_csv.as_posix() if write_metrics else "",
        "report_path": report_path.as_posix() if report_value else "",
        "checkpoint_path": checkpoint_path.as_posix(),
    }
    if write_report and report_value:
        _write_report(report_path, result)
    return result
