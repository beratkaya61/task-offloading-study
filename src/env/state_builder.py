import math

import numpy as np


def _resolve_norm(value, hint, fallback_cap):
    if hint is not None:
        return float(np.clip(hint, 0.0, 1.0))
    return min(1.0, value / fallback_cap)


def build_state(device, task, edge_servers, channel, ablation_flags=None):
    """
    Build the normalized state vector used by RL agents.

    Default layout:
    [snr, task_size, cpu_cycles, battery, edge_load, edge_energy, semantic_prior(6)]
    = 12 dimensions total.

    With deadline awareness enabled, three physical deadline features are appended:
    [deadline_norm, local_delay/deadline, edge75_delay/deadline].
    """
    if ablation_flags is None:
        ablation_flags = {}

    use_deadline_features = bool(ablation_flags.get("use_deadline_features", False))
    state_dim = 15 if use_deadline_features else 12

    if not device or not task:
        return np.zeros((state_dim,), dtype=np.float32)

    if edge_servers:
        closest_edge = min(edge_servers, key=lambda e: math.dist(device.location, e.location))
        datarate, _ = channel.calculate_datarate(device, closest_edge)
    else:
        closest_edge = None
        datarate = 10e6

    snr_norm = min(1.0, datarate / 50e6)
    size_norm = _resolve_norm(task.size_bits, getattr(task, "size_norm_hint", None), 10e6)
    cpu_norm = _resolve_norm(task.cpu_cycles, getattr(task, "cpu_norm_hint", None), 1e10)
    battery_norm = min(1.0, max(0.0, getattr(device, "battery", 10000.0) / 10000.0))
    load_norm = min(1.0, getattr(closest_edge, "current_load", 0.0) / 10.0) if closest_edge else 0.0

    if ablation_flags.get("disable_battery_awareness", False):
        battery_norm = 1.0
    if ablation_flags.get("disable_queue_awareness", False):
        load_norm = 0.0
    if ablation_flags.get("disable_mobility_features", False):
        snr_norm = 0.5

    if closest_edge:
        edge_energy_budget = max(1e-6, float(getattr(closest_edge, "energy_budget", 5000.0)))
        edge_remaining_energy = float(getattr(closest_edge, "remaining_energy", edge_energy_budget))
        edge_energy_norm = min(1.0, max(0.0, edge_remaining_energy / edge_energy_budget))
    else:
        edge_energy_norm = 1.0

    if not ablation_flags.get("disable_semantics", False) and not ablation_flags.get("disable_semantic_prior", False):
        from src.agents.semantic_prior import generate_action_prior

        prior_vector = generate_action_prior(task.semantic_analysis)
    else:
        prior_vector = np.zeros((6,), dtype=np.float32)

    physical_features = [snr_norm, size_norm, cpu_norm, battery_norm, load_norm, edge_energy_norm]

    if use_deadline_features:
        deadline = max(0.1, float(getattr(task, "deadline", 1.0)))
        local_delay = float(getattr(task, "cpu_cycles", 0.0)) / 1e9
        if closest_edge is not None:
            edge75_tx = 0.75 * float(getattr(task, "size_bits", 0.0)) / max(float(datarate), 1e-6)
            edge75_comp = 0.75 * float(getattr(task, "cpu_cycles", 0.0)) / 2e9
            edge75_queue = 0.015 * float(getattr(closest_edge, "queue_length", 0.0)) + 0.02 * float(getattr(closest_edge, "current_load", 0.0))
            edge75_delay = max(0.25 * local_delay, edge75_tx + edge75_comp + edge75_queue)
        else:
            edge75_delay = local_delay

        physical_features.extend(
            [
                min(1.0, deadline / 5.0),
                min(1.0, local_delay / deadline),
                min(1.0, edge75_delay / deadline),
            ]
        )

    state = np.concatenate(
        (
            np.array(physical_features, dtype=np.float32),
            prior_vector,
        )
    ).astype(np.float32)
    return state
