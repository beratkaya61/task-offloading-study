import math
import numpy as np


def build_state(device, task, edge_servers, channel, ablation_flags=None):
    """
    Build the normalized state vector used by RL agents.

    State layout:
    [snr, task_size, cpu_cycles, battery, edge_load, edge_energy, semantic_prior(6)]
    = 12 dimensions total
    """
    if ablation_flags is None:
        ablation_flags = {}

    if not device or not task:
        return np.zeros((12,), dtype=np.float32)

    if edge_servers:
        closest_edge = min(edge_servers, key=lambda e: math.dist(device.location, e.location))
        datarate, _ = channel.calculate_datarate(device, closest_edge)
    else:
        closest_edge = None
        datarate = 10e6

    snr_norm = min(1.0, datarate / 50e6)
    size_norm = min(1.0, task.size_bits / 10e6)
    cpu_norm = min(1.0, task.cpu_cycles / 1e10)
    battery_norm = min(1.0, max(0.0, getattr(device, 'battery', 10000.0) / 10000.0))
    load_norm = min(1.0, getattr(closest_edge, 'current_load', 0.0) / 10.0) if closest_edge else 0.0

    if ablation_flags.get('disable_battery_awareness', False):
        battery_norm = 1.0
    if ablation_flags.get('disable_queue_awareness', False):
        load_norm = 0.0
    if ablation_flags.get('disable_mobility_features', False):
        snr_norm = 0.5

    if closest_edge:
        edge_energy_budget = max(1e-6, float(getattr(closest_edge, 'energy_budget', 5000.0)))
        edge_remaining_energy = float(getattr(closest_edge, 'remaining_energy', edge_energy_budget))
        edge_energy_norm = min(1.0, max(0.0, edge_remaining_energy / edge_energy_budget))
    else:
        edge_energy_norm = 1.0

    if not ablation_flags.get('disable_semantics', False) and not ablation_flags.get('disable_semantic_prior', False):
        from src.agents.semantic_prior import generate_action_prior
        prior_vector = generate_action_prior(task.semantic_analysis)
    else:
        prior_vector = np.zeros((6,), dtype=np.float32)

    state = np.concatenate(
        (
            np.array(
                [snr_norm, size_norm, cpu_norm, battery_norm, load_norm, edge_energy_norm],
                dtype=np.float32,
            ),
            prior_vector,
        )
    ).astype(np.float32)
    return state
