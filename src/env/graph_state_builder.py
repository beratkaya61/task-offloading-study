import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


NODE_TYPE_IDS = {
    "device": 0,
    "task": 1,
    "edge": 2,
    "cloud": 3,
}

EDGE_TYPE_IDS = {
    "device_edge": 0,
    "edge_device": 1,
    "device_cloud": 2,
    "cloud_device": 3,
    "task_device": 4,
    "device_task": 5,
    "task_edge": 6,
    "edge_task": 7,
    "task_cloud": 8,
    "cloud_task": 9,
}

NODE_FEATURE_SCHEMA = [
    "is_device",
    "is_task",
    "is_edge",
    "is_cloud",
    "battery_norm",
    "cpu_capacity_norm",
    "load_norm",
    "queue_norm",
    "remaining_energy_norm",
    "mobility_speed_norm",
    "distance_to_device_norm",
    "task_size_norm",
    "task_cpu_norm",
    "deadline_tightness_norm",
    "semantic_priority",
    "semantic_confidence",
    "cloud_latency_norm",
]

EDGE_FEATURE_SCHEMA = [
    "is_device_edge",
    "is_device_cloud",
    "is_task_device",
    "is_task_edge",
    "is_task_cloud",
    "distance_norm",
    "datarate_norm",
    "tx_latency_norm",
    "queue_norm",
    "compute_latency_norm",
    "link_quality_norm",
    "offload_ratio_hint",
]

GLOBAL_FEATURE_SCHEMA = [
    "current_step_norm",
    "previous_action_norm",
    "num_edges_norm",
    "avg_edge_load_norm",
    "max_edge_queue_norm",
    "nearest_edge_distance_norm",
    "task_deadline_tightness_norm",
    "semantic_priority",
    "semantic_confidence",
    "trace_mode_flag",
]

ACTION_NAMES = ["local", "edge_25", "edge_50", "edge_75", "edge_100", "cloud"]


@dataclass
class GraphState:
    """
    Extensible graph observation contract for Phase 8 graph-aware policies.

    This object is intentionally framework-neutral. Later stages can convert it
    to PyTorch tensors or torch_geometric.data.Data without changing the builder.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    action_prior: np.ndarray
    action_mask: np.ndarray
    node_type_ids: np.ndarray
    edge_type_ids: np.ndarray
    node_id_map: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_state_reference: Optional[np.ndarray] = None

    def validate(self) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must be a 2D matrix")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        if self.edge_features.ndim != 2:
            raise ValueError("edge_features must be a 2D matrix")
        if self.edge_index.shape[1] != self.edge_features.shape[0]:
            raise ValueError("edge_index and edge_features disagree on num_edges")
        if self.node_type_ids.shape[0] != self.node_features.shape[0]:
            raise ValueError("node_type_ids length must match node count")
        if self.edge_type_ids.shape[0] != self.edge_features.shape[0]:
            raise ValueError("edge_type_ids length must match edge count")
        if self.action_prior.shape != (6,):
            raise ValueError("action_prior must have shape (6,)")
        if self.action_mask.shape != (6,):
            raise ValueError("action_mask must have shape (6,)")
        arrays = [
            self.node_features,
            self.edge_index.astype(np.float32),
            self.edge_features,
            self.global_features,
            self.action_prior,
            self.action_mask,
        ]
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("GraphState contains NaN or Inf values")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "global_features": self.global_features,
            "action_prior": self.action_prior,
            "action_mask": self.action_mask,
            "node_type_ids": self.node_type_ids,
            "edge_type_ids": self.edge_type_ids,
            "node_id_map": self.node_id_map,
            "metadata": self.metadata,
            "vector_state_reference": self.vector_state_reference,
        }


def build_graph_state(
    device: Any,
    task: Any,
    edge_servers: Optional[Iterable[Any]],
    cloud_server: Any = None,
    channel: Any = None,
    previous_action: Optional[int] = None,
    current_step: int = 0,
    ablation_flags: Optional[Dict[str, Any]] = None,
    semantic_analysis: Optional[Dict[str, Any]] = None,
    semantic_prior: Optional[np.ndarray] = None,
    trace_context: Optional[Dict[str, Any]] = None,
    normalization_config: Optional[Dict[str, float]] = None,
    topology_config: Optional[Dict[str, Any]] = None,
    vector_state_reference: Optional[np.ndarray] = None,
) -> GraphState:
    """
    Build a graph observation for one offloading decision.

    The initial topology is deliberately small and stable:
    one device node, one task node, N edge nodes, and one cloud node. The
    contract already carries action masks, semantic priors, metadata, and
    optional trace context so later Phase 8/9 work can extend it without
    changing every caller.
    """
    ablation_flags = ablation_flags or {}
    trace_context = trace_context or {}
    normalization = _default_normalization()
    if normalization_config:
        normalization.update(normalization_config)
    topology = _default_topology()
    if topology_config:
        topology.update(topology_config)

    edges = list(edge_servers or [])
    semantic = semantic_analysis if semantic_analysis is not None else getattr(task, "semantic_analysis", None)
    prior = _resolve_action_prior(semantic, semantic_prior, ablation_flags)
    action_mask = _build_action_mask(ablation_flags)

    node_features: List[np.ndarray] = []
    node_type_ids: List[int] = []
    node_id_map: Dict[str, int] = {}
    node_metadata: List[Dict[str, Any]] = []

    def add_node(name: str, node_type: str, features: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> int:
        index = len(node_features)
        node_features.append(features.astype(np.float32))
        node_type_ids.append(NODE_TYPE_IDS[node_type])
        node_id_map[name] = index
        node_metadata.append({"name": name, "type": node_type, **(extra or {})})
        return index

    device_index = add_node(
        "device",
        "device",
        _device_features(device, task, semantic, normalization),
        {"device_id": getattr(device, "id", None)},
    )
    task_index = add_node(
        "task",
        "task",
        _task_features(task, semantic, normalization),
        {"task_id": getattr(task, "id", None), "task_type": _task_type_name(task)},
    )

    edge_node_indices: List[int] = []
    link_cache: Dict[int, Dict[str, float]] = {}
    for position, edge in enumerate(edges):
        link = _link_metrics(device, edge, task, channel, normalization)
        link_cache[position] = link
        edge_node_indices.append(
            add_node(
                f"edge_{position}",
                "edge",
                _edge_server_features(edge, link, normalization),
                {"edge_id": getattr(edge, "id", position), "position": position},
            )
        )

    cloud_index = add_node(
        "cloud",
        "cloud",
        _cloud_features(cloud_server, task, normalization),
        {"cloud_present": cloud_server is not None},
    )

    edge_index_pairs: List[Tuple[int, int]] = []
    edge_features: List[np.ndarray] = []
    edge_type_ids: List[int] = []
    edge_metadata: List[Dict[str, Any]] = []

    def add_relation(src: int, dst: int, edge_type: str, features: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> None:
        edge_index_pairs.append((src, dst))
        edge_features.append(features.astype(np.float32))
        edge_type_ids.append(EDGE_TYPE_IDS[edge_type])
        edge_metadata.append({"type": edge_type, "src": src, "dst": dst, **(extra or {})})

    for position, edge_node_index in enumerate(edge_node_indices):
        link = link_cache[position]
        device_edge_features = _device_edge_features(link, edges[position], task, normalization, offload_ratio_hint=1.0)
        add_relation(device_index, edge_node_index, "device_edge", device_edge_features, {"edge_position": position})
        add_relation(edge_node_index, device_index, "edge_device", device_edge_features, {"edge_position": position})

        task_edge_features = _task_edge_features(link, edges[position], task, normalization, offload_ratio_hint=0.75)
        add_relation(task_index, edge_node_index, "task_edge", task_edge_features, {"edge_position": position})
        add_relation(edge_node_index, task_index, "edge_task", task_edge_features, {"edge_position": position})

    cloud_link = _cloud_link_metrics(device, task, cloud_server, normalization)
    device_cloud_features = _device_cloud_features(cloud_link, cloud_server, task, normalization)
    add_relation(device_index, cloud_index, "device_cloud", device_cloud_features)
    add_relation(cloud_index, device_index, "cloud_device", device_cloud_features)

    task_device_features = _task_device_features(device, task, normalization)
    add_relation(task_index, device_index, "task_device", task_device_features)
    add_relation(device_index, task_index, "device_task", task_device_features)

    task_cloud_features = _task_cloud_features(cloud_link, cloud_server, task, normalization)
    add_relation(task_index, cloud_index, "task_cloud", task_cloud_features)
    add_relation(cloud_index, task_index, "cloud_task", task_cloud_features)

    if edge_index_pairs:
        edge_index = np.array(edge_index_pairs, dtype=np.int64).T
        edge_feature_matrix = np.vstack(edge_features).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feature_matrix = np.zeros((0, len(EDGE_FEATURE_SCHEMA)), dtype=np.float32)

    global_features = _global_features(
        current_step=current_step,
        previous_action=previous_action,
        edge_servers=edges,
        link_cache=link_cache,
        task=task,
        semantic=semantic,
        trace_context=trace_context,
        normalization=normalization,
    )

    metadata = {
        "node_feature_schema": NODE_FEATURE_SCHEMA,
        "edge_feature_schema": EDGE_FEATURE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "action_names": ACTION_NAMES,
        "node_metadata": node_metadata,
        "edge_metadata": edge_metadata,
        "normalization": normalization,
        "topology": topology,
        "trace_context": trace_context,
        "previous_action": previous_action,
        "current_step": current_step,
        "phase": "phase_8_graph_state_builder",
    }

    graph_state = GraphState(
        node_features=np.vstack(node_features).astype(np.float32),
        edge_index=edge_index,
        edge_features=edge_feature_matrix,
        global_features=global_features.astype(np.float32),
        action_prior=prior.astype(np.float32),
        action_mask=action_mask.astype(np.float32),
        node_type_ids=np.array(node_type_ids, dtype=np.int64),
        edge_type_ids=np.array(edge_type_ids, dtype=np.int64),
        node_id_map=node_id_map,
        metadata=metadata,
        vector_state_reference=None if vector_state_reference is None else np.asarray(vector_state_reference, dtype=np.float32),
    )
    graph_state.validate()
    return graph_state


def _default_normalization() -> Dict[str, float]:
    return {
        "battery_j": 10000.0,
        "task_size_bits": 10e6,
        "task_cpu_cycles": 1e10,
        "edge_cpu_hz": 5e9,
        "cloud_cpu_hz": 5e9,
        "edge_load": 10.0,
        "edge_queue": 10.0,
        "cloud_queue": 20.0,
        "distance": 1000.0,
        "datarate": 50e6,
        "tx_latency": 2.0,
        "compute_latency": 5.0,
        "deadline_seconds": 5.0,
        "current_step": 1000.0,
        "cloud_latency": 1.0,
        "mobility_speed": 5.0,
    }


def _default_topology() -> Dict[str, Any]:
    return {
        "include_task_node": True,
        "include_cloud_node": True,
        "directed_edges": True,
        "edge_server_limit": None,
    }


def _resolve_action_prior(
    semantic_analysis: Optional[Dict[str, Any]],
    semantic_prior: Optional[np.ndarray],
    ablation_flags: Dict[str, Any],
) -> np.ndarray:
    if ablation_flags.get("disable_semantics", False) or ablation_flags.get("disable_semantic_prior", False):
        return np.zeros((6,), dtype=np.float32)
    if semantic_prior is not None:
        prior = np.asarray(semantic_prior, dtype=np.float32)
        if prior.shape != (6,):
            raise ValueError("semantic_prior must have shape (6,)")
        total = float(np.sum(prior))
        return prior / total if total > 0 else np.ones(6, dtype=np.float32) / 6.0
    from src.agents.semantic_prior import generate_action_prior

    return generate_action_prior(semantic_analysis)


def _build_action_mask(ablation_flags: Dict[str, Any]) -> np.ndarray:
    mask = np.ones((6,), dtype=np.float32)
    if ablation_flags.get("disable_partial_offloading", False):
        mask[1:4] = 0.0
    return mask


def _node_base(node_type: str) -> np.ndarray:
    features = np.zeros((len(NODE_FEATURE_SCHEMA),), dtype=np.float32)
    features[NODE_FEATURE_SCHEMA.index(f"is_{node_type}")] = 1.0
    return features


def _device_features(device: Any, task: Any, semantic: Optional[Dict[str, Any]], normalization: Dict[str, float]) -> np.ndarray:
    features = _node_base("device")
    battery_capacity = float(getattr(device, "battery_capacity", normalization["battery_j"]))
    battery_norm_denominator = max(1e-6, battery_capacity or normalization["battery_j"])
    features[NODE_FEATURE_SCHEMA.index("battery_norm")] = _clip01(float(getattr(device, "battery", normalization["battery_j"])) / battery_norm_denominator)
    velocity = getattr(device, "velocity", [0.0, 0.0])
    speed = math.sqrt(sum(float(component) ** 2 for component in velocity)) if velocity is not None else 0.0
    features[NODE_FEATURE_SCHEMA.index("mobility_speed_norm")] = _clip01(speed / max(1e-6, normalization["mobility_speed"]))
    features[NODE_FEATURE_SCHEMA.index("semantic_priority")] = _semantic_value(semantic, "priority_score", 0.0)
    features[NODE_FEATURE_SCHEMA.index("semantic_confidence")] = _semantic_value(semantic, "confidence", 0.0)
    if task is not None:
        features[NODE_FEATURE_SCHEMA.index("deadline_tightness_norm")] = _deadline_tightness(task, normalization)
    return features


def _task_features(task: Any, semantic: Optional[Dict[str, Any]], normalization: Dict[str, float]) -> np.ndarray:
    features = _node_base("task")
    features[NODE_FEATURE_SCHEMA.index("task_size_norm")] = _clip01(float(getattr(task, "size_bits", 0.0)) / max(1e-6, normalization["task_size_bits"]))
    features[NODE_FEATURE_SCHEMA.index("task_cpu_norm")] = _clip01(float(getattr(task, "cpu_cycles", 0.0)) / max(1e-6, normalization["task_cpu_cycles"]))
    features[NODE_FEATURE_SCHEMA.index("deadline_tightness_norm")] = _deadline_tightness(task, normalization)
    features[NODE_FEATURE_SCHEMA.index("semantic_priority")] = _semantic_value(semantic, "priority_score", 0.0)
    features[NODE_FEATURE_SCHEMA.index("semantic_confidence")] = _semantic_value(semantic, "confidence", 0.0)
    return features


def _edge_server_features(edge_server: Any, link: Dict[str, float], normalization: Dict[str, float]) -> np.ndarray:
    features = _node_base("edge")
    features[NODE_FEATURE_SCHEMA.index("cpu_capacity_norm")] = _clip01(float(getattr(edge_server, "max_freq", 0.0)) / max(1e-6, normalization["edge_cpu_hz"]))
    features[NODE_FEATURE_SCHEMA.index("load_norm")] = _clip01(float(getattr(edge_server, "current_load", 0.0)) / max(1e-6, normalization["edge_load"]))
    max_queue = float(getattr(edge_server, "max_queue_size", normalization["edge_queue"]))
    features[NODE_FEATURE_SCHEMA.index("queue_norm")] = _clip01(float(getattr(edge_server, "queue_length", 0.0)) / max(1e-6, max_queue))
    energy_budget = max(1e-6, float(getattr(edge_server, "energy_budget", 5000.0)))
    features[NODE_FEATURE_SCHEMA.index("remaining_energy_norm")] = _clip01(float(getattr(edge_server, "remaining_energy", energy_budget)) / energy_budget)
    features[NODE_FEATURE_SCHEMA.index("distance_to_device_norm")] = link["distance_norm"]
    return features


def _cloud_features(cloud_server: Any, task: Any, normalization: Dict[str, float]) -> np.ndarray:
    features = _node_base("cloud")
    features[NODE_FEATURE_SCHEMA.index("cpu_capacity_norm")] = _clip01(float(getattr(cloud_server, "cpu_freq", normalization["cloud_cpu_hz"])) / max(1e-6, normalization["cloud_cpu_hz"]))
    features[NODE_FEATURE_SCHEMA.index("load_norm")] = _clip01(float(getattr(cloud_server, "current_load", 0.0)) / max(1e-6, normalization["edge_load"]))
    features[NODE_FEATURE_SCHEMA.index("queue_norm")] = _clip01(float(getattr(cloud_server, "queue_length", 0.0)) / max(1e-6, normalization["cloud_queue"]))
    features[NODE_FEATURE_SCHEMA.index("cloud_latency_norm")] = _clip01(_cloud_latency(cloud_server) / max(1e-6, normalization["cloud_latency"]))
    if task is not None:
        features[NODE_FEATURE_SCHEMA.index("task_cpu_norm")] = _clip01(float(getattr(task, "cpu_cycles", 0.0)) / max(1e-6, normalization["task_cpu_cycles"]))
    return features


def _link_metrics(device: Any, edge_server: Any, task: Any, channel: Any, normalization: Dict[str, float]) -> Dict[str, float]:
    distance = _distance(device, edge_server)
    datarate = normalization["datarate"] * 0.2
    if channel is not None and device is not None and edge_server is not None:
        try:
            calculated = channel.calculate_datarate(device, edge_server)
            datarate = float(calculated[0])
        except Exception:
            datarate = normalization["datarate"] * 0.2
    tx_latency = float(getattr(task, "size_bits", 0.0)) / max(1e-6, datarate)
    edge_cpu = max(1e-6, float(getattr(edge_server, "max_freq", 2e9)))
    compute_latency = float(getattr(task, "cpu_cycles", 0.0)) / edge_cpu
    return {
        "distance": distance,
        "distance_norm": _clip01(distance / max(1e-6, normalization["distance"])),
        "datarate": datarate,
        "datarate_norm": _clip01(datarate / max(1e-6, normalization["datarate"])),
        "tx_latency": tx_latency,
        "tx_latency_norm": _clip01(tx_latency / max(1e-6, normalization["tx_latency"])),
        "compute_latency": compute_latency,
        "compute_latency_norm": _clip01(compute_latency / max(1e-6, normalization["compute_latency"])),
        "link_quality_norm": _clip01(datarate / max(1e-6, normalization["datarate"])),
    }


def _cloud_link_metrics(device: Any, task: Any, cloud_server: Any, normalization: Dict[str, float]) -> Dict[str, float]:
    latency = _cloud_latency(cloud_server)
    compute_latency = float(getattr(task, "cpu_cycles", 0.0)) / max(1e-6, float(getattr(cloud_server, "cpu_freq", normalization["cloud_cpu_hz"])))
    return {
        "distance": normalization["distance"],
        "distance_norm": 1.0,
        "datarate": normalization["datarate"] * 0.5,
        "datarate_norm": 0.5,
        "tx_latency": latency,
        "tx_latency_norm": _clip01(latency / max(1e-6, normalization["tx_latency"])),
        "compute_latency": compute_latency,
        "compute_latency_norm": _clip01(compute_latency / max(1e-6, normalization["compute_latency"])),
        "link_quality_norm": 0.5,
    }


def _device_edge_features(
    link: Dict[str, float],
    edge_server: Any,
    task: Any,
    normalization: Dict[str, float],
    offload_ratio_hint: float,
) -> np.ndarray:
    features = _edge_base("device_edge")
    _fill_common_edge_features(features, link, _queue_norm(edge_server, normalization), offload_ratio_hint)
    return features


def _task_edge_features(
    link: Dict[str, float],
    edge_server: Any,
    task: Any,
    normalization: Dict[str, float],
    offload_ratio_hint: float,
) -> np.ndarray:
    features = _edge_base("task_edge")
    _fill_common_edge_features(features, link, _queue_norm(edge_server, normalization), offload_ratio_hint)
    return features


def _device_cloud_features(link: Dict[str, float], cloud_server: Any, task: Any, normalization: Dict[str, float]) -> np.ndarray:
    features = _edge_base("device_cloud")
    _fill_common_edge_features(features, link, _cloud_queue_norm(cloud_server, normalization), 1.0)
    return features


def _task_device_features(device: Any, task: Any, normalization: Dict[str, float]) -> np.ndarray:
    local_compute = float(getattr(task, "cpu_cycles", 0.0)) / 1e9
    features = _edge_base("task_device")
    features[EDGE_FEATURE_SCHEMA.index("compute_latency_norm")] = _clip01(local_compute / max(1e-6, normalization["compute_latency"]))
    features[EDGE_FEATURE_SCHEMA.index("offload_ratio_hint")] = 0.0
    features[EDGE_FEATURE_SCHEMA.index("link_quality_norm")] = 1.0
    return features


def _task_cloud_features(link: Dict[str, float], cloud_server: Any, task: Any, normalization: Dict[str, float]) -> np.ndarray:
    features = _edge_base("task_cloud")
    _fill_common_edge_features(features, link, _cloud_queue_norm(cloud_server, normalization), 1.0)
    return features


def _edge_base(edge_family: str) -> np.ndarray:
    features = np.zeros((len(EDGE_FEATURE_SCHEMA),), dtype=np.float32)
    family_to_feature = {
        "device_edge": "is_device_edge",
        "device_cloud": "is_device_cloud",
        "task_device": "is_task_device",
        "task_edge": "is_task_edge",
        "task_cloud": "is_task_cloud",
    }
    feature_name = family_to_feature[edge_family]
    features[EDGE_FEATURE_SCHEMA.index(feature_name)] = 1.0
    return features


def _fill_common_edge_features(features: np.ndarray, link: Dict[str, float], queue_norm: float, offload_ratio_hint: float) -> None:
    features[EDGE_FEATURE_SCHEMA.index("distance_norm")] = link["distance_norm"]
    features[EDGE_FEATURE_SCHEMA.index("datarate_norm")] = link["datarate_norm"]
    features[EDGE_FEATURE_SCHEMA.index("tx_latency_norm")] = link["tx_latency_norm"]
    features[EDGE_FEATURE_SCHEMA.index("queue_norm")] = queue_norm
    features[EDGE_FEATURE_SCHEMA.index("compute_latency_norm")] = link["compute_latency_norm"]
    features[EDGE_FEATURE_SCHEMA.index("link_quality_norm")] = link["link_quality_norm"]
    features[EDGE_FEATURE_SCHEMA.index("offload_ratio_hint")] = _clip01(offload_ratio_hint)


def _global_features(
    current_step: int,
    previous_action: Optional[int],
    edge_servers: List[Any],
    link_cache: Dict[int, Dict[str, float]],
    task: Any,
    semantic: Optional[Dict[str, Any]],
    trace_context: Dict[str, Any],
    normalization: Dict[str, float],
) -> np.ndarray:
    features = np.zeros((len(GLOBAL_FEATURE_SCHEMA),), dtype=np.float32)
    edge_loads = [float(getattr(edge, "current_load", 0.0)) for edge in edge_servers]
    edge_queues = [_queue_norm(edge, normalization) for edge in edge_servers]
    distances = [link["distance_norm"] for link in link_cache.values()]
    features[GLOBAL_FEATURE_SCHEMA.index("current_step_norm")] = _clip01(float(current_step) / max(1e-6, normalization["current_step"]))
    features[GLOBAL_FEATURE_SCHEMA.index("previous_action_norm")] = 0.0 if previous_action is None else _clip01(float(previous_action) / 5.0)
    features[GLOBAL_FEATURE_SCHEMA.index("num_edges_norm")] = _clip01(float(len(edge_servers)) / 10.0)
    features[GLOBAL_FEATURE_SCHEMA.index("avg_edge_load_norm")] = _clip01(float(np.mean(edge_loads)) / max(1e-6, normalization["edge_load"])) if edge_loads else 0.0
    features[GLOBAL_FEATURE_SCHEMA.index("max_edge_queue_norm")] = max(edge_queues) if edge_queues else 0.0
    features[GLOBAL_FEATURE_SCHEMA.index("nearest_edge_distance_norm")] = min(distances) if distances else 1.0
    features[GLOBAL_FEATURE_SCHEMA.index("task_deadline_tightness_norm")] = _deadline_tightness(task, normalization)
    features[GLOBAL_FEATURE_SCHEMA.index("semantic_priority")] = _semantic_value(semantic, "priority_score", 0.0)
    features[GLOBAL_FEATURE_SCHEMA.index("semantic_confidence")] = _semantic_value(semantic, "confidence", 0.0)
    features[GLOBAL_FEATURE_SCHEMA.index("trace_mode_flag")] = 1.0 if trace_context else 0.0
    return features


def _queue_norm(server: Any, normalization: Dict[str, float]) -> float:
    max_queue = float(getattr(server, "max_queue_size", normalization["edge_queue"]))
    return _clip01(float(getattr(server, "queue_length", 0.0)) / max(1e-6, max_queue))


def _cloud_queue_norm(server: Any, normalization: Dict[str, float]) -> float:
    return _clip01(float(getattr(server, "queue_length", 0.0)) / max(1e-6, normalization["cloud_queue"]))


def _cloud_latency(cloud_server: Any) -> float:
    return float(getattr(cloud_server, "latency", getattr(cloud_server, "cloud_latency", 0.1)))


def _distance(device: Any, target: Any) -> float:
    device_location = getattr(device, "location", None)
    target_location = getattr(target, "location", None)
    if device_location is None or target_location is None:
        return 1000.0
    return float(math.dist(device_location, target_location))


def _deadline_tightness(task: Any, normalization: Dict[str, float]) -> float:
    deadline = float(getattr(task, "deadline", normalization["deadline_seconds"]))
    return _clip01(1.0 - (deadline / max(1e-6, normalization["deadline_seconds"])))


def _semantic_value(semantic: Optional[Dict[str, Any]], key: str, default: float) -> float:
    if not semantic:
        return float(default)
    return _clip01(float(semantic.get(key, default)))


def _task_type_name(task: Any) -> str:
    task_type = getattr(task, "task_type", None)
    return getattr(task_type, "name", str(task_type)) if task_type is not None else "UNKNOWN"


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
