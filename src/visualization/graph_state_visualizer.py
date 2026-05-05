from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.env.graph_state_builder import GraphState


NODE_COLORS = {
    "device": "#4C78A8",
    "task": "#F58518",
    "edge": "#54A24B",
    "cloud": "#B279A2",
}


def graph_state_to_networkx(graph_state: GraphState) -> nx.DiGraph:
    """Convert a Phase 8 GraphState into a NetworkX graph for visualization."""
    graph_state.validate()
    graph = nx.DiGraph()
    node_schema = graph_state.metadata.get("node_feature_schema", [])
    edge_schema = graph_state.metadata.get("edge_feature_schema", [])
    node_metadata = graph_state.metadata.get("node_metadata", [])
    edge_metadata = graph_state.metadata.get("edge_metadata", [])

    for index, features in enumerate(graph_state.node_features):
        meta = node_metadata[index] if index < len(node_metadata) else {}
        node_type = meta.get("type", "unknown")
        node_name = meta.get("name", f"node_{index}")
        graph.add_node(
            index,
            label=_node_label(node_name, node_type, features, node_schema),
            node_type=node_type,
            color=NODE_COLORS.get(node_type, "#9D9D9D"),
        )

    for edge_position, (src, dst) in enumerate(graph_state.edge_index.T):
        meta = edge_metadata[edge_position] if edge_position < len(edge_metadata) else {}
        features = graph_state.edge_features[edge_position]
        edge_type = meta.get("type", "edge")
        graph.add_edge(
            int(src),
            int(dst),
            label=_edge_label(edge_type, features, edge_schema),
            edge_type=edge_type,
        )

    return graph


def draw_graph_state(
    graph_state: GraphState,
    output_path: str,
    title: str = "Phase 8 GraphState",
    show_edge_labels: bool = True,
    figure_size=(12, 8),
) -> Path:
    """Render a GraphState as a PNG image and return the written path."""
    graph = graph_state_to_networkx(graph_state)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pos = _layout_positions(graph)
    node_colors = [graph.nodes[node].get("color", "#9D9D9D") for node in graph.nodes]
    node_labels = {node: graph.nodes[node].get("label", str(node)) for node in graph.nodes}

    plt.figure(figsize=figure_size)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=2600,
        linewidths=1.2,
        edgecolors="#2F2F2F",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        width=1.4,
        edge_color="#6B7280",
    )
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_color="#111827")

    if show_edge_labels:
        edge_labels = {(src, dst): data.get("label", "") for src, dst, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=6,
            label_pos=0.55,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
        )

    prior_text = _action_prior_text(graph_state)
    plt.title(title, fontsize=13, pad=16)
    plt.figtext(0.02, 0.02, prior_text, ha="left", va="bottom", fontsize=8, family="monospace")
    plt.axis("off")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output, dpi=180)
    plt.close()
    return output


def _layout_positions(graph: nx.DiGraph) -> Dict[int, Any]:
    typed_nodes: Dict[str, list] = {"device": [], "task": [], "edge": [], "cloud": []}
    for node, data in graph.nodes(data=True):
        typed_nodes.setdefault(data.get("node_type", "unknown"), []).append(node)

    positions: Dict[int, Any] = {}
    for node in typed_nodes.get("device", []):
        positions[node] = (0.0, 0.0)
    for node in typed_nodes.get("task", []):
        positions[node] = (0.0, 1.6)
    for node in typed_nodes.get("cloud", []):
        positions[node] = (3.1, 0.0)

    edge_nodes = typed_nodes.get("edge", [])
    if edge_nodes:
        y_values = np.linspace(1.0, -1.0, num=len(edge_nodes))
        for node, y_value in zip(edge_nodes, y_values):
            positions[node] = (1.55, float(y_value))

    missing_nodes = [node for node in graph.nodes if node not in positions]
    if missing_nodes:
        fallback = nx.spring_layout(graph.subgraph(missing_nodes), seed=42)
        positions.update(fallback)
    return positions


def _node_label(node_name: str, node_type: str, features: np.ndarray, schema: list) -> str:
    parts = [f"{node_name}", f"({node_type})"]
    if node_type == "device":
        parts.append(f"battery={_feature(features, schema, 'battery_norm'):.2f}")
        parts.append(f"speed={_feature(features, schema, 'mobility_speed_norm'):.2f}")
    elif node_type == "task":
        parts.append(f"size={_feature(features, schema, 'task_size_norm'):.2f}")
        parts.append(f"deadline={_feature(features, schema, 'deadline_tightness_norm'):.2f}")
        parts.append(f"priority={_feature(features, schema, 'semantic_priority'):.2f}")
    elif node_type == "edge":
        parts.append(f"load={_feature(features, schema, 'load_norm'):.2f}")
        parts.append(f"queue={_feature(features, schema, 'queue_norm'):.2f}")
        parts.append(f"energy={_feature(features, schema, 'remaining_energy_norm'):.2f}")
    elif node_type == "cloud":
        parts.append(f"queue={_feature(features, schema, 'queue_norm'):.2f}")
        parts.append(f"lat={_feature(features, schema, 'cloud_latency_norm'):.2f}")
    return "\n".join(parts)


def _edge_label(edge_type: str, features: np.ndarray, schema: list) -> str:
    if edge_type.startswith("device") or edge_type.startswith("task_edge"):
        return (
            f"{edge_type}\n"
            f"rate={_feature(features, schema, 'datarate_norm'):.2f} "
            f"q={_feature(features, schema, 'queue_norm'):.2f}"
        )
    return edge_type


def _action_prior_text(graph_state: GraphState) -> str:
    action_names = graph_state.metadata.get("action_names", [str(i) for i in range(6)])
    prior_parts = [
        f"{name}:{float(prob):.2f}/{int(mask)}"
        for name, prob, mask in zip(action_names, graph_state.action_prior, graph_state.action_mask)
    ]
    return "Action prior / mask: " + "  ".join(prior_parts)


def _feature(features: np.ndarray, schema: list, name: str, default: float = 0.0) -> float:
    try:
        return float(features[schema.index(name)])
    except (ValueError, IndexError):
        return default
