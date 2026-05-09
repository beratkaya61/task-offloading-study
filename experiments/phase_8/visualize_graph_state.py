import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.graph_state_builder import build_graph_state
from src.visualization.graph_state_visualizer import draw_graph_state


class DemoChannel:
    def calculate_datarate(self, device, edge_server):
        distance = ((device.location[0] - edge_server.location[0]) ** 2 + (device.location[1] - edge_server.location[1]) ** 2) ** 0.5
        datarate = max(5e6, 45e6 * (1.0 - min(distance, 1000.0) / 1200.0))
        return datarate, distance


def build_demo_graph_state():
    device = SimpleNamespace(
        id=1,
        battery=7200.0,
        battery_capacity=10000.0,
        location=[120.0, 180.0],
        velocity=[1.5, 0.5],
    )
    task = SimpleNamespace(
        id=701,
        size_bits=3.5e6,
        cpu_cycles=3.2e9,
        deadline=1.8,
        task_type=SimpleNamespace(name="CRITICAL"),
        semantic_analysis={
            "recommended_target": "edge",
            "confidence": 0.82,
            "priority_score": 0.9,
            "reason": "Critical task with tight latency and moderate data size.",
        },
    )
    edge_servers = [
        SimpleNamespace(
            id=1,
            location=[220.0, 220.0],
            max_freq=2.3e9,
            current_load=2.0,
            queue_length=1,
            max_queue_size=10,
            energy_budget=5000.0,
            remaining_energy=4400.0,
        ),
        SimpleNamespace(
            id=2,
            location=[640.0, 260.0],
            max_freq=2.6e9,
            current_load=5.0,
            queue_length=4,
            max_queue_size=10,
            energy_budget=5000.0,
            remaining_energy=3700.0,
        ),
        SimpleNamespace(
            id=3,
            location=[430.0, 780.0],
            max_freq=2.0e9,
            current_load=1.0,
            queue_length=0,
            max_queue_size=10,
            energy_budget=5000.0,
            remaining_energy=4800.0,
        ),
    ]
    cloud = SimpleNamespace(cpu_freq=5e9, current_load=2.0, queue_length=3)
    return build_graph_state(
        device=device,
        task=task,
        edge_servers=edge_servers,
        cloud_server=cloud,
        channel=DemoChannel(),
        previous_action=3,
        current_step=12,
        trace_context={"mode": "demo_synthetic"},
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize a sample Phase 8 GraphState.")
    parser.add_argument(
        "--output",
        default="results/phase_8/figures/sample_graph_state.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="Render detailed edge labels. Disabled by default for a cleaner topology view.",
    )
    args = parser.parse_args()
    graph_state = build_demo_graph_state()
    output_path = draw_graph_state(
        graph_state,
        args.output,
        title="Phase 8 GraphState: device-task-edge-cloud topology",
        show_edge_labels=args.show_edge_labels,
    )
    print(f"[OK] GraphState visualization written to {output_path}")


if __name__ == "__main__":
    main()
