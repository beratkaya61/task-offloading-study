import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.env.graph_state_builder import (
    EDGE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NODE_FEATURE_SCHEMA,
    build_graph_state,
)
from src.visualization.graph_state_visualizer import draw_graph_state


class DummyChannel:
    def calculate_datarate(self, device, edge_server):
        return 25e6, 100.0


def make_fixture():
    device = SimpleNamespace(
        id=1,
        battery=7500.0,
        battery_capacity=10000.0,
        location=[0.0, 0.0],
        velocity=[1.0, 2.0],
    )
    task = SimpleNamespace(
        id=42,
        size_bits=2e6,
        cpu_cycles=2e9,
        deadline=2.5,
        task_type=SimpleNamespace(name="CRITICAL"),
        semantic_analysis={
            "recommended_target": "edge",
            "confidence": 0.8,
            "priority_score": 0.9,
        },
    )
    edge_servers = [
        SimpleNamespace(
            id=1,
            location=[100.0, 0.0],
            max_freq=2e9,
            current_load=2.0,
            queue_length=1,
            max_queue_size=10,
            energy_budget=5000.0,
            remaining_energy=4500.0,
        ),
        SimpleNamespace(
            id=2,
            location=[300.0, 0.0],
            max_freq=2.5e9,
            current_load=4.0,
            queue_length=3,
            max_queue_size=10,
            energy_budget=5000.0,
            remaining_energy=3500.0,
        ),
    ]
    cloud = SimpleNamespace(cpu_freq=5e9, current_load=1.0, queue_length=2)
    return device, task, edge_servers, cloud


class GraphStateBuilderTest(unittest.TestCase):
    def test_graph_state_shapes_and_metadata(self):
        device, task, edge_servers, cloud = make_fixture()

        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
            previous_action=3,
            current_step=7,
            trace_context={"source": "unit_test"},
            vector_state_reference=np.zeros(12, dtype=np.float32),
        )

        self.assertEqual(graph_state.node_features.shape, (5, len(NODE_FEATURE_SCHEMA)))
        self.assertEqual(graph_state.edge_index.shape, (2, 14))
        self.assertEqual(graph_state.edge_features.shape, (14, len(EDGE_FEATURE_SCHEMA)))
        self.assertEqual(graph_state.global_features.shape, (len(GLOBAL_FEATURE_SCHEMA),))
        self.assertEqual(graph_state.action_prior.shape, (6,))
        self.assertEqual(graph_state.action_mask.tolist(), [1.0] * 6)
        self.assertEqual(graph_state.node_id_map["device"], 0)
        self.assertEqual(graph_state.node_id_map["task"], 1)
        self.assertIn("edge_0", graph_state.node_id_map)
        self.assertIn("cloud", graph_state.node_id_map)
        self.assertAlmostEqual(float(np.sum(graph_state.action_prior)), 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(graph_state.node_features)))
        self.assertTrue(np.all(np.isfinite(graph_state.edge_features)))
        self.assertEqual(graph_state.metadata["trace_context"]["source"], "unit_test")

    def test_ablation_flags_control_semantics_and_action_mask(self):
        device, task, edge_servers, cloud = make_fixture()

        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
            ablation_flags={
                "disable_semantics": True,
                "disable_partial_offloading": True,
            },
        )

        self.assertEqual(graph_state.action_prior.tolist(), [0.0] * 6)
        self.assertEqual(graph_state.action_mask.tolist(), [1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        graph_state.validate()

    def test_graph_state_visualizer_writes_png(self):
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "graph_state.png"
            written_path = draw_graph_state(graph_state, str(output_path), show_edge_labels=False)

            self.assertTrue(written_path.exists())
            self.assertGreater(written_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
