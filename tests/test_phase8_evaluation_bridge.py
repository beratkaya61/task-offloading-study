import unittest
from types import SimpleNamespace

import numpy as np

from src.agents.graph_policy import build_graph_policy_from_state
from src.agents.graph_policy_evaluator import GraphPolicyEnvAdapter, build_graph_state_from_env
from src.core.evaluation import evaluate_policy
from src.env.state_builder import build_state
from src.env.graph_state_builder import build_graph_state
from tests.test_graph_state_builder import DummyChannel, make_fixture


class SingleStepEnv:
    def __init__(self):
        self.action_space = SimpleNamespace()
        self.observation_space = SimpleNamespace()
        self._done = False

    def reset(self):
        self._done = False
        return np.zeros(12, dtype=np.float32), {}

    def step(self, action):
        self._done = True
        info = {"delay": 0.2, "energy": 0.1, "task_success": True}
        return np.zeros(12, dtype=np.float32), 1.0, True, False, info


class ConstantPolicy:
    def predict(self, obs, deterministic=True):
        return 0, None


class Phase8EvaluationBridgeTest(unittest.TestCase):
    def test_evaluate_policy_can_skip_csv_logging(self):
        result = evaluate_policy(
            SingleStepEnv(),
            ConstantPolicy(),
            num_episodes=2,
            run_name="unit_test_policy",
            csv_path="",
        )

        self.assertEqual(result["config_model_type"], "unit_test_policy")
        self.assertAlmostEqual(float(result["metric_success_rate"]), 1.0, places=5)
        self.assertEqual(int(result["metric_dominant_action"]), 0)

    def test_graph_policy_adapter_rebuilds_graph_from_live_env(self):
        device, task, edge_servers, cloud = make_fixture()
        env = SimpleNamespace(
            current_device=device,
            current_task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
            previous_action=2,
            current_step=4,
            ablation_flags={},
            trace_mode=False,
        )
        obs = build_state(device, task, edge_servers, DummyChannel(), {})
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )
        policy = build_graph_policy_from_state(graph_state, hidden_dim=16, message_passing_steps=1)
        adapter = GraphPolicyEnvAdapter(policy, env)

        rebuilt_graph = build_graph_state_from_env(env, obs)
        action, _ = adapter.predict(obs, deterministic=True)

        self.assertEqual(rebuilt_graph.vector_state_reference.shape, (12,))
        self.assertIn(action, range(6))


if __name__ == "__main__":
    unittest.main()
