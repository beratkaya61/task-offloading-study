import unittest

import torch

from src.agents.graph_policy import (
    apply_action_mask,
    build_graph_policy_from_state,
    fuse_semantic_prior_logits,
    graph_state_to_tensors,
)
from src.env.graph_state_builder import build_graph_state
from tests.test_graph_state_builder import DummyChannel, make_fixture


class GraphPolicyTest(unittest.TestCase):
    def test_graph_policy_forward_shapes(self):
        torch.manual_seed(7)
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )
        policy = build_graph_policy_from_state(graph_state, hidden_dim=32, message_passing_steps=2)

        output = policy(graph_state)

        self.assertEqual(output.logits.shape, (6,))
        self.assertEqual(output.masked_logits.shape, (6,))
        self.assertEqual(output.action_probabilities.shape, (6,))
        self.assertEqual(output.graph_embedding.shape, (32,))
        self.assertTrue(torch.all(torch.isfinite(output.logits)))
        self.assertTrue(torch.all(torch.isfinite(output.action_probabilities)))
        self.assertAlmostEqual(float(torch.sum(output.action_probabilities)), 1.0, places=5)

    def test_graph_policy_respects_action_mask(self):
        torch.manual_seed(7)
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
            ablation_flags={"disable_partial_offloading": True},
        )
        policy = build_graph_policy_from_state(graph_state, hidden_dim=32, message_passing_steps=1)

        output = policy(graph_state)

        self.assertEqual(float(output.action_probabilities[1]), 0.0)
        self.assertEqual(float(output.action_probabilities[2]), 0.0)
        self.assertEqual(float(output.action_probabilities[3]), 0.0)

    def test_graph_policy_predict_is_deterministic_with_fixed_weights(self):
        torch.manual_seed(13)
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )
        policy = build_graph_policy_from_state(graph_state, hidden_dim=32, message_passing_steps=2)

        action_a, _ = policy.predict(graph_state, deterministic=True)
        action_b, _ = policy.predict(graph_state, deterministic=True)

        self.assertEqual(action_a, action_b)
        self.assertIn(action_a, range(6))

    def test_graph_state_tensor_conversion_and_manual_mask(self):
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )
        tensors = graph_state_to_tensors(graph_state)
        logits = torch.zeros(6)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])

        masked_logits = apply_action_mask(logits, mask)

        self.assertEqual(tensors["node_features"].shape[0], graph_state.node_features.shape[0])
        self.assertLess(masked_logits[1].item(), -1e8)
        self.assertLess(masked_logits[3].item(), -1e8)

    def test_semantic_prior_fusion_modes_are_distinct(self):
        torch.manual_seed(21)
        device, task, edge_servers, cloud = make_fixture()
        graph_state = build_graph_state(
            device=device,
            task=task,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=DummyChannel(),
        )
        no_fusion_policy = build_graph_policy_from_state(
            graph_state,
            hidden_dim=32,
            message_passing_steps=1,
            semantic_prior_fusion="none",
        )
        state_dict = no_fusion_policy.state_dict()
        late_policy = build_graph_policy_from_state(
            graph_state,
            hidden_dim=32,
            message_passing_steps=1,
            semantic_prior_fusion="late",
            semantic_prior_weight=0.5,
        )
        late_policy.load_state_dict(state_dict)
        input_policy = build_graph_policy_from_state(
            graph_state,
            hidden_dim=32,
            message_passing_steps=1,
            semantic_prior_fusion="input",
        )
        input_policy.load_state_dict(state_dict)

        no_fusion_output = no_fusion_policy(graph_state)
        late_output = late_policy(graph_state)
        input_output = input_policy(graph_state)

        self.assertFalse(torch.allclose(no_fusion_output.logits, late_output.logits))
        self.assertFalse(torch.allclose(no_fusion_output.logits, input_output.logits))

    def test_late_fusion_boosts_high_prior_action(self):
        logits = torch.zeros(6)
        prior = torch.tensor([0.02, 0.03, 0.05, 0.10, 0.70, 0.10])

        fused_logits = fuse_semantic_prior_logits(logits, prior, weight=1.0)

        self.assertEqual(int(torch.argmax(fused_logits).item()), 4)


if __name__ == "__main__":
    unittest.main()
