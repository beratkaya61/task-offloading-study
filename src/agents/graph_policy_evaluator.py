from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agents.graph_policy import GraphPolicyNetwork
from src.env.graph_state_builder import build_graph_state


def build_graph_state_from_env(env, obs) -> object:
    trace_context = {"mode": "trace"} if getattr(env, "trace_mode", False) else {}
    obs_array = np.asarray(obs, dtype=np.float32) if obs is not None else None
    semantic_prior = None
    if obs_array is not None and obs_array.shape[0] >= 12:
        semantic_prior = obs_array[6:12]
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
        semantic_prior=semantic_prior,
        trace_context=trace_context,
        vector_state_reference=obs_array,
    )


def load_graph_policy_checkpoint(checkpoint_path: str, device: str = "cpu") -> GraphPolicyNetwork:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = GraphPolicyNetwork(
        node_feature_dim=int(checkpoint["node_feature_dim"]),
        edge_feature_dim=int(checkpoint["edge_feature_dim"]),
        global_feature_dim=int(checkpoint["global_feature_dim"]),
        hidden_dim=int(checkpoint.get("config", {}).get("graph_policy", {}).get("hidden_dim", 64)),
        message_passing_steps=int(checkpoint.get("config", {}).get("graph_policy", {}).get("message_passing_steps", 2)),
        semantic_prior_fusion=str(checkpoint.get("semantic_prior_fusion", "late")),
        semantic_prior_weight=float(checkpoint.get("semantic_prior_weight", 0.35)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class GraphPolicyEnvAdapter:
    """
    Adapts GraphPolicyNetwork to the existing vector-state evaluator.

    The evaluator still resets/steps the standard OffloadingEnv and passes the
    vector observation to `predict(...)`. This adapter rebuilds the graph view
    from the live environment state right before action selection, so the graph
    policy can be benchmarked under the same rollout logic as MLP PPO.
    """

    def __init__(self, model: GraphPolicyNetwork, env):
        self.model = model
        self.env = env

    def predict(self, obs, deterministic: bool = True):
        graph_state = build_graph_state_from_env(self.env, obs)
        return self.model.predict(graph_state, deterministic=deterministic)
