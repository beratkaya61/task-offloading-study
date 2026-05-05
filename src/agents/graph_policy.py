from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from src.env.graph_state_builder import GraphState


@dataclass
class GraphPolicyOutput:
    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_probabilities: torch.Tensor
    graph_embedding: torch.Tensor


class GraphPolicyNetwork(nn.Module):
    """
    PyTorch-only graph-aware policy for Phase 8.

    The first implementation intentionally avoids PyTorch Geometric so the
    project can test graph policy logic even when PyG wheels are unavailable.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
        num_actions: int = 6,
        semantic_prior_fusion: str = "late",
        semantic_prior_weight: float = 0.35,
    ):
        super().__init__()
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be >= 1")
        if semantic_prior_fusion not in {"none", "late"}:
            raise ValueError("semantic_prior_fusion must be 'none' or 'late'")

        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.num_actions = int(num_actions)
        self.semantic_prior_fusion = semantic_prior_fusion
        self.semantic_prior_weight = float(semantic_prior_weight)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        graph_state: GraphState,
        deterministic: bool = False,
    ) -> GraphPolicyOutput:
        tensors = graph_state_to_tensors(graph_state, device=next(self.parameters()).device)
        logits, graph_embedding = self._forward_tensors(tensors)
        masked_logits = apply_action_mask(logits, tensors["action_mask"])
        probabilities = torch.softmax(masked_logits, dim=-1)
        if deterministic:
            action_index = torch.argmax(probabilities, dim=-1)
            probabilities = torch.nn.functional.one_hot(action_index, num_classes=self.num_actions).float()
        return GraphPolicyOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_probabilities=probabilities,
            graph_embedding=graph_embedding,
        )

    def predict(self, graph_state: GraphState, deterministic: bool = True):
        output = self.forward(graph_state, deterministic=False)
        if deterministic:
            action = int(torch.argmax(output.action_probabilities, dim=-1).item())
        else:
            action = int(torch.multinomial(output.action_probabilities, num_samples=1).item())
        return action, output

    def _forward_tensors(self, tensors: Dict[str, torch.Tensor]):
        node_hidden = self.node_encoder(tensors["node_features"])
        edge_hidden = self.edge_encoder(tensors["edge_features"])
        edge_index = tensors["edge_index"]

        if edge_index.numel() > 0:
            src_indices = edge_index[0].long()
            dst_indices = edge_index[1].long()
            for _ in range(self.message_passing_steps):
                src_hidden = node_hidden[src_indices]
                dst_hidden = node_hidden[dst_indices]
                messages = self.message_mlp(torch.cat([src_hidden, dst_hidden, edge_hidden], dim=-1))
                aggregated = torch.zeros_like(node_hidden)
                aggregated.index_add_(0, dst_indices, messages)
                degree = torch.zeros((node_hidden.shape[0], 1), device=node_hidden.device, dtype=node_hidden.dtype)
                degree.index_add_(0, dst_indices, torch.ones((messages.shape[0], 1), device=node_hidden.device, dtype=node_hidden.dtype))
                aggregated = aggregated / degree.clamp_min(1.0)
                node_hidden = self.update_mlp(torch.cat([node_hidden, aggregated], dim=-1))

        graph_embedding = torch.mean(node_hidden, dim=0, keepdim=True)
        global_embedding = self.global_encoder(tensors["global_features"].unsqueeze(0))
        prior = tensors["action_prior"].unsqueeze(0)
        logits = self.policy_head(torch.cat([graph_embedding, global_embedding, prior], dim=-1))

        if self.semantic_prior_fusion == "late":
            logits = fuse_semantic_prior_logits(logits, prior, self.semantic_prior_weight)

        return logits.squeeze(0), graph_embedding.squeeze(0)


def graph_state_to_tensors(graph_state: GraphState, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    graph_state.validate()
    device = device or torch.device("cpu")
    return {
        "node_features": torch.as_tensor(graph_state.node_features, dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(graph_state.edge_index, dtype=torch.long, device=device),
        "edge_features": torch.as_tensor(graph_state.edge_features, dtype=torch.float32, device=device),
        "global_features": torch.as_tensor(graph_state.global_features, dtype=torch.float32, device=device),
        "action_prior": torch.as_tensor(graph_state.action_prior, dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(graph_state.action_mask, dtype=torch.float32, device=device),
    }


def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != action_mask.shape[-1]:
        raise ValueError("logits and action_mask must have the same action dimension")
    invalid = action_mask <= 0.0
    masked_logits = logits.clone()
    masked_logits[invalid] = -1e9
    return masked_logits


def fuse_semantic_prior_logits(logits: torch.Tensor, action_prior: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return logits
    safe_prior = action_prior.clamp_min(1e-6)
    prior_logits = torch.log(safe_prior)
    return logits + float(weight) * prior_logits


def build_graph_policy_from_state(
    graph_state: GraphState,
    hidden_dim: int = 64,
    message_passing_steps: int = 2,
    semantic_prior_fusion: str = "late",
    semantic_prior_weight: float = 0.35,
) -> GraphPolicyNetwork:
    return GraphPolicyNetwork(
        node_feature_dim=graph_state.node_features.shape[1],
        edge_feature_dim=graph_state.edge_features.shape[1],
        global_feature_dim=graph_state.global_features.shape[0],
        hidden_dim=hidden_dim,
        message_passing_steps=message_passing_steps,
        semantic_prior_fusion=semantic_prior_fusion,
        semantic_prior_weight=semantic_prior_weight,
    )
