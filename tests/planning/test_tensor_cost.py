"""Tests for the pure-tensor latent planning cost."""

import torch
from torch import nn

from stable_worldmodel.planning import (
    GoalMSE,
    LatentGoalCost,
    ShootingCostEvaluator,
)


class TinyLatentModel(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dim = dim
        self.encode_calls = 0

    def encode(self, info: dict) -> dict:
        self.encode_calls += 1
        values = info['pixels'].flatten(start_dim=2).mean(dim=-1)
        info['emb'] = values.unsqueeze(-1).expand(-1, -1, self.dim)
        return info

    def rollout_from_embeddings(
        self,
        emb: torch.Tensor,
        action_sequence: torch.Tensor,
        action_history: torch.Tensor | None = None,
        history_size: int | None = None,
    ) -> torch.Tensor:
        del action_history, history_size
        context = (
            emb.unsqueeze(1).expand(-1, action_sequence.size(1), -1, -1)
            if emb.ndim == 3
            else emb
        )
        steps = action_sequence.cumsum(dim=2).sum(dim=-1, keepdim=True)
        steps = steps.expand(-1, -1, -1, self.dim) * self.scale
        return torch.cat([context, context[:, :, -1:] + steps], dim=2)

    def rollout(self, info: dict, action_sequence: torch.Tensor) -> dict:
        if 'emb' not in info:
            batch = {
                'pixels': info['pixels'][:, 0],
            }
            info['emb'] = (
                self.encode(batch)['emb']
                .unsqueeze(1)
                .expand(-1, action_sequence.size(1), -1, -1)
            )
        info['predicted_emb'] = self.rollout_from_embeddings(
            info['emb'], action_sequence
        )
        return info


def test_latent_goal_cost_matches_dictionary_shooting_cost():
    torch.manual_seed(0)
    batch, samples, history, horizon, action_dim = 2, 5, 1, 3, 2
    pixels = torch.randn(batch, history, 3, 4, 4)
    goal = torch.randn(batch, history, 3, 4, 4)
    candidates = torch.randn(batch, samples, horizon, action_dim)
    model = TinyLatentModel()

    expanded = {
        'pixels': pixels[:, None].expand(-1, samples, -1, -1, -1, -1),
        'goal': goal[:, None].expand(-1, samples, -1, -1, -1, -1),
        'action': candidates,
    }
    reference = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        expanded, candidates
    )

    tensor_cost = LatentGoalCost(model)
    prepared = tensor_cost.prepare(
        {'pixels': pixels, 'goal': goal},
        device='cpu',
        dtype=torch.float32,
        action_dim=action_dim,
    )
    actual = tensor_cost(candidates, *prepared)

    torch.testing.assert_close(actual, reference)


def test_latent_goal_cost_encodes_once_per_prepare_not_per_candidate_call():
    model = TinyLatentModel()
    cost = LatentGoalCost(model)
    info = {
        'pixels': torch.randn(2, 1, 3, 4, 4),
        'goal': torch.randn(2, 1, 3, 4, 4),
    }
    prepared = cost.prepare(
        info, device='cpu', dtype=torch.float32, action_dim=2
    )
    candidates = torch.randn(2, 4, 3, 2)

    cost(candidates, *prepared)
    cost(candidates, *prepared)

    assert model.encode_calls == 2


def test_latent_goal_cost_validates_action_history_shape():
    model = TinyLatentModel()
    cost = LatentGoalCost(model)
    info = {
        'emb': torch.randn(2, 3, 4),
        'goal_emb': torch.randn(2, 1, 4),
        'action_history': torch.randn(2, 1, 2),
    }

    try:
        cost.prepare(info, device='cpu', dtype=torch.float32, action_dim=2)
    except ValueError as exc:
        assert 'action_history must have shape' in str(exc)
    else:
        raise AssertionError('invalid action history must be rejected')
