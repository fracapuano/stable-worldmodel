"""Tests for the LeWM terminal tensor cost."""

import pytest
import torch
from torch import nn

from stable_worldmodel.planning import (
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.planning.tensor_cost import LatentGoalCost


class TinyLatentModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dim = dim
        self.encoded_shapes = []

    def encode(self, info):
        self.encoded_shapes.append(info['pixels'].shape)
        values = info['pixels'].flatten(start_dim=2).mean(-1)
        info['emb'] = values[..., None].expand(-1, -1, self.dim)
        return info

    def rollout_from_embeddings(
        self,
        emb,
        actions,
        action_history=None,
        history_size=None,
        *,
        terminal_only=False,
    ):
        del action_history, history_size
        context = (
            emb[:, None].expand(-1, actions.size(1), -1, -1)
            if emb.ndim == 3
            else emb
        )
        steps = actions.cumsum(2).sum(-1, keepdim=True)
        steps = steps.expand(-1, -1, -1, self.dim) * self.scale
        rollout = torch.cat([context, context[:, :, -1:] + steps], dim=2)
        return rollout[:, :, -1] if terminal_only else rollout

    def rollout(self, info, actions):
        if 'emb' not in info:
            emb = self.encode({'pixels': info['pixels'][:, 0]})['emb']
            info['emb'] = emb[:, None].expand(-1, actions.size(1), -1, -1)
        info['predicted_emb'] = self.rollout_from_embeddings(
            info['emb'], actions
        )
        return info


def test_tensor_cost_matches_dictionary_path_and_encodes_once():
    torch.manual_seed(0)
    batch, samples, horizon, action_dim = 2, 5, 3, 2
    pixels = torch.randn(batch, 1, 3, 4, 4)
    goal = torch.randn(batch, 3, 3, 4, 4)
    candidates = torch.randn(batch, samples, horizon, action_dim)

    reference_model = TinyLatentModel()
    expanded = {
        'pixels': pixels[:, None].expand(-1, samples, -1, -1, -1, -1),
        'goal': goal[:, None].expand(-1, samples, -1, -1, -1, -1),
        'action': candidates,
    }
    expected = ShootingCostEvaluator(reference_model, GoalMSE()).get_cost(
        expanded, candidates
    )

    model = TinyLatentModel()
    cost = LatentGoalCost(model)
    prepared = cost.prepare(
        {'pixels': pixels, 'goal': goal},
        device='cpu',
        dtype=torch.float32,
        action_dim=action_dim,
    )
    actual = cost(candidates, *prepared)
    cost(candidates, *prepared)

    torch.testing.assert_close(actual, expected)
    assert model.encoded_shapes == [pixels.shape, goal[:, -1:].shape]
    assert prepared[1].shape == (batch, model.dim)


def test_tensor_cost_rejects_mismatched_action_history():
    cost = LatentGoalCost(TinyLatentModel())
    with pytest.raises(ValueError, match='action_history'):
        cost.prepare(
            {
                'emb': torch.randn(2, 3, 4),
                'goal_emb': torch.randn(2, 1, 4),
                'action_history': torch.randn(2, 1, 2),
            },
            device='cpu',
            dtype=torch.float32,
            action_dim=2,
        )
