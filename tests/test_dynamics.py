import pytest
import torch
from torch import nn

from stable_worldmodel.dynamics import DecodedDynamics


class AdditiveDynamics(nn.Module):
    def encode(self, info_dict):
        info_dict['emb'] = info_dict['pixels']
        return info_dict

    def rollout(self, info_dict, action_candidates):
        info_dict['predicted_emb'] = action_candidates + 1
        return info_dict


def test_decoded_dynamics_exposes_state_goal_and_decodes_rollout():
    decoder = nn.Linear(2, 1, bias=False)
    decoder.weight.data.fill_(2)
    dynamics = DecodedDynamics(AdditiveDynamics(), decoder)

    info = dynamics.encode({'state': torch.tensor([[3, 4]])})
    result = dynamics.rollout(info, torch.zeros(1, 2, 3, 2))

    assert torch.equal(info['emb'], torch.tensor([[3, 4]]))
    assert torch.equal(result['predicted_emb'], torch.full((1, 2, 3, 1), 4.0))
    assert not decoder.training
    assert not any(
        parameter.requires_grad for parameter in decoder.parameters()
    )


def test_decoded_dynamics_requires_state_and_rollout_output():
    dynamics = DecodedDynamics(AdditiveDynamics(), nn.Identity())
    with pytest.raises(KeyError, match='decoded goal encoding'):
        dynamics.encode({})

    class EmptyDynamics(AdditiveDynamics):
        def rollout(self, info_dict, action_candidates):
            return info_dict

    dynamics = DecodedDynamics(EmptyDynamics(), nn.Identity())
    with pytest.raises(KeyError, match='learned rollout did not populate'):
        dynamics.rollout({}, torch.zeros(1, 1, 1, 2))
