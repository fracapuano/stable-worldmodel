"""Behavioral tests for the device-resident CEM path."""

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from stable_worldmodel.planning import LatentGoalCost
from stable_worldmodel.planning.solver import AcceleratedCEMSolver, CEMSolver
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor


class TargetCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def prepare(self, info, *, device, dtype, action_dim):
        del action_dim
        return (info['target'].to(device=device, dtype=dtype),)

    def forward(self, candidates, target):
        return (
            (candidates - target[:, None] * self.scale).square().sum((-1, -2))
        )

    def get_cost(self, info, candidates):
        return (candidates - info['target']).square().sum((-1, -2))


def _configure(solver, n_envs=3):
    solver.configure(
        action_space=spaces.Box(
            -np.inf, np.inf, shape=(n_envs, 2), dtype=np.float32
        ),
        n_envs=n_envs,
        config=PlanConfig(horizon=4, receding_horizon=2),
    )


@pytest.mark.parametrize('manifest_streams', [False, True])
def test_eager_path_matches_reference_cem(manifest_streams):
    kwargs = dict(
        batch_size=2,
        num_samples=24,
        n_steps=4,
        topk=6,
        device='cpu',
        seed=17,
    )
    reference = CEMSolver(TargetCost(), **kwargs)
    accelerated = AcceleratedCEMSolver(
        TargetCost(), compile_kernel=False, **kwargs
    )
    _configure(reference)
    _configure(accelerated)
    info = {'target': torch.randn(3, 4, 2)}
    if manifest_streams:
        info.update(
            controller_seed=torch.tensor([101, 202, 303]),
            step_idx=torch.tensor([0, 5, 9]),
        )

    expected, actual = reference.solve(info), accelerated.solve(info)

    torch.testing.assert_close(actual['actions'], expected['actions'])
    torch.testing.assert_close(actual['var'][0], expected['var'][0])
    assert actual['costs'] == pytest.approx(expected['costs'])


def test_tensor_api_is_device_resident_and_deterministic(capsys):
    solver = AcceleratedCEMSolver(
        TargetCost(),
        num_samples=12,
        n_steps=3,
        topk=4,
        compile_kernel=False,
    )
    _configure(solver, n_envs=2)
    prepared = solver.prepare({'target': torch.randn(2, 4, 2)})
    noise = solver.sample_noise(2)

    first = solver.solve_tensors(prepared, noise=noise)
    second = solver.solve_tensors(prepared, noise=noise)

    torch.testing.assert_close(first['actions'], second['actions'])
    torch.testing.assert_close(first['costs'], second['costs'])
    assert first['actions'].device == noise.device
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize(
    'kwargs',
    [
        {'n_steps': 0},
        {'num_samples': 1, 'topk': 1},
        {'num_samples': 8, 'topk': 1},
        {'num_samples': 8, 'topk': 9},
        {'callbacks': [object()]},
    ],
)
def test_rejects_incompatible_settings(kwargs):
    with pytest.raises(ValueError):
        AcceleratedCEMSolver(TargetCost(), **kwargs)


def test_compile_failure_falls_back_to_eager(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise RuntimeError('synthetic compiler failure')

    monkeypatch.setattr(torch, 'compile', fail_compile)
    solver = AcceleratedCEMSolver(
        TargetCost(),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=True,
    )
    _configure(solver, 1)

    output = solver.solve({'target': torch.zeros(1, 4, 2)})

    assert output['compiled'] is False
    assert 'synthetic compiler failure' in output['compile_error']


def test_real_lewm_terminal_path_is_fully_capturable():
    model = LeWM(
        nn.Identity(),
        Predictor(
            num_frames=1,
            depth=1,
            heads=1,
            mlp_dim=8,
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dim_head=4,
        ),
        nn.Linear(2, 4),
    ).eval()
    solver = AcceleratedCEMSolver(
        LatentGoalCost(model),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, 1)

    output = solver.solve_tensors(
        solver.prepare(
            {
                'emb': torch.randn(1, 1, 4),
                'goal_emb': torch.randn(1, 1, 4),
            }
        )
    )

    assert output['actions'].shape == (1, 4, 2)
    assert output['compiled'] is True


def test_compiled_loop_reads_updated_parameters():
    cost = TargetCost()
    solver = AcceleratedCEMSolver(
        cost,
        num_samples=32,
        n_steps=3,
        topk=6,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, 1)
    info = {
        'target': torch.ones(1, 4, 2),
        'controller_seed': torch.tensor([88]),
    }

    with torch.no_grad():
        cost.scale.zero_()
    first = solver.solve(info)['actions']
    with torch.no_grad():
        cost.scale.fill_(2)
    second = solver.solve(info)['actions']

    assert not torch.equal(first, second)
