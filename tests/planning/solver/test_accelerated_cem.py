"""Parity and execution-contract tests for AcceleratedCEMSolver."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces
from torch import nn

from stable_worldmodel.planning.solver import (
    AcceleratedCEMSolver,
    CEMSolver,
)
from stable_worldmodel.policy import PlanConfig


class DictionaryQuadraticCost(nn.Module):
    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        return candidates.square().sum(dim=(-1, -2))

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        del info_dict
        return self(action_candidates)


class TensorQuadraticCost(DictionaryQuadraticCost):
    def prepare(
        self,
        info_dict: dict,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor]:
        del action_dim
        target = info_dict['target'].to(device=device, dtype=dtype)
        return (target,)

    def forward(
        self, candidates: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (candidates - target[:, None]).square().sum(dim=(-1, -2))


class DictionaryTargetCost(DictionaryQuadraticCost):
    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        target = info_dict['target']
        return (action_candidates - target).square().sum(dim=(-1, -2))


class ParameterizedTensorCost(TensorQuadraticCost):
    def __init__(self) -> None:
        super().__init__()
        self.target_scale = nn.Parameter(torch.ones(()))

    def forward(
        self, candidates: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (
            (candidates - target[:, None] * self.target_scale)
            .square()
            .sum(dim=(-1, -2))
        )


def _configure(solver, *, n_envs: int = 3) -> None:
    action_space = gym_spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(n_envs, 2),
        dtype=np.float32,
    )
    config = PlanConfig(horizon=4, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)


@pytest.mark.parametrize('manifest_streams', [False, True])
def test_accelerated_cem_matches_reference_eager(manifest_streams):
    kwargs = {
        'batch_size': 2,
        'num_samples': 24,
        'n_steps': 4,
        'topk': 6,
        'device': 'cpu',
        'seed': 17,
    }
    reference = CEMSolver(cost=DictionaryTargetCost(), **kwargs)
    accelerated = AcceleratedCEMSolver(
        cost=TensorQuadraticCost(), compile_kernel=False, **kwargs
    )
    _configure(reference)
    _configure(accelerated)
    info = {'target': torch.randn(3, 4, 2)}
    if manifest_streams:
        info['controller_seed'] = torch.tensor([101, 202, 303])
        info['step_idx'] = torch.tensor([0, 5, 9])

    expected = reference.solve(info)
    actual = accelerated.solve(info)

    torch.testing.assert_close(actual['actions'], expected['actions'])
    torch.testing.assert_close(actual['var'][0], expected['var'][0])
    assert actual['costs'] == pytest.approx(expected['costs'])
    assert actual['compiled'] is False


def test_accelerated_cem_rejects_iteration_callbacks():
    with pytest.raises(ValueError, match='per-iteration callbacks'):
        AcceleratedCEMSolver(cost=TensorQuadraticCost(), callbacks=[object()])


def test_accelerated_cem_falls_back_if_compile_fails(monkeypatch):
    def fail_compile(*args, **kwargs):
        del args, kwargs
        raise RuntimeError('synthetic compiler failure')

    monkeypatch.setattr(torch, 'compile', fail_compile)
    solver = AcceleratedCEMSolver(
        cost=TensorQuadraticCost(),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=True,
    )
    _configure(solver, n_envs=1)

    output = solver.solve({'target': torch.zeros(1, 4, 2)})

    assert output['actions'].shape == (1, 4, 2)
    assert output['compiled'] is False
    assert 'synthetic compiler failure' in output['compile_error']


def test_accelerated_cem_tensor_loop_is_fully_capturable():
    """AOT eager validates full-graph capture without a platform compiler."""
    solver = AcceleratedCEMSolver(
        cost=TensorQuadraticCost(),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, n_envs=1)

    output = solver.solve({'target': torch.zeros(1, 4, 2)})

    assert output['actions'].shape == (1, 4, 2)
    assert output['compiled'] is True
    assert output['compile_error'] is None


def test_compiled_kernel_reads_updated_model_parameters():
    cost = ParameterizedTensorCost()
    solver = AcceleratedCEMSolver(
        cost=cost,
        num_samples=32,
        n_steps=3,
        topk=6,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, n_envs=1)
    info = {
        'target': torch.ones(1, 4, 2),
        'controller_seed': torch.tensor([88]),
    }

    with torch.no_grad():
        cost.target_scale.zero_()
    first = solver.solve(info)['actions']
    with torch.no_grad():
        cost.target_scale.fill_(2.0)
    second = solver.solve(info)['actions']

    assert not torch.equal(first, second)
