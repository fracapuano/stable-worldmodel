"""Parity and execution-contract tests for AcceleratedCEMSolver."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces
from torch import nn

from stable_worldmodel.planning.solver import (
    AcceleratedCEMSolver,
    CEMSolver,
    PopulationAcceleratedCEMSolver,
)
from stable_worldmodel.planning.tensor_cost import PopulationLatentGoalCost
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor


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


class PopulationTargetCost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def prepare(
        self,
        info_dict: dict,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor]:
        del action_dim
        return (info_dict['target'].to(device=device, dtype=dtype),)

    def forward(
        self,
        candidates: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> torch.Tensor:
        scale = parameters[0].reshape(-1, 1, 1, 1, 1)
        return (
            (candidates - target[None, :, None] * scale)
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


def _reference_population_cem(noise, target, scales, *, topk, var_scale):
    population = scales.numel()
    mean = torch.zeros(
        population, target.size(0), target.size(1), target.size(2)
    )
    var = torch.full_like(mean, var_scale)
    costs = mean.new_zeros(mean.shape[:2])
    for step_noise in noise:
        sampled = step_noise.unsqueeze(0) * var.unsqueeze(2) + mean.unsqueeze(
            2
        )
        candidates = torch.cat([mean.unsqueeze(2), sampled[:, :, 1:]], dim=2)
        candidate_costs = (
            (
                candidates
                - target[None, :, None] * scales.reshape(-1, 1, 1, 1, 1)
            )
            .square()
            .sum(dim=(-1, -2))
        )
        values, indices = torch.topk(
            candidate_costs, k=topk, dim=2, largest=False
        )
        gather = indices[:, :, :, None, None].expand(
            -1, -1, -1, candidates.size(3), candidates.size(4)
        )
        elites = torch.gather(candidates, 2, gather)
        mean = elites.mean(dim=2)
        var = elites.std(dim=2)
        costs = values.mean(dim=2)
    return mean, var, costs


def test_population_cem_matches_independent_reference_with_common_noise():
    cost = PopulationTargetCost()
    solver = PopulationAcceleratedCEMSolver(
        cost,
        num_samples=16,
        n_steps=3,
        topk=4,
        var_scale=0.7,
        seed=9,
        compile_kernel=False,
    )
    _configure(solver, n_envs=2)
    target = torch.randn(2, 4, 2)
    scales = torch.tensor([0.5, 1.0, 1.5])
    noise = solver.sample_noise(task_batch_size=2)

    actual = solver.solve_population(
        {'target': target}, (scales,), noise=noise
    )
    expected = _reference_population_cem(
        noise, target, scales, topk=4, var_scale=0.7
    )

    torch.testing.assert_close(actual['actions'], expected[0])
    torch.testing.assert_close(actual['var'], expected[1])
    torch.testing.assert_close(actual['costs'], expected[2])


def test_population_cem_tensor_loop_is_fully_capturable():
    solver = PopulationAcceleratedCEMSolver(
        PopulationTargetCost(),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, n_envs=1)
    noise = solver.sample_noise(task_batch_size=1)

    output = solver.solve_population(
        {'target': torch.ones(1, 4, 2)},
        (torch.tensor([1.0, 2.0]),),
        noise=noise,
    )

    assert output['actions'].shape == (2, 1, 4, 2)
    assert output['compiled'] is True
    assert output['compile_error'] is None


def test_population_lewm_cem_graph_is_fully_capturable():
    predictor = Predictor(
        num_frames=2,
        depth=1,
        heads=2,
        mlp_dim=8,
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        dim_head=2,
    ).eval()
    model = LeWM(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
    ).eval()
    solver = PopulationAcceleratedCEMSolver(
        PopulationLatentGoalCost(model),
        num_samples=4,
        n_steps=1,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    action_space = gym_spaces.Box(
        low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32
    )
    solver.configure(
        action_space=action_space,
        n_envs=1,
        config=PlanConfig(horizon=2, receding_horizon=1),
    )
    parameters = tuple(
        value.detach().unsqueeze(0).expand(2, *value.shape)
        for value in predictor.parameters()
    )

    output = solver.solve_population(
        {
            'emb': torch.randn(1, 1, 4),
            'goal_emb': torch.randn(1, 1, 4),
        },
        parameters,
    )

    assert output['actions'].shape == (2, 1, 2, 4)
    assert output['compiled'] is True
    assert output['compile_error'] is None
