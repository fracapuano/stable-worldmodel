"""Behavioral tests for FastCEM over independent model instances."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor


def _model() -> LeWM:
    return LeWM(
        encoder=nn.Identity(),
        predictor=Predictor(
            num_frames=3,
            depth=1,
            heads=1,
            mlp_dim=8,
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dim_head=4,
            dropout=0.0,
            emb_dropout=0.0,
        ),
        action_encoder=nn.Linear(2, 4),
    ).eval()


def _configure(solver, tasks=2):
    solver.configure(
        action_space=spaces.Box(-1, 1, shape=(tasks, 2), dtype=np.float32),
        n_envs=tasks,
        config=PlanConfig(horizon=3, receding_horizon=1),
    )


class CountingEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(1, 1, bias=False)
        self.encode_batches = []
        self.rollout_calls = 0

    def encode(self, info):
        self.encode_batches.append(info['pixels'].size(0))
        info['emb'] = info['pixels']
        return info

    def rollout_from_embeddings(self, emb, actions, **kwargs):
        del kwargs
        self.rollout_calls += 1
        return emb[:, None, -1] + actions.sum(dim=(-1, -2))[..., None]


class FactorizedModel(nn.Module):
    """Tiny LeWM-like model with a low-rank residual held in buffers."""

    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(1, 1, bias=False)
        self.register_buffer('factor_a', torch.zeros(1, 1))
        self.register_buffer('factor_b', torch.zeros(1, 1))

    def encode(self, info):
        info['emb'] = info['pixels']
        return info

    def rollout_from_embeddings(self, emb, actions, **kwargs):
        del kwargs
        scale = (
            self.predictor.weight[0, 0]
            + (self.factor_a @ self.factor_b.T)[0, 0]
        )
        return emb[:, None, -1] + scale * actions.sum(dim=(-1, -2))[..., None]


def test_population_fast_cem_matches_independent_fast_cem_solves():
    torch.manual_seed(4)
    first = _model()
    second = deepcopy(first)
    with torch.no_grad():
        for parameter in second.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    models = (first, second)
    kwargs = {
        'batch_size': 2,
        'num_samples': 24,
        'n_steps': 3,
        'topk': 6,
        'compile_kernel': False,
    }
    serial = tuple(
        FastCEMSolver(ShootingCostEvaluator(model, GoalMSE()), **kwargs)
        for model in models
    )
    population = FastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()), **kwargs
    )
    for solver in (*serial, population):
        _configure(solver)

    info = {
        'emb': torch.randn(2, 2, 1, 4),
        'goal_emb': torch.randn(2, 2, 1, 4),
    }
    noise = population.sample_noise(2)

    actual = population.solve_population(info, models, noise=noise)
    assert population.population_backend == 'functional_vmap'
    expected = tuple(
        solver.solve_tensors(
            solver.prepare({key: value[index] for key, value in info.items()}),
            noise=noise,
        )
        for index, solver in enumerate(serial)
    )

    torch.testing.assert_close(
        actual['actions'],
        torch.stack(tuple(output['actions'] for output in expected)),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        actual['costs'],
        torch.stack(tuple(output['costs'] for output in expected)),
        rtol=3e-5,
        atol=3e-6,
    )


def test_population_models_keep_native_parameter_shapes():
    first = _model()
    second = deepcopy(first)
    expected = tuple(
        tuple(parameter.shape for parameter in model.parameters())
        for model in (first, second)
    )
    solver = FastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)

    output = solver.solve_population(
        {
            'emb': torch.randn(2, 1, 1, 4),
            'goal_emb': torch.randn(2, 1, 1, 4),
        },
        (first, second),
    )

    assert expected == tuple(
        tuple(parameter.shape for parameter in model.parameters())
        for model in (first, second)
    )
    assert output['population_backend'] == 'functional_vmap'


def test_population_stacks_complete_model_state():
    first = _model()
    second = deepcopy(first)
    with torch.no_grad():
        for parameter in second.predictor.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    solver = FastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)

    info = {
        'emb': torch.randn(2, 1, 1, 4),
        'goal_emb': torch.randn(2, 1, 1, 4),
    }
    prepared = solver.prepare_population(info, (first, second))
    population_state = prepared[3:]
    assert len(population_state) == len(
        tuple(first.parameters()) + tuple(first.buffers())
    )
    assert all(value.size(0) == 2 for value in population_state)

    output = solver.solve_population(info, (first, second))

    assert output['population_backend'] == 'functional_vmap'
    assert output['actions'].shape == (2, 1, 3, 2)


def test_population_fast_cem_compiles_as_one_full_graph():
    first = _model()
    second = deepcopy(first)
    solver = FastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, tasks=1)

    output = solver.solve_population(
        {
            'emb': torch.randn(2, 1, 1, 4),
            'goal_emb': torch.randn(2, 1, 1, 4),
        },
        (first, second),
    )

    assert output['actions'].shape == (2, 1, 3, 2)
    assert output['compiled'] is True
    assert output['population_backend'] == 'functional_vmap'


def test_population_fast_cem_requires_solver_model_first():
    first = _model()
    second = deepcopy(first)
    solver = FastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)

    with pytest.raises(ValueError, match='first population model'):
        solver.prepare_population(
            {
                'emb': torch.randn(1, 1, 1, 4),
                'goal_emb': torch.randn(1, 1, 1, 4),
            },
            (second,),
        )


def test_population_preparation_uses_each_models_ordinary_encoder():
    models = tuple(CountingEncoderModel().eval() for _ in range(3))
    solver = FastCEMSolver(
        ShootingCostEvaluator(models[0], GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=2)

    current, goal, history, *_state = solver.prepare_population(
        {
            'pixels': torch.randn(3, 2, 1, 1),
            'goal': torch.randn(3, 2, 1, 1),
        },
        models,
    )

    # One functional/vmapped Python dispatch covers the whole population.
    assert models[0].encode_batches == [2, 2]
    assert all(not model.encode_batches for model in models[1:])
    assert current.shape == (3, 2, 1, 1)
    assert goal.shape == (3, 2, 1)
    assert history.shape == (3, 2, 0, 2)


def test_factorized_population_matches_serial_factor_forwards():
    torch.manual_seed(12)
    model = FactorizedModel().eval()
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=16,
        n_steps=3,
        topk=4,
        compile_kernel=False,
    )
    _configure(solver, tasks=2)
    factors = {
        'factor_a': torch.tensor([[[0.25]], [[-0.50]], [[1.00]]]),
        'factor_b': torch.tensor([[[0.75]], [[0.40]], [[-0.20]]]),
    }
    info = {
        'emb': torch.randn(3, 2, 1, 1),
        'goal_emb': torch.randn(3, 2, 1, 1),
    }
    noise = solver.sample_noise(2)

    actual = solver.solve_factorized_population(info, factors, noise=noise)
    expected = []
    for index in range(3):
        model.factor_a.copy_(factors['factor_a'][index])
        model.factor_b.copy_(factors['factor_b'][index])
        member_info = {key: value[index] for key, value in info.items()}
        expected.append(
            solver.solve_tensors(
                solver.prepare(member_info),
                noise=noise,
            )
        )

    torch.testing.assert_close(
        actual['actions'],
        torch.stack([output['actions'] for output in expected]),
    )
    torch.testing.assert_close(
        actual['costs'],
        torch.stack([output['costs'] for output in expected]),
    )
    assert actual['population_backend'] == 'factorized_vmap'


def test_factorized_population_prepares_only_low_rank_state():
    model = FactorizedModel().eval()
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)
    factors = {
        'factor_a': torch.randn(4, 1, 1),
        'factor_b': torch.randn(4, 1, 1),
    }

    prepared = solver.prepare_factorized_population(
        {
            'emb': torch.randn(4, 1, 1, 1),
            'goal_emb': torch.randn(4, 1, 1, 1),
        },
        factors,
    )

    assert len(prepared[3:]) == len(factors)
    assert tuple(value.shape for value in prepared[3:]) == (
        (4, 1, 1),
        (4, 1, 1),
    )
    assert tuple(model.predictor.weight.shape) == (1, 1)


def test_factorized_population_compiles_as_one_full_graph():
    model = FactorizedModel().eval()
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, tasks=1)

    output = solver.solve_factorized_population(
        {
            'emb': torch.randn(2, 1, 1, 1),
            'goal_emb': torch.randn(2, 1, 1, 1),
        },
        {
            'factor_a': torch.randn(2, 1, 1),
            'factor_b': torch.randn(2, 1, 1),
        },
    )

    assert output['actions'].shape == (2, 1, 3, 2)
    assert output['compiled'] is True
    assert output['population_backend'] == 'factorized_vmap'
