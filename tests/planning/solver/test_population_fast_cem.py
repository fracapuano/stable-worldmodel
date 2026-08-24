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


def test_population_fast_cem_explains_cuda_launch_limit(monkeypatch):
    models = (_model(), _model())
    solver = FastCEMSolver(
        ShootingCostEvaluator(models[0], GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)
    prepared = solver.prepare_population(
        {
            'emb': torch.randn(2, 1, 1, 4),
            'goal_emb': torch.randn(2, 1, 1, 4),
        },
        models,
    )
    assert solver._population_loop is not None

    def fail(*_args):
        raise RuntimeError('CUDA error: invalid argument')

    monkeypatch.setattr(solver._population_loop.cost, '_call_population', fail)
    candidates = torch.randn(2, 1, 4, 3, 2)
    with pytest.raises(RuntimeError) as raised:
        solver._population_loop.cost(candidates, *prepared)

    message = str(raised.value)
    assert 'population=2, tasks=1, samples=4' in message
    assert 'effective_attention_batch=8' in message
    assert 'No automatic tiling or fallback is performed' in message
    assert raised.value.__cause__ is not None


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


@pytest.mark.parametrize('cached_key', ['emb', 'goal_emb'])
def test_population_preparation_accepts_independently_cached_embeddings(
    cached_key,
):
    models = tuple(CountingEncoderModel().eval() for _ in range(2))
    solver = FastCEMSolver(
        ShootingCostEvaluator(models[0], GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=1)
    pixels = torch.randn(2, 1, 1, 1)
    goal = torch.randn(2, 1, 1, 1)
    info = (
        {'emb': pixels.clone(), 'goal': goal}
        if cached_key == 'emb'
        else {'pixels': pixels, 'goal_emb': goal.clone()}
    )

    current, prepared_goal, history, *_state = solver.prepare_population(
        info, models
    )

    torch.testing.assert_close(current, pixels)
    torch.testing.assert_close(prepared_goal, goal[:, :, 0])
    assert history.shape == (2, 1, 0, 2)
