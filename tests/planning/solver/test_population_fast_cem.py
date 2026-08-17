"""Behavioral tests for FastCEM's model-population dimension."""

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
    PopulationFastCEMSolver,
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


def _stack_predictors(models, names):
    states = tuple(dict(model.named_parameters()) for model in models)
    return tuple(
        torch.stack(tuple(state[name].detach() for state in states))
        for name in names
    )


class CountingEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(1, 1, bias=False)
        self.encode_batches = []

    @property
    def population_predictor_parameter_names(self):
        return ('predictor.weight',)

    def encode(self, info):
        self.encode_batches.append(info['pixels'].size(0))
        info['emb'] = info['pixels']
        return info

    def rollout_from_embeddings(self, emb, actions, **kwargs):
        del kwargs
        return emb[:, None, -1] + actions.sum((-1, -2), keepdim=True)

    def rollout_population_from_embeddings(
        self, emb, actions, parameters, **kwargs
    ):
        del parameters, kwargs
        return emb[:, :, None, -1] + actions.sum((-1, -2), keepdim=True)


def test_population_fast_cem_matches_independent_fast_cem_solves():
    torch.manual_seed(4)
    first = _model()
    second = deepcopy(first)
    with torch.no_grad():
        for parameter in second.predictor.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    kwargs = {
        'batch_size': 2,
        'num_samples': 24,
        'n_steps': 3,
        'topk': 6,
        'compile_kernel': False,
    }
    serial = tuple(
        FastCEMSolver(ShootingCostEvaluator(model, GoalMSE()), **kwargs)
        for model in (first, second)
    )
    population = PopulationFastCEMSolver(
        ShootingCostEvaluator(first, GoalMSE()), **kwargs
    )
    for solver in (*serial, population):
        _configure(solver)

    info = {
        'emb': torch.randn(2, 2, 1, 4),
        'goal_emb': torch.randn(2, 2, 1, 4),
    }
    parameters = _stack_predictors(
        (first, second), population.predictor_parameter_names
    )
    noise = population.sample_noise(2)

    actual = population.solve_population(info, parameters, noise=noise)
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


def test_population_fast_cem_compiles_as_one_full_graph():
    model = _model()
    solver = PopulationFastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, tasks=1)
    info = {
        'emb': torch.randn(2, 1, 1, 4),
        'goal_emb': torch.randn(2, 1, 1, 4),
    }
    parameters = tuple(
        value.detach()[None].expand(2, *value.shape).clone()
        for name, value in model.named_parameters()
        if name in solver.predictor_parameter_names
    )

    output = solver.solve_population(info, parameters)

    assert output['actions'].shape == (2, 1, 3, 2)
    assert output['compiled'] is True


def test_population_fast_cem_rejects_wrong_population_parameters():
    model = _model()
    solver = PopulationFastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
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
        }
    )

    with pytest.raises(ValueError, match='common population axis'):
        solver.solve_population_tensors(
            (torch.randn(2, 1), torch.randn(3, 1)), prepared
        )


def test_population_preparation_encodes_all_worlds_in_one_batch():
    model = CountingEncoderModel()
    solver = PopulationFastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    _configure(solver, tasks=2)

    current, goal, history = solver.prepare_population(
        {
            'pixels': torch.randn(3, 2, 1, 1),
            'goal': torch.randn(3, 2, 1, 1),
        }
    )

    assert model.encode_batches == [6, 6]
    assert current.shape == (3, 2, 1, 1)
    assert goal.shape == (3, 2, 1)
    assert history.shape == (3, 2, 0, 2)
