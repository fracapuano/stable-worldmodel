"""Behavioral tests for the device-resident CEM path."""

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from stable_worldmodel.planning import (
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.planning.solver import CEMSolver, FastCEMSolver
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
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


class TinyDynamics(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dim = dim

    def encode(self, info):
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
        if emb.ndim == 3:
            emb = emb[:, None].expand(-1, actions.size(1), -1, -1)
        steps = actions.cumsum(2).sum(-1, keepdim=True) * self.scale
        steps = steps.expand(-1, -1, -1, self.dim)
        rollout = torch.cat([emb, emb[:, :, -1:] + steps], dim=2)
        return rollout[:, :, -1] if terminal_only else rollout

    def rollout(self, info, actions):
        if 'emb' not in info:
            emb = self.encode({'pixels': info['pixels'][:, 0]})['emb']
            info['emb'] = emb[:, None].expand(-1, actions.size(1), -1, -1)
        info['predicted_emb'] = self.rollout_from_embeddings(
            info['emb'], actions
        )
        return info


class ReferenceOnlyCost(nn.Module):
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
    fast = FastCEMSolver(TargetCost(), compile_kernel=False, **kwargs)
    _configure(reference)
    _configure(fast)
    info = {'target': torch.randn(3, 4, 2)}
    if manifest_streams:
        info.update(
            controller_seed=torch.tensor([101, 202, 303]),
            step_idx=torch.tensor([0, 5, 9]),
        )

    expected, actual = reference.solve(info), fast.solve(info)

    torch.testing.assert_close(actual['actions'], expected['actions'])
    torch.testing.assert_close(actual['var'][0], expected['var'][0])
    assert actual['costs'] == pytest.approx(expected['costs'])


def test_standard_shooting_cost_automatically_uses_fast_path():
    torch.manual_seed(0)
    model = TinyDynamics()
    kwargs = dict(
        batch_size=2,
        num_samples=24,
        n_steps=4,
        topk=6,
        device='cpu',
        seed=17,
    )
    reference = CEMSolver(ShootingCostEvaluator(model, GoalMSE()), **kwargs)
    cost = ShootingCostEvaluator(model, GoalMSE())
    fast = FastCEMSolver(cost, compile_kernel=False, **kwargs)
    _configure(reference)
    _configure(fast)
    info = {
        'pixels': torch.randn(3, 2, 3, 2, 2),
        'goal': torch.randn(3, 2, 3, 2, 2),
        'action': torch.zeros(3, 1, 2),
        'controller_seed': torch.tensor([101, 202, 303]),
        'step_idx': torch.tensor([0, 5, 9]),
    }

    expected = reference.solve(dict(info))
    actual = fast.solve(dict(info))

    assert fast.cost is cost
    torch.testing.assert_close(actual['actions'], expected['actions'])
    torch.testing.assert_close(actual['var'][0], expected['var'][0])
    torch.testing.assert_close(
        torch.tensor(actual['costs']),
        torch.tensor(expected['costs']),
        rtol=2e-6,
        atol=1e-7,
    )


def test_rejects_callbacks_instead_of_falling_back():
    with pytest.raises(ValueError, match='does not support callbacks'):
        FastCEMSolver(TargetCost(), callbacks=[object()])


@pytest.mark.parametrize(
    'cost',
    [
        ReferenceOnlyCost(),
        ShootingCostEvaluator(nn.Identity(), GoalMSE()),
    ],
)
def test_rejects_incompatible_cost_or_model_instead_of_falling_back(cost):
    with pytest.raises(TypeError, match='use CEMSolver for other costs'):
        FastCEMSolver(cost)


def test_standard_cost_supports_receding_horizon_and_warm_start():
    class RecordingSolver(FastCEMSolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_actions = []

        def solve(self, info, init_action=None):
            self.init_actions.append(
                None if init_action is None else init_action.clone()
            )
            return super().solve(info, init_action)

    solver = RecordingSolver(
        ShootingCostEvaluator(TinyDynamics(), GoalMSE()),
        num_samples=8,
        n_steps=2,
        topk=2,
        compile_kernel=False,
    )
    events = []
    policy = WorldModelPolicy(
        solver,
        PlanConfig(horizon=4, receding_horizon=2, warm_start=True),
        on_plan=events.append,
    )
    env = type('VectorEnv', (), {})()
    env.num_envs = 1
    env.action_space = spaces.Box(-1, 1, shape=(1, 2), dtype=np.float32)
    env.single_action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    policy.set_env(env)

    def info(step):
        return {
            'pixels': torch.randn(1, 1, 3, 2, 2),
            'goal': torch.randn(1, 1, 3, 2, 2),
            'action': torch.zeros(1, 1, 2),
            'controller_seed': torch.tensor([101]),
            'step_idx': torch.tensor([step]),
        }

    policy.get_action(info(0))
    policy.get_action(info(1))
    assert len(events) == 1
    policy.get_action(info(2))

    assert len(events) == 2
    assert solver.init_actions[0] is None
    assert solver.init_actions[1].shape == (1, 2, 2)
    torch.testing.assert_close(
        solver.init_actions[1], events[0]['solver_output']['actions'][:, 2:]
    )


def test_tensor_api_is_device_resident_and_deterministic(capsys):
    solver = FastCEMSolver(
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
    ],
)
def test_rejects_incompatible_settings(kwargs):
    with pytest.raises(ValueError):
        FastCEMSolver(TargetCost(), **kwargs)


def test_compile_failure_falls_back_to_eager(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise RuntimeError('synthetic compiler failure')

    monkeypatch.setattr(torch, 'compile', fail_compile)
    solver = FastCEMSolver(
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


def test_real_lewm_standard_cost_is_fully_capturable(monkeypatch):
    def reject_while_loop(*_args, **_kwargs):
        raise AssertionError('compiled FastCEM must use the static loop')

    monkeypatch.setattr(torch, 'while_loop', reject_while_loop)
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
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        num_samples=4,
        n_steps=2,
        topk=2,
        compile_kernel=True,
        compile_backend='aot_eager',
        compile_fallback=False,
    )
    _configure(solver, 1)

    candidates = torch.randn(1, 4, 4, 2)
    prepared = solver.prepare(
        {
            'emb': torch.randn(1, 1, 4),
            'goal_emb': torch.randn(1, 1, 4),
        }
    )
    current, goal, history = prepared
    reference = model.rollout(
        {
            'pixels': torch.empty(1, 4, 1, 1),
            'emb': current[:, None].expand(-1, 4, -1, -1),
            'action_history': history[:, None].expand(-1, 4, -1, -1),
        },
        candidates,
    )['predicted_emb'][:, :, -1]
    actual = solver._tensor_cost(candidates, *prepared)
    torch.testing.assert_close(
        actual, (reference - goal[:, None]).square().sum(-1)
    )

    output = solver.solve_tensors(prepared)

    assert output['actions'].shape == (1, 4, 2)
    assert output['compiled'] is True


def test_compiled_loop_reads_updated_parameters():
    cost = TargetCost()
    solver = FastCEMSolver(
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
