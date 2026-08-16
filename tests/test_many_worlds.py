"""End-to-end tests for population planning with serial Gym realization."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from stable_worldmodel import ManyWorlds, PlanConfig
from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationTask
from stable_worldmodel.world.many_worlds import _same_device

ENV_ID = 'ManyWorldsTest-v0'


class PopulationPlanEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            'emb': gym.spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32),
            'goal_emb': gym.spaces.Box(
                -100.0, 100.0, shape=(1,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.Box(-3.0, 3.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self.state = 0.0
        self.goal = 0.0
        self.reset_count = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            'emb': np.asarray([self.state], dtype=np.float32),
            'goal_emb': np.asarray([self.goal], dtype=np.float32),
        }

    def _info(self) -> dict[str, object]:
        return {
            'state': np.asarray([self.state], dtype=np.float32),
            'collision': False,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.state = float(np.asarray(options.get('state', [0.0])).item())
        self.goal = float(
            np.asarray(options.get('target_state', [1.0])).item()
        )
        self.reset_count += 1
        return self._observation(), self._info()

    def step(self, action):
        self.state += float(np.asarray(action).item())
        return self._observation(), 0.0, False, False, self._info()


if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=PopulationPlanEnv)


class DirectionalWorldModel(nn.Module):
    """A predictor sign determines which physical direction looks positive."""

    def __init__(self, direction: float, *, shared: float = 0.0) -> None:
        super().__init__()
        self.predictor = nn.Linear(1, 1, bias=False)
        self.predictor.weight.data.fill_(direction)
        self.shared = nn.Parameter(
            torch.tensor(shared, dtype=torch.float32), requires_grad=False
        )

    @property
    def population_predictor_parameter_names(self) -> tuple[str, ...]:
        return ('predictor.weight',)

    def encode(self, value):
        return value

    def rollout_from_embeddings(
        self,
        emb,
        action_sequence,
        action_history=None,
        history_size=None,
    ):
        del action_history, history_size
        direction = self.predictor.weight.reshape(1, 1, 1, 1)
        return emb[:, None, -1:, :] + direction * action_sequence.cumsum(dim=2)

    def rollout_population_from_embeddings(
        self,
        emb,
        action_sequence,
        predictor_parameters,
        action_history=None,
        history_size=None,
    ):
        del action_history, history_size
        direction = predictor_parameters[0][:, 0, 0].reshape(-1, 1, 1, 1, 1)
        initial = emb[None, :, None, -1:, :]
        return initial + direction * action_sequence.cumsum(dim=3)


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        split='fitness',
        environment=ENV_ID,
        tasks=(
            EvaluationTask(
                environment_seed=4,
                controller_seed=17,
                layout_seed=4,
                start=(0.0,),
                goal=(1.0,),
                observation_noise_seed=0,
            ),
        ),
    )


def _many_worlds(models) -> ManyWorlds:
    return ManyWorlds(
        models,
        ENV_ID,
        config=PlanConfig(horizon=2, receding_horizon=2),
        cem_kwargs={
            'num_samples': 64,
            'n_steps': 3,
            'topk': 8,
            'seed': 17,
            'compile_kernel': False,
        },
        model_names=('positive', 'negative'),
        num_envs=1,
        max_episode_steps=2,
        add_pixels=False,
    )


def test_many_worlds_batches_planning_then_realizes_each_plan() -> None:
    world = _many_worlds(
        [DirectionalWorldModel(1.0), DirectionalWorldModel(-1.0)]
    )
    try:
        result = world.evaluate(
            protocol=_protocol(), eval_budget=2, record=True
        )

        assert result.planned_actions.shape == (2, 1, 2, 1)
        assert result.planner_costs.shape == (2, 1)
        assert result.environment_actions.shape == (2, 1, 2, 1)
        assert result.population_size == 2
        assert [item.backend for item in result.evaluations] == [
            'positive',
            'negative',
        ]

        positive_final = result.evaluations[0].episodes[0].steps[-1]
        negative_final = result.evaluations[1].episodes[0].steps[-1]
        assert float(positive_final.next_observation['state'].item()) > 0.0
        assert float(negative_final.next_observation['state'].item()) < 0.0

        # One reset produces the shared planning observation, then the same
        # task is reset once before each model's Gym rollout.
        assert world.envs.envs[0].unwrapped.reset_count == 3
        assert world.policy is world._planner_policy
    finally:
        world.close()


def test_many_worlds_rejects_changes_outside_population_parameters() -> None:
    with pytest.raises(ValueError, match='predictor-only variation'):
        _many_worlds(
            [
                DirectionalWorldModel(1.0, shared=0.0),
                DirectionalWorldModel(-1.0, shared=1.0),
            ]
        )


def test_many_worlds_requires_budget_to_fit_one_plan() -> None:
    world = _many_worlds(
        [DirectionalWorldModel(1.0), DirectionalWorldModel(-1.0)]
    )
    try:
        with pytest.raises(ValueError, match='fit in one plan'):
            world.evaluate(protocol=_protocol(), eval_budget=3)
    finally:
        world.close()


def test_many_worlds_accepts_implicit_cuda_device_index() -> None:
    assert _same_device(torch.device('cuda:0'), torch.device('cuda'))
    assert not _same_device(torch.device('cuda:1'), torch.device('cuda:0'))


def test_many_worlds_chunks_population_with_one_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [
        DirectionalWorldModel(1.0),
        DirectionalWorldModel(-1.0),
        DirectionalWorldModel(0.5),
    ]
    world = ManyWorlds(
        models,
        ENV_ID,
        config=PlanConfig(horizon=2, receding_horizon=2),
        cem_kwargs={
            'num_samples': 16,
            'n_steps': 2,
            'topk': 4,
            'compile_kernel': False,
        },
        population_batch_size=2,
        num_envs=1,
        max_episode_steps=2,
        add_pixels=False,
    )
    solve_sizes = []
    prepare_calls = 0
    original_solve = world.solver.solve_population_tensors
    original_prepare = world.solver.cost.prepare

    def solve(parameters, prepared, **kwargs):
        solve_sizes.append(parameters[0].size(0))
        return original_solve(parameters, prepared, **kwargs)

    def prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(world.solver, 'solve_population_tensors', solve)
    monkeypatch.setattr(world.solver.cost, 'prepare', prepare)
    try:
        result = world.plan(protocol=_protocol(), eval_budget=2)
        assert result['actions'].shape == (3, 1, 2, 1)
        assert solve_sizes == [2, 2]
        assert prepare_calls == 1
    finally:
        world.close()


def test_many_worlds_realizes_bfloat16_plans() -> None:
    world = _many_worlds(
        [
            DirectionalWorldModel(1.0).to(torch.bfloat16),
            DirectionalWorldModel(-1.0).to(torch.bfloat16),
        ]
    )
    try:
        result = world.evaluate(protocol=_protocol(), eval_budget=2)
        assert result.environment_actions.dtype == np.float32
        assert len(result.evaluations) == 2
    finally:
        world.close()


def test_many_worlds_rejects_receding_horizon_beyond_plan() -> None:
    with pytest.raises(ValueError, match='cannot exceed'):
        ManyWorlds(
            [DirectionalWorldModel(1.0), DirectionalWorldModel(-1.0)],
            ENV_ID,
            config=PlanConfig(horizon=1, receding_horizon=2),
            cem_kwargs={'compile_kernel': False},
            add_pixels=False,
        )
