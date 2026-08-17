"""End-to-end tests for population-parallel closed-loop evaluation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationTask
from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    PopulationFastCEMSolver,
    ShootingCostEvaluator,
)
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
        return self._observation(), self._info()

    def step(self, action):
        self.state += float(np.asarray(action).item())
        return self._observation(), 0.0, self.state > 0.0, False, self._info()


if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=PopulationPlanEnv)


class DirectionalWorldModel(nn.Module):
    """The predictor sign changes which real action appears goal-directed."""

    def __init__(self, direction: float, *, shared: float = 0.0) -> None:
        super().__init__()
        self.predictor = nn.Linear(1, 1, bias=False)
        self.predictor.weight.data.fill_(direction)
        self.shared = nn.Parameter(
            torch.tensor(shared, dtype=torch.float32), requires_grad=False
        )
        self.serial_calls = 0
        self.population_calls = 0

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
        *,
        terminal_only=False,
    ):
        del action_history, history_size
        self.serial_calls += 1
        direction = self.predictor.weight.reshape(1, 1, 1, 1)
        delta = action_sequence.sum(dim=-1, keepdim=True)
        future = emb[:, None, -1:, :] + direction * delta.cumsum(dim=2)
        if terminal_only:
            return future[:, :, -1]
        context = emb[:, None].expand(-1, action_sequence.size(1), -1, -1)
        return torch.cat([context, future], dim=2)

    def rollout_population_from_embeddings(
        self,
        emb,
        action_sequence,
        predictor_parameters,
        action_history=None,
        history_size=None,
        *,
        terminal_only=False,
    ):
        del action_history, history_size
        self.population_calls += 1
        direction = predictor_parameters[0][:, 0, 0].reshape(-1, 1, 1, 1, 1)
        if emb.ndim == 3:
            emb = emb[None].expand(action_sequence.size(0), -1, -1, -1)
        delta = action_sequence.sum(dim=-1, keepdim=True)
        future = emb[:, :, None, -1:, :] + direction * delta.cumsum(dim=3)
        if terminal_only:
            return future[:, :, :, -1]
        context = emb[:, :, None].expand(
            -1, -1, action_sequence.size(2), -1, -1
        )
        return torch.cat([context, future], dim=3)


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
                name='right',
            ),
            EvaluationTask(
                environment_seed=5,
                controller_seed=23,
                layout_seed=5,
                start=(0.5,),
                goal=(-1.0,),
                observation_noise_seed=0,
                name='left',
            ),
        ),
    )


def _world(
    direction: float,
    *,
    shared: float = 0.0,
    config: swm.PlanConfig | None = None,
) -> tuple[swm.World, DirectionalWorldModel]:
    config = config or swm.PlanConfig(
        horizon=2,
        receding_horizon=1,
        warm_start=True,
    )
    model = DirectionalWorldModel(direction, shared=shared).eval()
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        batch_size=2,
        num_samples=64,
        n_steps=3,
        topk=8,
        compile_kernel=False,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, history_keys=('emb',)
    )
    world = swm.World(
        ENV_ID,
        num_envs=2,
        max_episode_steps=4,
        add_pixels=False,
    )
    world.set_policy(policy)
    return world, model


def _many_worlds(**kwargs):
    first, first_model = _world(1.0, **kwargs)
    second, second_model = _world(-1.0, **kwargs)
    first_model.train()
    second_model.train()
    many = swm.ManyWorlds.init(
        worlds=[first, second], model_names=['positive', 'negative']
    )
    assert not any(model.training for model in many.models)
    solver = PopulationFastCEMSolver.from_fast_cem(first.policy.solver)
    return many, solver, (first_model, second_model)


def test_many_worlds_matches_individual_closed_loop_evaluations() -> None:
    many, solver, _ = _many_worlds()
    protocol = _protocol()
    try:
        batched = many.evaluate(
            protocol, solver=solver, eval_budget=4, record=True
        )
        serial = tuple(
            world.evaluate(
                protocol=protocol,
                eval_budget=4,
                backend=name,
                record=True,
            )
            for world, name in zip(many.worlds, many.model_names, strict=True)
        )

        assert batched.planned_actions.shape == (4, 2, 2, 2, 1)
        assert batched.planner_costs.shape == (4, 2, 2)
        assert batched.planner_variances.shape == (4, 2, 2, 2, 1)
        assert batched.environment_actions.shape == (2, 2, 4, 1)
        assert batched.population_size == 2
        assert batched.planning_calls == 4

        for population_result, serial_result in zip(
            batched.evaluations, serial, strict=True
        ):
            assert population_result.backend == serial_result.backend
            assert (
                population_result.protocol_digest
                == serial_result.protocol_digest
            )
            for actual_episode, expected_episode in zip(
                population_result.episodes,
                serial_result.episodes,
                strict=True,
            ):
                assert actual_episode.length == expected_episode.length
                assert actual_episode.success == expected_episode.success
                np.testing.assert_allclose(
                    [step.action for step in actual_episode.steps],
                    [step.action for step in expected_episode.steps],
                    rtol=1e-6,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    actual_episode.steps[-1].next_observation['state'],
                    expected_episode.steps[-1].next_observation['state'],
                    rtol=1e-6,
                    atol=1e-7,
                )
    finally:
        many.close()


def test_many_worlds_uses_population_rollout_not_model_loop() -> None:
    many, solver, models = _many_worlds()
    before = tuple(
        {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        for model in models
    )
    try:
        result = many.evaluate(_protocol(), solver=solver, eval_budget=3)

        assert result.planning_calls == 3
        assert models[0].population_calls == 3 * solver.n_steps
        assert models[1].population_calls == 0
        assert [model.serial_calls for model in models] == [0, 0]
        for model, expected in zip(models, before, strict=True):
            for name, value in model.state_dict().items():
                assert torch.equal(value, expected[name])
    finally:
        many.close()


def test_many_worlds_supports_history_action_blocks_and_replanning() -> None:
    config = swm.PlanConfig(
        horizon=2,
        receding_horizon=1,
        history_len=3,
        action_block=2,
        warm_start=True,
    )
    many, solver, _ = _many_worlds(config=config)
    protocol = _protocol()
    try:
        batched = many.evaluate(
            protocol, solver=solver, eval_budget=4, record=True
        )
        serial = tuple(
            world.evaluate(protocol=protocol, eval_budget=4, record=True)
            for world in many.worlds
        )

        assert batched.planning_calls == 2
        assert batched.planned_actions.shape == (2, 2, 2, 2, 2)
        for actual, expected in zip(batched.evaluations, serial, strict=True):
            for actual_episode, expected_episode in zip(
                actual.episodes, expected.episodes, strict=True
            ):
                np.testing.assert_allclose(
                    [step.action for step in actual_episode.steps],
                    [step.action for step in expected_episode.steps],
                    rtol=1e-6,
                    atol=1e-7,
                )
    finally:
        many.close()


def test_many_worlds_infers_budget_and_accepts_named_constructor() -> None:
    many, solver, _ = _many_worlds()
    try:
        result = many.evaluate(_protocol(), solver=solver)
        assert result.environment_actions.shape[2] == 4
        assert [item.backend for item in result.evaluations] == [
            'positive',
            'negative',
        ]
    finally:
        many.close()


def test_many_worlds_rejects_changes_outside_population_parameters() -> None:
    first, _ = _world(1.0, shared=0.0)
    second, _ = _world(-1.0, shared=1.0)
    many = swm.ManyWorlds(worlds=[first, second])
    solver = PopulationFastCEMSolver.from_fast_cem(first.policy.solver)
    try:
        with pytest.raises(ValueError, match='predictor-only variation'):
            many.evaluate(_protocol(), solver=solver, eval_budget=1)
    finally:
        many.close()


def test_many_worlds_rejects_mismatched_world_configuration() -> None:
    first, _ = _world(1.0)
    second, _ = _world(
        -1.0,
        config=swm.PlanConfig(horizon=3, receding_horizon=1),
    )
    try:
        with pytest.raises(ValueError, match='same PlanConfig'):
            swm.ManyWorlds(worlds=[first, second])
    finally:
        first.close()
        second.close()


def test_many_worlds_accepts_implicit_cuda_device_index() -> None:
    assert _same_device(torch.device('cuda:0'), torch.device('cuda'))
    assert not _same_device(torch.device('cuda:1'), torch.device('cuda:0'))
