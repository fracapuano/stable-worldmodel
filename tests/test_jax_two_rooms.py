"""Optional integration coverage for the envX ManyWorlds boundary."""

from __future__ import annotations

import importlib
import math
from copy import deepcopy
from importlib.util import find_spec

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

import stable_worldmodel as swm
from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationTask
from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.world.jax_two_rooms import JaxTwoRoomsRollout

pytestmark = pytest.mark.skipif(
    find_spec('envx') is None,
    reason='envX optional dependency is not installed',
)


class TinyTwoRoomsModel(nn.Module):
    def __init__(self, direction: float) -> None:
        super().__init__()
        self.direction = nn.Parameter(
            torch.tensor(direction), requires_grad=False
        )

    def encode(self, value):
        pixels = value['pixels'].float()
        dims = tuple(range(2, pixels.ndim))
        return {'emb': pixels.mean(dim=dims, keepdim=False)[..., None]}

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
        delta = action_sequence.sum(dim=-1, keepdim=True)
        future = emb[:, None, -1:] + self.direction * delta.cumsum(dim=2)
        if terminal_only:
            return future[:, :, -1]
        context = emb[:, None].expand(-1, action_sequence.size(1), -1, -1)
        return torch.cat([context, future], dim=2)


def _protocol(*, score: str = 'distance') -> EvaluationProtocol:
    return EvaluationProtocol(
        split='fitness',
        environment='swm/TwoRoom-v1',
        metadata=(('score', score),),
        tasks=(
            EvaluationTask(
                environment_seed=3,
                controller_seed=11,
                layout_seed=3,
                start=(40.0, 49.0),
                goal=(180.0, 49.0),
                observation_noise_seed=0,
                name='right',
            ),
            EvaluationTask(
                environment_seed=5,
                controller_seed=13,
                layout_seed=5,
                start=(180.0, 49.0),
                goal=(40.0, 49.0),
                observation_noise_seed=0,
                name='left',
            ),
        ),
    )


def _world(
    direction: float,
    *,
    receding_horizon: int = 1,
    max_episode_steps: int = 4,
    process=None,
    extra_wrappers=None,
) -> swm.World:
    model = TinyTwoRoomsModel(direction).eval()
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        batch_size=2,
        num_samples=4,
        n_steps=1,
        topk=2,
        compile_kernel=False,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=2,
            receding_horizon=receding_horizon,
            history_len=3,
            action_block=2,
        ),
        process=process,
    )
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=2,
        image_shape=(8, 8),
        max_episode_steps=max_episode_steps,
        extra_wrappers=extra_wrappers,
    )
    world.set_policy(policy)
    return world


def _evaluate_fixed_plan(
    protocol: EvaluationProtocol,
    actions: torch.Tensor,
    *,
    max_episode_steps: int = 4,
):
    world = _world(
        1.0,
        receding_horizon=2,
        max_episode_steps=max_episode_steps,
    )
    many = swm.ManyWorlds(worlds=[world])
    solver = world.policy.solver
    planned = actions.reshape(1, len(protocol.tasks), 2, 4)

    def fixed_plan(_info, _models, *, noise=None, init_action=None):
        del noise, init_action
        return {
            'actions': planned,
            'costs': torch.zeros(1, len(protocol.tasks)),
            'mean': [planned],
            'var': [torch.ones_like(planned)],
            'compiled': False,
            'compile_error': None,
            'population_backend': 'fixed-test',
            'solve_time_seconds': 0.0,
        }

    solver.solve_population = fixed_plan
    try:
        return many.evaluate(protocol, solver=solver, eval_budget=4)
    finally:
        many.close()


def test_jax_rollout_uses_torch_dlpack_and_preserves_task_state() -> None:
    world = _world(1.0)
    protocol = _protocol()
    try:
        world._reset_from_protocol(protocol, eval_budget=4)
        rollout = JaxTwoRoomsRollout(
            population_size=3,
            num_tasks=2,
            eval_budget=4,
        )
        initial_state = rollout.initial_state(world)
        actions = torch.zeros(3, 2, 4, 2)
        outcome = rollout(initial_state, actions)

        starts = torch.tensor([task.start for task in protocol.tasks])
        goals = torch.tensor([task.goal for task in protocol.tasks])
        expected = torch.linalg.vector_norm(starts - goals, dim=-1)
        assert outcome.final_observations.shape == (3, 2, 10)
        assert outcome.successes.shape == (3, 2)
        assert outcome.final_distances.shape == (3, 2)
        torch.testing.assert_close(
            outcome.final_distances,
            expected.expand(3, 2),
        )
        assert not outcome.successes.any()
    finally:
        world.close()


def test_many_worlds_envx_returns_device_resident_population_scores() -> None:
    worlds = [_world(1.0), _world(-1.0)]
    many = swm.ManyWorlds(worlds=worlds)
    assert many.bootstrap_simulator_count == many.num_tasks == 2
    assert many.simulator_count == many.population_size * many.num_tasks == 4

    def unexpected_reset(*_args, **_kwargs):
        pytest.fail('envX must only reset the first task-sized Gym pool')

    worlds[1].envs.reset = unexpected_reset
    try:
        result = many.evaluate(
            _protocol(),
            solver=worlds[0].policy.solver,
            eval_budget=4,
        )

        assert result.planning_calls == 1
        assert result.planned_actions.shape == (1, 2, 2, 2, 4)
        assert result.environment_actions.shape == (2, 2, 4, 2)
        assert torch.is_tensor(result.environment_actions)
        assert result.task_successes.shape == (2, 2)
        assert result.task_final_distances.shape == (2, 2)
        assert result.scores.shape == (2,)
        assert result.scores.device == result.planned_actions.device
        torch.testing.assert_close(
            result.scores,
            1.0
            - result.task_final_distances.float().mean(dim=1)
            / math.hypot(224, 224),
        )
        # envX outcomes belong to the result object; they must not be published
        # as the state of Gym environments which were never stepped.
        starts = np.asarray([task.start for task in _protocol().tasks])
        actual = np.stack(
            [
                np.asarray(wrapped.unwrapped.agent_position)
                for wrapped in worlds[0].envs.envs
            ]
        )
        np.testing.assert_allclose(actual, starts)
        np.testing.assert_allclose(
            np.asarray(worlds[0].infos['state'])[:, -1], starts
        )
        assert worlds[1].infos == {}
    finally:
        many.close()


def test_many_worlds_accepts_one_population_forward_world() -> None:
    world = _world(1.0, receding_horizon=2)
    solver = world.policy.solver

    class Population:
        model = solver.cost.model
        population_size = 3
        backend = 'test-functional-vmap'
        compiled = False

        def forward(self, *args, module, **kwargs):
            parameters, buffers = stack_module_state(
                [deepcopy(module) for _ in range(self.population_size)]
            )

            def call(member_parameters, member_buffers):
                return functional_call(
                    module,
                    (member_parameters, member_buffers),
                    args,
                    kwargs,
                    strict=True,
                )

            return vmap(call)(parameters, buffers)

    population = Population()
    many = swm.ManyWorlds.from_population(world, population_size=3)
    try:
        result = many.evaluate(
            _protocol(),
            solver=solver,
            population=population,
            eval_budget=4,
        )
        assert many.bootstrap_simulator_count == 2
        assert many.simulator_count == 6
        assert result.planned_actions.shape == (1, 3, 2, 2, 4)
        assert result.task_final_distances.shape == (3, 2)
        assert result.population_backend == 'test-functional-vmap'
        assert result.compiled is False
    finally:
        many.close()


def test_many_worlds_envx_supports_success_score() -> None:
    successes = torch.tensor([[True, False], [True, True]])
    distances = torch.zeros(2, 2)
    name, scores = swm.ManyWorlds._score_two_rooms(
        _protocol(score='success'), successes, distances
    )
    assert name == 'success'
    torch.testing.assert_close(scores, torch.tensor([0.5, 1.0]))


def test_many_worlds_envx_stops_metrics_at_first_terminal() -> None:
    protocol = EvaluationProtocol(
        split='fitness',
        environment='swm/TwoRoom-v1',
        metadata=(('score', 'distance'),),
        tasks=(
            EvaluationTask(
                environment_seed=3,
                controller_seed=11,
                layout_seed=3,
                start=(20.0, 49.0),
                goal=(40.0, 49.0),
                observation_noise_seed=0,
                name='right',
            ),
            EvaluationTask(
                environment_seed=5,
                controller_seed=13,
                layout_seed=5,
                start=(40.0, 49.0),
                goal=(20.0, 49.0),
                observation_noise_seed=0,
                name='left',
            ),
        ),
    )
    actions = torch.tensor(
        [
            [
                [[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            ]
        ]
    )
    starts = torch.tensor([task.start for task in protocol.tasks])
    goals = torch.tensor([task.goal for task in protocol.tasks])
    full_plan_endpoint = starts + 5.0 * actions[0].sum(dim=1)
    assert torch.all(
        torch.linalg.vector_norm(full_plan_endpoint - goals, dim=1) > 16.0
    )

    result = _evaluate_fixed_plan(protocol, actions)
    assert result.task_successes.all()
    torch.testing.assert_close(
        result.task_final_distances, torch.full((1, 2), 15.0)
    )
    for episode in result.evaluations[0].episodes:
        assert episode.success
        assert episode.length == 1
        assert episode.episode_return == 0.0
        assert episode.path_cost == 5.0
        assert episode.control_cost == 1.0
        assert episode.collisions == 0


def test_many_worlds_envx_honors_world_episode_limit() -> None:
    protocol = _protocol()
    actions = torch.zeros(1, 2, 4, 2)

    result = _evaluate_fixed_plan(
        protocol,
        actions,
        max_episode_steps=2,
    )
    assert not result.task_successes.any()
    torch.testing.assert_close(
        result.task_final_distances, torch.full((1, 2), 140.0)
    )
    for episode in result.evaluations[0].episodes:
        assert episode.length == 2
        assert not episode.success
        assert episode.path_cost == 0.0
        assert episode.control_cost == 0.0


def test_many_worlds_envx_requires_common_world_episode_limit() -> None:
    worlds = [
        _world(1.0, max_episode_steps=2),
        _world(-1.0, max_episode_steps=3),
    ]
    try:
        with pytest.raises(
            ValueError, match='same positive max_episode_steps'
        ):
            swm.ManyWorlds(worlds=worlds)
    finally:
        for world in worlds:
            world.close()


def test_many_worlds_envx_counts_collisions() -> None:
    protocol = EvaluationProtocol(
        split='fitness',
        environment='swm/TwoRoom-v1',
        metadata=(('score', 'distance'),),
        tasks=(
            EvaluationTask(
                environment_seed=3,
                controller_seed=11,
                layout_seed=3,
                start=(90.0, 30.0),
                goal=(20.0, 200.0),
                observation_noise_seed=0,
                name='low-side',
            ),
            EvaluationTask(
                environment_seed=5,
                controller_seed=13,
                layout_seed=5,
                start=(134.0, 30.0),
                goal=(200.0, 200.0),
                observation_noise_seed=0,
                name='high-side',
            ),
        ),
    )
    actions = torch.tensor(
        [
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
            ]
        ]
    )

    result = _evaluate_fixed_plan(protocol, actions)
    for episode in result.evaluations[0].episodes:
        assert episode.collisions == 2
        assert episode.length == 4
        assert episode.path_cost == 10.5
        assert episode.control_cost == 4.0


def test_many_worlds_preserves_the_fast_cem_callback_contract() -> None:
    worlds = [_world(1.0), _world(-1.0)]
    events = [[], []]
    for index, world in enumerate(worlds):
        world.policy.on_plan = events[index].append
    many = swm.ManyWorlds(worlds=worlds)
    try:
        many.evaluate(
            _protocol(), solver=worlds[0].policy.solver, eval_budget=2
        )

        for population_events in events:
            assert len(population_events) == 1
            solver_output = population_events[0]['solver_output']
            assert solver_output['solve_time_seconds'] > 0
            assert solver_output['population_backend'] == 'functional_vmap'
    finally:
        many.close()


def test_many_worlds_rejects_numpy_action_postprocessing() -> None:
    class NumpyActionProcess:
        def transform(self, value):
            return value

        def inverse_transform(self, value):
            return value.detach().cpu().numpy()

    process = {'action': NumpyActionProcess()}
    worlds = [_world(1.0, process=process), _world(-1.0, process=process)]
    many = swm.ManyWorlds(worlds=worlds)
    try:
        with pytest.raises(TypeError, match='must return a torch.Tensor'):
            many.evaluate(
                _protocol(), solver=worlds[0].policy.solver, eval_budget=2
            )
    finally:
        many.close()


def test_many_worlds_rejects_custom_wrapper_stacks() -> None:
    class UnsupportedWrapper(gym.Wrapper):
        pass

    worlds = [
        _world(1.0, extra_wrappers=[UnsupportedWrapper]),
        _world(-1.0, extra_wrappers=[UnsupportedWrapper]),
    ]
    try:
        with pytest.raises(ValueError, match='custom wrapper'):
            swm.ManyWorlds(worlds=worlds)
    finally:
        for world in worlds:
            world.close()


def test_many_worlds_requires_envx(monkeypatch) -> None:
    module = importlib.import_module('stable_worldmodel.world.many_worlds')
    monkeypatch.setattr(module, 'find_spec', lambda _name: None)
    worlds = [_world(1.0), _world(-1.0)]
    try:
        with pytest.raises(ImportError, match='requires envX'):
            swm.ManyWorlds(worlds=worlds)
    finally:
        for world in worlds:
            world.close()
