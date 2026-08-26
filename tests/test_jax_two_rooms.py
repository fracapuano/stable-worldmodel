"""Optional integration coverage for the envX ManyWorlds boundary."""

from __future__ import annotations

import math
from importlib.util import find_spec

import numpy as np
import pytest
import torch
from torch import nn

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


def _world(direction: float, *, receding_horizon: int = 1) -> swm.World:
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
    )
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=2,
        image_shape=(8, 8),
        max_episode_steps=4,
    )
    world.set_policy(policy)
    return world


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
    assert many.simulator_backend == 'envx'
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

        assert result.simulator_backend == 'envx'
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

    def evaluate(backend: str):
        world = _world(1.0, receding_horizon=2)
        many = swm.ManyWorlds(worlds=[world], simulator_backend=backend)
        solver = world.policy.solver
        planned = actions.reshape(1, 2, 2, 4)

        def fixed_plan(_info, _models, *, noise=None, init_action=None):
            del noise, init_action
            return {
                'actions': planned,
                'costs': torch.zeros(1, 2),
                'mean': [planned],
                'var': [torch.ones_like(planned)],
                'compiled': False,
                'compile_error': None,
                'population_backend': 'fixed-test',
            }

        solver.solve_population = fixed_plan
        try:
            return many.evaluate(protocol, solver=solver, eval_budget=4)
        finally:
            many.close()

    gym_result = evaluate('gym')
    envx_result = evaluate('envx')
    torch.testing.assert_close(
        envx_result.task_successes, gym_result.task_successes
    )
    torch.testing.assert_close(
        envx_result.task_final_distances,
        gym_result.task_final_distances,
        rtol=0,
        atol=1e-6,
    )
    assert envx_result.task_successes.all()
    torch.testing.assert_close(
        envx_result.task_final_distances, torch.full((1, 2), 15.0)
    )

    for gym_episode, envx_episode in zip(
        gym_result.evaluations[0].episodes,
        envx_result.evaluations[0].episodes,
        strict=True,
    ):
        assert envx_episode.success == gym_episode.success
        assert envx_episode.length == gym_episode.length == 1
        assert envx_episode.episode_return == gym_episode.episode_return == 0.0
        assert envx_episode.path_cost == gym_episode.path_cost == 5.0
        assert envx_episode.control_cost == gym_episode.control_cost == 1.0
        assert envx_episode.collisions == gym_episode.collisions == 0


@pytest.mark.parametrize('configured_score', ['distance', 'success'])
def test_many_worlds_two_room_score_matches_gym_backend(
    configured_score,
) -> None:
    protocol = _protocol(score=configured_score)
    backend_results = {}
    for backend in ('gym', 'envx'):
        worlds = [_world(1.0), _world(-1.0)]
        many = swm.ManyWorlds(worlds=worlds, simulator_backend=backend)
        try:
            backend_results[backend] = many.evaluate(
                protocol,
                solver=worlds[0].policy.solver,
                # One complete action block keeps both paths open-loop and
                # isolates simulator/score parity from replanning semantics.
                eval_budget=2,
            )
        finally:
            many.close()

    gym_result = backend_results['gym']
    envx_result = backend_results['envx']
    assert gym_result.score_name == envx_result.score_name == configured_score
    torch.testing.assert_close(
        gym_result.task_final_distances,
        envx_result.task_final_distances,
        rtol=1e-6,
        atol=2e-5,
    )
    torch.testing.assert_close(
        gym_result.scores,
        envx_result.scores,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(gym_result.fitness, envx_result.fitness)
