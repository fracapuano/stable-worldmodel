import numpy as np
import pytest
import torch

import stable_worldmodel as swm
from stable_worldmodel.backends import (
    DecodedModelBackend,
    OracleModelBackend,
    replay_simulator,
)
from stable_worldmodel.evaluation import (
    EvaluationManifest,
    TaskKey,
)
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.planning.solver.callbacks import CandidateTraceRecorder


class ToyLatentDynamics(torch.nn.Module):
    def encode(self, value):
        value['emb'] = value['latent']
        return value

    def rollout(self, info, actions):
        info['predicted_emb'] = actions[..., :2]
        return info


def test_decoded_backend_uses_physical_goal_and_decodes_rollout():
    decoder = torch.nn.Linear(2, 2)
    with torch.no_grad():
        decoder.weight.copy_(2.0 * torch.eye(2))
        decoder.bias.copy_(torch.tensor([1.0, -1.0]))
    backend = DecodedModelBackend(ToyLatentDynamics(), decoder)

    goal = backend.encode({'state': np.asarray([[4.0, 5.0]])})['emb']
    result = backend.rollout(
        {}, torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]])
    )['predicted_emb']

    assert torch.equal(goal, torch.tensor([[4.0, 5.0]]))
    assert torch.equal(
        result,
        torch.tensor([[[[5.0, 5.0], [9.0, 9.0]]]]),
    )
    assert not any(parameter.requires_grad for parameter in decoder.parameters())


def test_tworoom_oracle_matches_one_step_and_full_replay():
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))
    try:
        for seed in (0, 1, 2, 7, 31):
            world.reset(seed=seed, options={'variation': ['all']})
            env = world.envs.envs[0].unwrapped
            generator = torch.Generator().manual_seed(91 + seed)
            actions = 2 * torch.rand(32, 2, generator=generator) - 1

            replay = replay_simulator(env, actions)
            predicted = [env.get_oracle_state()]
            for action in actions:
                predicted.append(env.oracle_transition(predicted[-1], action))
            predicted = torch.stack(predicted)

            assert torch.equal(predicted[1], replay[1])
            assert torch.equal(predicted, replay)
    finally:
        world.close()


def test_oracle_rollout_honors_action_blocks_without_mutating_env():
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))
    try:
        world.reset(seed=8)
        env = world.envs.envs[0].unwrapped
        initial = env.get_oracle_state()
        actions = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        replay = replay_simulator(env, actions)

        backend = OracleModelBackend()
        backend.bind_envs(world.envs)
        candidates = actions.reshape(1, 1, 2, 4)
        result = backend.rollout(
            {'env_index': torch.tensor([[[0]]])}, candidates
        )['predicted_emb']

        assert torch.equal(result[0, 0], replay[[0, 2, 4]])
        assert torch.equal(env.get_oracle_state(), initial)
    finally:
        world.close()


def test_manifest_world_evaluation_is_deterministic_and_traced():
    tasks = (
        TaskKey(
            environment_seed=1,
            controller_seed=101,
            layout_seed=1,
            start=(50.0, 50.0),
            goal=(175.0, 175.0),
            observation_noise_seed=201,
        ),
        TaskKey(
            environment_seed=2,
            controller_seed=102,
            layout_seed=2,
            start=(50.0, 175.0),
            goal=(175.0, 50.0),
            observation_noise_seed=202,
        ),
    )
    manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=tasks,
    )
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=len(tasks),
        image_shape=(32, 32),
        max_episode_steps=5,
    )
    backend = OracleModelBackend()
    solver = CEMSolver(
        cost=ShootingCostEvaluator(backend, GoalMSE()),
        batch_size=1,
        num_samples=8,
        n_steps=2,
        topk=2,
        callbacks=[CandidateTraceRecorder()],
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=2, receding_horizon=1, warm_start=False),
    )
    world.set_policy(policy)
    try:
        first = world.evaluate(
            manifest=manifest,
            eval_budget=3,
            backend='oracle',
            record=True,
        )
        second = world.evaluate(
            manifest=manifest,
            eval_budget=3,
            backend='oracle',
            record=True,
        )
    finally:
        world.close()

    first_actions = [
        [np.asarray(step.action).tolist() for step in episode.steps]
        for episode in first.episodes
    ]
    second_actions = [
        [np.asarray(step.action).tolist() for step in episode.steps]
        for episode in second.episodes
    ]
    assert first.task_keys == manifest.task_keys
    assert first.controller_seeds == manifest.controller_seeds
    assert first_actions == second_actions
    assert len(first.model_queries) == 3
    assert (
        'model_queries' in first.model_queries[0]['solver_output']['callbacks']
    )
    candidate_history = first.model_queries[0]['solver_output']['callbacks'][
        'model_queries'
    ]
    assert len(candidate_history) == 2
    assert all(len(batch) == 1 for batch in candidate_history)
    assert all(set(batch[0]) == {'candidates'} for batch in candidate_history)
    assert first.backend_type.endswith('.OracleModelBackend')


def test_manifest_start_and_goal_override_conflicting_options(monkeypatch):
    task = TaskKey(
        environment_seed=1,
        controller_seed=2,
        layout_seed=1,
        start=(50.0, 50.0),
        goal=(175.0, 175.0),
        observation_noise_seed=3,
        options=(
            ('state', (40.0, 40.0)),
            ('target_state', (160.0, 160.0)),
        ),
    )
    manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(task,),
    )
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))
    monkeypatch.setattr(world, '_run', lambda **kwargs: None)
    monkeypatch.setattr(world, '_manifest_backend_type', lambda: 'test')
    try:
        world.evaluate(manifest=manifest, eval_budget=1)
        assert np.allclose(world.infos['state'][0], task.start)
        assert np.allclose(world.infos['goal_state'][0], task.goal)
    finally:
        world.close()


def test_manifest_without_recording_skips_full_info_snapshots(monkeypatch):
    class FullSnapshotForbidden:
        def __deepcopy__(self, memo):
            raise AssertionError('full info snapshot was created')

    task = TaskKey(
        environment_seed=1,
        controller_seed=2,
        layout_seed=1,
        start=(50.0, 50.0),
        goal=(175.0, 175.0),
        observation_noise_seed=3,
    )
    manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(task,),
    )
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))

    def reset(**kwargs):
        world.infos = {
            'state': np.zeros((1, 1, 2), dtype=np.float32),
            'pixels': [FullSnapshotForbidden()],
        }
        world.terminateds = np.zeros(1, dtype=bool)
        world.truncateds = np.zeros(1, dtype=bool)

    monkeypatch.setattr(world, 'reset', reset)
    monkeypatch.setattr(world, '_run', lambda **kwargs: None)
    monkeypatch.setattr(world, '_manifest_backend_type', lambda: 'test')
    try:
        result = world.evaluate(manifest=manifest, eval_budget=1, record=False)
        assert result.episodes[0].steps == ()
    finally:
        world.close()


@pytest.mark.parametrize(
    ('task', 'error', 'match'),
    [
        (
            TaskKey(
                environment_seed=1,
                controller_seed=2,
                layout_seed=99,
                start=(50.0, 50.0),
                goal=(175.0, 175.0),
                observation_noise_seed=3,
            ),
            ValueError,
            'layout_seed must equal environment_seed',
        ),
        (
            TaskKey(
                environment_seed=1,
                controller_seed=2,
                layout_seed=1,
                start=(50.0, 50.0),
                goal=(175.0, 175.0),
                observation_noise_seed=3,
                dynamics_parameters=(('observation_noise_std', 0.05),),
            ),
            NotImplementedError,
            'observation noise is not applied',
        ),
        (
            TaskKey(
                environment_seed=1,
                controller_seed=2,
                layout_seed=1,
                start=(50.0, 50.0),
                goal=(175.0, 175.0),
                observation_noise_seed=3,
                dynamics_parameters=(('agent.speed', 7.0),),
            ),
            ValueError,
            'variation_values',
        ),
    ],
)
def test_manifest_evaluation_rejects_unapplied_task_fields(task, error, match):
    manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(task,),
    )
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))
    try:
        with pytest.raises(error, match=match):
            world.evaluate(manifest=manifest, eval_budget=1)
    finally:
        world.close()
