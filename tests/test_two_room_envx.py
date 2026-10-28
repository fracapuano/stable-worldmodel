"""TwoRoom uses the common native pool without changing frozen tasks."""

from importlib.util import find_spec

import numpy as np
import pytest
import torch

from stable_worldmodel.envs.two_room.env import TwoRoomEnv
from stable_worldmodel.world.envx import EnvXPool

pytestmark = pytest.mark.skipif(
    find_spec('envx') is None, reason='envX is optional'
)


@pytest.mark.parametrize('seed', [3, 8])
@pytest.mark.parametrize('varied', [False, True])
def test_native_reset_pixels_and_physics_match_gym(seed, varied):
    options = {'state': [40.0, 49.0], 'target_state': [180.0, 49.0]}
    if varied:
        options['variation'] = [
            'wall.axis',
            'wall.thickness',
            'door.number',
            'door.position',
            'door.size',
            'agent.radius',
            'agent.speed',
            'target.radius',
            'agent.color',
            'target.color',
            'wall.color',
            'door.color',
            'background.color',
            'rendering.render_target',
        ]
    gym = TwoRoomEnv()
    pool = EnvXPool(
        'swm/TwoRoom-v1',
        1,
        image_shape=(224, 224),
        max_episode_steps=50,
        device='cpu',
    )
    try:
        _, reference = gym.reset(seed=seed, options=options)
        _, actual = pool.reset(seed=[seed], options=[options])
        state = pool.snapshot()['inputs']['extras']
        assert int(state['wall_axis'][0]) == gym.wall_axis
        assert int(state['wall_thickness'][0]) == gym.wall_thickness
        assert int(state['num_doors'][0]) == gym.num_doors
        np.testing.assert_array_equal(
            np.asarray(state['door_positions'][0, : gym.num_doors]),
            gym.door_positions,
        )
        for key, expected in [
            ('pixels', gym.render()),
            ('goal', reference['goal']),
        ]:
            delta = actual[key][0, 0].cpu().numpy().astype(
                int
            ) - expected.astype(int)
            # Torch/JAX exp round differently at uint8 boundaries.
            assert np.abs(delta).max() <= 1
        actions = (
            np.random.default_rng(17)
            .uniform(-2, 2, (20, 2))
            .astype(np.float32)
        )
        for action in actions:
            _, _, done, _, reference = gym.step(action)
            _, _, terminated, _, actual = pool.step(action[None])
            np.testing.assert_allclose(
                actual['state'][0, 0], reference['state'], atol=1e-5
            )
            np.testing.assert_allclose(
                actual['proprio'][0, 0], reference['proprio'], atol=1e-5
            )
            assert bool(terminated[0]) == done
            if done:
                break
    finally:
        pool.close()
        gym.close()


def test_snapshot_remains_usable_after_reset_and_without_resetting_consumer():
    pool = EnvXPool(
        'swm/TwoRoom-v1',
        1,
        image_shape=None,
        max_episode_steps=4,
        device='cpu',
    )
    consumer = pool._with_num_envs(1)
    try:
        first = {'state': [40, 49], 'target_state': [180, 49]}
        pool.reset(seed=3, options=[first])
        snapshot = pool.snapshot()
        actions = torch.ones(2, 1, 4, 2)
        expected = consumer.rollout(actions, initial_state=snapshot)
        pool.reset(seed=5, options=[dict(first, state=[180, 49])])
        actual = consumer.rollout(actions, initial_state=snapshot)
        torch.testing.assert_close(
            actual.final_observations, expected.final_observations
        )
        torch.testing.assert_close(
            actual.final_distances, expected.final_distances
        )
        assert not np.array_equal(
            snapshot['inputs']['qpos'], pool.snapshot()['inputs']['qpos']
        )
        with pytest.raises(ValueError, match='one row per task'):
            consumer.rollout(
                actions,
                initial_state={
                    'inputs': {'qpos': np.zeros((2, 2))},
                    'contract': {},
                },
            )
    finally:
        consumer.close()
        pool.close()
