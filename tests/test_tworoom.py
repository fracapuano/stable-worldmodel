import gymnasium as gym
import numpy as np

import stable_worldmodel  # noqa: F401


def test_tworoom_success_radius_is_strictly_eight_pixels():
    env = gym.make('swm/TwoRoom-v1')
    try:
        for distance, expected in ((7.99, True), (8.0, False)):
            env.reset(
                seed=0,
                options={
                    'variation': [],
                    'state': np.array([50.0, 50.0], dtype=np.float32),
                    'target_state': np.array(
                        [50.0 + distance, 50.0], dtype=np.float32
                    ),
                },
            )
            _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
            assert terminated is expected
            assert np.isclose(info['distance_to_target'], distance)
    finally:
        env.close()
