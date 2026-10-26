import numpy as np
import pytest

from stable_worldmodel.envs.pusht.env import PushT


@pytest.mark.parametrize('scale', [20, 60])
def test_pusht_default_tee_respects_block_scale_variation(scale):
    env = PushT()
    try:
        env.reset(
            seed=0,
            options={'variation_values': {'block.scale': scale}},
        )
        x_coordinates = [
            vertex.x
            for shape in env.block.shapes
            for vertex in shape.get_vertices()
        ]

        assert max(x_coordinates) - min(x_coordinates) == pytest.approx(
            4 * scale
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    'shape,angle,expected_success,expected_distance',
    [
        ('L', np.pi, False, 180),
        ('T', np.pi, False, 180),
        ('Z', np.pi, True, 180),
        ('square', np.pi / 2, True, 90),
        ('I', np.pi, True, 180),
        ('small_tee', np.pi, False, 180),
        ('+', np.pi / 2, True, 90),
    ],
)
def test_pusht_eval_state_respects_block_shape_rotation_symmetry(
    shape,
    angle,
    expected_success,
    expected_distance,
):
    env = PushT()
    env.variation_space.set_value({'block.shape': env.shapes.index(shape)})

    goal_state = np.array([100.0, 100.0, 256.0, 256.0, 0.0, 0.0, 0.0])
    cur_state = goal_state.copy()
    cur_state[4] = angle

    success, distance = env.eval_state(goal_state, cur_state)

    assert bool(success) is expected_success
    assert distance == pytest.approx(expected_distance)
