"""Configuration matching respects shoulder periodicity and wrist limits."""

from types import SimpleNamespace

import numpy as np
import pytest

CASES = [
    pytest.param([0, 0], [0, 0], 0, True, id='same-configuration'),
    pytest.param(
        [np.pi - 0.01, 0],
        [-np.pi + 0.01, 0],
        0.02,
        True,
        id='positive-shoulder-seam',
    ),
    pytest.param(
        [-np.pi + 0.01, 0],
        [np.pi - 0.01, 0],
        0.02,
        True,
        id='negative-shoulder-seam',
    ),
    pytest.param([4 * np.pi, 0], [0, 0], 0, True, id='full-shoulder-turns'),
    pytest.param([0, 0], [-2 * np.pi, 0], 0, True, id='goal-full-turn'),
    pytest.param([np.pi, 0], [0, 0], np.pi, False, id='half-shoulder-turn'),
    pytest.param(
        [0, np.deg2rad(160)],
        [0, np.deg2rad(-160)],
        np.deg2rad(320),
        False,
        id='wrist-limits-do-not-wrap',
    ),
    pytest.param([0.02, -0.04], [0, 0], 0.04, True, id='worst-joint-error'),
    pytest.param([0.049, 0], [0, 0], 0.049, True, id='shoulder-inside'),
    pytest.param([0, 0.049], [0, 0], 0.049, True, id='wrist-inside'),
    pytest.param([0.05, 0], [0, 0], 0.05, False, id='shoulder-threshold'),
    pytest.param([0, 0.05], [0, 0], 0.05, False, id='wrist-threshold'),
    pytest.param([0.051, 0], [0, 0], 0.051, False, id='shoulder-outside'),
    pytest.param([0, 0.051], [0, 0], 0.051, False, id='wrist-outside'),
]


@pytest.mark.parametrize('state,goal,expected,success', CASES)
def test_envx_reacher_configuration_distance(state, goal, expected, success):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    from stable_worldmodel.world.envx import _Engine

    engine = _Engine.__new__(_Engine)
    engine.name = 'reacher'
    engine.adapter = SimpleNamespace(vector=lambda value: value)
    # Include a second, matching task to check reduction stays per task.
    states = jnp.asarray([state, [0, 0]], dtype=jnp.float32)
    goals = jnp.asarray([goal, [0, 0]], dtype=jnp.float32)
    _, distance, hits = jax.jit(engine._metrics)(
        states, goals, {'threshold': 0.05}
    )
    np.testing.assert_allclose(distance, [expected, 0], atol=1e-6)
    np.testing.assert_array_equal(hits, [success, True])


@pytest.mark.parametrize('state,goal,expected,success', CASES)
def test_native_reacher_configuration_termination(
    state, goal, expected, success
):
    pytest.importorskip('dm_control.suite.reacher')
    from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import (
        ReacherQPosMatchTask,
    )

    task = ReacherQPosMatchTask.__new__(ReacherQPosMatchTask)
    task.target_qpos = np.asarray(goal)
    task.qpos_threshold = 0.05
    physics = SimpleNamespace(data=SimpleNamespace(qpos=np.asarray(state)))
    assert task.get_termination(physics) == (0.0 if success else None)


def test_native_reacher_without_goal_does_not_terminate():
    pytest.importorskip('dm_control.suite.reacher')
    from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import (
        ReacherQPosMatchTask,
    )

    task = ReacherQPosMatchTask.__new__(ReacherQPosMatchTask)
    task.target_qpos = None
    assert task.get_termination(None) is None
