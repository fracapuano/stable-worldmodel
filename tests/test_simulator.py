import numpy as np
import pytest
import torch

import stable_worldmodel as swm
from stable_worldmodel.simulator import (
    SimulatorDynamics,
    simulator_goal_encode,
)


@pytest.fixture
def two_room_world():
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(32, 32))
    try:
        yield world
    finally:
        world.close()


def test_tworoom_transition_matches_step_and_has_no_side_effects(
    two_room_world,
):
    for seed in (0, 1, 2, 7, 31):
        two_room_world.reset(seed=seed, options={'variation': ['all']})
        simulator = two_room_world.envs.envs[0].unwrapped
        dynamics = SimulatorDynamics(simulator)
        generator = torch.Generator().manual_seed(91 + seed)
        actions = 2 * torch.rand(32, 2, generator=generator) - 1

        predicted_state = torch.as_tensor(two_room_world.infos['state'][0, -1])
        for action in actions:
            simulator_state = simulator.agent_position.clone()
            predicted_next = dynamics.predict(predicted_state, action)

            assert torch.equal(simulator.agent_position, simulator_state)

            simulator.step(action.numpy())
            assert torch.equal(predicted_next, simulator.agent_position)
            predicted_state = predicted_next


def test_simulator_rollout_honors_action_blocks_without_mutating_env(
    two_room_world,
):
    two_room_world.reset(seed=8)
    simulator = two_room_world.envs.envs[0].unwrapped
    dynamics = SimulatorDynamics(simulator)
    initial = simulator.agent_position.clone()
    actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    candidates = actions.reshape(1, 1, 2, 4)

    result = dynamics.rollout({'state': initial.reshape(1, 1, 2)}, candidates)[
        'predicted_emb'
    ]

    expected = [initial]
    state = initial
    for action in actions:
        state = simulator.get_next_state(state, action)
        expected.append(state)

    assert torch.equal(result[0, 0], torch.stack(expected)[[0, 2, 4]])
    assert torch.equal(simulator.agent_position, initial)


def test_tworoom_transition_supports_planner_batches(two_room_world):
    two_room_world.reset(seed=12)
    simulator = two_room_world.envs.envs[0].unwrapped
    dynamics = SimulatorDynamics(simulator)
    initial = simulator.agent_position.clone()
    states = initial.expand(3, 5, -1).clone()
    actions = torch.linspace(-1, 1, states.numel()).reshape_as(states)

    predicted = dynamics.predict(states, actions)
    expected = torch.stack(
        [
            simulator.get_next_state(state, action)
            for state, action in zip(
                states.reshape(-1, 2), actions.reshape(-1, 2)
            )
        ]
    ).reshape_as(states)

    assert torch.equal(predicted, expected)
    assert torch.equal(simulator.agent_position, initial)


def test_rollout_decodes_actions_and_batches_simulator_calls():
    class BatchSimulator:
        class ActionSpace:
            shape = (2,)

        action_space = ActionSpace()

        def __init__(self):
            self.calls = 0

        def get_next_state(self, state, action):
            self.calls += 1
            assert state.shape == action.shape == (2, 5, 2)
            return state + action

    simulator = BatchSimulator()
    dynamics = SimulatorDynamics(simulator, decode_action=lambda x: x * 2)
    candidates = torch.full((2, 5, 3, 2), 0.25)

    trajectory = dynamics.rollout({'state': torch.zeros(2, 2)}, candidates)[
        'predicted_emb'
    ]

    assert simulator.calls == 3
    assert torch.equal(
        trajectory[:, :, :, 0],
        torch.tensor([0.0, 0.5, 1.0, 1.5]).expand(2, 5, -1),
    )


def test_simulator_dynamics_integrates_with_world_model_planning(
    two_room_world,
):
    two_room_world.reset(
        seed=3,
        options={
            'state': np.asarray([50.0, 50.0], dtype=np.float32),
            'target_state': np.asarray([175.0, 175.0], dtype=np.float32),
        },
    )
    simulator = two_room_world.envs.envs[0]
    dynamics = SimulatorDynamics(simulator)
    solver = swm.planning.CEMSolver(
        cost=swm.planning.ShootingCostEvaluator(
            dynamics,
            swm.planning.GoalMSE(),
            encode_goal=simulator_goal_encode,
        ),
        batch_size=1,
        num_samples=8,
        n_steps=2,
        topk=2,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=2, receding_horizon=1, warm_start=False),
    )
    two_room_world.set_policy(policy)

    action = policy.get_action(dict(two_room_world.infos))

    assert action.shape == (1, 2)
    assert np.isfinite(action).all()


def test_simulator_dynamics_rejects_env_without_get_next_state():
    class NotSimulatable:
        action_space = object()

    with pytest.raises(TypeError, match='Simulatable contract'):
        SimulatorDynamics(NotSimulatable())
