"""Plan in TwoRoom with its exact simulator dynamics."""

import stable_worldmodel as swm
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.simulator import simulator_goal_encode


def main():
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=1,
        image_shape=(64, 64),
        max_episode_steps=50,
    )
    dynamics = swm.SimulatorDynamics(world.envs.envs[0])
    solver = CEMSolver(
        cost=ShootingCostEvaluator(
            dynamics, GoalMSE(), encode_goal=simulator_goal_encode
        ),
        batch_size=1,
        num_samples=64,
        n_steps=3,
        topk=8,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=1, warm_start=False),
    )
    world.set_policy(policy)

    try:
        results = world.evaluate(episodes=10, seed=0)
        print(f'success rate: {results["success_rate"]:.1f}%')
    finally:
        world.close()


if __name__ == '__main__':
    main()
