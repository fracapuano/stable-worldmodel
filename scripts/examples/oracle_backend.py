"""Cheap end-to-end TwoRoom exact-backend smoke example."""

import stable_worldmodel as swm
from stable_worldmodel.evaluation import EvaluationManifest, TaskKey
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)


def main():
    tasks = tuple(
        TaskKey(
            name=f'oracle-smoke-{index}',
            environment_seed=10 + index,
            controller_seed=20 + index,
            layout_seed=30 + index,
            start=start,
            goal=goal,
            observation_noise_seed=40 + index,
        )
        for index, (start, goal) in enumerate(
            [
                ((50.0, 50.0), (175.0, 175.0)),
                ((50.0, 175.0), (175.0, 50.0)),
            ]
        )
    )
    manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=tasks,
        metadata=(('purpose', 'oracle-backend-smoke'),),
    )
    backend = swm.OracleModelBackend()
    cost = ShootingCostEvaluator(backend, GoalMSE())
    solver = CEMSolver(
        cost=cost,
        batch_size=2,
        num_samples=32,
        n_steps=3,
        topk=4,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=3, receding_horizon=1, warm_start=False),
    )
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=len(tasks),
        image_shape=(64, 64),
        max_episode_steps=20,
    )
    world.set_policy(policy)
    try:
        results = world.evaluate(
            manifest=manifest,
            eval_budget=10,
            backend='oracle',
            record=False,
        )
        print(f'oracle success rate: {results.success_rate:.1f}%')
    finally:
        world.close()


if __name__ == '__main__':
    main()
