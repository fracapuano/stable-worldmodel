"""Plan in TwoRoom with exact simulator dynamics and a pretrained LeWM.

Oracle and LeWM share one ``EvaluationProtocol`` (same ordered tasks,
controller seeds, and CEM loop) so the only swapped piece is the dynamics.
"""

import stable_worldmodel as swm
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.simulator import simulator_goal_encode

from commons import (
    DEVICE,
    EVAL_BUDGET,
    PLAN_CONFIG,
    load_lewm,
    load_protocol,
    make_world,
    pixel_transforms,
    video_dir,
)

VIDEO_DIR = video_dir('multiple_dynamics')


def evaluate(world, cost, *, protocol, transform=None, name, device='cpu'):
    policy = swm.policy.WorldModelPolicy(
        solver=CEMSolver(
            cost=cost,
            batch_size=1,
            num_samples=64,
            n_steps=3,
            topk=8,
            device=device,
        ),
        config=PLAN_CONFIG,
        transform=transform,
    )
    world.set_policy(policy)
    results = world.evaluate(
        protocol=protocol,
        eval_budget=EVAL_BUDGET,
        backend=name,
        video=VIDEO_DIR,
    )
    print(f'{name} success rate: {results.success_rate:.1f}%')
    print(f'{name} videos: {VIDEO_DIR / name}')
    return results


def report(protocol, oracle, lewm):
    print(f'protocol digest: {protocol.digest}')
    print(f'{"task":<10} {"oracle":<8} {"lewm":<8}')
    rows = zip(protocol.tasks, oracle.episodes, lewm.episodes)
    for task, left, right in rows:
        print(f'{task.name:<10} {left.success!s:<8} {right.success!s:<8}')


def main():
    protocol = load_protocol()
    world = make_world(protocol)

    try:
        oracle = swm.SimulatorDynamics(world.envs.envs[0])
        oracle_results = evaluate(
            world,
            ShootingCostEvaluator(
                oracle, GoalMSE(), encode_goal=simulator_goal_encode
            ),
            protocol=protocol,
            name='oracle',
            device='cpu',
        )

        lewm = load_lewm(DEVICE)
        lewm_results = evaluate(
            world,
            ShootingCostEvaluator(lewm, GoalMSE()),
            protocol=protocol,
            transform=pixel_transforms(),
            name='lewm',
            device=DEVICE,
        )
    finally:
        world.close()

    report(protocol, oracle_results, lewm_results)


if __name__ == '__main__':
    main()
