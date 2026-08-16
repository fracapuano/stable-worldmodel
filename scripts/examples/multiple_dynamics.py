"""Plan in TwoRoom with exact simulator dynamics and a pretrained LeWM.

Oracle and LeWM share one ``EvaluationProtocol`` (same ordered tasks,
controller seeds, and CEM loop) so the only swapped piece is the dynamics.
"""

import sys
import tempfile
from pathlib import Path

# Prefer this checkout over a separately installed stable_worldmodel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.evaluation import (
    EvaluationProtocol,
    EvaluationTask,
    assert_paired,
)
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.simulator import simulator_goal_encode

VIDEO_DIR = Path(__file__).resolve().parent / 'videos' / 'multiple_dynamics'
CHECKPOINT = 'quentinll/lewm-tworooms'
ENV_ID = 'swm/TwoRoom-v1'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EVAL_BUDGET = 50
# Same CEM loop for both dynamics: 5 blocked actions, execute 1 block, replan.
PLAN_CONFIG = swm.PlanConfig(
    horizon=5,
    receding_horizon=1,
    history_len=3,
    action_block=5,
)
# Shared layout (environment_seed == layout_seed) so one SimulatorDynamics
# is valid for every parallel task. Starts are in the left room, goals right.
TASK_STARTS_GOALS = (
    ((40.0, 50.0), (170.0, 180.0)),
    ((50.0, 80.0), (180.0, 40.0)),
    ((45.0, 120.0), (165.0, 60.0)),
    ((55.0, 160.0), (175.0, 140.0)),
    ((35.0, 70.0), (185.0, 110.0)),
    ((60.0, 40.0), (160.0, 170.0)),
    ((42.0, 100.0), (178.0, 90.0)),
    ((48.0, 140.0), (172.0, 50.0)),
    ((38.0, 175.0), (168.0, 155.0)),
    ((52.0, 55.0), (182.0, 125.0)),
)


def make_protocol() -> EvaluationProtocol:
    layout_seed = 0
    tasks = tuple(
        EvaluationTask(
            environment_seed=layout_seed,
            controller_seed=1_000 + i,
            layout_seed=layout_seed,
            start=start,
            goal=goal,
            observation_noise_seed=2_000 + i,
            name=f'task-{i}',
        )
        for i, (start, goal) in enumerate(TASK_STARTS_GOALS)
    )
    return EvaluationProtocol(
        split='validation',
        environment=ENV_ID,
        tasks=tasks,
        metadata=(('example', 'multiple_dynamics'),),
    )


def freeze_protocol(protocol: EvaluationProtocol) -> EvaluationProtocol:
    """Round-trip through the immutable on-disk format."""
    with tempfile.TemporaryDirectory() as tmp:
        path = protocol.write(Path(tmp) / 'tworoom_oracle_gap.json')
        loaded = EvaluationProtocol.read(path)
    assert_paired(protocol, loaded)
    if loaded.digest != protocol.digest:
        raise ValueError('reloaded protocol digest does not match')
    return loaded


def pixel_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=224),
        ]
    )


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol = freeze_protocol(make_protocol())
    world = swm.World(
        ENV_ID,
        num_envs=len(protocol.tasks),
        image_shape=(224, 224),
        max_episode_steps=EVAL_BUDGET,
    )

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

        lewm = swm.wm.utils.load_pretrained(CHECKPOINT).to(device).eval()
        lewm.requires_grad_(False)
        lewm_results = evaluate(
            world,
            ShootingCostEvaluator(lewm, GoalMSE()),
            protocol=protocol,
            transform={
                'pixels': pixel_transform(),
                'goal': pixel_transform(),
            },
            name='lewm',
            device=device,
        )
    finally:
        world.close()

    report(protocol, oracle_results, lewm_results)


if __name__ == '__main__':
    main()
