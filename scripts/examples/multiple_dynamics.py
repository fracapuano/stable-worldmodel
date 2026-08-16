"""Plan in TwoRoom with exact simulator dynamics and a pretrained LeWM."""

import sys
from pathlib import Path

# Prefer this checkout over a separately installed stable_worldmodel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.simulator import simulator_goal_encode

CHECKPOINT = 'quentinll/lewm-tworooms'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pixel_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=224),
        ]
    )


def evaluate(world, cost, *, config, transform=None, name, device='cpu'):
    policy = swm.policy.WorldModelPolicy(
        solver=CEMSolver(
            cost=cost,
            batch_size=1,
            num_samples=64,
            n_steps=3,
            topk=8,
            device=device,
        ),
        config=config,
        transform=transform,
    )
    world.set_policy(policy)
    results = world.evaluate(episodes=10, seed=0)
    print(f'{name} success rate: {results["success_rate"]:.1f}%')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=50,
    )

    try:
        oracle = swm.SimulatorDynamics(world.envs.envs[0])
        evaluate(
            world,
            ShootingCostEvaluator(
                oracle, GoalMSE(), encode_goal=simulator_goal_encode
            ),
            config=swm.PlanConfig(
                horizon=5, receding_horizon=1, warm_start=False
            ),
            name='oracle',
            device='cpu',
        )

        lewm = swm.wm.utils.load_pretrained(CHECKPOINT).to(device).eval()
        lewm.requires_grad_(False)
        evaluate(
            world,
            ShootingCostEvaluator(lewm, GoalMSE()),
            config=swm.PlanConfig(
                horizon=5,
                receding_horizon=1,
                history_len=3,
                warm_start=False,
            ),
            transform={
                'pixels': pixel_transform(),
                'goal': pixel_transform(),
            },
            name='lewm',
            device=device,
        )
    finally:
        world.close()


if __name__ == '__main__':
    main()
