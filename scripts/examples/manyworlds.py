"""Benchmark sequential and population-batched FastCEM on ten LeWMs.

The sequential condition recreates, evaluates, and closes one weight-bound
``World`` at a time. The ManyWorlds condition creates the same number of
weight-bound ``World`` instances and evaluates all of them through the model
population dimension of their existing ``FastCEMSolver``.

Run the default benchmark with::

    python scripts/examples/manyworlds.py

The pretrained checkpoint and PushT simulator require the ``train`` and
``env`` extras, respectively (for example, ``pip install -e '.[train,env]'``).
The checkpoint is downloaded on first use. Use smaller CEM settings for a
quick smoke run, for example::

    python scripts/examples/manyworlds.py \
        --worlds 2 --eval-budget 5 --num-samples 8 --n-steps 2 --topk 2

Setup and evaluation time are reported separately. Checkpoint download and one
untimed CPU load are excluded, while every timed condition still instantiates
its own model copies.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationTask
from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

CHECKPOINT = 'quentinll/lewm-pusht'
ENV_ID = 'swm/PushT-v1'
IMAGE_SHAPE = (224, 224)

# The checkpoint was trained on five-step action blocks.
PLAN_CONFIG = PlanConfig(
    horizon=5,
    receding_horizon=1,
    history_len=3,
    action_block=5,
    warm_start=True,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Statistics of the PushT action column used to train the checkpoint.
ACTION_MEAN = [[-0.00781276, 0.00686056]]
ACTION_STD = [[0.20824118, 0.20649128]]

# One deterministic PushT task is repeated across model copies. PushT reads
# ``goal_state`` from reset options; ``goal`` also remains in the immutable
# protocol so the compared conditions have exactly the same task identity.
START_STATE = (256.0, 400.0, 300.0, 200.0, 0.0, 0.0, 0.0)
GOAL_STATE = (256.0, 300.0, 256.0, 256.0, 0.785398, 0.0, 0.0)


@dataclass(frozen=True)
class Measurement:
    setup_seconds: float
    evaluation_seconds: float
    planning_calls: int
    success_rates: tuple[float, ...]

    @property
    def total_seconds(self) -> float:
        return self.setup_seconds + self.evaluation_seconds


def default_device() -> str:
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    return str(accelerator or 'cpu')


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'xpu':
        torch.xpu.synchronize(device)


def release_accelerator_memory(device: str | torch.device) -> None:
    gc.collect()
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'xpu':
        torch.xpu.empty_cache()


def make_protocol(checkpoint: str = CHECKPOINT) -> EvaluationProtocol:
    task = EvaluationTask(
        environment_seed=7,
        controller_seed=19,
        layout_seed=7,
        start=START_STATE,
        goal=GOAL_STATE,
        observation_noise_seed=23,
        options=(('goal_state', GOAL_STATE),),
        name='pusht-benchmark',
    )
    return EvaluationProtocol(
        split='benchmark',
        environment=ENV_ID,
        tasks=(task,),
        metadata=(('checkpoint', checkpoint),),
    )


def pixel_transforms() -> dict[str, object]:
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=IMAGE_SHAPE[0]),
        ]
    )
    return {'pixels': transform, 'goal': transform}


def action_process() -> dict[str, object]:
    return {
        'action': swm.data.ZScoreScaler(
            mean=ACTION_MEAN,
            std=ACTION_STD,
        )
    }


def load_model(checkpoint: str, device: str):
    model = swm.wm.utils.load_pretrained(checkpoint).to(device).eval()
    model.requires_grad_(False)
    return model


def make_world(
    checkpoint: str,
    protocol: EvaluationProtocol,
    args: argparse.Namespace,
):
    model = load_model(checkpoint, args.device)
    calls = {'count': 0}

    def on_plan(_event) -> None:
        calls['count'] += 1

    solver = FastCEMSolver(
        cost=ShootingCostEvaluator(model, GoalMSE()),
        batch_size=len(protocol.tasks),
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        topk=args.topk,
        device=args.device,
        compile_kernel=args.compile_kernel,
    )
    policy = WorldModelPolicy(
        solver=solver,
        config=PLAN_CONFIG,
        process=action_process(),
        transform=pixel_transforms(),
        on_plan=on_plan,
    )
    world = swm.World(
        ENV_ID,
        num_envs=len(protocol.tasks),
        image_shape=IMAGE_SHAPE,
        max_episode_steps=args.eval_budget,
    )
    world.set_policy(policy)
    return world, calls


def benchmark_sequential(
    protocol: EvaluationProtocol,
    args: argparse.Namespace,
) -> Measurement:
    setup_seconds = 0.0
    evaluation_seconds = 0.0
    planning_calls = 0
    success_rates = []

    for index in range(args.worlds):
        world = None
        try:
            synchronize(args.device)
            started = time.perf_counter()
            world, calls = make_world(args.checkpoint, protocol, args)
            synchronize(args.device)
            setup_seconds += time.perf_counter() - started

            synchronize(args.device)
            started = time.perf_counter()
            result = world.evaluate(
                protocol=protocol,
                eval_budget=args.eval_budget,
                backend=f'sequential-{index}',
            )
            synchronize(args.device)
            evaluation_seconds += time.perf_counter() - started

            planning_calls += calls['count']
            success_rates.append(result.success_rate)
        finally:
            if world is not None:
                world.close()
            del world
            release_accelerator_memory(args.device)

    return Measurement(
        setup_seconds=setup_seconds,
        evaluation_seconds=evaluation_seconds,
        planning_calls=planning_calls,
        success_rates=tuple(success_rates),
    )


def benchmark_many_worlds(
    protocol: EvaluationProtocol,
    args: argparse.Namespace,
) -> Measurement:
    worlds = []
    many = None
    try:
        synchronize(args.device)
        started = time.perf_counter()
        for _ in range(args.worlds):
            world, _calls = make_world(args.checkpoint, protocol, args)
            worlds.append(world)
        many = swm.ManyWorlds.init(
            worlds=worlds,
            model_names=[f'model-{index}' for index in range(args.worlds)],
        )
        synchronize(args.device)
        setup_seconds = time.perf_counter() - started

        # This is the same FastCEMSolver already bound to the first World.
        solver = worlds[0].policy.solver
        synchronize(args.device)
        started = time.perf_counter()
        result = many.evaluate(
            protocol,
            solver=solver,
            eval_budget=args.eval_budget,
        )
        synchronize(args.device)
        evaluation_seconds = time.perf_counter() - started

        return Measurement(
            setup_seconds=setup_seconds,
            evaluation_seconds=evaluation_seconds,
            planning_calls=result.planning_calls,
            success_rates=tuple(
                evaluation.success_rate for evaluation in result.evaluations
            ),
        )
    finally:
        if many is not None:
            many.close()
        else:
            for world in worlds:
                world.close()
        release_accelerator_memory(args.device)


def print_report(
    sequential: Measurement,
    batched: Measurement,
    args: argparse.Namespace,
) -> None:
    print()
    print(
        f'{"condition":<18} {"setup (s)":>12} {"eval (s)":>12} '
        f'{"total (s)":>12} {"plan calls":>12}'
    )
    print('-' * 70)
    for name, measurement in (
        ('sequential', sequential),
        ('many-worlds', batched),
    ):
        print(
            f'{name:<18} {measurement.setup_seconds:>12.3f} '
            f'{measurement.evaluation_seconds:>12.3f} '
            f'{measurement.total_seconds:>12.3f} '
            f'{measurement.planning_calls:>12}'
        )

    evaluation_speedup = (
        sequential.evaluation_seconds / batched.evaluation_seconds
    )
    total_speedup = sequential.total_seconds / batched.total_seconds
    print()
    print(f'evaluation speedup: {evaluation_speedup:.2f}x')
    print(f'end-to-end speedup: {total_speedup:.2f}x')
    print(
        'mean success rate: '
        f'{sum(sequential.success_rates) / args.worlds:.1f}% sequential, '
        f'{sum(batched.success_rates) / args.worlds:.1f}% many-worlds'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--worlds', type=int, default=10)
    parser.add_argument('--eval-budget', type=int, default=10)
    parser.add_argument('--num-samples', type=int, default=64)
    parser.add_argument('--n-steps', type=int, default=3)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--device', default=default_device())
    parser.add_argument(
        '--compile',
        action=argparse.BooleanOptionalAction,
        default=None,
        dest='compile_kernel',
        help='compile FastCEM; default is automatic on CUDA',
    )
    args = parser.parse_args()
    if args.worlds < 1 or args.eval_budget < 1:
        parser.error('--worlds and --eval-budget must be positive')
    if args.n_steps < 1 or not 2 <= args.topk <= args.num_samples:
        parser.error('require n-steps >= 1 and 2 <= topk <= num-samples')
    return args


def main() -> None:
    args = parse_args()
    protocol = make_protocol(args.checkpoint)

    print(f'checkpoint: {args.checkpoint}')
    print(f'device: {args.device}')
    print(f'world models: {args.worlds}')
    print(
        'CEM: '
        f'{args.num_samples} samples, {args.topk} elites, '
        f'{args.n_steps} iterations'
    )
    print(f'evaluation budget: {args.eval_budget} environment steps')
    print(f'protocol: {protocol.digest}')

    # Exclude network download and first-time Python/model import work from
    # both conditions. Timed setup still loads a fresh model per World.
    print('priming checkpoint cache...')
    probe = swm.wm.utils.load_pretrained(args.checkpoint)
    del probe
    gc.collect()

    print('running traditional sequential evaluation...')
    sequential = benchmark_sequential(protocol, args)
    print('running ManyWorlds population evaluation...')
    batched = benchmark_many_worlds(protocol, args)
    print_report(sequential, batched, args)


if __name__ == '__main__':
    main()
