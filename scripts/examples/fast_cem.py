"""Compare FastCEM budgets on one learned model and fixed task protocol.

Edit ``PLANNERS`` to explore the success/compute trade-off, then run::

    python scripts/examples/fast_cem.py

Pass ``--video`` to save one agent/goal video per task and planner.
"""

import argparse

from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.policy import WorldModelPolicy

from commons import (
    ACTION_PROCESS,
    CHECKPOINT,
    DEVICE,
    EVAL_BUDGET,
    PLAN_CONFIG,
    load_lewm,
    load_protocol,
    make_world,
    pixel_transforms,
    video_dir,
)

VIDEO_DIR = video_dir('fast_cem')

PLANNERS = (
    dict(name='small', num_samples=32, n_steps=2, topk=4),
    dict(name='large', num_samples=128, n_steps=5, topk=16),
)


def evaluate(
    world,
    model,
    eval_protocol,
    *,
    name,
    num_samples,
    n_steps,
    topk,
    device,
    video,
):
    metrics = {'queries': 0, 'task_plans': 0, 'solve_seconds': 0.0}

    def on_plan(event):
        metrics['queries'] += 1
        metrics['task_plans'] += len(event['env_indices'])
        metrics['solve_seconds'] += event['solver_output'][
            'solve_time_seconds'
        ]

    policy = WorldModelPolicy(
        solver=FastCEMSolver(
            cost=ShootingCostEvaluator(model, GoalMSE()),
            batch_size=len(eval_protocol.tasks),
            num_samples=num_samples,
            n_steps=n_steps,
            topk=topk,
            device=device,
        ),
        config=PLAN_CONFIG,
        process=ACTION_PROCESS,
        transform=pixel_transforms(),
        on_plan=on_plan,
    )
    world.set_policy(policy)
    results = world.evaluate(
        protocol=eval_protocol,
        eval_budget=EVAL_BUDGET,
        backend=name,
        video=VIDEO_DIR if video else None,
    )
    successes = sum(ep.success for ep in results.episodes)
    predictor_requests = metrics['queries'] * n_steps * PLAN_CONFIG.horizon
    candidate_rollouts = metrics['task_plans'] * num_samples * n_steps
    mean_solve_ms = (
        1_000 * metrics['solve_seconds'] / max(metrics['queries'], 1)
    )
    print(
        f'{name} ({num_samples}/{topk}/{n_steps}) '
        f'success: {results.success_rate:.1f}% '
        f'({successes}/{len(results.episodes)}), '
        f'plan req: {metrics["queries"]}, '
        f'task plans: {metrics["task_plans"]}, '
        f'pred req: {predictor_requests}, '
        f'rollouts: {candidate_rollouts}, '
        f'mean solve: {mean_solve_ms:.1f} ms'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    protocol = load_protocol()
    print(f'loading {CHECKPOINT} on {DEVICE}')
    model = load_lewm(DEVICE)
    world = make_world(protocol)
    try:
        for planner in PLANNERS:
            evaluate(
                world,
                model,
                protocol,
                device=DEVICE,
                video=args.video,
                **planner,
            )
    finally:
        world.close()

    print(f'protocol: {protocol.digest}')
    print('S/K/I = samples / top-k elites / CEM iterations')


if __name__ == '__main__':
    main()
