"""Compare FastCEM budgets on one learned model and fixed task protocol.

Edit ``PLANNERS`` to explore the success/compute trade-off, then run::

    python scripts/examples/fast_cem.py

Pass ``--video`` to save one agent/goal video per task and planner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Prefer this checkout over a separately installed stable_worldmodel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import stable_worldmodel as swm
from stable_worldmodel.planning import (
    AcceleratedCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from multiple_dynamics import (
    CHECKPOINT,
    ENV_ID,
    EVAL_BUDGET,
    PLAN_CONFIG,
    freeze_protocol,
    make_protocol,
    pixel_transform,
)

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'fast_cem'


@dataclass(frozen=True)
class FastCEMConfig:
    """Search budget plus the closed-loop MPC configuration it serves."""

    name: str
    num_samples: int
    n_steps: int
    topk: int
    mpc: swm.PlanConfig = PLAN_CONFIG
    batch_size: int = 10
    var_scale: float = 1.0
    seed: int = 1234
    compile_kernel: bool | None = None


# These differ only in search budget. CUDA compiles by default; MPS/CPU use
# the same device-resident tensor loop eagerly.
PLANNERS = (
    FastCEMConfig(name='small', num_samples=32, n_steps=2, topk=4),
    FastCEMConfig(name='large', num_samples=128, n_steps=5, topk=16),
)


@dataclass
class PlannerMetrics:
    """Turn policy planning events into exact model-workload counts."""

    config: FastCEMConfig
    queries: int = 0
    task_plans: int = 0
    solver_batches: int = 0
    solve_seconds: list[float] = field(default_factory=list)
    compiled_queries: int = 0
    compile_error: str | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        tasks = len(event['env_indices'])
        self.queries += 1
        self.task_plans += tasks
        self.solver_batches += math.ceil(tasks / self.config.batch_size)

        output = event['solver_output']
        self.solve_seconds.append(float(output['solve_time_seconds']))
        self.compiled_queries += int(bool(output.get('compiled', False)))
        self.compile_error = output.get('compile_error') or self.compile_error

    def summary(self, results, evaluation_seconds: float) -> dict[str, Any]:
        config = self.config
        rollouts = self.task_plans * config.num_samples * config.n_steps
        steady = self.solve_seconds[1:] or self.solve_seconds
        return {
            'success_rate': results.success_rate,
            'successes': sum(ep.success for ep in results.episodes),
            'tasks': len(results.episodes),
            'planner_queries': self.queries,
            'task_plans': self.task_plans,
            'candidate_rollouts': rollouts,
            'latent_predictions': rollouts * config.mpc.horizon,
            # LatentGoalCost encodes current/goal once per solver batch.
            'encoder_forwards': 2 * self.solver_batches,
            'predictor_forwards': (
                self.solver_batches * config.n_steps * config.mpc.horizon
            ),
            'planner_seconds': sum(self.solve_seconds),
            'first_query_seconds': self.solve_seconds[0],
            'steady_query_seconds': sum(steady) / len(steady),
            'evaluation_seconds': evaluation_seconds,
            'compiled_queries': self.compiled_queries,
            'compile_error': self.compile_error,
        }


def evaluate(world, model, protocol, config, *, device, video):
    """Compose the public model, planner, policy, and evaluation APIs."""
    metrics = PlannerMetrics(config)
    solver = AcceleratedCEMSolver(
        cost=ShootingCostEvaluator(model, GoalMSE()),
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        var_scale=config.var_scale,
        n_steps=config.n_steps,
        topk=config.topk,
        device=device,
        seed=config.seed,
        compile_kernel=config.compile_kernel,
    )
    if not solver.fast_path_enabled:
        raise RuntimeError(solver.fallback_reason)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config.mpc,
        # Training-set stats for CHECKPOINT: CEM searches normalized actions;
        # the policy maps its chosen action back to TwoRoom's [-1, 1] space.
        process={
            'action': swm.data.ZScoreScaler(
                mean=[[0.00309341, -0.05298233]],
                std=[[0.8674758, 0.86776555]],
            )
        },
        transform={
            'pixels': pixel_transform(),
            'goal': pixel_transform(),
        },
        on_plan=metrics,
    )
    world.set_policy(policy)

    synchronize(device)
    started = time.perf_counter()
    results = world.evaluate(
        protocol=protocol,
        eval_budget=EVAL_BUDGET,
        backend=config.name,
        video=OUTPUT_DIR / 'videos' if video else None,
    )
    synchronize(device)
    return metrics.summary(results, time.perf_counter() - started)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def choose_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def report(protocol, summaries) -> None:
    print(f'protocol: {protocol.digest}')
    print(
        f'{"planner":<9} {"S/K/I":>11} {"success":>13} '
        f'{"plan req":>9} {"pred req":>9} {"rollouts":>10} '
        f'{"latent":>10} {"steady ms":>10}'
    )
    for config, result in zip(PLANNERS, summaries):
        budget = f'{config.num_samples}/{config.topk}/{config.n_steps}'
        success = (
            f'{result["success_rate"]:.1f}% '
            f'({result["successes"]}/{result["tasks"]})'
        )
        print(
            f'{config.name:<9} {budget:>11} {success:>13} '
            f'{result["planner_queries"]:>9,d} '
            f'{result["predictor_forwards"]:>9,d} '
            f'{result["candidate_rollouts"]:>10,d} '
            f'{result["latent_predictions"]:>10,d} '
            f'{1_000 * result["steady_query_seconds"]:>10.1f}'
        )
    print('\nS/K/I = samples / top-k elites / CEM iterations')
    print(f'full metrics: {OUTPUT_DIR / "summary.json"}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    device = choose_device(args.device)
    protocol = freeze_protocol(make_protocol())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'loading {CHECKPOINT} on {device}')
    model = swm.wm.utils.load_pretrained(CHECKPOINT).to(device).eval()
    model.requires_grad_(False)
    world = swm.World(
        ENV_ID,
        num_envs=len(protocol.tasks),
        image_shape=(224, 224),
        max_episode_steps=EVAL_BUDGET,
    )
    try:
        summaries = [
            evaluate(
                world,
                model,
                protocol,
                config,
                device=device,
                video=args.video,
            )
            for config in PLANNERS
        ]
    finally:
        world.close()

    output = {
        'checkpoint': CHECKPOINT,
        'device': str(device),
        'protocol_digest': protocol.digest,
        'eval_budget': EVAL_BUDGET,
        'planners': [
            {'config': asdict(config), 'result': result}
            for config, result in zip(PLANNERS, summaries)
        ],
    }
    (OUTPUT_DIR / 'summary.json').write_text(
        json.dumps(output, indent=2) + '\n'
    )
    report(protocol, summaries)


if __name__ == '__main__':
    main()
