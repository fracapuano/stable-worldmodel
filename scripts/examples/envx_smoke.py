"""Reproducible GPU checks using public World/ManyWorlds and frozen witnesses.

Run once per environment in a fresh process. Only timing instrumentation uses
internal methods. No evolving-wm import, simulator implementation or task suite
loader is needed. Checkpoint directories contain weights.pt and config.json.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--environment', choices=('pusht', 'reacher', 'cube'), required=True
    )
    parser.add_argument('--gpu', required=True, help='Physical GPU UUID')
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    locks = Path('/tmp/evolving-wm-gpu-locks')
    locks.mkdir(exist_ok=True)
    lock = (locks / (args.gpu + '.lock')).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader',
        ],
        text=True,
    )
    if any(row.startswith(args.gpu + ',') for row in apps.splitlines()):
        raise RuntimeError('selected GPU already has compute processes')
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MUJOCO_GL='disable',
        OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        'environment': args.environment,
        'gpu_uuid': args.gpu,
        'started_unix': time.time(),
    }

    def save(**values):
        evidence.update(values)
        args.output.write_text(
            json.dumps(evidence, indent=2, allow_nan=False) + '\n'
        )
        print(json.dumps(values, allow_nan=False), flush=True)

    worlds = []
    try:
        import copy

        import jax
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from torchvision.transforms import v2
        from transformers import ViTConfig, ViTModel

        import stable_worldmodel as swm
        from stable_worldmodel.data import ZScoreScaler
        from stable_worldmodel.evaluation import EvaluationProtocol
        from stable_worldmodel.planning import (
            FastCEMSolver,
            GoalMSE,
            ShootingCostEvaluator,
        )
        from stable_worldmodel.wm.lewm import LeWM

        device = 'cuda:0'
        root = Path(__file__).resolve().parents[2]
        protocol = EvaluationProtocol.read(
            root / 'tests/data/envx' / (args.environment + '.json')
        )
        settings = json.loads(
            (root / 'tests/data/envx/planner.json').read_text()
        )
        normalization = settings['action_normalization'][args.environment]
        save(
            status='loading_model',
            protocol_digest=protocol.digest,
            provenance=dict(protocol.metadata),
            planner=settings,
            versions={
                name: importlib.metadata.version(name)
                for name in (
                    'torch',
                    'jax',
                    'jaxlib',
                    'mujoco',
                    'mujoco-mjx',
                    'warp-lang',
                    'transformers',
                )
            },
            jax_devices=[str(d) for d in jax.devices()],
            gpu_name=torch.cuda.get_device_name(),
        )
        config = json.loads((args.checkpoint_dir / 'config.json').read_text())
        # This is the exact stable-pretraining vit_hf(tiny) configuration used
        # by these releases. Strict checkpoint loading checks its schema.
        encoder_config = config['encoder']
        assert encoder_config['size'] == 'tiny'
        encoder = ViTModel(
            ViTConfig(
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
                intermediate_size=768,
                patch_size=encoder_config['patch_size'],
                image_size=encoder_config['image_size'],
            ),
            add_pooling_layer=False,
            use_mask_token=encoder_config['use_mask_token'],
        )
        model = LeWM(
            encoder=encoder,
            **{
                key: instantiate(value)
                for key, value in config.items()
                if key not in ('encoder', '_target_')
            },
        )
        weights = args.checkpoint_dir / 'weights.pt'
        model.load_state_dict(
            torch.load(weights, map_location='cpu', weights_only=True),
            strict=True,
        )
        model = model.to(device).eval().requires_grad_(False)
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                v2.Resize(settings['image_size']),
            ]
        )
        save(
            weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            status='constructing_worlds',
        )
        for member in range(2):
            world = swm.World(
                protocol.environment,
                len(protocol.tasks),
                image_shape=(settings['image_size'],) * 2,
                max_episode_steps=settings['task_steps'],
                execution_backend='envx',
                device=device,
            )
            bound = model if member == 0 else copy.deepcopy(model)
            solver = FastCEMSolver(
                ShootingCostEvaluator(bound, GoalMSE()),
                batch_size=len(protocol.tasks),
                num_samples=settings['cem_num_samples'],
                n_steps=settings['cem_steps'],
                topk=settings['cem_topk'],
                var_scale=settings['cem_var_scale'],
                seed=settings['cem_seed'],
                device=device,
                compile_mode='default',
            )
            world.set_policy(
                swm.policy.WorldModelPolicy(
                    solver,
                    swm.PlanConfig(
                        horizon=settings['horizon'],
                        receding_horizon=settings['receding_horizon'],
                        action_block=settings['action_block'],
                        history_len=settings['history_len'],
                    ),
                    process={
                        'action': ZScoreScaler(
                            mean=[normalization['mean']],
                            std=[normalization['std']],
                        )
                    },
                    transform={'pixels': transform, 'goal': transform},
                )
            )
            worlds.append(world)

        world = worlds[0]
        phases = []

        def timed(obj, method):
            original = getattr(obj, method)

            def call(*positional, **keyword):
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = original(*positional, **keyword)
                torch.cuda.synchronize()
                phases.append(
                    {'phase': method, 'seconds': time.perf_counter() - started}
                )
                return result

            setattr(obj, method, call)

        for obj, method in (
            (world.envs, 'reset'),
            (world.envs, 'rollout'),
            (world.policy.solver, 'solve'),
            (world.policy.solver, 'solve_population'),
        ):
            timed(obj, method)
        save(status='restoring_and_rendering')
        world._reset_from_protocol(protocol, settings['task_steps'])
        for label, which in (('start', 0), ('goal', 1)):
            inputs = world.envs._inputs(which=which)
            engine = world.envs._engine(world.num_envs)
            state = engine.restore(inputs)
            if args.environment == 'pusht':
                np.testing.assert_allclose(
                    np.asarray(engine.adapter.vector(state)),
                    np.asarray(inputs['qpos']),
                    atol=1e-6,
                    rtol=0,
                )
            else:
                np.testing.assert_array_equal(
                    np.asarray(state.qpos), np.asarray(inputs['qpos'])
                )
                np.testing.assert_array_equal(
                    np.asarray(state.qvel), np.asarray(inputs['qvel'])
                )
        save(
            reset_fidelity=True,
            start_goal_images_differ=bool(
                (world.infos['pixels'] != world.infos['goal']).any()
            ),
            reset_cold=phases.copy(),
            status='replaying_witnesses',
        )
        phases.clear()
        plans = torch.zeros(
            1,
            world.num_envs,
            settings['task_steps'],
            world.envs.single_action_space.shape[0],
            device=device,
        )
        lengths = []
        for i, task in enumerate(protocol.tasks):
            reference = torch.tensor(
                dict(task.options)['reference_actions'], device=device
            )
            plans[0, i, : len(reference)] = reference
            lengths.append(len(reference))
        witnesses = []
        for _ in range(3):
            outcome = world.envs.rollout(
                plans, limits=np.asarray([lengths], np.int32)
            )
            witnesses.append(
                {
                    'success': outcome.successes.tolist(),
                    'distance': outcome.final_distances.tolist(),
                    'length': outcome.lengths.tolist(),
                }
            )
        independence = world.envs.rollout(
            torch.cat((plans, torch.zeros_like(plans)), dim=0)
        )
        assert not torch.allclose(
            independence.final_observations[0],
            independence.final_observations[1],
        )
        save(
            witnesses=witnesses,
            witness_physics_timings=phases.copy(),
            independent_plans=True,
            status='world_evaluations',
        )
        phases.clear()
        ordinary = []
        for run in range(3):
            begin = len(phases)
            result = world.evaluate(
                protocol=protocol, eval_budget=settings['task_steps']
            )
            assert result.task_keys == protocol.task_keys
            assert result.metadata['planning_calls'] == 1
            ordinary.append(
                {
                    'temperature': 'cold_planner' if run == 0 else 'warm',
                    'phases': phases[begin:],
                    'result': result.to_dict(),
                }
            )
            save(world_runs=ordinary, status='world_evaluations')
        many = swm.ManyWorlds(worlds, model_names=('release-a', 'release-b'))
        population_runs = []
        save(status='population_evaluations')
        for run in range(3):
            begin = len(phases)
            result = many.evaluate(
                protocol,
                solver=world.policy.solver,
                eval_budget=settings['task_steps'],
            )
            assert result.planning_calls == 1
            assert result.task_final_distances.is_cuda
            assert all(
                value.task_keys == protocol.task_keys
                for value in result.evaluations
            )
            torch.testing.assert_close(
                result.scores, -result.task_final_distances.mean(1)
            )
            population_runs.append(
                {
                    'temperature': 'cold_population' if run == 0 else 'warm',
                    'phases': phases[begin:],
                    'success': result.task_successes.tolist(),
                    'distances': result.task_final_distances.tolist(),
                    'scores': result.scores.tolist(),
                    'compiled': result.compiled,
                    'compile_error': result.compile_error,
                    'population_backend': result.population_backend,
                }
            )
            save(population_runs=population_runs)
        save(
            status='passed',
            finished_unix=time.time(),
            limitation='Two source tasks and population 2; no full-suite or training throughput claim. GPU results use tolerances, not bitwise equality.',
        )
    except BaseException as error:
        save(
            status='failed',
            error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        for world in worlds:
            world.close()
            world.close()


if __name__ == '__main__':
    main()
