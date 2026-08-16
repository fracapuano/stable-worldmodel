import json

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

import stable_worldmodel as swm
from stable_worldmodel.evaluation import (
    EvaluationProtocol,
    EvaluationTask,
    assert_paired,
)
from stable_worldmodel.planning.solver.callbacks import CandidateTraceRecorder


def make_protocol(*, controller_seed: int = 20) -> EvaluationProtocol:
    return EvaluationProtocol(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(
            EvaluationTask(
                environment_seed=10,
                controller_seed=controller_seed,
                layout_seed=10,
                start=(40.0, 50.0),
                goal=(170.0, 180.0),
                observation_noise_seed=30,
                name='paired-task',
            ),
        ),
    )


def test_protocol_round_trip_and_v1_compatibility(tmp_path):
    protocol = make_protocol()
    path = protocol.write(tmp_path / 'validation.json')

    assert EvaluationProtocol.read(path) == protocol
    assert protocol.write(path) == path

    payload = json.loads(path.read_text())
    assert set(payload) == {
        'digest',
        'environment',
        'metadata',
        'schema_version',
        'split',
        'tasks',
        'version',
    }


def test_protocol_integrity_checks(tmp_path):
    protocol = make_protocol()
    payload = protocol.to_dict()
    payload['tasks'][0]['goal'] = [171.0, 180.0]
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match='task_key'):
        EvaluationProtocol.read(path)

    other = make_protocol(controller_seed=21)
    with pytest.raises(ValueError, match='controller seeds'):
        assert_paired(protocol, other)


class ZeroPolicy:
    def set_env(self, env):
        self.env = env

    def get_action(self, infos):
        return np.zeros(self.env.action_space.shape, dtype=np.float32)


def test_world_evaluates_an_ordered_protocol():
    protocol = make_protocol()
    world = swm.World(
        protocol.environment,
        num_envs=1,
        image_shape=None,
        max_episode_steps=10,
        add_pixels=False,
    )
    world.set_policy(ZeroPolicy())
    try:
        results = world.evaluate(
            protocol=protocol,
            eval_budget=3,
            record=True,
            backend='zero',
        )
    finally:
        world.close()

    assert results.protocol_digest == protocol.digest
    assert results.task_keys == protocol.task_keys
    assert results.controller_seeds == protocol.controller_seeds
    assert results.backend == 'zero'
    assert results.episodes[0].length == 3
    assert len(results.episodes[0].steps) == 3


class QuadraticCost:
    def get_cost(self, info_dict, action_candidates):
        return action_candidates.square().sum(dim=(-1, -2))


def run_seeded_cem(batch_size: int):
    recorder = CandidateTraceRecorder()
    solver = swm.planning.CEMSolver(
        cost=QuadraticCost(),
        batch_size=batch_size,
        num_samples=8,
        n_steps=2,
        topk=2,
        callbacks=[recorder],
    )
    solver.configure(
        action_space=gym_spaces.Box(
            low=-1, high=1, shape=(3, 1), dtype=np.float32
        ),
        n_envs=3,
        config=swm.PlanConfig(horizon=2, receding_horizon=1, warm_start=False),
    )
    output = solver.solve(
        {
            'state': torch.zeros(3, 1),
            'controller_seed': torch.tensor([[11], [22], [33]]),
            'step_idx': torch.tensor([[0], [0], [0]]),
        }
    )
    initial_candidates = torch.cat(
        [batch[0]['candidates'] for batch in recorder.history]
    )
    return output['actions'], initial_candidates


def test_controller_rng_is_independent_of_cem_batching():
    single_actions, single_candidates = run_seeded_cem(batch_size=1)
    batch_actions, batch_candidates = run_seeded_cem(batch_size=3)

    assert torch.equal(single_candidates, batch_candidates)
    assert torch.equal(single_actions, batch_actions)


def test_controller_rng_preserves_frozen_evaluation_stream():
    solver = swm.planning.CEMSolver(
        cost=QuadraticCost(),
        num_samples=8,
        n_steps=2,
        topk=2,
    )
    solver.configure(
        action_space=gym_spaces.Box(
            low=-1, high=1, shape=(1, 1), dtype=np.float32
        ),
        n_envs=1,
        config=swm.PlanConfig(horizon=2, receding_horizon=1, warm_start=False),
    )
    actual = solver._sample_task_candidates(
        torch.tensor([[11]]), torch.tensor([[2]]), 0, 3
    )
    stream_seed = 11 + 1_000_003 * 2 + 10_007 * 3
    generator = torch.Generator().manual_seed(stream_seed)
    expected = torch.randn(8, 2, 1, generator=generator, dtype=solver.dtype)

    assert torch.equal(actual, expected)
