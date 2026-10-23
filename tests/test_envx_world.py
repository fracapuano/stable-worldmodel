"""Frozen-task contracts and the public World/ManyWorlds envX boundary."""

from __future__ import annotations

import os
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationTask
from stable_worldmodel.planning import (
    FastCEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.world.envx import ENVIRONMENTS, _Engine, task_inputs

HAS_ENVX = find_spec('envx') is not None
DEVICE = os.environ.get('SWM_ENVX_TEST_DEVICE', 'cpu')
integration = pytest.mark.skipif(
    not HAS_ENVX, reason='optional envX dependency'
)


def protocol(name='pusht', count=2):
    if name == 'cube':
        return EvaluationProtocol.read(
            Path(__file__).parent / 'data/envx_tasks/cube.json'
        )
    tasks = []
    for i in range(count):
        if name == 'pusht':
            start = (100.0 + i * 20, 100, 250, 250, 0, 0, 0)
            goal = (160.0 + i * 20, 100, 250, 250, 0, 0, 0)
            options = (('goal_state', goal), ('variation', []))
        else:
            start, goal = (0.2 + i * 0.1, -0.3), (0.6, -0.7)
            options = (
                (
                    'reacher_task',
                    {
                        'start_qpos': start,
                        'start_qvel': [0.3, -0.2],
                        'goal_qpos': goal,
                        'goal_qvel': [-0.1, 0.1],
                    },
                ),
            )
        tasks.append(
            EvaluationTask(i, i + 11, i, start, goal, i, options=options)
        )
    environment = next(k for k, v in ENVIRONMENTS.items() if v == name)
    return EvaluationProtocol('test', environment, tuple(tasks))


def world_for(protocol, *, pixels=False, cap=7, **kwargs):
    return swm.World(
        protocol.environment,
        len(protocol.tasks),
        image_shape=(16, 16) if pixels else None,
        add_pixels=pixels,
        device=DEVICE,
        max_episode_steps=cap,
        **kwargs,
    )


def test_contract_uses_existing_full_reacher_state_without_mutation():
    task = protocol('reacher').tasks[0]
    options = swm.World._task_reset_options(task)
    original = deepcopy(options)
    start, goal, contract = task_inputs('reacher', options)
    np.testing.assert_array_equal(
        start['qvel'], np.asarray([0.3, -0.2], np.float32)
    )
    np.testing.assert_array_equal(
        goal['qpos'], np.asarray(task.goal, np.float32)
    )
    assert contract['repeat'] == 2
    assert options['reacher_task'] == original['reacher_task']
    options['reacher_task']['start_qpos'] = [1, 2]
    with pytest.raises(ValueError, match='disagrees'):
        task_inputs('reacher', options)


def test_missing_full_state_is_an_error_instead_of_random_reset():
    with pytest.raises(ValueError, match='qvel'):
        task_inputs('reacher', {'state': [0, 0], 'target_state': [1, 1]})
    with pytest.raises(ValueError, match='qpos'):
        task_inputs('cube', {'state': [0, 0, 0], 'target_state': [1, 1, 1]})


@integration
def test_contract_metrics_and_wrapped_angle():
    import jax.numpy as jnp

    for name, states, goals, expected in (
        (
            'cube',
            [[0, 0, 0], [0.041, 0, 0]],
            [[0.04, 0, 0], [0, 0, 0]],
            [True, False],
        ),
        ('reacher', [[0, 0], [0.051, 0]], [[0.049, 0], [0, 0]], [True, False]),
        (
            'pusht',
            [[0, 0, 0, 0, 0.01, 10, 0], [20, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 2 * np.pi - 0.01, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
            [True, False],
        ),
    ):
        engine = _Engine.__new__(_Engine)
        engine.name = name
        engine.adapter = SimpleNamespace(vector=lambda value: value)
        _, distances, success = engine._metrics(
            jnp.asarray(states),
            jnp.asarray(goals),
            {
                'threshold': 0.04 if name == 'cube' else 0.05,
                'position_threshold': 20,
                'angle_threshold': np.pi / 9,
            },
        )
        np.testing.assert_array_equal(success, expected)
        if name == 'pusht':
            # Fitness keeps the declared raw state7 L2 distance, including
            # velocity and raw angle, even when angular success wraps.
            np.testing.assert_allclose(
                distances[0],
                np.linalg.norm(np.asarray(states[0]) - goals[0]),
                rtol=1e-6,
            )


@integration
def test_first_success_inside_repeated_control_freezes_outcome():
    import jax.numpy as jnp

    engine = _Engine.__new__(_Engine)
    engine.name = 'reacher'
    engine.adapter = SimpleNamespace(
        restore=lambda inputs: jnp.zeros((1, 2)),
        vector=lambda state: state,
        advance=lambda state, action, contract: state + action,
    )
    result, simulator_state = engine._run(
        {'goal': jnp.asarray([[0.1, 0.0]])},
        {'repeat': jnp.asarray([2]), 'threshold': jnp.asarray([0.05])},
        jnp.asarray([[[0.1, 0], [1, 0], [-1, 0]]]),
        jnp.asarray([3]),
    )
    states, success, distances, _, lengths, paths, controls, _ = result
    np.testing.assert_allclose(states, [[0.1, 0]])
    np.testing.assert_allclose(simulator_state, states)
    np.testing.assert_array_equal(success, [True])
    np.testing.assert_array_equal(lengths, [1])
    np.testing.assert_array_equal(distances, [0])
    np.testing.assert_allclose(paths, [0.1])
    np.testing.assert_allclose(controls, [0.01])


@integration
def test_completed_task_retains_terminal_infos_while_other_task_runs():
    p = protocol()
    world = world_for(p, pixels=True, cap=5)
    try:
        world._reset_from_protocol(p, 5)
        active = np.ones(2, bool)
        terminal = None
        for _ in range(5):
            _, rewards, done, truncated, infos = world.envs.step(
                torch.tensor([[0.5, 0], [0, 0]], device=DEVICE), mask=active
            )
            if not active[0]:
                assert rewards[0] == 0
                assert not done[0] and not truncated[0]
            if done[0] and terminal is None:
                terminal = {
                    key: value[0].clone()
                    if torch.is_tensor(value)
                    else value[0].copy()
                    for key, value in infos.items()
                }
            active &= ~(done | truncated)
        assert terminal is not None
        assert infos['terminated'][0]
        assert infos['step_idx'][0, 0] < infos['step_idx'][1, 0] == 5
        for key, value in terminal.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(infos[key][0], value)
            else:
                np.testing.assert_array_equal(infos[key][0], value)
    finally:
        world.close()


@integration
def test_reacher_success_on_first_repeat_retains_physics_and_pixels():
    if DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    p = protocol('reacher', count=1)
    task = p.tasks[0]
    payload = dict(task.options)['reacher_task']
    payload.update(start_qpos=[0, 0], start_qvel=[1, 1])
    task = replace(task, start=(0, 0))
    p = replace(p, tasks=(task,))
    world = world_for(p, pixels=True)
    try:
        world._reset_from_protocol(p, 2)
        pool = world.envs
        engine = pool._engine(1)
        first = jax.jit(engine.adapter.advance)(
            pool._state,
            jax.device_put(jnp.zeros((1, 2)), pool.jax_device),
            pool._contract,
        )
        goal = tuple(np.asarray(first.qpos)[0].tolist())
        payload.update(
            goal_qpos=goal, goal_qvel=np.asarray(first.qvel)[0].tolist()
        )
        task = replace(
            task,
            goal=goal,
            options=(
                ('reacher_task', payload),
                ('task_contract', {'threshold': 0.001}),
            ),
        )
        world._reset_from_protocol(replace(p, tasks=(task,)), 2)
        _, _, done, _, infos = pool.step(torch.zeros((1, 2), device=DEVICE))
        assert done.all()
        np.testing.assert_allclose(np.asarray(pool._state.qpos), [goal])
        np.testing.assert_allclose(
            np.asarray(pool._state.qvel), np.asarray(first.qvel), rtol=1e-5
        )
        torch.testing.assert_close(infos['pixels'], infos['goal'])
        assert infos['distance_to_target'].item() < 0.001
    finally:
        world.close()


@integration
def test_implicit_cuda_uses_current_torch_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    import jax

    jax.local_devices(backend='gpu')
    previous = torch.cuda.current_device()
    world = None
    try:
        torch.cuda.set_device(1)
        p = protocol(count=1)
        world = swm.World(p.environment, 1, add_pixels=False, device='cuda')
        world._reset_from_protocol(p, 2)
        assert world.envs.device == torch.device('cuda:1')
        assert world.envs.jax_device == jax.local_devices(backend='gpu')[1]
        plans = torch.zeros((1, 1, 2, 2), device='cuda:1')
        outcome = world.envs.rollout(plans)
        assert outcome.final_observations.device == plans.device
        assert outcome.final_distances.device == plans.device
    finally:
        if world is not None:
            world.close()
        torch.cuda.set_device(previous)


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_frozen_reference_plans_succeed(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX integration runs on CUDA')
    p = EvaluationProtocol.read(
        Path(__file__).parent / 'data/envx_tasks' / (name + '.json')
    )
    references = [dict(task.options)['reference_actions'] for task in p.tasks]
    lengths = [len(actions) for actions in references]
    budget = max(lengths)
    world = world_for(p, cap=budget)
    try:
        world._reset_from_protocol(p, budget)
        plans = torch.zeros(
            1,
            len(p.tasks),
            budget,
            world.envs.single_action_space.shape[0],
            device=DEVICE,
        )
        for i, actions in enumerate(references):
            plans[0, i, : len(actions)] = torch.tensor(actions, device=DEVICE)
        for _ in range(3):
            outcome = world.envs.rollout(
                plans, limits=np.asarray([lengths], np.int32)
            )
            assert outcome.successes.all().item()
    finally:
        world.close()


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_reset_fidelity_population_order_independence_and_budget(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX integration runs on CUDA')
    p = protocol(name)
    world = world_for(p)
    try:
        world._reset_from_protocol(p, 5)
        pool = world.envs
        state = pool._state
        if name == 'cube':
            np.testing.assert_array_equal(
                np.asarray(state.qpos),
                np.asarray(
                    [dict(t.options)['cube_qpos'] for t in p.tasks], np.float32
                ),
            )
            np.testing.assert_array_equal(
                np.asarray(state.qvel),
                np.asarray(
                    [dict(t.options)['cube_qvel'] for t in p.tasks], np.float32
                ),
            )
        elif name == 'reacher':
            np.testing.assert_array_equal(
                np.asarray(state.qpos),
                np.asarray([t.start for t in p.tasks], np.float32),
            )
            np.testing.assert_array_equal(
                np.asarray(state.qvel),
                np.asarray([[0.3, -0.2]] * 2, np.float32),
            )
        else:
            torch.testing.assert_close(
                world.infos['state'][:, 0].cpu(),
                torch.tensor([t.start for t in p.tasks]),
            )
        plans = torch.zeros(
            3, 2, 5, world.envs.single_action_space.shape[0], device=DEVICE
        )
        plans[0, :, :, 0] = 0.3
        plans[1, :, :, 0] = -0.3
        plans[2, :, :, 1] = 0.2
        combined = pool.rollout(plans)
        assert not torch.allclose(
            combined.final_observations[0], combined.final_observations[1]
        )
        for i in range(3):
            single = pool.rollout(plans[i : i + 1])
            torch.testing.assert_close(
                combined.final_observations[i],
                single.final_observations[0],
                rtol=2e-4,
                atol=2e-4,
            )
        from dataclasses import replace

        reversed_protocol = replace(p, tasks=tuple(reversed(p.tasks)))
        world._reset_from_protocol(reversed_protocol, 5)
        reordered = pool.rollout(plans.flip(1))
        torch.testing.assert_close(
            combined.final_distances,
            reordered.final_distances.flip(1),
            rtol=2e-4,
            atol=2e-4,
        )
        world._reset_from_protocol(p, 5)
        limited = pool.rollout(plans, max_steps=2)
        assert torch.all(limited.lengths <= 2)
        assert (
            combined.final_distances.device.type == torch.device(DEVICE).type
        )
        assert world.infos['task_key'] == [[k] for k in p.task_keys]
    finally:
        world.close()
        world.close()
    assert world.infos == {}
    assert world.envs._state is None
    assert not world.envs._engines
    with pytest.raises(RuntimeError, match='closed'):
        world._reset_from_protocol(p, 5)


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_goal_pixels_are_rendered_from_restored_goal_state(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    p = protocol(name)
    world = world_for(p, pixels=True)
    try:
        world._reset_from_protocol(p, 3)
        pixels, goal = world.infos['pixels'], world.infos['goal']
        assert pixels.shape == goal.shape == (2, 1, 16, 16, 3)
        assert not torch.equal(pixels, goal)
        pool = world.envs
        renderer = pool._engine(2, render=True)
        expected = torch.from_dlpack(
            renderer.render(renderer.restore(pool._inputs(which=1)))
        )
        torch.testing.assert_close(
            goal[:, 0].float(),
            expected.float(),
            rtol=0,
            atol=0 if name == 'pusht' else 2,
        )
    finally:
        world.close()


@integration
def test_full_population_allocation_failure_propagates_without_retry(
    monkeypatch,
):
    p = protocol()
    world = world_for(p)
    try:
        world._reset_from_protocol(p, 2)
        counts = []
        failure = RuntimeError('simulator device allocation failed')

        def allocate(count):
            counts.append(count)
            raise failure

        monkeypatch.setattr(world.envs, '_engine', allocate)
        plans = torch.zeros(17, 2, 2, 2, device=DEVICE)
        with pytest.raises(RuntimeError) as caught:
            world.envs.rollout(plans)
        assert caught.value is failure
        assert counts == [34]
    finally:
        world.close()


class TinyModel(nn.Module):
    def __init__(self, direction=1.0):
        super().__init__()
        self.direction = nn.Parameter(
            torch.tensor(direction), requires_grad=False
        )

    def encode(self, value):
        key = 'pixels' if 'pixels' in value else 'state'
        pixels = value[key].float()
        return {'emb': pixels.flatten(2).mean(-1, keepdim=True)}

    def rollout_from_embeddings(
        self,
        emb,
        action_sequence,
        action_history=None,
        history_size=None,
        *,
        terminal_only=False,
    ):
        delta = action_sequence.sum(-1, keepdim=True)
        future = emb[:, None, -1:] + self.direction * delta.cumsum(2)
        if terminal_only:
            return future[:, :, -1]
        return torch.cat(
            (emb[:, None].expand(-1, action_sequence.size(1), -1, -1), future),
            dim=2,
        )


def attach_policy(
    world, *, horizon=4, receding=4, action_block=1, direction=1.0
):
    model = TinyModel(direction).to(DEVICE)
    solver = FastCEMSolver(
        ShootingCostEvaluator(model, GoalMSE()),
        batch_size=world.num_envs,
        num_samples=4,
        n_steps=1,
        topk=2,
        device=DEVICE,
        compile_kernel=False,
    )
    world.set_policy(
        swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=horizon,
                receding_horizon=receding,
                action_block=action_block,
            ),
        )
    )
    return solver


@integration
def test_classic_cem_uses_world_step_loop(monkeypatch):
    from stable_worldmodel.planning import CEMSolver

    class Cost:
        def get_cost(self, info, actions):
            return actions.square().sum((-1, -2))

    p = protocol()
    world = world_for(p)
    solver = CEMSolver(Cost(), num_samples=4, n_steps=1, topk=2, device=DEVICE)
    world.set_policy(
        swm.policy.WorldModelPolicy(
            solver, swm.PlanConfig(horizon=2, receding_horizon=2)
        )
    )

    def unexpected(*args, **kwargs):
        raise AssertionError('complete-plan execution requires FastCEM')

    monkeypatch.setattr(world.policy, '_get_action_plan', unexpected)
    try:
        result = world.evaluate(protocol=p, eval_budget=2)
        assert result.task_keys == p.task_keys
        assert all(row.length == 2 for row in result.episodes)
    finally:
        world.close()


@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_policy_attachment_does_not_construct_either_simulator(
    name, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise AssertionError('policy attachment must only need control spaces')

    monkeypatch.setattr('stable_worldmodel.world.world._make_env', unexpected)
    monkeypatch.setattr('stable_worldmodel.world.envx.EnvXPool', unexpected)
    world = world_for(protocol(name))
    try:
        attach_policy(world)
        assert world.num_envs == 2
        assert world.policy.env.single_action_space.shape == (
            5 if name == 'cube' else 2,
        )
    finally:
        world.close()


class DefaultTaskEnv(gym.Env):
    """Distinguish the ordinary environment task from frozen task execution."""

    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    observation_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def __init__(self):
        self.options = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.options.append(options)
        return np.zeros(2, np.float32), {'terminated': False}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, True, {'terminated': False}

    def close(self):
        self.closed = True


def test_ordinary_evaluation_keeps_environment_defaults(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('ordinary evaluation must not require envX')

    envs = [DefaultTaskEnv(), DefaultTaskEnv()]
    pending = iter(envs)
    monkeypatch.setattr(
        'stable_worldmodel.world.world._make_env',
        lambda *a, **kw: next(pending),
    )
    monkeypatch.setattr('stable_worldmodel.world.envx.EnvXPool', unexpected)
    world = world_for(protocol())
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    try:
        result = world.evaluate(
            episodes=2, options={'ordinary_task': 'original'}
        )
        assert result['success_rate'] == 0
        assert all(
            row['ordinary_task'] == 'original'
            for e in envs
            for row in e.options
        )
    finally:
        world.close()
    assert all(e.closed for e in envs)


@integration
def test_world_returns_to_ordinary_task_after_protocol(monkeypatch):
    envs = [DefaultTaskEnv(), DefaultTaskEnv()]
    pending = iter(envs)
    monkeypatch.setattr(
        'stable_worldmodel.world.world._make_env',
        lambda *a, **kw: next(pending),
    )
    world = world_for(protocol())
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    try:
        frozen = world.evaluate(protocol=protocol(), eval_budget=2)
        assert frozen.task_keys == protocol().task_keys
        assert all(not e.options for e in envs)
        ordinary = world.evaluate(
            episodes=2, options={'ordinary_task': 'original'}
        )
        assert ordinary['success_rate'] == 0
        assert world.policy.env is world.envs
    finally:
        world.close()
        world.close()
    assert all(e.closed for e in envs)


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_blocked_plan_executes_25_controls_in_one_planning_call(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    p = protocol(name)
    worlds = [world_for(p, pixels=True, cap=25) for _ in range(2)]
    solvers = [
        attach_policy(w, horizon=5, receding=5, action_block=5) for w in worlds
    ]
    try:
        result = worlds[0].evaluate(protocol=p, eval_budget=25)
        assert result.metadata['planning_calls'] == 1
        assert all(row.length <= 25 for row in result.episodes)
        many = swm.ManyWorlds(worlds)
        result = many.evaluate(p, solver=solvers[0], eval_budget=25)
        assert result.planning_calls == 1
        assert result.environment_actions.shape == (
            2,
            2,
            25,
            5 if name == 'cube' else 2,
        )
        assert all(
            row.length <= 25
            for ev in result.evaluations
            for row in ev.episodes
        )
    finally:
        for world in worlds:
            world.close()


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_public_world_and_manyworlds_evaluation(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    p = protocol(name)
    worlds = [world_for(p, pixels=True), world_for(p, pixels=True)]
    solvers = [
        attach_policy(w, direction=float(1 - i * 2))
        for i, w in enumerate(worlds)
    ]
    many = swm.ManyWorlds(worlds)
    try:
        events = []
        worlds[0].policy.on_plan = events.append
        result = worlds[0].evaluate(protocol=p, eval_budget=4)
        assert result.task_keys == p.task_keys
        assert result.metadata['planning_calls'] == len(events) == 1
        assert all(1 <= row.length <= 4 for row in result.episodes)
        population = many.evaluate(p, solver=solvers[0], eval_budget=4)
        assert population.task_final_distances.shape == (2, 2)
        torch.testing.assert_close(
            population.scores, -population.task_final_distances.mean(1)
        )
        assert all(
            value.task_keys == p.task_keys for value in population.evaluations
        )
    finally:
        many.close()


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_world_mpc_and_records_support_other_budgets(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    p = protocol(name)
    world = world_for(p, pixels=True)
    attach_policy(world, horizon=2, receding=1)
    try:
        result = world.evaluate(protocol=p, eval_budget=3, record=True)
        assert len(result.model_queries) >= 1
        assert all(len(row.steps) == row.length for row in result.episodes)
        assert all(row.length <= 3 for row in result.episodes)
    finally:
        world.close()


@integration
@pytest.mark.parametrize('name', ['pusht', 'reacher', 'cube'])
def test_population_model_and_success_scoring(name):
    if name != 'pusht' and DEVICE == 'cpu':
        pytest.skip('MJX rendering requires CUDA')
    from dataclasses import replace

    from torch.func import functional_call, stack_module_state, vmap

    p = replace(protocol(name), metadata=(('score', 'success'),))
    world = world_for(p, pixels=True, cap=25)
    solver = attach_policy(world, horizon=5, receding=5, action_block=5)

    class Population:
        model = solver.cost.model
        population_size = 3
        backend = 'test-functional-vmap'
        compiled = False

        def forward(self, *args, module, **kwargs):
            parameters, buffers = stack_module_state(
                [deepcopy(module) for _ in range(self.population_size)]
            )

            def call(parameters, buffers):
                return functional_call(
                    module, (parameters, buffers), args, kwargs, strict=True
                )

            return vmap(call)(parameters, buffers)

    many = swm.ManyWorlds.from_population(world, population_size=3)
    try:
        result = many.evaluate(
            p, solver=solver, population=Population(), eval_budget=25
        )
        assert result.planning_calls == 1
        assert result.task_final_distances.shape == (3, 2)
        assert result.population_backend == 'test-functional-vmap'
        torch.testing.assert_close(
            result.scores, result.task_successes.float().mean(1)
        )
    finally:
        many.close()


@integration
def test_pusht_complete_velocity_and_integer_state_restore():
    from dataclasses import replace

    p = protocol(count=1)
    task = p.tasks[0]
    options = dict(task.options)
    options['simulator_state'] = {
        'start': {
            'qpos': task.start,
            'qvel': [2, -3, 0.4],
            'last_action': [120, 130],
            'n_contacts': 3,
            'time': 7,
        }
    }
    p = replace(p, tasks=(replace(task, options=tuple(options.items())),))
    world = world_for(p)
    try:
        world._reset_from_protocol(p, 4)
        state = world.envs._state
        np.testing.assert_allclose(np.asarray(state.block_vel), [[2, -3]])
        np.testing.assert_allclose(np.asarray(state.block_angular_vel), [0.4])
        np.testing.assert_allclose(np.asarray(state.last_action), [[120, 130]])
        assert np.asarray(state.time).dtype == np.int32
        assert np.asarray(state.n_contacts).dtype == np.int32
        np.testing.assert_array_equal(state.time, [7])
    finally:
        world.close()


@integration
def test_mixed_physics_intervals_are_rejected():
    from dataclasses import replace

    p = protocol()
    changed = replace(
        p.tasks[1],
        options=(*p.tasks[1].options, ('task_contract', {'repeat': 2})),
    )
    p = replace(p, tasks=(p.tasks[0], changed))
    world = world_for(p)
    try:
        with pytest.raises(ValueError, match='share one physics repeat'):
            world._reset_from_protocol(p, 4)
    finally:
        world.close()
