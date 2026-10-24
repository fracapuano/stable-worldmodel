"""Device-resident frozen-task execution behind World and ManyWorlds."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

ENVIRONMENTS = {
    'swm/PushT-v1': 'pusht',
    'swm/ReacherDMControl-v0': 'reacher',
    'swm/OGBCube-v0': 'cube',
}


def action_space_for(env_name):
    width = 5 if ENVIRONMENTS[env_name] == 'cube' else 2
    return gym.spaces.Box(-1, 1, (width,), np.float32)


def _vector(value, width, name):
    array = np.asarray(value, np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f'{name} requires {width} finite values')
    return array


def task_inputs(name, options):
    """Interpret existing reset options; never mutate a frozen task."""
    state = dict(options.get('simulator_state', {}))
    start, goal = dict(state.get('start', {})), dict(state.get('goal', {}))
    contract = dict(options.get('task_contract', {}))
    if name == 'pusht':
        start.setdefault('qpos', options['state'])
        goal.setdefault(
            'qpos', options.get('goal_state', options['target_state'])
        )
        for value in (start, goal):
            value.setdefault('qvel', [0, 0, 0])
        width, velocity_width = 7, 3
    elif name == 'reacher':
        payload = options.get('reacher_task', {})
        for value, prefix, position in (
            (start, 'start', options['state']),
            (goal, 'goal', options['target_state']),
        ):
            value.setdefault('qpos', payload.get(prefix + '_qpos', position))
            if 'qvel' not in value:
                value['qvel'] = payload.get(
                    prefix + '_qvel', options.get(prefix + '_qvel')
                )
        width = velocity_width = 2
    else:
        for value, prefix in ((start, 'cube'), (goal, 'cube_goal')):
            for field in ('qpos', 'qvel'):
                if field not in value:
                    value[field] = options.get(prefix + '_' + field)
        width, velocity_width = 21, 20
    for label, value in (('start', start), ('goal', goal)):
        value['qpos'] = _vector(value.get('qpos'), width, label + ' qpos')
        value['qvel'] = _vector(
            value.get('qvel'), velocity_width, label + ' qvel'
        )
        expected = options['state' if label == 'start' else 'target_state']
        actual = value['qpos'][14:17] if name == 'cube' else value['qpos']
        if not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError(
                f'{label} simulator state disagrees with task coordinates'
            )
        if name == 'cube' and not np.isclose(
            np.linalg.norm(value['qpos'][17:21]), 1, atol=1e-4
        ):
            raise ValueError(f'{label} Cube quaternion must be normalized')
    defaults = {
        'repeat': 2 if name == 'reacher' else 1,
        'threshold': 0.04 if name == 'cube' else 0.05,
        'position_threshold': 20.0,
        'angle_threshold': float(np.pi / 9),
        'relative': True,
        'action_scale': 100.0,
        'clip_target': False,
    }
    unknown = contract.keys() - defaults.keys()
    if unknown:
        raise ValueError(
            f'unsupported task_contract options: {sorted(unknown)}'
        )
    defaults.update(contract)
    repeat = defaults['repeat']
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError('task_contract repeat must be a positive integer')
    for key in (
        'threshold',
        'position_threshold',
        'angle_threshold',
        'action_scale',
    ):
        if not np.isfinite(defaults[key]) or defaults[key] <= 0:
            raise ValueError(
                f'task_contract {key} must be finite and positive'
            )
    return start, goal, defaults


@dataclass(frozen=True)
class EnvXOutcome:
    final_observations: torch.Tensor
    successes: torch.Tensor
    final_distances: torch.Tensor
    returns: torch.Tensor
    lengths: torch.Tensor
    path_costs: torch.Tensor
    control_costs: torch.Tensor
    collisions: torch.Tensor
    final_rewards: torch.Tensor
    records: tuple = ()


class _Engine:
    def __init__(self, name, count, device, image_shape, kwargs):
        import envx
        import jax

        from ._envx_compat import StateAdapter, make_env

        del envx  # Import failure identifies the optional dependency early.
        self.name, self.count, self.device = name, count, device
        config = dict(kwargs, num_envs=count, observation_type='state')
        if name == 'pusht':
            if image_shape and image_shape[0] != image_shape[1]:
                raise ValueError(
                    'pinned envX PushT renderer requires square images'
                )
            config['observation_size'] = image_shape[0] if image_shape else 224
        elif name == 'reacher':
            config.update(devices=[device])
            config.setdefault(
                'physics_backend',
                'warp' if device.platform == 'gpu' else 'jax',
            )
        else:
            config.update(device=device)
            config.setdefault('naconmax', max(512, 32 * count))
            config.setdefault('njmax', 1024)
        if image_shape and name != 'pusht':
            config.update(observation_type='pixels', image_size=image_shape)
        with jax.default_device(device):
            env, params = make_env(name, **config)
            self.adapter = StateAdapter(env, params, name)
        self.restore = jax.jit(self.adapter.restore)
        self.render = jax.jit(self.adapter.render) if image_shape else None
        self.run = jax.jit(self._run, static_argnames=('record',))

    def _metrics(self, state, goals, contract):
        import jax.numpy as jnp

        values = self.adapter.vector(state)
        if self.name == 'reacher':
            distance = jnp.max(jnp.abs(values - goals), axis=-1)
            success = distance < contract['threshold']
        elif self.name == 'cube':
            distance = jnp.linalg.norm(values - goals[:, :3], axis=-1)
            success = distance <= contract['threshold']
        else:
            distance = jnp.linalg.norm(values - goals, axis=-1)
            position = jnp.linalg.norm(values[:, :4] - goals[:, :4], axis=-1)
            angle = jnp.mod(jnp.abs(values[:, 4] - goals[:, 4]), 2 * jnp.pi)
            angle = jnp.minimum(angle, 2 * jnp.pi - angle)
            success = (position < contract['position_threshold']) & (
                angle < contract['angle_threshold']
            )
        return values, distance, success

    def _run(
        self,
        inputs,
        contract,
        actions,
        limits,
        initial_state=None,
        *,
        record=False,
    ):
        import jax
        import jax.numpy as jnp

        from ._envx_compat import retain_state

        state = (
            self.adapter.restore(inputs)
            if initial_state is None
            else initial_state
        )
        values, distance, _ = self._metrics(state, inputs['goal'], contract)
        zeros = jnp.zeros(actions.shape[0], jnp.float32)
        initial = (
            state,
            values,
            distance,
            jnp.zeros_like(zeros, dtype=bool),
            jnp.zeros_like(limits),
            zeros,
            zeros,
            zeros,
            zeros,
        )

        def step(carry, item):
            (
                state,
                values,
                distance,
                success,
                lengths,
                returns,
                path,
                control,
                final_rewards,
            ) = carry
            index, action = item
            active = (~success) & (index < limits)
            lengths += active.astype(jnp.int32)
            control += jnp.where(active, jnp.square(action).sum(-1), 0)

            def repeat_step(repeat_index, inner):
                current, values, distance, success, path = inner
                next_state = self.adapter.advance(current, action, contract)
                next_values, next_distance, hit = self._metrics(
                    next_state, inputs['goal'], contract
                )
                live = (
                    active & (~success) & (repeat_index < contract['repeat'])
                )
                path += jnp.where(
                    live, jnp.linalg.norm(next_values - values, axis=-1), 0
                )
                values = jnp.where(live[:, None], next_values, values)
                distance = jnp.where(live, next_distance, distance)
                success |= live & hit
                # Retain the full terminal simulator state so rendering and
                # subsequent masked steps agree with the reported outcome.
                current = retain_state(current, next_state, live)
                return current, values, distance, success, path

            state, values, distance, success, path = jax.lax.fori_loop(
                0,
                jnp.max(contract['repeat']),
                repeat_step,
                (state, values, distance, success, path),
            )
            reward = (
                -distance
                if self.name == 'pusht'
                else success.astype(jnp.float32)
            )
            returns += jnp.where(active, reward, 0)
            final_rewards = jnp.where(active, reward, final_rewards)
            trace = None
            if record:
                trace = (
                    {'state': values, 'distance_to_target': distance[:, None]},
                    reward,
                )
                if self.render is not None:
                    trace[0]['pixels'] = self.render(state)
            return (
                state,
                values,
                distance,
                success,
                lengths,
                returns,
                path,
                control,
                final_rewards,
            ), trace

        final, trace = jax.lax.scan(
            step,
            initial,
            (jnp.arange(actions.shape[1]), jnp.swapaxes(actions, 0, 1)),
        )
        (
            _,
            values,
            distance,
            success,
            lengths,
            returns,
            path,
            control,
            final_rewards,
        ) = final
        result = (
            (
                values,
                success,
                distance,
                returns,
                lengths,
                path,
                control,
                jnp.zeros_like(lengths),
                final_rewards,
            ),
            final[0],
        )
        return (*result, trace) if record else result

    def close(self):
        self.restore = self.render = self.run = None
        self.adapter.close()


class EnvXPool:
    """Internal task pool. Physics and pixels remain on the selected device."""

    execution_backend = 'envx'

    def __init__(
        self,
        env_name,
        num_envs,
        *,
        image_shape,
        max_episode_steps,
        device,
        image_transform=None,
        goal_transform=None,
        envx_options=None,
    ):
        self.device = torch.device('cuda')
        if env_name not in ENVIRONMENTS:
            raise ValueError(f'envX World does not support {env_name!r}')
        if min(num_envs, max_episode_steps) < 1:
            raise ValueError(
                'environment count and episode limit must be positive'
            )
        self.env_name, self.name = env_name, ENVIRONMENTS[env_name]
        self.num_envs, self.max_episode_steps = num_envs, max_episode_steps
        self._jax_device = None
        self.image_shape, self.image_transform, self.goal_transform = (
            image_shape,
            image_transform,
            goal_transform,
        )
        self.envx_options = dict(envx_options or {})
        managed = {
            'num_envs',
            'device',
            'devices',
            'observation_type',
            'render',
            'image_size',
            'observation_size',
        }
        if managed.intersection(self.envx_options):
            raise ValueError(
                'configure envX batch, device and images through World arguments'
            )
        if (
            self.name == 'cube'
            and self.envx_options.get('env_type', 'single') != 'single'
        ):
            raise ValueError('the Cube task adapter supports env_type=single')
        self._engines = {}
        self._renderer = None
        self._closed = False
        self.rows = None
        self._reset_inputs = {}
        self._contract = None
        self.single_action_space = action_space_for(env_name)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.seeds = np.zeros(num_envs, np.int64)
        if device is not None:
            self.set_device(device)

    def set_device(self, device):
        device = torch.device(device)
        if device.type not in ('cuda', 'cpu'):
            raise ValueError('envX supports CPU or CUDA devices')
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device != self.device:
            if self._engines or self._renderer is not None:
                raise ValueError(
                    'cannot change an initialized envX pool device'
                )
            self._jax_device = None
            self.device = device

    @property
    def jax_device(self):
        if self._jax_device is None:
            self.set_device(self.device)
            os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            from .jax_two_rooms import JaxTwoRoomsRollout

            JaxTwoRoomsRollout._validate_cuda_packages()
            try:
                import envx
                import jax
            except ImportError as error:
                raise ImportError(
                    'Frozen envX evaluation requires Python 3.11 or 3.12 and '
                    'the pinned dependencies in stable-worldmodel[envx]'
                ) from error
            del envx
            self._jax_device = jax.local_devices(
                backend='gpu' if self.device.type == 'cuda' else 'cpu'
            )[self.device.index or 0]
        return self._jax_device

    def _engine(self, count, *, render=False):
        if self._closed:
            raise RuntimeError('envX World is closed')
        if render and count == self.num_envs:
            if self._renderer is None:
                self._renderer = _Engine(
                    self.name,
                    count,
                    self.jax_device,
                    self.image_shape,
                    self.envx_options,
                )
            return self._renderer
        key = (count, render)
        if key not in self._engines:
            self._engines[key] = _Engine(
                self.name,
                count,
                self.jax_device,
                self.image_shape if render else None,
                self.envx_options,
            )
        return self._engines[key]

    def _inputs(self, repeat=1, which=0):
        import jax

        if which in self._reset_inputs:
            import jax.numpy as jnp

            return jax.tree.map(
                lambda value: jnp.tile(
                    value, (repeat,) + (1,) * (value.ndim - 1)
                ),
                self._reset_inputs[which],
            )
        rows = [row[which] for row in self.rows]
        extras = [set(row) - {'qpos', 'qvel'} for row in rows]
        if any(keys != extras[0] for keys in extras):
            raise ValueError('simulator input fields must match across tasks')
        allowed = {
            'time',
            'act',
            'ctrl',
            'qacc_warmstart',
            'qfrc_applied',
            'xfrc_applied',
            'eq_active',
            'mocap_pos',
            'mocap_quat',
            'userdata',
            'history',
            'plugin_state',
        }
        if self.name == 'pusht':
            allowed = {'time', 'last_action', 'n_contacts'}
        if extras[0] - allowed:
            raise ValueError(
                f'unsupported simulator fields: {sorted(extras[0] - allowed)}'
            )

        # Tile only the task axis: candidate-major/task-minor matches Torch.
        def batch(values):
            value = np.asarray(values, np.float32)
            if not np.isfinite(value).all():
                raise ValueError('simulator inputs must be finite')
            return jax.device_put(
                value,
                self.jax_device,
            )

        goals = [row[1]['qpos'] for row in self.rows]
        if self.name == 'cube':
            goals = [value[14:21] for value in goals]
        self._reset_inputs[which] = {
            'qpos': batch([r['qpos'] for r in rows]),
            'qvel': batch([r['qvel'] for r in rows]),
            'goal': batch(goals),
            'extras': {
                key: batch([r[key] for r in rows]) for key in extras[0]
            },
        }
        return self._inputs(repeat=repeat, which=which)

    def reset(self, seed=None, options=None):
        if self._closed:
            raise RuntimeError('envX World is closed')
        if not isinstance(options, list) or len(options) != self.num_envs:
            raise ValueError(
                'envX reset requires one complete reset option per task'
            )
        _ = self.jax_device
        import jax

        if any(
            row.get('variation') or row.get('variation_values')
            for row in options
        ):
            raise NotImplementedError(
                'envX task execution does not yet support Gym variations'
            )
        self.rows = [task_inputs(self.name, row) for row in options]
        if len({row[2]['repeat'] for row in self.rows}) != 1:
            raise ValueError(
                'envX task batches must share one physics repeat count'
            )
        self._reset_inputs.clear()
        self._contract = {
            key: jax.device_put(
                np.asarray([r[2][key] for r in self.rows]), self.jax_device
            )
            for key in self.rows[0][2]
        }
        self.seeds = np.asarray(seed, np.int64)
        inputs = self._inputs()
        engine = self._engine(self.num_envs)
        state = engine.restore(inputs)
        self._state = state
        self._steps = np.zeros(self.num_envs, np.int32)
        vector = torch.from_dlpack(engine.adapter.vector(state))
        info = {
            'state': vector[:, None],
            'action': np.zeros(
                (self.num_envs, 1, *self.single_action_space.shape), np.float32
            ),
            'step_idx': np.zeros((self.num_envs, 1), np.int64),
            'terminated': np.zeros(self.num_envs, bool),
        }
        target = torch.from_dlpack(inputs['goal'])
        if self.name == 'cube':
            target = target[:, :3]
        info['target_state'] = target[:, None]
        if not self.image_shape:
            info['goal'] = target[:, None]
        if self.image_shape:
            renderer = self._engine(self.num_envs, render=True)
            for key, which, transform in (
                ('pixels', 0, self.image_transform),
                ('goal', 1, self.goal_transform),
            ):
                pixels = torch.from_dlpack(
                    renderer.render(
                        renderer.restore(self._inputs(which=which))
                    )
                )
                info[key] = (transform(pixels) if transform else pixels)[
                    :, None
                ]
        self._infos = info
        return None, info

    def step(self, actions, mask=None):
        import jax

        engine = self._engine(self.num_envs)
        actions = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        if tuple(actions.shape) != (
            self.num_envs,
            *self.single_action_space.shape,
        ):
            raise ValueError('actions do not match the envX task batch')
        active = (
            np.ones(self.num_envs, bool)
            if mask is None
            else np.asarray(mask, bool).copy()
        )
        active &= self._steps < self.max_episode_steps
        actions = torch.where(
            torch.as_tensor(active, device=self.device)[:, None], actions, 0
        )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError('envX actions must be finite')
        contract = self._contract
        values, self._state = engine.run(
            self._inputs(),
            contract,
            jax.dlpack.from_dlpack(actions.contiguous())[:, None],
            jax.device_put(active.astype(np.int32), self.jax_device),
            self._state,
        )
        vector, success, distance, rewards, *_ = (
            torch.from_dlpack(v) for v in values
        )
        self._steps += active
        self._infos.update(
            state=vector[:, None],
            distance_to_target=distance[:, None],
            action=torch.where(
                torch.as_tensor(active, device=self.device)[:, None, None],
                actions[:, None],
                torch.as_tensor(self._infos['action'], device=self.device),
            ),
            step_idx=self._steps[:, None].copy(),
        )
        if self.image_shape:
            renderer = self._engine(self.num_envs, render=True)
            pixels = torch.from_dlpack(renderer.render(self._state))
            self._infos['pixels'] = (
                self.image_transform(pixels)
                if self.image_transform
                else pixels
            )[:, None]
        terminated = success.cpu().numpy() & active
        truncated = (
            (self._steps >= self.max_episode_steps) & active & ~terminated
        )
        self._infos['terminated'] = np.where(
            active, terminated, self._infos['terminated']
        )
        return None, rewards.cpu().numpy(), terminated, truncated, self._infos

    def rollout(self, actions, *, max_steps=None, limits=None, record=False):
        import jax
        import jax.numpy as jnp

        if self.rows is None:
            raise RuntimeError(
                'reset the envX task pool before executing plans'
            )
        if (
            not torch.is_tensor(actions)
            or actions.device.type != self.device.type
            or (
                self.device.index is not None
                and actions.device.index != self.device.index
            )
        ):
            raise ValueError(f'envX action tensors must be on {self.device}')
        population, tasks, horizon = actions.shape[:3]
        if (
            tasks != self.num_envs
            or actions.shape[3:] != self.single_action_space.shape
            or horizon < 1
            or population < 1
        ):
            raise ValueError(
                'plans must have shape (population, tasks, time, action)'
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError('envX actions must be finite')
        if max_steps is not None and max_steps < 1:
            raise ValueError('max_steps must be positive')
        limit = min(horizon, self.max_episode_steps, max_steps or horizon)
        if limits is None:
            limits = np.full((population, tasks), limit, np.int32)
        limits = jnp.minimum(jax.device_put(limits, self.jax_device), limit)
        if limits.shape != (population, tasks):
            raise ValueError('plan limits must have shape (population, tasks)')
        plans = actions.detach().float().contiguous()
        # A contiguous Torch view can still begin at an unaligned storage
        # offset. XLA assumes aligned buffers; copy only such views.
        alignment = 256 if self.device.type == 'cuda' else 64
        if plans.data_ptr() % alignment:
            plans = plans.clone()
        engine = self._engine(
            population * tasks, render=record and bool(self.image_shape)
        )
        contract = {
            key: jnp.tile(value, population)
            for key, value in self._contract.items()
        }
        result = engine.run(
            self._inputs(repeat=population),
            contract,
            jax.dlpack.from_dlpack(plans).reshape(
                population * tasks, horizon, -1
            ),
            limits.reshape(-1),
            record=record,
        )
        values = result[0]
        records = ()
        if record:
            from stable_worldmodel.evaluation.records import rollout_records

            observations, rewards = result[2]
            observations = {
                key: torch.from_dlpack(value).reshape(
                    horizon, population, tasks, *value.shape[2:]
                )
                for key, value in observations.items()
            }
            if 'pixels' in observations and self.image_transform:
                pixels = observations['pixels']
                pixels = self.image_transform(pixels.flatten(0, 2))
                observations['pixels'] = pixels.reshape(
                    horizon, population, tasks, *pixels.shape[1:]
                )
            records = rollout_records(
                self._infos,
                plans,
                torch.from_dlpack(values[4]).reshape(population, tasks),
                torch.from_dlpack(values[1]).reshape(population, tasks),
                observations,
                torch.from_dlpack(rewards).reshape(horizon, population, tasks),
            )
        return EnvXOutcome(
            *(
                torch.from_dlpack(v).reshape(population, tasks, *v.shape[1:])
                for v in values
            ),
            records=records,
        )

    def close(self):
        if not self._closed:
            for engine in self._engines.values():
                engine.close()
            if self._renderer:
                self._renderer.close()
            self._engines.clear()
            self._reset_inputs.clear()
            self._contract = None
            self._renderer = self.rows = self._state = self._infos = None
            self._closed = True
