"""Device-resident frozen-task execution behind World and ManyWorlds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

ENVIRONMENTS = {
    'swm/TwoRoom-v1': 'two-rooms',
    'swm/PushT-v1': 'pusht',
    'swm/ReacherDMControl-v0': 'reacher',
    'swm/OGBCube-v0': 'cube',
}


def _validate_cuda_packages() -> None:
    torch_cuda = torch.version.cuda
    if not torch.cuda.is_available() or torch_cuda is None:
        return
    installed = []
    for major in (12, 13):
        try:
            version(f'jax-cuda{major}-plugin')
        except PackageNotFoundError:
            continue
        installed.append(major)
    expected = int(torch_cuda.split('.', maxsplit=1)[0])
    if installed and installed != [expected]:
        choices = ', '.join(map(str, installed))
        raise RuntimeError(
            f'PyTorch uses CUDA {expected}, but JAX CUDA plugin(s) '
            f'{choices} are installed. Install only '
            'the envX/JAX extra matching PyTorch to avoid cuDNN '
            'sublibrary conflicts.'
        )


def action_space_for(env_name):
    width = 5 if ENVIRONMENTS[env_name] == 'cube' else 2
    return gym.spaces.Box(-1, 1, (width,), np.float32)


def _vector(value, width, name):
    array = np.asarray(value, np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f'{name} requires {width} finite values')
    return array


def _two_room_inputs(options, seed, config):
    from stable_worldmodel import spaces
    from stable_worldmodel.envs.two_room.env import (
        DEFAULT_VARIATIONS,
        make_variation_space,
    )

    space = make_variation_space()
    if config.get('init_value') is not None:
        space.set_init_value(config['init_value'])
    spaces.reset_variation_space(space, seed, options, DEFAULT_VARIATIONS)
    values = space.value
    count = int(values['door']['number'])
    extras = {
        'door_positions': np.where(
            np.arange(3) < count, values['door']['position'], 49
        ).astype(np.float32),
        'door_sizes': np.where(
            np.arange(3) < count, values['door']['size'], 14
        ).astype(np.float32),
        'agent_radius': float(values['agent']['radius'].item()),
        'target_radius': float(values['target']['radius'].item()),
        'speed': float(values['agent']['speed'].item()),
        'wall_axis': int(values['wall']['axis']),
        'wall_thickness': int(values['wall']['thickness']),
        'num_doors': count,
        'render_target': bool(values['rendering']['render_target'])
        or bool(config.get('render_target', False)),
        **{
            name + '_color': values[name]['color']
            for name in ('agent', 'target', 'wall', 'door', 'background')
        },
    }
    start = dict(
        qpos=_vector(options['state'], 2, 'start'),
        qvel=np.zeros(0, np.float32),
        **extras,
    )
    goal = dict(start, qpos=_vector(options['target_state'], 2, 'goal'))
    return start, goal


def task_inputs(name, options, *, seed=None, config=None):
    """Interpret existing reset options; never mutate a frozen task."""
    state = dict(options.get('simulator_state', {}))
    start, goal = dict(state.get('start', {})), dict(state.get('goal', {}))
    contract = dict(options.get('task_contract', {}))
    if name == 'two-rooms':
        start, goal = _two_room_inputs(options, seed, config or {})
        width, velocity_width = 2, 0
    elif name == 'pusht':
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
        'threshold': (config or {}).get('success_radius', 16.0)
        if name == 'two-rooms'
        else 0.04
        if name == 'cube'
        else 0.05,
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


class _Engine:
    def __init__(self, name, count, device, image_shape, kwargs):
        import envx
        import jax

        from ._envx_compat import StateAdapter, make_env

        del envx  # Import failure identifies the optional dependency early.
        self.name, self.count, self.device = name, count, device
        config = dict(kwargs, num_envs=count, observation_type='state')
        if name == 'two-rooms':
            for key in (
                'success_radius',
                'render_target',
                'init_value',
                'image_resample',
            ):
                config.pop(key, None)
        elif name == 'pusht':
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
        if image_shape and name not in {'pusht', 'two-rooms'}:
            config.update(observation_type='pixels', image_size=image_shape)
        with jax.default_device(device):
            env, params = make_env(name, **config)
            self.adapter = StateAdapter(env, params, name)
        self.restore = jax.jit(self.adapter.restore)
        self.render = jax.jit(self.adapter.render) if image_shape else None
        self.run = jax.jit(self._run)

    def _metrics(self, state, goals, contract):
        import jax.numpy as jnp

        values = self.adapter.vector(state)
        if self.name == 'two-rooms':
            distance = jnp.linalg.norm(values[:, :2] - goals, axis=-1)
            success = distance < contract['threshold']
        elif self.name == 'reacher':
            delta = values - goals
            # Only the shoulder is periodic; the wrist has joint limits.
            shoulder = jnp.arctan2(jnp.sin(delta[:, 0]), jnp.cos(delta[:, 0]))
            distance = jnp.maximum(jnp.abs(shoulder), jnp.abs(delta[:, 1]))
            success = distance < contract['threshold']
        elif self.name == 'cube':
            distance = jnp.linalg.norm(values - goals[:, :3], axis=-1)
            success = distance <= contract['threshold']
        else:
            position = jnp.linalg.norm(values[:, :4] - goals[:, :4], axis=-1)
            angle = jnp.mod(jnp.abs(values[:, 4] - goals[:, 4]), 2 * jnp.pi)
            angle = jnp.minimum(angle, 2 * jnp.pi - angle)
            distance = jnp.hypot(position, jnp.rad2deg(angle))
            success = (position < contract['position_threshold']) & (
                angle < contract['angle_threshold']
            )
        return values, distance, success

    def _run(self, inputs, contract, actions, limits, initial_state=None):
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
            jnp.zeros(actions.shape[0], jnp.int32),
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
                collisions,
            ) = carry
            index, action = item
            active = (~success) & (index < limits)
            lengths += active.astype(jnp.int32)
            control += jnp.where(active, jnp.square(action).sum(-1), 0)

            def repeat_step(repeat_index, inner):
                current, values, distance, success, path, collisions = inner
                next_state, collided = self.adapter.advance(
                    current, action, contract
                )
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
                collisions += (live & collided).astype(jnp.int32)
                success |= live & hit
                # Retain the full terminal simulator state so rendering and
                # subsequent masked steps agree with the reported outcome.
                current = retain_state(current, next_state, live)
                return current, values, distance, success, path, collisions

            state, values, distance, success, path, collisions = (
                jax.lax.fori_loop(
                    0,
                    jnp.max(contract['repeat']),
                    repeat_step,
                    (state, values, distance, success, path, collisions),
                )
            )
            reward = (
                jnp.zeros_like(distance)
                if self.name == 'two-rooms'
                else -distance
                if self.name == 'pusht'
                else success.astype(jnp.float32)
            )
            returns += jnp.where(active, reward, 0)
            final_rewards = jnp.where(active, reward, final_rewards)
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
                collisions,
            ), None

        final, _ = jax.lax.scan(
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
            collisions,
        ) = final
        return (
            values,
            success,
            distance,
            returns,
            lengths,
            path,
            control,
            collisions,
            final_rewards,
        ), final[0]

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

    def _with_num_envs(self, num_envs):
        return type(self)(
            self.env_name,
            num_envs,
            image_shape=self.image_shape,
            max_episode_steps=self.max_episode_steps,
            device=self.device,
            image_transform=self.image_transform,
            goal_transform=self.goal_transform,
            envx_options=self.envx_options,
        )

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
            _validate_cuda_packages()
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
        if render:
            if self._renderer is None:
                self._renderer = _Engine(
                    self.name,
                    self.num_envs,
                    self.jax_device,
                    self.image_shape,
                    self.envx_options,
                )
            return self._renderer
        if count not in self._engines:
            self._engines[count] = _Engine(
                self.name, count, self.jax_device, None, self.envx_options
            )
        return self._engines[count]

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
        if self.name == 'two-rooms':
            allowed = set(rows[0]) - {'qpos', 'qvel'}
        elif self.name == 'pusht':
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

        if self.name != 'two-rooms' and any(
            row.get('variation') or row.get('variation_values')
            for row in options
        ):
            raise NotImplementedError(
                'envX task execution does not yet support Gym variations'
            )
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, (int, np.integer)):
            seed = [int(seed) + i for i in range(self.num_envs)]
        if len(seed) != self.num_envs:
            raise ValueError('reset requires one seed per task')
        self.rows = [
            task_inputs(self.name, row, seed=seed[i], config=self.envx_options)
            for i, row in enumerate(options)
        ]
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
        self.seeds = np.asarray(
            [0 if value is None else value for value in seed], np.int64
        )
        inputs = self._inputs()
        engine = self._engine(self.num_envs)
        state = engine.restore(inputs)
        self._state = state
        self._steps = np.zeros(self.num_envs, np.int32)
        vector = torch.from_dlpack(engine.adapter.vector(state))
        if self.name == 'two-rooms':
            vector = vector[:, :2]
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
        if self.name == 'two-rooms':
            info['proprio'] = vector[:, None]
            info['goal_state'] = target[:, None]
            info['distance_to_target'] = torch.linalg.vector_norm(
                vector - target, dim=-1
            )[:, None]
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
                        renderer.restore(self._inputs(which=which)),
                        self._inputs(which=which),
                    )
                )
                pixels = self._resize_pixels(pixels)
                info[key] = (transform(pixels) if transform else pixels)[
                    :, None
                ]
        self._infos = info
        return None, info

    def _resize_pixels(self, pixels):
        if tuple(pixels.shape[1:3]) == tuple(self.image_shape):
            return pixels
        from torchvision.transforms.functional import InterpolationMode, resize

        from stable_worldmodel.wrapper.default import _resolve_resample

        mode = _resolve_resample(self.envx_options.get('image_resample'))
        # PIL's default for RGB images is bicubic. Tensor interpolation keeps
        # the common resize modes on the simulator device.
        mode = InterpolationMode.BICUBIC if mode is None else mode
        return resize(
            pixels.permute(0, 3, 1, 2),
            list(self.image_shape),
            interpolation=mode,
            antialias=True,
        ).permute(0, 2, 3, 1)

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
        if self.name == 'two-rooms':
            vector = vector[:, :2]
        if self.name == 'two-rooms':
            self._infos['proprio'] = vector[:, None]
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
            pixels = torch.from_dlpack(
                renderer.render(self._state, self._inputs())
            )
            pixels = self._resize_pixels(pixels)
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

    def snapshot(self):
        """Return immutable batched reset inputs, independent of later resets."""
        if self.rows is None or self._closed:
            raise RuntimeError(
                'reset the open envX task pool before taking a snapshot'
            )
        return {'inputs': self._inputs(), 'contract': self._contract}

    def rollout(
        self, actions, *, initial_state=None, max_steps=None, limits=None
    ):
        import jax
        import jax.numpy as jnp

        initial = self.snapshot() if initial_state is None else initial_state
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
        if any(value.shape[0] != tasks for value in jax.tree.leaves(initial)):
            raise ValueError('snapshot must contain one row per task')
        plans = actions.detach().float().contiguous()
        # A contiguous Torch view can still begin at an unaligned storage
        # offset. XLA assumes aligned buffers; copy only such views.
        alignment = 256 if self.device.type == 'cuda' else 64
        if plans.data_ptr() % alignment:
            plans = plans.clone()
        engine = self._engine(population * tasks)
        contract = {
            key: jnp.tile(value, population)
            for key, value in initial['contract'].items()
        }
        inputs = jax.tree.map(
            lambda value: jnp.tile(
                value, (population,) + (1,) * (value.ndim - 1)
            ),
            initial['inputs'],
        )
        values, _ = engine.run(
            inputs,
            contract,
            jax.dlpack.from_dlpack(plans).reshape(
                population * tasks, horizon, -1
            ),
            limits.reshape(-1),
        )
        return EnvXOutcome(
            *(
                torch.from_dlpack(v).reshape(population, tasks, *v.shape[1:])
                for v in values
            )
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
