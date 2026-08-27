"""envX-backed TwoRooms rollouts for population planning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from stable_worldmodel.world.world import World


@dataclass(frozen=True)
class JaxTwoRoomsOutcome:
    """Small device-resident result returned by one compiled rollout."""

    final_observations: torch.Tensor
    successes: torch.Tensor
    final_distances: torch.Tensor
    returns: torch.Tensor
    lengths: torch.Tensor
    path_costs: torch.Tensor
    control_costs: torch.Tensor
    collisions: torch.Tensor


class JaxTwoRoomsRollout:
    """Execute ``population * tasks`` TwoRooms plans in one JAX program.

    The Gym environments are only used to construct the initial task states.
    Planned actions cross directly from Torch to JAX through DLPack and only
    the small reduced outputs cross back the same way.
    """

    def __init__(
        self,
        *,
        population_size: int,
        num_tasks: int,
        eval_budget: int,
        max_episode_steps: int | None = None,
    ) -> None:
        # JAX otherwise reserves most of the accelerator on first use, which
        # is hostile to the Torch world models sharing the same device.
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        self._validate_cuda_packages()
        try:
            import envx
            import jax
            import jax.numpy as jnp
            from envx.tworooms import EnvState
        except ImportError as error:
            raise ImportError(
                'envX ManyWorlds rollouts require Python 3.11 or 3.12 and '
                'the pinned envX revision documented in docs/api/world.md'
            ) from error

        self._jax = jax
        self._jnp = jnp
        self._state_type = EnvState
        self.population_size = int(population_size)
        self.num_tasks = int(num_tasks)
        self.num_envs = self.population_size * self.num_tasks
        self.eval_budget = int(eval_budget)
        if max_episode_steps is None:
            max_episode_steps = self.eval_budget
        self.max_episode_steps = min(self.eval_budget, int(max_episode_steps))
        if self.max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        self.env, params = envx.make(
            'two-rooms',
            num_envs=self.num_envs,
            observation_type='state',
        )
        self.params = params.replace(
            max_steps_in_episode=self.max_episode_steps
        )
        key = jax.random.key(2)

        def score(initial_state, actions, params):
            _, trajectory = self.env.rollout(
                key,
                initial_state,
                actions,
                params,
            )
            positions = trajectory.observation[..., :2]
            previous = self._jnp.concatenate(
                (initial_state.agent_position[None], positions[:-1]), axis=0
            )
            # envX exposes every transition in the fixed-length scan. Include
            # the first terminal transition, then exclude the rest of the plan
            # from the effective episode and all of its reported statistics.
            completed_before = self._jnp.concatenate(
                (
                    self._jnp.zeros_like(trajectory.done[:1]),
                    self._jnp.cumsum(trajectory.done[:-1], axis=0),
                ),
                axis=0,
            )
            active = completed_before == 0
            lengths = active.sum(axis=0)
            final_index = lengths - 1
            environment_index = self._jnp.arange(self.num_envs)
            return (
                trajectory.observation[final_index, environment_index],
                (trajectory.info['success'] & active).any(axis=0),
                trajectory.info['distance_to_target'][
                    final_index, environment_index
                ],
                (trajectory.reward * active).sum(axis=0),
                lengths,
                (
                    self._jnp.linalg.norm(positions - previous, axis=-1)
                    * active
                ).sum(axis=0),
                (self._jnp.square(actions).sum(axis=-1) * active).sum(axis=0),
                (trajectory.info['collided'] * active).sum(axis=0),
            )

        # The outer JIT lets XLA discard the unused trajectory instead of
        # materializing time * population * tasks worth of state and info.
        self._score = jax.jit(score)

    @staticmethod
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

    def _repeat_tasks(self, value: Any):
        value = self._jnp.asarray(value)
        repeats = (self.population_size,) + (1,) * value.ndim
        return self._jnp.tile(value, repeats).reshape(
            self.num_envs, *value.shape[1:]
        )

    @staticmethod
    def _scalar(value: Any) -> float:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return float(np.asarray(value).reshape(-1)[0])

    def initial_state(self, world: World):
        """Copy the ``tasks``-sized Gym reset state into one JAX pytree."""
        fields: dict[str, list[Any]] = {
            'agent_position': [],
            'target_position': [],
            'door_positions': [],
            'door_sizes': [],
            'agent_radius': [],
            'target_radius': [],
            'speed': [],
            'wall_axis': [],
            'wall_thickness': [],
            'num_doors': [],
        }
        for wrapped in world.envs.envs:
            env = wrapped.unwrapped
            num_doors = int(env.num_doors)
            door_positions = np.full(3, 49.0, dtype=np.float32)
            door_sizes = np.full(3, 14.0, dtype=np.float32)
            door_positions[:num_doors] = np.asarray(
                env.door_positions, dtype=np.float32
            )
            door_sizes[:num_doors] = np.asarray(
                env.door_sizes, dtype=np.float32
            )
            fields['agent_position'].append(
                np.asarray(env.agent_position, dtype=np.float32)
            )
            fields['target_position'].append(
                np.asarray(env.target_position, dtype=np.float32)
            )
            fields['door_positions'].append(door_positions)
            fields['door_sizes'].append(door_sizes)
            fields['agent_radius'].append(float(env.agent_radius))
            fields['target_radius'].append(
                self._scalar(env.variation_space['target']['radius'].value)
            )
            fields['speed'].append(float(env.agent_speed))
            fields['wall_axis'].append(int(env.wall_axis))
            fields['wall_thickness'].append(int(env.wall_thickness))
            fields['num_doors'].append(num_doors)

        arrays = {
            name: self._repeat_tasks(np.asarray(values))
            for name, values in fields.items()
        }
        success_distances = {
            float(wrapped.unwrapped.success_radius)
            for wrapped in world.envs.envs
        }
        if len(success_distances) != 1:
            raise ValueError(
                'envX requires every task to use the same success radius'
            )
        self.params = self.params.replace(
            success_distance=success_distances.pop()
        )
        return self._state_type(
            **arrays,
            last_action=self._jnp.zeros((self.num_envs, 2), self._jnp.float32),
            time=self._jnp.zeros((self.num_envs,), self._jnp.int32),
        )

    def __call__(
        self,
        initial_state,
        actions: torch.Tensor,
    ) -> JaxTwoRoomsOutcome:
        """Roll out ``(P, T, time, 2)`` actions without a host copy."""
        expected = (
            self.population_size,
            self.num_tasks,
            self.eval_budget,
            2,
        )
        if tuple(actions.shape) != expected:
            raise ValueError(f'expected action plans with shape {expected}')
        if actions.device.type not in {'cpu', 'cuda'}:
            raise ValueError(
                'envX DLPack rollouts require Torch tensors on CPU or CUDA, '
                f'got {actions.device}'
            )
        if (
            actions.device.type == 'cuda'
            and self._jax.default_backend() != 'gpu'
        ):
            raise RuntimeError(
                'Torch plans are on CUDA but JAX has no GPU backend; install '
                "the envX CUDA extra matching PyTorch's CUDA major as "
                'documented in docs/api/world.md'
            )

        time_major = (
            actions.detach()
            .to(dtype=torch.float32)
            .permute(2, 0, 1, 3)
            .reshape(self.eval_budget, self.num_envs, 2)
            .contiguous()
        )
        jax_actions = self._jax.dlpack.from_dlpack(time_major)
        outputs = self._score(initial_state, jax_actions, self.params)

        def to_torch(value):
            return torch.utils.dlpack.from_dlpack(value).reshape(
                self.population_size, self.num_tasks, *value.shape[1:]
            )

        return JaxTwoRoomsOutcome(
            final_observations=to_torch(outputs[0]),
            successes=to_torch(outputs[1]),
            final_distances=to_torch(outputs[2]),
            returns=to_torch(outputs[3]),
            lengths=to_torch(outputs[4]),
            path_costs=to_torch(outputs[5]),
            control_costs=to_torch(outputs[6]),
            collisions=to_torch(outputs[7]),
        )


__all__ = ['JaxTwoRoomsOutcome', 'JaxTwoRoomsRollout']
