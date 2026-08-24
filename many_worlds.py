"""Population evaluation of multiple weight-bound Worlds with FastCEM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from stable_worldmodel.buffer import HistoryBuffer
from stable_worldmodel.evaluation import (
    EpisodeResult,
    EvaluationProtocol,
    EvaluationResults,
    StepRecord,
)
from stable_worldmodel.evaluation.records import recordable
from stable_worldmodel.planning import FastCEMSolver
from stable_worldmodel.policy import ACTION_HISTORY_KEY, WorldModelPolicy

from .env_pool import EnvPool
from .world import World


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices while allowing an implicit accelerator index."""
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def _equivalent(left: Any, right: Any) -> bool:
    """Structural equality for policy preprocessing configuration."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    if hasattr(left, '__dict__'):
        return _equivalent(vars(left), vars(right))
    try:
        result = left == right
        return (
            bool(result)
            if not isinstance(result, np.ndarray)
            else result.all()
        )
    except (TypeError, ValueError):
        return repr(left) == repr(right)


@dataclass(frozen=True)
class ManyWorldsEvaluationResults:
    """Population planner tensors and ordinary per-World evaluation results.

    The leading dimension of planner tensors is the number of synchronized
    planning calls. The remaining leading dimensions are ``population`` and
    ``tasks``. Planner tensors stay on the solver device. envX actions and
    scores also stay device-resident; the Gym backend returns NumPy actions.
    """

    model_names: tuple[str, ...]
    planned_actions: torch.Tensor
    planner_costs: torch.Tensor
    planner_variances: torch.Tensor
    environment_actions: np.ndarray | torch.Tensor
    evaluations: tuple[EvaluationResults, ...]
    compiled: bool
    compile_error: str | None
    population_backend: str
    simulator_backend: str = 'gym'
    task_successes: torch.Tensor | None = None
    task_final_distances: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    score_name: str | None = None

    @property
    def population_size(self) -> int:
        return len(self.model_names)

    @property
    def planning_calls(self) -> int:
        return self.planned_actions.size(0)

    @property
    def task_returns(self) -> np.ndarray:
        """Realized returns shaped ``(population, tasks)``."""
        return np.asarray(
            [
                [episode.episode_return for episode in result.episodes]
                for result in self.evaluations
            ],
            dtype=np.float64,
        )

    @property
    def fitness(self) -> np.ndarray:
        """Configured score, or mean return for the Gym backend, ``(P,)``."""
        if self.scores is not None:
            return self.scores.detach().cpu().numpy()
        return self.task_returns.mean(axis=1)

    @property
    def success_rates(self) -> np.ndarray:
        """Success percentage per population member, ``(P,)``."""
        return np.asarray(
            [result.success_rate for result in self.evaluations],
            dtype=np.float64,
        )


class ManyWorlds:
    """Compose weight-bound :class:`World` instances into one population run.

    Each input ``World`` must already have a ``WorldModelPolicy`` backed by a
    compatible ``FastCEMSolver``. This preserves the invariant that one World
    owns one concrete model instance and weight set. ``evaluate`` replaces the
    individual solver calls with one population FastCEM graph. TwoRooms uses
    one open-loop envX rollout by default; other environments retain the
    closed-loop Gymnasium evaluator.

    Every LeWM parameter and persistent buffer may differ across Worlds. The
    model architecture, policy preprocessing, action space, and MPC
    configuration must remain identical so the population stays one tensor
    program.
    """

    def __init__(
        self,
        worlds: Sequence[World],
        *,
        model_names: Sequence[str] | None = None,
        simulator_backend: str = 'auto',
    ) -> None:
        worlds = tuple(worlds)
        if not worlds:
            raise ValueError('ManyWorlds requires at least one World')
        if not all(isinstance(world, World) for world in worlds):
            raise TypeError('worlds must contain only World instances')
        if len({id(world) for world in worlds}) != len(worlds):
            raise ValueError('worlds must contain distinct World instances')

        self.worlds = worlds
        self.policies = tuple(self._policy_for(world) for world in worlds)
        self.models = tuple(
            self._model_for(policy) for policy in self.policies
        )
        for model in self.models:
            model.eval()
        self._validate_world_shapes()
        self.simulator_backend = self._resolve_simulator_backend(
            simulator_backend
        )
        self.envs = (
            EnvPool.from_envs(
                [env for world in self.worlds for env in world.envs.envs]
            )
            if self.simulator_backend == 'gym'
            else self.worlds[0].envs
        )
        self._flat_infos: dict[str, Any] = {}
        self._jax_rollout = None

        if model_names is None:
            model_names = tuple(
                str(getattr(world, 'name', f'world-{index}'))
                for index, world in enumerate(worlds)
            )
        else:
            model_names = tuple(str(name) for name in model_names)
        if len(model_names) != len(worlds):
            raise ValueError('model_names must contain one name per World')
        if len(set(model_names)) != len(model_names):
            raise ValueError('model_names must be unique')
        self.model_names = tuple(model_names)

    @classmethod
    def init(
        cls,
        *,
        worlds: Sequence[World],
        model_names: Sequence[str] | None = None,
        simulator_backend: str = 'auto',
    ) -> Self:
        """Named constructor matching ``ManyWorlds.init(worlds=[...])``."""
        return cls(
            worlds=worlds,
            model_names=model_names,
            simulator_backend=simulator_backend,
        )

    def _resolve_simulator_backend(self, requested: str) -> str:
        if requested not in {'auto', 'gym', 'envx'}:
            raise ValueError(
                "simulator_backend must be 'auto', 'gym', or 'envx'"
            )
        first_env = self.worlds[0].envs.envs[0]
        spec = getattr(first_env, 'spec', None)
        names = {
            getattr(spec, 'id', None),
            getattr(first_env.unwrapped, 'env_name', None),
        }
        is_two_rooms = bool({'swm/TwoRoom-v1', 'TwoRoom'} & names)
        if requested == 'auto':
            return 'envx' if is_two_rooms else 'gym'
        if requested == 'envx' and not is_two_rooms:
            raise ValueError(
                'the envX ManyWorlds backend currently supports only '
                'swm/TwoRoom-v1'
            )
        return requested

    @staticmethod
    def _policy_for(world: World) -> WorldModelPolicy:
        policy = world.policy
        if not isinstance(policy, WorldModelPolicy):
            raise TypeError(
                'each World must have an attached WorldModelPolicy before it '
                'is passed to ManyWorlds'
            )
        if not isinstance(policy.solver, FastCEMSolver):
            raise TypeError(
                'each World policy must use FastCEMSolver so its model and '
                'planning configuration are explicit'
            )
        return policy

    @staticmethod
    def _model_for(policy: WorldModelPolicy) -> nn.Module:
        model = getattr(policy.solver.cost, 'model', None)
        if not isinstance(model, nn.Module):
            raise TypeError(
                'each FastCEM cost must expose its world model as cost.model'
            )
        return model

    @property
    def population_size(self) -> int:
        return len(self.worlds)

    @property
    def num_tasks(self) -> int:
        return self.worlds[0].num_envs

    @property
    def simulator_count(self) -> int:
        """Number of independently simulated population/task states."""
        if self.simulator_backend == 'envx':
            return self.population_size * self.num_tasks
        return self.envs.num_envs

    @property
    def bootstrap_simulator_count(self) -> int:
        """Gym envs stepped/reset for observations by this evaluator."""
        return (
            self.num_tasks
            if self.simulator_backend == 'envx'
            else self.simulator_count
        )

    @property
    def config(self):
        return self.policies[0].cfg

    def close(self) -> None:
        """Close every owned World; safe when the Worlds are already closed."""
        for world in self.worlds:
            world.close()

    def _validate_world_shapes(self) -> None:
        first_world = self.worlds[0]
        first_policy = self.policies[0]
        first_model = self.models[0]
        first_solver = first_policy.solver
        first_space = first_world.envs.single_action_space
        if not isinstance(first_space, gym.spaces.Box):
            raise TypeError(
                'population FastCEM requires a continuous Box action space'
            )

        for index, (world, policy, model) in enumerate(
            zip(
                self.worlds[1:],
                self.policies[1:],
                self.models[1:],
                strict=True,
            ),
            start=1,
        ):
            if world.num_envs != first_world.num_envs:
                raise ValueError(
                    f'World {index} has {world.num_envs} envs; expected '
                    f'{first_world.num_envs}'
                )
            if policy.cfg != first_policy.cfg:
                raise ValueError('all Worlds must use the same PlanConfig')
            if policy.history_keys != first_policy.history_keys:
                raise ValueError('all Worlds must use the same history_keys')
            if not _equivalent(policy.process, first_policy.process):
                raise ValueError('all Worlds must use identical preprocessing')
            if not _equivalent(policy.transform, first_policy.transform):
                raise ValueError('all Worlds must use identical transforms')
            space = world.envs.single_action_space
            if (
                type(space) is not type(first_space)
                or space.shape != first_space.shape
                or not np.array_equal(space.low, first_space.low)
                or not np.array_equal(space.high, first_space.high)
            ):
                raise ValueError('all Worlds must use the same action space')
            if type(model) is not type(first_model):
                raise TypeError(
                    'all world models must have the same concrete type'
                )
            for attribute in (
                'batch_size',
                'num_samples',
                'var_scale',
                'n_steps',
                'topk',
                'device',
                'dtype',
            ):
                if getattr(policy.solver, attribute) != getattr(
                    first_solver, attribute
                ):
                    raise ValueError(
                        f'all Worlds must use the same FastCEM {attribute}'
                    )

    def _validate_population(self, solver: FastCEMSolver) -> None:
        if not isinstance(solver, FastCEMSolver):
            raise TypeError('solver must be a FastCEMSolver')
        if getattr(solver._tensor_cost, 'model', None) is not self.models[0]:
            raise ValueError("solver must use the first World's FastCEM model")

        base_state = self.models[0].state_dict()
        expected_device = torch.device(solver.device)
        for index, model in enumerate(self.models):
            state = model.state_dict()
            if state.keys() != base_state.keys():
                raise ValueError(
                    f'model {index} has a different state-dict schema'
                )
            for name, reference in base_state.items():
                value = state[name]
                if value.shape != reference.shape:
                    raise ValueError(
                        f'model {index} state tensor {name!r} has shape '
                        f'{tuple(value.shape)}, expected {tuple(reference.shape)}'
                    )
                if value.dtype != reference.dtype or not _same_device(
                    value.device, expected_device
                ):
                    raise ValueError(
                        f'model {index} state tensor {name!r} must use '
                        f'{expected_device}/{reference.dtype}'
                    )

    def _infer_eval_budget(self) -> int:
        budgets = []
        for world in self.worlds:
            env = world.envs.envs[0]
            budget = getattr(env, '_max_episode_steps', None)
            if budget is None:
                budget = getattr(
                    getattr(env, 'spec', None), 'max_episode_steps', None
                )
            budgets.append(budget)
        if any(value is None for value in budgets) or len(set(budgets)) != 1:
            raise ValueError(
                'eval_budget is required when Worlds do not share one explicit '
                'max_episode_steps value'
            )
        return int(budgets[0])

    def _reset(self, protocol: EvaluationProtocol, eval_budget: int) -> None:
        seeds = []
        options = []
        for world in self.worlds:
            first_env = world.envs.envs[0]
            spec = getattr(first_env, 'spec', None)
            env_id = getattr(spec, 'id', None)
            accepted_names = {
                env_id,
                getattr(first_env.unwrapped, 'env_name', None),
            }
            if protocol.environment not in accepted_names:
                raise ValueError(
                    f'protocol environment {protocol.environment!r} does not '
                    f'match World environment {env_id!r}'
                )
            if len(protocol.tasks) != world.num_envs:
                raise ValueError(
                    'protocol evaluation requires one World env per task: '
                    f'{len(protocol.tasks)} tasks != {world.num_envs} envs'
                )
            if eval_budget < 1:
                raise ValueError(
                    'protocol evaluation requires eval_budget >= 1'
                )
            seeds.extend(task.environment_seed for task in protocol.tasks)
            options.extend(
                world._task_reset_options(task) for task in protocol.tasks
            )

        _, infos = self.envs.reset(seed=seeds, options=options)
        infos['controller_seed'] = np.tile(
            np.asarray(protocol.controller_seeds, dtype=np.int64),
            self.population_size,
        )[:, None]
        infos['task_key'] = [
            [key]
            for _ in range(self.population_size)
            for key in protocol.task_keys
        ]
        self._sync_world_batches(infos=infos)
        for policy in self.policies:
            if hasattr(policy, 'reset_state'):
                policy.reset_state()

    @staticmethod
    def _slice_batch(value: Any, start: int, end: int) -> Any:
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return value[start:end]
        if isinstance(value, list):
            return value[start:end]
        return value

    def _sync_world_batches(
        self,
        *,
        infos: dict[str, Any],
        rewards: np.ndarray | None = None,
        terminateds: np.ndarray | None = None,
        truncateds: np.ndarray | None = None,
        actions: np.ndarray | None = None,
    ) -> None:
        """Expose flat ``P*T`` pool state through the constituent Worlds."""
        self._flat_infos = infos
        tasks = self.num_tasks
        for population_index, world in enumerate(self.worlds):
            start = population_index * tasks
            end = start + tasks
            world.infos = {
                key: self._slice_batch(value, start, end)
                for key, value in infos.items()
            }
            world.envs._stacked_infos = world.infos
            world.envs.seeds = self.envs.seeds[start:end].copy()
            world.rewards = None if rewards is None else rewards[start:end]
            world.terminateds = (
                np.zeros(tasks, dtype=bool)
                if terminateds is None
                else terminateds[start:end]
            )
            world.truncateds = (
                np.zeros(tasks, dtype=bool)
                if truncateds is None
                else truncateds[start:end]
            )
            if actions is not None:
                world.actions = np.asarray(actions[population_index]).copy()

    def _combined_info(self) -> dict[str, Any]:
        return self._flat_infos

    def _population_info(self, flat_info: dict[str, Any]) -> dict[str, Any]:
        population, tasks = self.population_size, self.num_tasks
        expected = population * tasks
        result = {}
        for key, value in flat_info.items():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                if value.shape[0] == expected:
                    result[key] = value.reshape(
                        population, tasks, *value.shape[1:]
                    )
            elif isinstance(value, list) and len(value) == expected:
                result[key] = tuple(
                    tuple(value[p * tasks : (p + 1) * tasks])
                    for p in range(population)
                )
        return result

    @staticmethod
    def _info_at(
        world: World, key: str, index: int, default: Any = None
    ) -> Any:
        value = world.infos.get(key)
        if value is None:
            return default
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return value[index]
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    @classmethod
    def _snapshot(cls, world: World, index: int) -> dict[str, Any]:
        result = {}
        for key, value in world.infos.items():
            if key.startswith('_'):
                continue
            if torch.is_tensor(value):
                result[key] = value[index].detach().cpu().clone()
            elif isinstance(value, np.ndarray):
                result[key] = value[index].copy()
            elif isinstance(value, list):
                result[key] = deepcopy(value[index])
            else:
                result[key] = deepcopy(value)
        return result

    @classmethod
    def _state_at(cls, world: World, index: int) -> Any:
        value = cls._info_at(world, 'state', index)
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        return deepcopy(value)

    def _prepare_step_info(
        self, history: HistoryBuffer | None
    ) -> dict[str, Any]:
        flat = self.policies[0]._prepare_info(self._combined_info())
        if history is not None:
            history.append(
                {
                    key: flat[key]
                    for key in (*self.policies[0].history_keys, 'action')
                }
            )
        return flat

    def _planning_info(
        self, flat: dict[str, Any], history: HistoryBuffer | None
    ) -> dict[str, Any]:
        if history is not None:
            n_frames = min(
                self.config.history_len,
                max(history.num_strided()),
            )
            values = history.get(n_frames)
            for key in self.policies[0].history_keys:
                flat[key] = values[key]
            if 'action' in values:
                flat[ACTION_HISTORY_KEY] = values['action']
        return self._population_info(flat)

    def _environment_plan(self, planned: torch.Tensor) -> np.ndarray:
        population, tasks = planned.shape[:2]
        raw_shape = self.worlds[0].envs.single_action_space.shape
        raw_dim = int(np.prod(raw_shape))
        keep = self.config.receding_horizon
        actions_per_plan = keep * self.config.action_block
        selected = planned[:, :, :keep].reshape(
            population, tasks, actions_per_plan, raw_dim
        )
        selected = selected.detach()
        if selected.dtype == torch.bfloat16:
            selected = selected.float()
        actions = selected.cpu().numpy()
        process = self.policies[0].process
        if 'action' in process:
            actions = np.asarray(
                process['action'].inverse_transform(
                    actions.reshape(-1, raw_dim)
                )
            ).reshape(actions.shape)
        return actions.reshape(population, tasks, actions_per_plan, *raw_shape)

    def _environment_plan_tensor(
        self, planned: torch.Tensor, eval_budget: int
    ) -> torch.Tensor:
        """Expand a complete blocked plan on-device for an envX rollout."""
        population, tasks = planned.shape[:2]
        raw_shape = self.worlds[0].envs.single_action_space.shape
        raw_dim = int(np.prod(raw_shape))
        actions = planned.reshape(
            population,
            tasks,
            self.config.plan_len,
            raw_dim,
        )[:, :, :eval_budget]
        actions = actions.detach().float()
        process = self.policies[0].process
        if 'action' in process:
            actions = (
                process['action']
                .inverse_transform(actions.reshape(-1, raw_dim))
                .reshape(population, tasks, eval_budget, raw_dim)
            )
        return actions.reshape(population, tasks, eval_budget, *raw_shape)

    def _step_worlds(self, actions: np.ndarray, alive: np.ndarray) -> None:
        flat_actions = actions.reshape(
            self.simulator_count, *actions.shape[2:]
        )
        flat_alive = alive.reshape(self.simulator_count)
        _, rewards, terminateds, truncateds, infos = self.envs.step(
            flat_actions,
            mask=None if flat_alive.all() else flat_alive,
        )
        self._sync_world_batches(
            infos=infos,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            actions=actions,
        )

    @staticmethod
    def _select_tasks(value: Any, world: int, tasks: list[int]) -> Any:
        """Select one World's active tasks from a population value."""
        if torch.is_tensor(value):
            index = torch.as_tensor(tasks, device=value.device)
            return value[world].index_select(0, index)
        if isinstance(value, np.ndarray):
            return value[world, tasks]
        if isinstance(value, (tuple, list)):
            member = value[world]
            return type(member)(member[index] for index in tasks)
        return value

    def _notify_plans(
        self,
        info: dict[str, Any],
        output: dict[str, Any],
        alive: np.ndarray,
        model_queries: list[list[dict[str, Any]]],
        *,
        record: bool,
        selected_horizon: int | None = None,
    ) -> None:
        """Preserve each policy's ordinary post-plan observer contract."""
        if selected_horizon is None:
            selected_horizon = self.config.receding_horizon
        for world_index, policy in enumerate(self.policies):
            task_indices = np.flatnonzero(alive[world_index]).tolist()
            if not task_indices:
                continue
            controller_input = {
                key: self._select_tasks(value, world_index, task_indices)
                for key, value in info.items()
            }
            actions = self._select_tasks(
                output['actions'], world_index, task_indices
            )
            solver_output = {
                'actions': actions,
                'costs': self._select_tasks(
                    output['costs'], world_index, task_indices
                ),
                'mean': [
                    self._select_tasks(
                        output['mean'][0], world_index, task_indices
                    )
                ],
                'var': [
                    self._select_tasks(
                        output['var'][0], world_index, task_indices
                    )
                ],
                'compiled': output['compiled'],
                'compile_error': output['compile_error'],
            }
            event = {
                'env_indices': tuple(task_indices),
                'controller_input': controller_input,
                'solver_output': solver_output,
                'selected_plan': actions[:, :selected_horizon],
            }
            if record:
                model_queries[world_index].append(recordable(event))
            if policy.on_plan is not None:
                policy.on_plan(event)

    def _reset_envx(
        self, protocol: EvaluationProtocol, eval_budget: int
    ) -> dict[str, Any]:
        """Reset only the task-sized Gym pool used for initial/goal pixels."""
        first = self.worlds[0]
        first._reset_from_protocol(protocol, eval_budget)
        for world in self.worlds[1:]:
            env = world.envs.envs[0]
            spec = getattr(env, 'spec', None)
            accepted_names = {
                getattr(spec, 'id', None),
                getattr(env.unwrapped, 'env_name', None),
            }
            if protocol.environment not in accepted_names:
                raise ValueError(
                    f'protocol environment {protocol.environment!r} does not '
                    f'match World environment {getattr(spec, "id", None)!r}'
                )
            if world.num_envs != len(protocol.tasks):
                raise ValueError(
                    'protocol evaluation requires one World env per task'
                )
        for policy in self.policies[1:]:
            policy.reset_state()

        # Pixel preprocessing is performed for T initial/goal frames, not P*T.
        task_info = self.policies[0]._prepare_info(first.infos)
        population_info = {}
        for key, value in task_info.items():
            if torch.is_tensor(value) and value.size(0) == self.num_tasks:
                population_info[key] = value.unsqueeze(0).expand(
                    self.population_size, *value.shape
                )
            elif (
                isinstance(value, np.ndarray)
                and value.shape[0] == self.num_tasks
            ):
                population_info[key] = np.broadcast_to(
                    value[None], (self.population_size, *value.shape)
                )
        return population_info

    @staticmethod
    def _score_envx(
        protocol: EvaluationProtocol,
        successes: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        score_name = str(dict(protocol.metadata).get('score', 'distance'))
        if score_name == 'success':
            return score_name, successes.float().mean(dim=1)
        if score_name == 'distance':
            image_diagonal = float(np.hypot(224, 224))
            return score_name, 1.0 - distances.float().mean(
                dim=1
            ) / image_diagonal
        raise ValueError(
            "envX protocol metadata score must be 'distance' or 'success', "
            f'got {score_name!r}'
        )

    def _update_world_envx_info(
        self,
        final_observations: torch.Tensor,
        successes: torch.Tensor,
        distances: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expose final envX metrics through the existing World info API."""
        observations = final_observations.detach().cpu().numpy()
        success_values = successes.detach().cpu().numpy().astype(bool)
        distance_values = distances.detach().cpu().numpy()
        final_actions = actions[:, :, -1].detach().cpu().numpy()
        for index, world in enumerate(self.worlds):
            info = dict(self.worlds[0].infos)
            info['state'] = observations[index, :, None, :2]
            info['proprio'] = observations[index, :, None, :2]
            info['distance_to_target'] = distance_values[index, :, None]
            info['terminated'] = success_values[index, :, None]
            world.infos = info
            world.envs._stacked_infos = info
            world.rewards = np.zeros(self.num_tasks, dtype=np.float32)
            world.terminateds = success_values[index]
            world.truncateds = np.zeros(self.num_tasks, dtype=bool)
            world.actions = final_actions[index]
        return observations, success_values, distance_values

    @torch.inference_mode()
    def _evaluate_envx(
        self,
        protocol: EvaluationProtocol,
        *,
        solver: FastCEMSolver,
        eval_budget: int,
        record: bool,
    ) -> ManyWorldsEvaluationResults:
        """Plan once and score every full plan in one compiled envX rollout."""
        if record:
            raise ValueError(
                'envX ManyWorlds rollouts do not materialize per-step records'
            )
        if eval_budget > self.config.plan_len:
            raise ValueError(
                f'eval_budget {eval_budget} exceeds the generated plan length '
                f'{self.config.plan_len}'
            )

        info = self._reset_envx(protocol, eval_budget)
        seeds = info.get('controller_seed')
        noise = solver._batch_noise(
            self.num_tasks,
            0,
            None if seeds is None else seeds[0],
            np.zeros((self.num_tasks, 1), dtype=np.int64),
        )
        output = solver.solve_population(info, self.models, noise=noise)
        planned = output['actions']
        model_queries: list[list[dict[str, Any]]] = [
            [] for _ in range(self.population_size)
        ]
        self._notify_plans(
            info,
            output,
            np.ones((self.population_size, self.num_tasks), dtype=bool),
            model_queries,
            record=False,
            selected_horizon=self.config.horizon,
        )
        actions = self._environment_plan_tensor(planned, eval_budget)

        from .jax_two_rooms import JaxTwoRoomsRollout

        if (
            self._jax_rollout is None
            or self._jax_rollout.eval_budget != eval_budget
        ):
            self._jax_rollout = JaxTwoRoomsRollout(
                population_size=self.population_size,
                num_tasks=self.num_tasks,
                eval_budget=eval_budget,
            )
        initial_state = self._jax_rollout.initial_state(self.worlds[0])
        outcome = self._jax_rollout(initial_state, actions)
        score_name, scores = self._score_envx(
            protocol, outcome.successes, outcome.final_distances
        )
        _, success_values, distance_values = self._update_world_envx_info(
            outcome.final_observations,
            outcome.successes,
            outcome.final_distances,
            actions,
        )
        control_costs = actions.float().square().sum(dim=(2, 3)).cpu().numpy()
        returns = outcome.returns.detach().cpu().numpy()
        path_costs = outcome.path_costs.detach().cpu().numpy()
        collisions = outcome.collisions.detach().cpu().numpy()

        evaluations = tuple(
            EvaluationResults(
                backend=self.model_names[population_index],
                backend_type=(
                    f'{type(self.models[population_index]).__module__}.'
                    f'{type(self.models[population_index]).__qualname__}'
                ),
                protocol_digest=protocol.digest,
                episodes=tuple(
                    EpisodeResult(
                        task_key=task.key,
                        environment_seed=task.environment_seed,
                        controller_seed=task.controller_seed,
                        success=bool(
                            success_values[population_index, task_index]
                        ),
                        episode_return=float(
                            returns[population_index, task_index]
                        ),
                        length=eval_budget,
                        path_cost=float(
                            path_costs[population_index, task_index]
                        ),
                        control_cost=float(
                            control_costs[population_index, task_index]
                        ),
                        collisions=int(
                            collisions[population_index, task_index]
                        ),
                        constraint_violations=0,
                    )
                    for task_index, task in enumerate(protocol.tasks)
                ),
                model_queries=tuple(model_queries[population_index]),
                metadata={
                    'split': protocol.split,
                    'environment': protocol.environment,
                    'eval_budget': eval_budget,
                    'population_index': population_index,
                    'simulator_backend': 'envx',
                    'mean_final_distance': float(
                        distance_values[population_index].mean()
                    ),
                },
            )
            for population_index in range(self.population_size)
        )
        return ManyWorldsEvaluationResults(
            model_names=self.model_names,
            planned_actions=planned.unsqueeze(0),
            planner_costs=output['costs'].unsqueeze(0),
            planner_variances=output['var'][0].unsqueeze(0),
            environment_actions=actions,
            evaluations=evaluations,
            compiled=bool(output['compiled']),
            compile_error=output['compile_error'],
            population_backend=output['population_backend'] or 'unknown',
            simulator_backend='envx',
            task_successes=outcome.successes,
            task_final_distances=outcome.final_distances,
            scores=scores,
            score_name=score_name,
        )

    @torch.inference_mode()
    def evaluate(
        self,
        protocol: EvaluationProtocol,
        *,
        solver: FastCEMSolver,
        eval_budget: int | None = None,
        record: bool = False,
    ) -> ManyWorldsEvaluationResults:
        """Evaluate all Worlds with population-batched FastCEM.

        This is the population analogue of
        ``tuple(world.evaluate(protocol=protocol) for world in worlds)``.
        There is no per-World solver call or model loop: CEM and model ranking
        both carry the population axis on the accelerator. TwoRooms plans are
        executed by envX; other environments use one flat ``P*T`` Gym pool.
        """
        if not isinstance(protocol, EvaluationProtocol):
            raise TypeError('protocol must be an EvaluationProtocol')
        if eval_budget is None:
            eval_budget = self._infer_eval_budget()
        if eval_budget < 1:
            raise ValueError('eval_budget must be positive')

        solver.configure(
            action_space=self.worlds[0].envs.action_space,
            n_envs=self.num_tasks,
            config=self.config,
        )
        self._validate_population(solver)
        if self.simulator_backend == 'envx':
            return self._evaluate_envx(
                protocol,
                solver=solver,
                eval_budget=eval_budget,
                record=record,
            )
        self._reset(protocol, eval_budget)

        population, tasks = self.population_size, self.num_tasks
        alive = np.ones((population, tasks), dtype=bool)
        returns = np.zeros((population, tasks), dtype=np.float64)
        lengths = np.zeros((population, tasks), dtype=np.int64)
        path_costs = np.zeros((population, tasks), dtype=np.float64)
        control_costs = np.zeros((population, tasks), dtype=np.float64)
        collisions = np.zeros((population, tasks), dtype=np.int64)
        violations = np.zeros((population, tasks), dtype=np.int64)
        successes = np.zeros((population, tasks), dtype=bool)
        previous_states = [
            [self._state_at(world, task) for task in range(tasks)]
            for world in self.worlds
        ]
        previous_records = (
            [
                [self._snapshot(world, task) for task in range(tasks)]
                for world in self.worlds
            ]
            if record
            else None
        )
        step_records = [[[] for _ in range(tasks)] for _ in range(population)]
        model_queries: list[list[dict[str, Any]]] = [
            [] for _ in range(population)
        ]

        history = None
        if self.config.history_len > 1:
            max_len = self.config.history_max_len
            if max_len is None:
                max_len = (
                    self.config.history_len - 1
                ) * self.config.action_block + 1
            history = HistoryBuffer(
                n_envs=population * tasks,
                max_len=max_len,
                action_block=self.config.action_block,
                block_keys=('action',),
            )

        next_init = None
        action_plan = None
        plan_offset = 0
        planned_actions = []
        planner_costs = []
        planner_variances = []
        compiled = []
        executed_actions = []

        for _decision in range(eval_budget):
            flat_info = self._prepare_step_info(history)
            if action_plan is None or plan_offset >= action_plan.shape[2]:
                info = self._planning_info(flat_info, history)
                seeds = info.get('controller_seed')
                noise = solver._batch_noise(
                    tasks,
                    0,
                    None if seeds is None else seeds[0],
                    np.full((tasks, 1), _decision, dtype=np.int64),
                )
                output = solver.solve_population(
                    info,
                    self.models,
                    noise=noise,
                    init_action=next_init,
                )
                planned = output['actions']
                planned_actions.append(planned)
                planner_costs.append(output['costs'])
                planner_variances.append(output['var'][0])
                compiled.append(bool(output['compiled']))
                self._notify_plans(
                    info,
                    output,
                    alive,
                    model_queries,
                    record=record,
                )
                action_plan = self._environment_plan(planned)
                plan_offset = 0

                rest = planned[:, :, self.config.receding_horizon :]
                next_init = (
                    rest
                    if self.config.warm_start and rest.size(2) > 0
                    else None
                )

            actions = action_plan[:, :, plan_offset]
            plan_offset += 1
            recorded_actions = np.asarray(actions).copy()
            if np.issubdtype(recorded_actions.dtype, np.floating):
                recorded_actions[~alive] = np.nan
            executed_actions.append(recorded_actions)

            active_before = alive.copy()
            self._step_worlds(actions, active_before)
            for world_index, world in enumerate(self.worlds):
                for task_index in np.where(active_before[world_index])[0]:
                    current_state = self._state_at(world, int(task_index))
                    action = np.asarray(
                        actions[world_index, task_index]
                    ).copy()
                    reward = float(world.rewards[task_index])
                    returns[world_index, task_index] += reward
                    lengths[world_index, task_index] += 1
                    successes[world_index, task_index] |= bool(
                        world.terminateds[task_index]
                    )
                    control_costs[world_index, task_index] += float(
                        np.square(action).sum()
                    )

                    before = previous_states[world_index][task_index]
                    if before is not None and current_state is not None:
                        delta = np.asarray(current_state) - np.asarray(before)
                        path_costs[world_index, task_index] += float(
                            np.linalg.norm(delta)
                        )
                    collisions[world_index, task_index] += int(
                        np.asarray(
                            self._info_at(
                                world, 'collision', int(task_index), False
                            )
                        ).any()
                    )
                    violations[world_index, task_index] += int(
                        np.asarray(
                            self._info_at(
                                world,
                                'constraint_violation',
                                int(task_index),
                                False,
                            )
                        ).any()
                    )

                    if record:
                        current_record = self._snapshot(world, int(task_index))
                        step_records[world_index][task_index].append(
                            StepRecord(
                                decision=int(
                                    lengths[world_index, task_index] - 1
                                ),
                                observation=previous_records[world_index][
                                    task_index
                                ],
                                action=action,
                                next_observation=current_record,
                                reward=reward,
                                cost=-reward,
                                terminated=bool(world.terminateds[task_index]),
                                truncated=bool(world.truncateds[task_index]),
                            )
                        )
                        previous_records[world_index][task_index] = (
                            current_record
                        )
                    previous_states[world_index][task_index] = current_state

                done = active_before[world_index] & (
                    world.terminateds | world.truncateds
                )
                alive[world_index, done] = False

            if not alive.any():
                break

        evaluations = tuple(
            EvaluationResults(
                backend=self.model_names[world_index],
                backend_type=(
                    f'{type(self.models[world_index]).__module__}.'
                    f'{type(self.models[world_index]).__qualname__}'
                ),
                protocol_digest=protocol.digest,
                episodes=tuple(
                    EpisodeResult(
                        task_key=task.key,
                        environment_seed=task.environment_seed,
                        controller_seed=task.controller_seed,
                        success=bool(successes[world_index, task_index]),
                        episode_return=float(returns[world_index, task_index]),
                        length=int(lengths[world_index, task_index]),
                        path_cost=float(path_costs[world_index, task_index]),
                        control_cost=float(
                            control_costs[world_index, task_index]
                        ),
                        collisions=int(collisions[world_index, task_index]),
                        constraint_violations=int(
                            violations[world_index, task_index]
                        ),
                        steps=tuple(step_records[world_index][task_index]),
                    )
                    for task_index, task in enumerate(protocol.tasks)
                ),
                model_queries=tuple(model_queries[world_index]),
                metadata={
                    'split': protocol.split,
                    'environment': protocol.environment,
                    'eval_budget': eval_budget,
                    'population_index': world_index,
                },
            )
            for world_index in range(population)
        )

        return ManyWorldsEvaluationResults(
            model_names=self.model_names,
            planned_actions=torch.stack(planned_actions),
            planner_costs=torch.stack(planner_costs),
            planner_variances=torch.stack(planner_variances),
            environment_actions=np.stack(executed_actions, axis=2),
            evaluations=evaluations,
            compiled=all(compiled),
            compile_error=solver.compile_error,
            population_backend=solver.population_backend or 'unknown',
            simulator_backend='gym',
        )


__all__ = ['ManyWorlds', 'ManyWorldsEvaluationResults']
