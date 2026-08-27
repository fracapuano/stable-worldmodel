"""Population evaluation of multiple weight-bound Worlds with FastCEM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker, TimeLimit
from torch import nn
from typing_extensions import Self

from stable_worldmodel.envs.two_room.env import TwoRoomEnv
from stable_worldmodel.evaluation import (
    EpisodeResult,
    EvaluationProtocol,
    EvaluationResults,
)
from stable_worldmodel.planning import FastCEMSolver
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.wrapper import (
    AddPixelsWrapper,
    EnsureInfoKeysWrapper,
    EverythingToInfoWrapper,
    MegaWrapper,
    ResizeGoalWrapper,
)

from .world import World

_ENVX_WRAPPERS = (
    MegaWrapper,
    ResizeGoalWrapper,
    EnsureInfoKeysWrapper,
    EverythingToInfoWrapper,
    AddPixelsWrapper,
    TimeLimit,
    OrderEnforcing,
    PassiveEnvChecker,
)


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
    ``tasks``. Planner tensors, actions, and TwoRoom scores stay on the solver
    device.
    """

    model_names: tuple[str, ...]
    planned_actions: torch.Tensor
    planner_costs: torch.Tensor
    planner_variances: torch.Tensor
    environment_actions: torch.Tensor
    evaluations: tuple[EvaluationResults, ...]
    compiled: bool
    compile_error: str | None
    population_backend: str
    task_successes: torch.Tensor
    task_final_distances: torch.Tensor
    scores: torch.Tensor
    score_name: str

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
        """TwoRoom score per population member, shaped ``(P,)``."""
        return self.scores.detach().cpu().numpy()

    @property
    def success_rates(self) -> np.ndarray:
        """Success percentage per population member, ``(P,)``."""
        return np.asarray(
            [result.success_rate for result in self.evaluations],
            dtype=np.float64,
        )


class ManyWorlds:
    """Evaluate weight-bound TwoRoom Worlds in one envX population run.

    Each input ``World`` must already have a ``WorldModelPolicy`` backed by a
    compatible ``FastCEMSolver``. This preserves the invariant that one World
    owns one concrete model instance and weight set. ``evaluate`` replaces the
    individual solver calls with one population FastCEM graph and executes the
    complete plans in one open-loop envX rollout.

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
        self._validate_envx_worlds()
        self._envx_episode_limit = self._common_episode_limit()
        if find_spec('envx') is None:
            raise ImportError(
                'ManyWorlds requires envX. Install the pinned envX revision '
                'documented in docs/api/world.md.'
            )
        self.envs = self.worlds[0].envs
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
    ) -> Self:
        """Named constructor matching ``ManyWorlds.init(worlds=[...])``."""
        return cls(worlds=worlds, model_names=model_names)

    def _common_episode_limit(self) -> int:
        limits = {
            getattr(getattr(env, 'spec', None), 'max_episode_steps', None)
            for world in self.worlds
            for env in world.envs.envs
        }
        if None in limits or len(limits) != 1:
            raise ValueError(
                'envX requires every World task to use the same positive '
                'max_episode_steps'
            )
        limit = int(limits.pop())
        if limit < 1:
            raise ValueError(
                'envX requires every World task to use the same positive '
                'max_episode_steps'
            )
        return limit

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

    def _validate_envx_worlds(self) -> None:
        """Reject every simulator configuration outside the tested envX path."""
        for world_index, world in enumerate(self.worlds):
            for task_index, wrapped in enumerate(world.envs.envs):
                env_id = getattr(getattr(wrapped, 'spec', None), 'id', None)
                if env_id != 'swm/TwoRoom-v1':
                    raise ValueError(
                        'ManyWorlds currently supports only '
                        f'swm/TwoRoom-v1, got {env_id!r}'
                    )
                current = wrapped
                wrapper_types = []
                while isinstance(current, gym.Wrapper):
                    wrapper_types.append(type(current))
                    if type(current) not in _ENVX_WRAPPERS:
                        wrapper = (
                            f'{type(current).__module__}.'
                            f'{type(current).__qualname__}'
                        )
                        raise ValueError(
                            'ManyWorlds supports only the standard TwoRoom '
                            'wrapper stack; custom wrapper '
                            f'{wrapper} was found in World {world_index}, '
                            f'task {task_index}'
                        )
                    current = current.env
                if tuple(wrapper_types) != _ENVX_WRAPPERS:
                    actual = ' -> '.join(
                        wrapper.__qualname__ for wrapper in wrapper_types
                    )
                    raise ValueError(
                        'ManyWorlds requires the exact standard TwoRoom '
                        f'wrapper stack, got {actual}'
                    )
                if type(current) is not TwoRoomEnv:
                    env_type = (
                        f'{type(current).__module__}.'
                        f'{type(current).__qualname__}'
                    )
                    raise ValueError(
                        'ManyWorlds currently supports only '
                        f'swm/TwoRoom-v1, got {env_type}'
                    )

        action_space = self.worlds[0].envs.single_action_space
        if action_space.shape != (2,):
            raise ValueError(
                'envX TwoRoom requires an unmodified action shape of (2,)'
            )

    @property
    def population_size(self) -> int:
        return len(self.worlds)

    @property
    def num_tasks(self) -> int:
        return self.worlds[0].num_envs

    @property
    def simulator_count(self) -> int:
        """Number of independently simulated population/task states."""
        return self.population_size * self.num_tasks

    @property
    def bootstrap_simulator_count(self) -> int:
        """Gym envs stepped/reset for observations by this evaluator."""
        return self.num_tasks

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
            transformed = process['action'].inverse_transform(
                actions.reshape(-1, raw_dim)
            )
            if not torch.is_tensor(transformed):
                raise TypeError(
                    'ManyWorlds action inverse_transform must return a '
                    'torch.Tensor so the envX hot path remains device-resident'
                )
            if transformed.device != actions.device:
                raise ValueError(
                    'ManyWorlds action inverse_transform must preserve the '
                    f'input device {actions.device}, got {transformed.device}'
                )
            actions = transformed.reshape(
                population, tasks, eval_budget, raw_dim
            )
        return actions.reshape(population, tasks, eval_budget, *raw_shape)

    def _notify_plans(
        self,
        info: dict[str, Any],
        output: dict[str, Any],
        *,
        selected_horizon: int,
    ) -> None:
        """Preserve each policy's ordinary post-plan observer contract."""
        for world_index, policy in enumerate(self.policies):
            controller_input = {
                key: value[world_index] for key, value in info.items()
            }
            actions = output['actions'][world_index]
            solver_output = {
                'actions': actions,
                'costs': output['costs'][world_index],
                'mean': [output['mean'][0][world_index]],
                'var': [output['var'][0][world_index]],
                'compiled': output['compiled'],
                'compile_error': output['compile_error'],
                'population_backend': output['population_backend'],
                'solve_time_seconds': output['solve_time_seconds'],
            }
            event = {
                'env_indices': tuple(range(self.num_tasks)),
                'controller_input': controller_input,
                'solver_output': solver_output,
                'selected_plan': actions[:, :selected_horizon],
            }
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
    def _score_two_rooms(
        protocol: EvaluationProtocol,
        successes: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        """Aggregate per-task TwoRoom outcomes on the solver device."""
        score_name = str(dict(protocol.metadata).get('score', 'distance'))
        if score_name == 'success':
            return score_name, successes.float().mean(dim=1)
        if score_name == 'distance':
            image_diagonal = float(np.hypot(224, 224))
            return score_name, 1.0 - distances.float().mean(
                dim=1
            ) / image_diagonal
        raise ValueError(
            "TwoRoom protocol metadata score must be 'distance' or 'success', "
            f'got {score_name!r}'
        )

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
        self._notify_plans(
            info,
            output,
            selected_horizon=self.config.horizon,
        )
        actions = self._environment_plan_tensor(planned, eval_budget)

        from .jax_two_rooms import JaxTwoRoomsRollout

        episode_limit = min(eval_budget, self._envx_episode_limit)
        if (
            self._jax_rollout is None
            or self._jax_rollout.eval_budget != eval_budget
            or self._jax_rollout.max_episode_steps != episode_limit
        ):
            self._jax_rollout = JaxTwoRoomsRollout(
                population_size=self.population_size,
                num_tasks=self.num_tasks,
                eval_budget=eval_budget,
                max_episode_steps=episode_limit,
            )
        initial_state = self._jax_rollout.initial_state(self.worlds[0])
        outcome = self._jax_rollout(initial_state, actions)
        score_name, scores = self._score_two_rooms(
            protocol, outcome.successes, outcome.final_distances
        )
        success_values = outcome.successes.detach().cpu().numpy().astype(bool)
        distance_values = outcome.final_distances.detach().cpu().numpy()
        returns = outcome.returns.detach().cpu().numpy()
        lengths = outcome.lengths.detach().cpu().numpy()
        path_costs = outcome.path_costs.detach().cpu().numpy()
        control_costs = outcome.control_costs.detach().cpu().numpy()
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
                        length=int(lengths[population_index, task_index]),
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
                model_queries=(),
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

        There is no per-World solver call or model loop: CEM, model ranking,
        action postprocessing, and TwoRoom execution stay tensorized across
        the population on the accelerator.
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
        return self._evaluate_envx(
            protocol,
            solver=solver,
            eval_budget=eval_budget,
            record=record,
        )


__all__ = ['ManyWorlds', 'ManyWorldsEvaluationResults']
