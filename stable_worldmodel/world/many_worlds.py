"""Closed-loop evaluation of multiple weight-bound Worlds with FastCEM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from stable_worldmodel.buffer import HistoryBuffer
from stable_worldmodel.evaluation import (
    EpisodeResult,
    EvaluationProtocol,
    EvaluationResults,
    StepRecord,
)
from stable_worldmodel.planning import FastCEMSolver
from stable_worldmodel.policy import ACTION_HISTORY_KEY, WorldModelPolicy

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
    ``tasks``. Planner tensors stay on the solver device; realized environment
    actions are copied to the host once per planning call and accumulated here.
    """

    model_names: tuple[str, ...]
    planned_actions: torch.Tensor
    planner_costs: torch.Tensor
    planner_variances: torch.Tensor
    environment_actions: np.ndarray
    evaluations: tuple[EvaluationResults, ...]
    compiled: bool
    compile_error: str | None

    @property
    def population_size(self) -> int:
        return len(self.model_names)

    @property
    def planning_calls(self) -> int:
        return self.planned_actions.size(0)


class ManyWorlds:
    """Compose weight-bound :class:`World` instances into one population run.

    Each input ``World`` must already have a ``WorldModelPolicy`` backed by a
    compatible ``FastCEMSolver``. This preserves the invariant that one World
    owns one concrete model instance and weight set. ``evaluate`` replaces the
    individual solver calls with one population FastCEM graph at every MPC
    decision, then steps the real Gymnasium environments with the independently
    selected plans.

    The current population kernel supports LeWM populations whose predictor
    parameters may differ while encoder, action encoder, projections, buffers,
    policy preprocessing, action space, and MPC configuration are identical.
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

    def _validate_population(self, solver: FastCEMSolver) -> None:
        if not isinstance(solver, FastCEMSolver):
            raise TypeError('solver must be a FastCEMSolver')
        if solver._tensor_cost.model is not self.models[0]:
            raise ValueError("solver must use the first World's FastCEM model")

        names = solver.population_parameter_names
        base_state = self.models[0].state_dict()
        base_parameters = dict(self.models[0].named_parameters())
        missing = sorted(set(names) - set(base_parameters))
        if missing:
            raise KeyError(
                'population parameters are absent from the first model: '
                + ', '.join(missing[:5])
            )
        population_names = set(names)
        expected_device = torch.device(solver.device)

        for index, model in enumerate(self.models):
            state = model.state_dict()
            if state.keys() != base_state.keys():
                raise ValueError(
                    f'model {index} has a different state-dict schema'
                )
            parameters = dict(model.named_parameters())
            for name in names:
                value = parameters[name]
                reference = base_parameters[name]
                if value.shape != reference.shape:
                    raise ValueError(
                        f'model {index} parameter {name!r} has shape '
                        f'{tuple(value.shape)}, expected {tuple(reference.shape)}'
                    )
                if value.dtype != solver.dtype or not _same_device(
                    value.device, expected_device
                ):
                    raise ValueError(
                        f'model {index} parameter {name!r} must use '
                        f'{expected_device}/{solver.dtype}'
                    )
            changed_shared = next(
                (
                    name
                    for name, expected in base_state.items()
                    if name not in population_names
                    and not torch.equal(state[name], expected)
                ),
                None,
            )
            if changed_shared is not None:
                raise ValueError(
                    'ManyWorlds currently supports predictor-only variation; '
                    f'model {index} changes shared tensor {changed_shared!r}'
                )

    def _stack_parameters(
        self, solver: FastCEMSolver
    ) -> tuple[torch.Tensor, ...]:
        by_model = tuple(
            dict(model.named_parameters()) for model in self.models
        )
        return tuple(
            torch.stack(
                tuple(parameters[name].detach() for parameters in by_model)
            )
            for name in solver.population_parameter_names
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
        for world in self.worlds:
            world._reset_from_protocol(protocol, eval_budget)

    def _combined_info(self) -> dict[str, Any]:
        keys = self.worlds[0].infos.keys()
        if any(world.infos.keys() != keys for world in self.worlds[1:]):
            raise ValueError(
                'World info dictionaries must have identical keys'
            )
        combined = {}
        for key in keys:
            values = tuple(world.infos[key] for world in self.worlds)
            first = values[0]
            if torch.is_tensor(first):
                combined[key] = torch.cat(values, dim=0)
            elif isinstance(first, np.ndarray):
                combined[key] = np.concatenate(values, axis=0)
            elif isinstance(first, list):
                combined[key] = [item for value in values for item in value]
            else:
                combined[key] = first
        return combined

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

    def _step_worlds(self, actions: np.ndarray, alive: np.ndarray) -> None:
        for index, world in enumerate(self.worlds):
            world.actions = np.asarray(actions[index]).copy()
            mask = alive[index] if not alive[index].all() else None
            (
                _,
                world.rewards,
                world.terminateds,
                world.truncateds,
                world.infos,
            ) = world.envs.step(actions[index], mask=mask)

    @torch.inference_mode()
    def evaluate(
        self,
        protocol: EvaluationProtocol,
        *,
        solver: FastCEMSolver,
        eval_budget: int | None = None,
        record: bool = False,
    ) -> ManyWorldsEvaluationResults:
        """Evaluate all Worlds with one population FastCEM call per replan.

        This is the population analogue of
        ``tuple(world.evaluate(protocol=protocol) for world in worlds)``. Real
        Gym environments still execute on the host, but there is no per-model
        world-model or CEM forward loop: every refinement step carries a model
        population dimension on the accelerator.
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
        parameters = self._stack_parameters(solver)
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
                    parameters,
                    noise=noise,
                    init_action=next_init,
                )
                planned = output['actions']
                planned_actions.append(planned)
                planner_costs.append(output['costs'])
                planner_variances.append(output['var'][0])
                compiled.append(bool(output['compiled']))
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
        )


__all__ = ['ManyWorlds', 'ManyWorldsEvaluationResults']
