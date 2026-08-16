"""Population-batched planning with serial Gymnasium realization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationResults
from stable_worldmodel.planning import (
    PopulationAcceleratedCEMSolver,
    PopulationLatentGoalCost,
)
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

from .world import World


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices while allowing an implicit accelerator index."""
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


@dataclass(frozen=True)
class ManyWorldsEvaluationResults:
    """Planner tensors and realized outcomes for a model population.

    ``planned_actions``, ``planner_costs``, and ``planner_variances`` remain on
    the planner device. ``environment_actions`` is the single bulk host copy
    consumed by the serial Gymnasium rollouts.
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


class _OpenLoopPlanPolicy:
    """Serve one fixed plan batch through the existing ``World`` loop."""

    def __init__(self, actions: np.ndarray, solver: Any) -> None:
        self.actions = np.asarray(actions)
        self.solver = solver
        self.env: Any | None = None
        self._step = 0

    def set_env(self, env: Any) -> None:
        if self.actions.shape[0] != env.num_envs:
            raise ValueError(
                'plan task batch does not match the environment pool: '
                f'{self.actions.shape[0]} != {env.num_envs}'
            )
        self.env = env
        self.reset_state()

    def reset_state(self, env_ids: list[int] | None = None) -> None:
        if env_ids is not None:
            raise ValueError('open-loop plans only support full-batch reset')
        self._step = 0

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        del info_dict, kwargs
        if self._step >= self.actions.shape[1]:
            raise RuntimeError('evaluation budget exceeded the fixed plan')
        action = self.actions[:, self._step].copy()
        self._step += 1
        return action


class ManyWorlds(World):
    """Evaluate compatible world models with one population-batched CEM.

    The models share the observation encoder, action encoder, projection head,
    and every other non-population tensor. Only the parameter names exposed by
    ``population_predictor_parameter_names`` may differ. CEM maintains an
    independent distribution for every model and task while sharing common
    random numbers across models.

    This first implementation plans once, transfers the selected action plans
    to the host in one batch, and then realizes each model's plans serially
    through the inherited Gymnasium ``World`` evaluation loop. The evaluation
    budget must fit within one receding-horizon plan.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        env_name: str,
        *,
        config: PlanConfig,
        solver: PopulationAcceleratedCEMSolver | None = None,
        cem_kwargs: Mapping[str, Any] | None = None,
        parameter_names: Sequence[str] | None = None,
        model_names: Sequence[str] | None = None,
        population_batch_size: int | None = None,
        process: Mapping[str, Any] | None = None,
        transform: Mapping[str, Any] | None = None,
        history_keys: tuple[str, ...] = ('pixels',),
        num_envs: int = 1,
        **world_kwargs: Any,
    ) -> None:
        models = tuple(models)
        if not models:
            raise ValueError('ManyWorlds requires at least one model')
        if not all(isinstance(model, nn.Module) for model in models):
            raise TypeError('every model must be a torch.nn.Module')
        if any(type(model) is not type(models[0]) for model in models[1:]):
            raise TypeError('all models must have the same concrete type')
        if solver is not None and cem_kwargs:
            raise ValueError('pass either solver or cem_kwargs, not both')

        for model in models:
            model.eval()
        self.models = models
        self.config = config
        if config.receding_horizon > config.horizon:
            raise ValueError(
                'receding_horizon cannot exceed the CEM planning horizon: '
                f'{config.receding_horizon} > {config.horizon}'
            )
        if population_batch_size is None:
            population_batch_size = len(models)
        if (
            isinstance(population_batch_size, bool)
            or not isinstance(population_batch_size, int)
            or population_batch_size < 1
        ):
            raise ValueError(
                'population_batch_size must be a positive integer'
            )
        self.population_batch_size = min(population_batch_size, len(models))
        self.process = dict(process or {})
        self.transform = dict(transform or {})

        if solver is None:
            kwargs = dict(cem_kwargs or {})
            try:
                first_parameter = next(models[0].parameters())
            except StopIteration as exc:
                raise ValueError(
                    'population models must have parameters'
                ) from exc
            kwargs.setdefault('device', first_parameter.device)
            solver = PopulationAcceleratedCEMSolver(
                PopulationLatentGoalCost(models[0]), **kwargs
            )
        if not isinstance(solver, PopulationAcceleratedCEMSolver):
            raise TypeError('solver must be a PopulationAcceleratedCEMSolver')
        self.solver = solver

        if parameter_names is None:
            parameter_names = getattr(
                models[0], 'population_predictor_parameter_names', None
            )
        if parameter_names is None:
            parameter_names = getattr(
                solver.cost, 'predictor_parameter_names', None
            )
        if parameter_names is None:
            raise TypeError(
                'models or solver cost must expose population parameter names'
            )
        self.parameter_names = tuple(parameter_names)
        if not self.parameter_names:
            raise ValueError('population parameter names cannot be empty')
        cost_parameter_names = getattr(
            solver.cost, 'predictor_parameter_names', None
        )
        if cost_parameter_names is not None and tuple(
            cost_parameter_names
        ) != tuple(self.parameter_names):
            raise ValueError(
                'parameter_names must use the order required by the solver cost'
            )

        if model_names is None:
            model_names = tuple(
                f'model-{index}' for index in range(len(models))
            )
        else:
            model_names = tuple(str(name) for name in model_names)
        if len(model_names) != len(models):
            raise ValueError('model_names must contain one name per model')
        if len(set(model_names)) != len(model_names):
            raise ValueError('model_names must be unique')
        self.model_names = tuple(model_names)

        self._validate_model_population()
        super().__init__(env_name, num_envs=num_envs, **world_kwargs)
        self._planner_policy = WorldModelPolicy(
            solver=solver,
            config=config,
            process=self.process,
            transform=self.transform,
            history_keys=history_keys,
        )
        self.set_policy(self._planner_policy)

    @property
    def population_size(self) -> int:
        return len(self.models)

    def _validate_model_population(self) -> None:
        base_state = self.models[0].state_dict()
        base_parameters = dict(self.models[0].named_parameters())
        missing = sorted(set(self.parameter_names) - set(base_parameters))
        if missing:
            raise KeyError(
                'population parameter names are absent from the first model: '
                + ', '.join(missing[:5])
            )
        solver_device = torch.device(self.solver.device)
        wrong_device = [
            name
            for name, value in base_parameters.items()
            if not _same_device(value.device, solver_device)
        ]
        if wrong_device:
            raise ValueError(
                'models must already reside on the solver device; '
                f'{wrong_device[0]!r} is on '
                f'{base_parameters[wrong_device[0]].device}, expected '
                f'{solver_device}'
            )

        population_names = set(self.parameter_names)
        for index, model in enumerate(self.models[1:], start=1):
            state = model.state_dict()
            if state.keys() != base_state.keys():
                raise ValueError(
                    f'model {index} has a different state-dict schema'
                )
            parameters = dict(model.named_parameters())
            for name in self.parameter_names:
                actual = parameters[name]
                expected = base_parameters[name]
                if actual.shape != expected.shape:
                    raise ValueError(
                        f'model {index} parameter {name!r} has shape '
                        f'{tuple(actual.shape)}, expected {tuple(expected.shape)}'
                    )
                if (
                    actual.dtype != expected.dtype
                    or actual.device != expected.device
                ):
                    raise ValueError(
                        f'model {index} parameter {name!r} must use '
                        f'{expected.device}/{expected.dtype}'
                    )

            changed_shared = [
                name
                for name, expected in base_state.items()
                if name not in population_names
                and not torch.equal(state[name], expected)
            ]
            if changed_shared:
                raise ValueError(
                    'ManyWorlds currently supports predictor-only variation; '
                    f'model {index} changes shared tensor '
                    f'{changed_shared[0]!r}'
                )

    def _stack_population_parameters(
        self, start: int, end: int, *, pad_to: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        by_model = [
            dict(model.named_parameters()) for model in self.models[start:end]
        ]
        stacked = []
        for name in self.parameter_names:
            values = [parameters[name].detach() for parameters in by_model]
            value = torch.stack(values)
            if pad_to is not None and value.size(0) < pad_to:
                padding = value[-1:].expand(
                    pad_to - value.size(0), *value.shape[1:]
                )
                value = torch.cat([value, padding], dim=0)
            if value.dtype != self.solver.dtype:
                value = value.to(dtype=self.solver.dtype)
            stacked.append(value)
        return tuple(stacked)

    def _solve_population(
        self,
        prepared: tuple[torch.Tensor, ...],
        noise: torch.Tensor,
    ) -> dict[str, Any]:
        outputs = []
        chunk_size = self.population_batch_size
        for start in range(0, self.population_size, chunk_size):
            end = min(start + chunk_size, self.population_size)
            parameters = self._stack_population_parameters(
                start, end, pad_to=chunk_size
            )
            output = self.solver.solve_population_tensors(
                parameters, prepared, noise=noise
            )
            actual_size = end - start
            outputs.append(
                {
                    'actions': output['actions'][:actual_size],
                    'costs': output['costs'][:actual_size],
                    'var': output['var'][:actual_size],
                    'compiled': output['compiled'],
                }
            )

        return {
            'actions': torch.cat(
                [output['actions'] for output in outputs], dim=0
            ),
            'costs': torch.cat([output['costs'] for output in outputs], dim=0),
            'var': torch.cat([output['var'] for output in outputs], dim=0),
            'compiled': all(output['compiled'] for output in outputs),
            'compile_error': self.solver.compile_error,
        }

    @property
    def actions_per_plan(self) -> int:
        """Number of Gymnasium steps emitted by one planning decision."""
        return self.config.receding_horizon * self.config.action_block

    def _validate_eval_budget(self, eval_budget: int) -> None:
        if eval_budget > self.actions_per_plan:
            raise ValueError(
                'ManyWorlds currently requires the evaluation budget to fit '
                f'in one plan: {eval_budget} > {self.actions_per_plan}'
            )

    def _protocol_noise(
        self, protocol: EvaluationProtocol, *, decision_index: int = 0
    ) -> torch.Tensor:
        iterations = []
        for step in range(self.solver.n_steps):
            rows = []
            for task in protocol.tasks:
                stream_seed = (
                    task.controller_seed
                    + 1_000_003 * decision_index
                    + 10_007 * step
                ) % (2**63 - 1)
                generator = torch.Generator(
                    device=self.solver.device
                ).manual_seed(stream_seed)
                rows.append(
                    torch.randn(
                        self.solver.num_samples,
                        self.solver.horizon,
                        self.solver.action_dim,
                        generator=generator,
                        device=self.solver.device,
                        dtype=self.solver.dtype,
                    )
                )
            iterations.append(torch.stack(rows))
        return torch.stack(iterations)

    @torch.inference_mode()
    def plan(
        self,
        protocol: EvaluationProtocol,
        *,
        eval_budget: int,
        noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Plan for every model and task without transferring solver output."""
        self._validate_eval_budget(eval_budget)
        self._reset_from_protocol(protocol, eval_budget)
        prepared_info = self._planner_policy._prepare_info(self.infos)
        if noise is None:
            noise = self._protocol_noise(protocol)
        prepared = self.solver.cost.prepare(
            prepared_info,
            device=self.solver.device,
            dtype=self.solver.dtype,
            action_dim=self.solver.action_dim,
        )
        return self._solve_population(prepared, noise)

    def _environment_actions(
        self, planned_actions: torch.Tensor, eval_budget: int
    ) -> np.ndarray:
        self._validate_eval_budget(eval_budget)

        population, tasks = planned_actions.shape[:2]
        raw_shape = self.envs.single_action_space.shape
        raw_dim = int(np.prod(raw_shape))
        selected = planned_actions[:, :, : self.config.receding_horizon]
        selected = selected.reshape(
            population, tasks, self.actions_per_plan, raw_dim
        )[:, :, :eval_budget]
        selected = selected.detach()
        if selected.dtype == torch.bfloat16:
            selected = selected.float()
        actions = selected.cpu().numpy()
        if 'action' in self.process:
            flat = actions.reshape(-1, raw_dim)
            flat = self.process['action'].inverse_transform(flat)
            actions = np.asarray(flat).reshape(actions.shape)
        return actions.reshape(population, tasks, eval_budget, *raw_shape)

    @torch.inference_mode()
    def evaluate(
        self,
        *,
        protocol: EvaluationProtocol,
        eval_budget: int,
        noise: torch.Tensor | None = None,
        record: bool = False,
    ) -> ManyWorldsEvaluationResults:
        """Plan in one population batch, then realize plans model by model."""
        planner_output = self.plan(
            protocol, eval_budget=eval_budget, noise=noise
        )
        environment_actions = self._environment_actions(
            planner_output['actions'], eval_budget
        )

        evaluations = []
        try:
            for model_name, actions in zip(
                self.model_names, environment_actions, strict=True
            ):
                self.set_policy(_OpenLoopPlanPolicy(actions, self.solver))
                evaluations.append(
                    super().evaluate(
                        protocol=protocol,
                        eval_budget=eval_budget,
                        record=record,
                        backend=model_name,
                    )
                )
        finally:
            self.set_policy(self._planner_policy)

        return ManyWorldsEvaluationResults(
            model_names=self.model_names,
            planned_actions=planner_output['actions'],
            planner_costs=planner_output['costs'],
            planner_variances=planner_output['var'],
            environment_actions=environment_actions,
            evaluations=tuple(evaluations),
            compiled=bool(planner_output['compiled']),
            compile_error=planner_output['compile_error'],
        )


__all__ = ['ManyWorlds', 'ManyWorldsEvaluationResults']
