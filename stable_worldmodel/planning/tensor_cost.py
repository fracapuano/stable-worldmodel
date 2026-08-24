"""Model-agnostic tensor costs and functional population execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

_CUDA_GRID_Y_LIMIT = 65_535
PopulationCall = Callable[..., Any]


class TensorPlanningCost(nn.Module, ABC):
    """Pure-tensor planning cost understood by :class:`FastCEMSolver`.

    Implementations own every model-specific decision: which task inputs are
    required, how candidate-independent context is prepared, how a rollout is
    produced, and how that rollout is scored. The population executor only
    stacks module state and maps these tensor operations over population.
    """

    @abstractmethod
    def prepare(
        self,
        info: dict[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare one task batch for repeated candidate scoring."""

    @abstractmethod
    def prepare_population(
        self,
        info: dict[str, Any],
        *,
        call_population: PopulationCall,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare a population/task batch.

        ``call_population(module, *tensors)`` executes ``module`` once under
        ``functional_call``/``vmap`` using the complete stacked state of the
        population costs. An adapter uses it only when preparation itself
        depends on each member's parameters, such as observation encoding.
        Every returned tensor must start with ``(population, tasks)``.
        """


def is_tensor_planning_cost(cost: Any) -> bool:
    """Whether ``cost`` provides the scalar FastCEM tensor contract."""
    return (
        isinstance(cost, nn.Module)
        and callable(getattr(cost, 'prepare', None))
        and type(cost).forward is not nn.Module.forward
    )


def is_population_tensor_cost(cost: Any) -> bool:
    """Whether ``cost`` additionally supports population preparation."""
    return is_tensor_planning_cost(cost) and callable(
        getattr(cost, 'prepare_population', None)
    )


class FunctionalPopulationCost(nn.Module):
    """Evaluate independent tensor costs with one population ``vmap``.

    Members must use one tensor program (the same concrete cost type and state
    schema), while their complete parameters and persistent buffers may differ.
    No model architecture, rollout convention, or objective is assumed here.
    """

    def __init__(self, costs: Sequence[nn.Module]) -> None:
        super().__init__()
        costs = tuple(costs)
        if not costs:
            raise ValueError('tensor cost population cannot be empty')
        if not all(is_population_tensor_cost(cost) for cost in costs):
            raise TypeError(
                'every population member must implement the tensor cost '
                'prepare(...), prepare_population(...), and forward(...) '
                'contract'
            )
        first_type = type(costs[0])
        if any(type(cost) is not first_type for cost in costs[1:]):
            raise TypeError(
                'all population tensor costs must have the same concrete type'
            )
        self.template = costs[0]
        self._population_size = len(costs)
        self._members = costs
        self._parameter_names: tuple[str, ...] = ()
        self._buffer_names: tuple[str, ...] = ()

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def backend(self) -> str:
        return 'functional_vmap'

    @property
    def state_size(self) -> int:
        return len(self._parameter_names) + len(self._buffer_names)

    def matches(self, costs: Sequence[nn.Module]) -> bool:
        costs = tuple(costs)
        return len(costs) == len(self._members) and all(
            cost is member
            for cost, member in zip(costs, self._members, strict=True)
        )

    def stack_state(
        self, costs: Sequence[nn.Module]
    ) -> tuple[torch.Tensor, ...]:
        """Stack complete cost state on a leading population axis."""
        costs = tuple(costs)
        if len(costs) != self.population_size:
            raise ValueError(
                f'expected {self.population_size} tensor costs, got '
                f'{len(costs)}'
            )
        parameters, buffers = stack_module_state(costs)
        parameter_names = tuple(parameters)
        buffer_names = tuple(buffers)
        self._validate_state_schema(parameter_names, buffer_names)
        return tuple(
            value.detach()
            for value in (*parameters.values(), *buffers.values())
        )

    def _validate_state_schema(
        self,
        parameter_names: tuple[str, ...],
        buffer_names: tuple[str, ...],
    ) -> None:
        if self._parameter_names and parameter_names != self._parameter_names:
            raise ValueError('population parameter schemas changed')
        if self._buffer_names and buffer_names != self._buffer_names:
            raise ValueError('population buffer schemas changed')
        self._parameter_names = parameter_names
        self._buffer_names = buffer_names

    def _unpack_state(
        self, state: tuple[torch.Tensor, ...]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if len(state) != self.state_size:
            raise ValueError(
                f'expected {self.state_size} population state tensors, got '
                f'{len(state)}'
            )
        split = len(self._parameter_names)
        parameters = dict(
            zip(self._parameter_names, state[:split], strict=True)
        )
        buffers = dict(zip(self._buffer_names, state[split:], strict=True))
        return parameters, buffers

    @staticmethod
    def _call_population(module, parameters, buffers, *args):
        def call_member(member_parameters, member_buffers, *member_args):
            return functional_call(
                module,
                (member_parameters, member_buffers),
                member_args,
                strict=True,
            )

        return vmap(
            call_member,
            in_dims=(0, 0, *(0 for _ in args)),
            randomness='different',
        )(parameters, buffers, *args)

    def prepare(
        self,
        info: dict[str, Any],
        state: tuple[torch.Tensor, ...],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, ...]:
        """Delegate population preparation to the tensor-cost adapter."""
        parameters, buffers = self._unpack_state(state)

        def call_population(module, *args):
            return self._call_population(module, parameters, buffers, *args)

        prepared = self.template.prepare_population(
            info,
            call_population=call_population,
            device=device,
            dtype=dtype,
            action_dim=action_dim,
        )
        if not isinstance(prepared, tuple) or not prepared:
            raise TypeError(
                'prepare_population must return a non-empty tuple of tensors'
            )
        if any(not torch.is_tensor(value) for value in prepared):
            raise TypeError('population preparation returned a non-tensor')
        self._validate_prepared_inputs(prepared)
        return (*prepared, *state)

    def _split_prepared(
        self, prepared: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if self.state_size:
            if len(prepared) <= self.state_size:
                raise ValueError('prepared population inputs are missing')
            return prepared[: -self.state_size], prepared[-self.state_size :]
        return prepared, ()

    def _validate_prepared_inputs(
        self, prepared: tuple[torch.Tensor, ...]
    ) -> int:
        first = prepared[0]
        if first.ndim < 2 or first.size(0) != self.population_size:
            raise ValueError(
                'population-prepared tensors must start with population and '
                'task axes'
            )
        tasks = first.size(1)
        if any(
            value.ndim < 2
            or value.size(0) != self.population_size
            or value.size(1) != tasks
            for value in prepared
        ):
            raise ValueError(
                'all population-prepared tensors must share population and '
                'task axes'
            )
        return tasks

    def validate_prepared(self, prepared: tuple[torch.Tensor, ...]) -> int:
        """Validate a prepared tuple and return its task count."""
        inputs, state = self._split_prepared(prepared)
        tasks = self._validate_prepared_inputs(inputs)
        if any(
            value.ndim < 1 or value.size(0) != self.population_size
            for value in state
        ):
            raise ValueError(
                'stacked cost state must carry the population axis'
            )
        return tasks

    def forward(self, candidates, *prepared):
        inputs, state = self._split_prepared(prepared)
        parameters, buffers = self._unpack_state(state)
        try:
            return self._call_population(
                self.template,
                parameters,
                buffers,
                candidates,
                *inputs,
            )
        except RuntimeError as error:
            if 'CUDA error: invalid argument' not in str(error):
                raise
            population, tasks, samples = candidates.shape[:3]
            effective_batch = population * tasks * samples
            max_population = _CUDA_GRID_Y_LIMIT // (tasks * samples)
            max_samples = _CUDA_GRID_Y_LIMIT // (population * tasks)
            raise RuntimeError(
                'CUDA rejected the unsplit population FastCEM tensor-cost '
                'launch. A common cause is a batched kernel, including '
                'scaled-dot-product attention, exceeding CUDA grid limits: '
                f'population={population}, tasks={tasks}, samples={samples}, '
                f'effective_candidate_batch={effective_batch}. No automatic '
                'tiling or fallback is performed. For the usual CUDA grid-y '
                f'limit of {_CUDA_GRID_Y_LIMIT}, use population <= '
                f'{max_population} with the current tasks/samples, or samples '
                f'<= {max_samples} with the current population/tasks.'
            ) from error


__all__ = [
    'FunctionalPopulationCost',
    'PopulationCall',
    'TensorPlanningCost',
    'is_population_tensor_cost',
    'is_tensor_planning_cost',
]
