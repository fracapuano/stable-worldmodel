"""Fast CEM with a device-resident refinement loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from stable_worldmodel.planning.tensor_cost import (
    FunctionalPopulationCost,
    is_population_tensor_cost,
    is_tensor_planning_cost,
)
from stable_worldmodel.protocols import Costable

from .callbacks import Callback
from .cem import CEMSolver
from .utils import prepare_init_action

logger = logging.getLogger(__name__)


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


class _CEMLoop(nn.Module):
    """The fixed-shape portion captured by ``torch.compile``."""

    def __init__(self, cost: nn.Module, n_steps: int, topk: int) -> None:
        super().__init__()
        self.cost, self.n_steps, self.topk = cost, n_steps, topk

    def _step(self, mean, std, noise, *prepared):
        candidates = torch.addcmul(mean[:, None], noise, std[:, None])
        candidates[:, 0].copy_(mean)
        costs = self.cost(candidates, *prepared)
        values, indices = costs.topk(self.topk, dim=1, largest=False)
        indices = indices[:, :, None, None].expand(
            -1, -1, candidates.size(2), candidates.size(3)
        )
        elite_std, elite_mean = torch.std_mean(
            candidates.gather(1, indices), dim=1
        )
        return elite_mean, elite_std, values.mean(1)

    def eager(self, mean, std, noise, *prepared):
        cost = mean.new_zeros(mean.shape[:-2])
        for step_noise in noise:
            mean, std, cost = self._step(mean, std, step_noise, *prepared)
        return mean, std, cost

    def forward(self, mean, std, noise, *prepared):
        step = torch.zeros((), dtype=torch.int64, device=mean.device)
        cost = mean.new_zeros(mean.shape[:-2])

        def cond(step, _mean, _std, _cost):
            return step < self.n_steps

        def body(step, mean, std, _cost):
            step_noise = noise.index_select(0, step[None]).squeeze(0)
            mean, std, cost = self._step(mean, std, step_noise, *prepared)
            return step + 1, mean, std, cost

        _, mean, std, cost = torch.while_loop(
            cond, body, (step, mean, std, cost)
        )
        return mean, std, cost


class _PopulationCEMLoop(_CEMLoop):
    """Fixed-shape CEM loop over ``(population, tasks, samples, ...)``."""

    def _step(self, mean, std, noise, *prepared):
        # Common random numbers across models, independent distributions.
        candidates = torch.addcmul(
            mean[:, :, None], noise[None], std[:, :, None]
        )
        candidates[:, :, 0].copy_(mean)
        costs = self.cost(candidates, *prepared)
        values, indices = costs.topk(self.topk, dim=2, largest=False)
        indices = indices[:, :, :, None, None].expand(
            -1, -1, -1, candidates.size(3), candidates.size(4)
        )
        elite_std, elite_mean = torch.std_mean(
            candidates.gather(2, indices), dim=2
        )
        return elite_mean, elite_std, values.mean(2)

    def forward(self, mean, std, noise, *prepared):
        # ``functional_call``/``vmap`` cannot currently be nested inside a
        # ``torch.while_loop`` capture. n_steps is static, so Dynamo safely
        # unrolls the population path into one graph.
        return self.eager(mean, std, noise, *prepared)


class FastCEMSolver(CEMSolver):
    """Device-resident CEM for explicit pure-tensor planning costs.

    Tensor costs use the same constructor and :meth:`solve` contract as
    :class:`CEMSolver`. Population-capable tensor costs can use
    :meth:`solve_population` with independent cost modules. Complete cost state
    and CEM tensors carry a population axis, so all candidate slices are ranked
    by one functional population call.
    Incompatible costs and per-iteration callbacks are rejected at construction
    rather than silently running reference CEM.
    """

    def __init__(
        self,
        cost: Costable | nn.Module,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks: list[Callback] | None = None,
        *,
        compile_kernel: bool | None = None,
        compile_mode: str = 'reduce-overhead',
        compile_backend: str | Callable[..., Any] | None = None,
        compile_fallback: bool = True,
    ) -> None:
        if n_steps < 1 or num_samples < 2 or not 2 <= topk <= num_samples:
            raise ValueError('invalid CEM steps, samples, or top-k')
        if callbacks:
            raise ValueError('FastCEMSolver does not support callbacks')
        super().__init__(
            cost,
            batch_size,
            num_samples,
            var_scale,
            n_steps,
            topk,
            device,
            seed,
            callbacks,
        )
        self.compile_kernel = compile_kernel
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.compile_fallback = compile_fallback
        self._tensor_cost = self._adapt_cost(cost)
        self._loop = _CEMLoop(self._tensor_cost, n_steps, topk)
        self._population_loop: _PopulationCEMLoop | None = None
        self._population_costs: tuple[nn.Module, ...] = ()
        self._compiled_loop = None
        self._compiled_population_loop = None
        self._compile_error = None
        self._population_compile_error = None

    @staticmethod
    def _adapt_cost(cost):
        if is_tensor_planning_cost(cost):
            return cost
        raise TypeError(
            'FastCEMSolver requires a tensor cost implementing prepare(...) '
            'and forward(...); use CEMSolver for other costs or provide a '
            'model-specific tensor-cost adapter'
        )

    @property
    def tensor_cost(self) -> nn.Module:
        """The pure-tensor planning cost executed by this solver."""
        return self._tensor_cost

    @property
    def compilation_enabled(self) -> bool:
        return (
            self.compile_kernel
            if self.compile_kernel is not None
            else torch.device(self.device).type == 'cuda'
        )

    @property
    def compile_error(self) -> str | None:
        return self._population_compile_error or self._compile_error

    @property
    def population_backend(self) -> str | None:
        """Active population execution backend, if costs are bound."""
        if self._population_loop is None:
            return None
        return self._population_loop.cost.backend

    def prepare(self, info: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        """Encode inputs once and move them to the planning device."""
        return self._tensor_cost.prepare(
            info,
            device=self.device,
            dtype=self.dtype,
            action_dim=self.action_dim,
        )

    def _bind_population_costs(
        self, costs: Sequence[nn.Module]
    ) -> FunctionalPopulationCost:
        """Bind independent tensor costs to one population CEM graph."""
        costs = tuple(costs)
        if not costs:
            raise ValueError('tensor cost population cannot be empty')
        if not is_population_tensor_cost(self._tensor_cost):
            raise TypeError(
                'population FastCEM requires a tensor cost implementing '
                'prepare_population(...)'
            )
        if costs[0] is not self._tensor_cost:
            raise ValueError(
                'the first population tensor cost must be the solver cost'
            )
        if (
            self._population_loop is not None
            and self._population_loop.cost.matches(costs)
        ):
            assert self._population_loop is not None
            return self._population_loop.cost

        expected_device = torch.device(self.device)
        for index, cost in enumerate(costs):
            if not is_population_tensor_cost(cost):
                raise TypeError(
                    'population members must be population tensor costs'
                )
            if cost.training:
                raise ValueError(
                    f'population tensor cost {index} must be in evaluation mode'
                )
            wrong_device = next(
                (
                    value.device
                    for value in (
                        *tuple(cost.parameters()),
                        *tuple(cost.buffers()),
                    )
                    if not _same_device(value.device, expected_device)
                ),
                None,
            )
            if wrong_device is not None:
                raise ValueError(
                    f'population tensor cost {index} must reside on '
                    f'{expected_device}, got {wrong_device}'
                )

        population_cost = FunctionalPopulationCost(costs)
        self._population_loop = _PopulationCEMLoop(
            population_cost, self.n_steps, self.topk
        )
        self._population_costs = costs
        self._compiled_population_loop = None
        self._population_compile_error = None
        return population_cost

    def prepare_population(
        self,
        info: dict[str, Any],
        costs: Sequence[nn.Module],
    ) -> tuple[torch.Tensor, ...]:
        """Stack tensor-cost state and prepare the population input batch."""
        cost = self._bind_population_costs(costs)
        state = cost.stack_state(costs)
        return cost.prepare(
            info,
            state,
            device=self.device,
            dtype=self.dtype,
            action_dim=self.action_dim,
        )

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Pre-generate the complete deterministic CEM noise stream."""
        shape = (
            batch_size,
            self.num_samples,
            self.horizon,
            self.action_dim,
        )
        return torch.stack(
            [
                torch.randn(
                    shape,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.n_steps)
            ]
        )

    def _batch_noise(self, batch_size, start, seeds, decisions):
        if seeds is None:
            return self.sample_noise(batch_size)
        return torch.stack(
            [
                torch.stack(
                    [
                        self._sample_task_candidates(
                            seeds, decisions, task, step
                        )
                        for task in range(start, start + batch_size)
                    ]
                )
                for step in range(self.n_steps)
            ]
        )

    def _run_loop(
        self,
        loop,
        compiled_attribute: str,
        error_attribute: str,
        label: str,
        *args,
    ):
        error = getattr(self, error_attribute)
        if not self.compilation_enabled or error is not None:
            return (*loop.eager(*args), False)
        try:
            compiled = getattr(self, compiled_attribute)
            if compiled is None:
                options = {
                    'fullgraph': True,
                    'dynamic': False,
                    'mode': self.compile_mode,
                }
                if self.compile_backend is not None:
                    options['backend'] = self.compile_backend
                compiled = torch.compile(loop, **options)
                setattr(self, compiled_attribute, compiled)
            return (*compiled(*args), True)
        except Exception as exc:
            if not self.compile_fallback:
                raise
            error = f'{type(exc).__name__}: {exc}'
            setattr(self, error_attribute, error)
            logger.warning(f'{label} compilation failed; using eager: {error}')
            return (*loop.eager(*args), False)

    def _run(self, mean, std, noise, prepared):
        return self._run_loop(
            self._loop,
            '_compiled_loop',
            '_compile_error',
            'FastCEM',
            mean,
            std,
            noise,
            *prepared,
        )

    def _initial_population_distribution(
        self,
        population: int,
        tasks: int,
        init_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if init_action is not None:
            init_action = init_action.to(device=self.device, dtype=self.dtype)
            if init_action.ndim == 3:
                init_action = init_action[None].expand(population, -1, -1, -1)
            if init_action.ndim != 4 or init_action.shape[:2] != (
                population,
                tasks,
            ):
                raise ValueError(
                    'init_action must have shape (tasks, time, action_dim) or '
                    '(population, tasks, time, action_dim)'
                )
            if init_action.size(-1) != self.action_dim:
                raise ValueError('init_action has the wrong action dimension')
            init_action = init_action[:, :, : self.horizon]
            init_action = init_action.flatten(0, 1)

        mean, std = self.init_action_distrib(population * tasks, init_action)
        mean = mean.to(device=self.device, dtype=self.dtype)
        std = std.to(device=self.device, dtype=self.dtype)
        return (
            mean.unflatten(0, (population, tasks)),
            std.unflatten(0, (population, tasks)),
        )

    def _run_population(self, mean, std, noise, prepared):
        if self._population_loop is None:
            raise RuntimeError('population tensor costs have not been bound')
        return self._run_loop(
            self._population_loop,
            '_compiled_population_loop',
            '_population_compile_error',
            'FastCEM population',
            mean,
            std,
            noise,
            *prepared,
        )

    @torch.inference_mode()
    def solve_tensors(self, prepared, *, noise=None, init_action=None):
        """Solve one prepared batch without transferring results to the CPU."""
        batch_size = prepared[0].size(0)
        if noise is None:
            noise = self.sample_noise(batch_size)
        expected = (
            self.n_steps,
            batch_size,
            self.num_samples,
            self.horizon,
            self.action_dim,
        )
        if noise.shape != expected:
            raise ValueError(f'expected noise shape {expected}')

        mean, std = self.init_action_distrib(batch_size, init_action)
        mean = mean.to(device=self.device, dtype=self.dtype)
        std = std.to(device=self.device, dtype=self.dtype)
        mean, std, costs, compiled = self._run(mean, std, noise, prepared)
        return {
            'actions': mean,
            'costs': costs,
            'mean': [mean],
            'var': [std],
            'compiled': compiled,
            'compile_error': self._compile_error,
        }

    @torch.inference_mode()
    def solve_population_tensors(
        self,
        costs: Sequence[nn.Module],
        prepared: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve a cost population without transferring results to the CPU."""
        device = torch.device(self.device)
        costs = tuple(costs)
        population = len(costs)
        population_cost = self._bind_population_costs(costs)
        if not prepared or any(
            not torch.is_tensor(value) for value in prepared
        ):
            raise TypeError('prepared must be a non-empty tuple of tensors')
        tasks = population_cost.validate_prepared(prepared)
        if any(not _same_device(value.device, device) for value in prepared):
            raise ValueError(
                'prepared inputs must already reside on the solver device'
            )
        if noise is None:
            noise = self.sample_noise(tasks)
        expected = (
            self.n_steps,
            tasks,
            self.num_samples,
            self.horizon,
            self.action_dim,
        )
        if tuple(noise.shape) != expected:
            raise ValueError(f'expected population noise shape {expected}')
        if not _same_device(noise.device, device):
            raise ValueError('noise must already reside on the solver device')

        mean, std = self._initial_population_distribution(
            population, tasks, init_action
        )
        mean, std, costs, compiled = self._run_population(
            mean, std, noise, prepared
        )
        return {
            'actions': mean,
            'costs': costs,
            'mean': [mean],
            'var': [std],
            'compiled': compiled,
            'compile_error': self._population_compile_error,
            'population_backend': self.population_backend,
        }

    @torch.inference_mode()
    def solve_population(
        self,
        info: dict[str, Any],
        costs: Sequence[nn.Module],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve one population using this solver's existing configuration."""
        return self.solve_population_tensors(
            costs,
            self.prepare_population(info, costs),
            noise=noise,
            init_action=init_action,
        )

    @torch.inference_mode()
    def solve(
        self, info: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve through the standard planner interface."""
        started = time.perf_counter()
        total = len(next(iter(info.values())))
        init_action = prepare_init_action(
            self.cost,
            info,
            init_action,
            self.horizon,
            total,
            self.action_dim,
        )
        outputs = []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_info = {key: value[start:end] for key, value in info.items()}
            outputs.append(
                self.solve_tensors(
                    self.prepare(batch_info),
                    noise=self._batch_noise(
                        end - start,
                        start,
                        info.get('controller_seed'),
                        info.get('step_idx'),
                    ),
                    init_action=init_action[start:end],
                )
            )

        actions = torch.cat([output['actions'] for output in outputs])
        std = torch.cat([output['var'][0] for output in outputs])
        costs = torch.cat([output['costs'] for output in outputs])
        actions, std = actions.cpu(), std.cpu()
        return {
            'actions': actions,
            'costs': costs.cpu().tolist(),
            'mean': [actions],
            'var': [std],
            'solve_time_seconds': time.perf_counter() - started,
            'compiled': any(output['compiled'] for output in outputs),
            'compile_error': self._compile_error,
        }


__all__ = ['FastCEMSolver']
