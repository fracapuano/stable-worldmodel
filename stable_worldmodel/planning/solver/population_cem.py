"""Population-parallel FastCEM for compatible LeWM world models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from loguru import logger as logging
from torch import nn

from stable_worldmodel.planning.tensor_cost import (
    _LeWMTerminalCost,
    _PopulationLeWMTerminalCost,
)

from .fast_cem import FastCEMSolver


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


class _PopulationCEMLoop(nn.Module):
    """Fixed-shape CEM loop over ``(population, tasks, samples, ...)``."""

    def __init__(self, cost: nn.Module, n_steps: int, topk: int) -> None:
        super().__init__()
        self.cost, self.n_steps, self.topk = cost, n_steps, topk

    def _step(self, mean, std, noise, parameters, prepared):
        # Noise has no population dimension on purpose: every model receives
        # common random numbers while retaining its own distribution state.
        candidates = torch.addcmul(
            mean[:, :, None], noise[None], std[:, :, None]
        )
        candidates[:, :, 0].copy_(mean)
        costs = self.cost(candidates, parameters, *prepared)
        values, indices = costs.topk(self.topk, dim=2, largest=False)
        indices = indices[:, :, :, None, None].expand(
            -1, -1, -1, candidates.size(3), candidates.size(4)
        )
        elite_std, elite_mean = torch.std_mean(
            candidates.gather(2, indices), dim=2
        )
        return elite_mean, elite_std, values.mean(2)

    def eager(self, mean, std, noise, parameters, *prepared):
        cost = mean.new_zeros(mean.shape[:2])
        for step_noise in noise:
            mean, std, cost = self._step(
                mean, std, step_noise, parameters, prepared
            )
        return mean, std, cost

    def forward(self, mean, std, noise, parameters, *prepared):
        step = torch.zeros((), dtype=torch.int64, device=mean.device)
        cost = mean.new_zeros(mean.shape[:2])

        def cond(step, _mean, _std, _cost):
            return step < self.n_steps

        def body(step, mean, std, _cost):
            step_noise = noise.index_select(0, step[None]).squeeze(0)
            mean, std, cost = self._step(
                mean, std, step_noise, parameters, prepared
            )
            return step + 1, mean, std, cost

        _, mean, std, cost = torch.while_loop(
            cond, body, (step, mean, std, cost)
        )
        return mean, std, cost


class PopulationFastCEMSolver(FastCEMSolver):
    """FastCEM with one independent distribution per model and task.

    The solver consumes a conventional
    ``ShootingCostEvaluator(lewm, GoalMSE())`` just like
    :class:`FastCEMSolver`. It adapts that cost to a population tensor kernel
    and receives the independently varying predictor parameters from
    :class:`~stable_worldmodel.world.ManyWorlds`.

    Candidate actions, predictor parameters, CEM state, and costs remain on the
    configured accelerator. All models use the same task-specific CEM noise,
    making paired comparisons independent of population order and size.
    """

    def __init__(
        self,
        cost,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks=None,
        *,
        compile_kernel: bool | None = None,
        compile_mode: str = 'reduce-overhead',
        compile_backend: str | Callable[..., Any] | None = None,
        compile_fallback: bool = True,
    ) -> None:
        super().__init__(
            cost=cost,
            batch_size=batch_size,
            num_samples=num_samples,
            var_scale=var_scale,
            n_steps=n_steps,
            topk=topk,
            device=device,
            seed=seed,
            callbacks=callbacks,
            compile_kernel=compile_kernel,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            compile_fallback=compile_fallback,
        )
        self._loop = _PopulationCEMLoop(self._tensor_cost, n_steps, topk)
        self._compiled_loop = None

    @staticmethod
    def _adapt_cost(cost):
        adapted = FastCEMSolver._adapt_cost(cost)
        if isinstance(adapted, _PopulationLeWMTerminalCost):
            return adapted
        if isinstance(adapted, _LeWMTerminalCost):
            return _PopulationLeWMTerminalCost(
                adapted.model, history_size=adapted.history_size
            )
        if callable(getattr(adapted, 'prepare_population', None)):
            return adapted
        raise TypeError(
            'PopulationFastCEMSolver requires population-aware tensor cost '
            'preparation or a compatible LeWM ShootingCostEvaluator'
        )

    @classmethod
    def from_fast_cem(
        cls, solver: FastCEMSolver, *, seed: int = 1234
    ) -> PopulationFastCEMSolver:
        """Copy the search and compilation configuration of a FastCEM solver."""
        if not isinstance(solver, FastCEMSolver):
            raise TypeError('solver must be a FastCEMSolver')
        return cls(
            cost=solver.cost,
            batch_size=solver.batch_size,
            num_samples=solver.num_samples,
            var_scale=solver.var_scale,
            n_steps=solver.n_steps,
            topk=solver.topk,
            device=solver.device,
            seed=seed,
            compile_kernel=solver.compile_kernel,
            compile_mode=solver.compile_mode,
            compile_backend=solver.compile_backend,
            compile_fallback=solver.compile_fallback,
        )

    @property
    def predictor_parameter_names(self) -> tuple[str, ...]:
        return tuple(self._tensor_cost.predictor_parameter_names)

    def prepare_population(
        self, info: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        """Encode all population/task inputs in one shared encoder call."""
        return self._tensor_cost.prepare_population(
            info,
            device=self.device,
            dtype=self.dtype,
            action_dim=self.action_dim,
        )

    def _initial_distribution(
        self,
        population: int,
        tasks: int,
        init_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (population, tasks, self.horizon, self.action_dim)
        if init_action is None:
            mean = torch.zeros(shape, device=self.device, dtype=self.dtype)
        else:
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
            remaining = self.horizon - init_action.size(2)
            mean = torch.cat(
                [
                    init_action,
                    init_action.new_zeros(
                        population, tasks, remaining, self.action_dim
                    ),
                ],
                dim=2,
            )
        return mean, torch.full_like(mean, self.var_scale)

    @staticmethod
    def _validate_parameters(
        parameters: tuple[torch.Tensor, ...], device: torch.device
    ) -> int:
        if not parameters:
            raise ValueError('predictor parameters cannot be empty')
        if any(not torch.is_tensor(value) for value in parameters):
            raise TypeError('predictor parameters must be a tuple of tensors')
        population = parameters[0].size(0)
        if population < 1 or any(
            value.size(0) != population for value in parameters
        ):
            raise ValueError(
                'all predictor parameters need one common population axis'
            )
        wrong_device = next(
            (
                value.device
                for value in parameters
                if not _same_device(value.device, device)
            ),
            None,
        )
        if wrong_device is not None:
            raise ValueError(
                'population parameters must already reside on the solver '
                f'device; expected {device}, got {wrong_device}'
            )
        return population

    def _run_population(self, mean, std, noise, parameters, prepared):
        if not self.compilation_enabled or self._compile_error is not None:
            return (
                *self._loop.eager(mean, std, noise, parameters, *prepared),
                False,
            )
        try:
            if self._compiled_loop is None:
                options = {
                    'fullgraph': True,
                    'dynamic': False,
                    'mode': self.compile_mode,
                }
                if self.compile_backend is not None:
                    options['backend'] = self.compile_backend
                self._compiled_loop = torch.compile(self._loop, **options)
            return (
                *self._compiled_loop(mean, std, noise, parameters, *prepared),
                True,
            )
        except Exception as exc:
            if not self.compile_fallback:
                raise
            self._compile_error = f'{type(exc).__name__}: {exc}'
            logging.warning(
                'Population FastCEM compilation failed; using eager: '
                f'{self._compile_error}'
            )
            return (
                *self._loop.eager(mean, std, noise, parameters, *prepared),
                False,
            )

    @torch.inference_mode()
    def solve_population_tensors(
        self,
        parameters: tuple[torch.Tensor, ...],
        prepared: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve a model population without transferring results to the CPU."""
        device = torch.device(self.device)
        population = self._validate_parameters(parameters, device)
        if not prepared or any(
            not torch.is_tensor(value) for value in prepared
        ):
            raise TypeError('prepared must be a non-empty tuple of tensors')
        if prepared[0].size(0) != population:
            raise ValueError(
                'prepared inputs and parameters disagree on population'
            )
        tasks = prepared[0].size(1)
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

        mean, std = self._initial_distribution(population, tasks, init_action)
        mean, std, costs, compiled = self._run_population(
            mean, std, noise, parameters, prepared
        )
        return {
            'actions': mean,
            'costs': costs,
            'mean': [mean],
            'var': [std],
            'compiled': compiled,
            'compile_error': self._compile_error,
        }

    @torch.inference_mode()
    def solve_population(
        self,
        info: dict[str, Any],
        parameters: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self.solve_population_tensors(
            parameters,
            self.prepare_population(info),
            noise=noise,
            init_action=init_action,
        )

    def solve(self, info, init_action=None, **kwargs):
        parameters = kwargs.pop('predictor_parameters', None)
        if parameters is None:
            raise TypeError(
                'PopulationFastCEMSolver.solve requires '
                'predictor_parameters=...; use ManyWorlds.evaluate for the '
                'high-level API'
            )
        return self.solve_population(
            info, parameters, init_action=init_action, **kwargs
        )


# Temporary compatibility for the pre-rebase branch name.
PopulationAcceleratedCEMSolver = PopulationFastCEMSolver


__all__ = ['PopulationFastCEMSolver']
