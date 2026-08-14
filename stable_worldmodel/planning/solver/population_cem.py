"""Population-batched, accelerator-resident CEM for LeWM post-training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from loguru import logger as logging
from torch import nn

from .cem import CEMSolver


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


class _PopulationCEMTensorKernel(nn.Module):
    """Optimize one independent CEM distribution per model and task."""

    def __init__(self, cost: nn.Module, *, n_steps: int, topk: int) -> None:
        super().__init__()
        self.cost = cost
        self.n_steps = n_steps
        self.topk = topk

    def _step(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        predictor_parameters: tuple[torch.Tensor, ...],
        prepared: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``noise`` is shared across population members (common random
        # numbers), while every member retains its own CEM mean and variance.
        sampled = noise.unsqueeze(0) * var.unsqueeze(2) + mean.unsqueeze(2)
        candidates = torch.cat([mean.unsqueeze(2), sampled[:, :, 1:]], dim=2)
        costs = self.cost(candidates, predictor_parameters, *prepared)
        topk_values, topk_indices = torch.topk(
            costs, k=self.topk, dim=2, largest=False
        )
        gather_index = topk_indices[:, :, :, None, None].expand(
            -1, -1, -1, candidates.size(3), candidates.size(4)
        )
        elites = torch.gather(candidates, 2, gather_index)
        return (
            elites.mean(dim=2),
            elites.std(dim=2),
            topk_values.mean(dim=2),
        )

    def forward_eager(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        predictor_parameters: tuple[torch.Tensor, ...],
        *prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_cost = mean.new_zeros(mean.shape[:2])
        for step in range(self.n_steps):
            mean, var, final_cost = self._step(
                mean,
                var,
                noise[step],
                predictor_parameters,
                prepared,
            )
        return mean, var, final_cost

    def forward(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        predictor_parameters: tuple[torch.Tensor, ...],
        *prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_step = torch.zeros((), dtype=torch.int64, device=mean.device)
        initial_cost = mean.new_zeros(mean.shape[:2])

        def cond(
            step: torch.Tensor,
            _mean: torch.Tensor,
            _var: torch.Tensor,
            _cost: torch.Tensor,
        ) -> torch.Tensor:
            return step < self.n_steps

        def body(
            step: torch.Tensor,
            loop_mean: torch.Tensor,
            loop_var: torch.Tensor,
            _cost: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            noise_index = step.reshape(1, 1, 1, 1, 1).expand(
                1, *noise.shape[1:]
            )
            step_noise = torch.gather(noise, 0, noise_index).squeeze(0)
            next_mean, next_var, next_cost = self._step(
                loop_mean,
                loop_var,
                step_noise,
                predictor_parameters,
                prepared,
            )
            return step + 1, next_mean, next_var, next_cost

        _, mean, var, final_cost = torch.while_loop(
            cond, body, (initial_step, mean, var, initial_cost)
        )
        return mean, var, final_cost


class PopulationAcceleratedCEMSolver(CEMSolver):
    """CEM over a population of predictor parameter tensors.

    Candidate predictor tensors and CEM state remain on the configured device.
    A solve returns device tensors and performs no accelerator-to-host copy.
    Population chunking belongs to the caller so the same fixed graph can be
    reused for every chunk.  CUDA callers should use a fixed chunk size (and
    pad a final partial chunk when necessary) to avoid shape recompilation.

    All candidates in a call receive identical CEM noise.  Passing the same
    tensor returned by :meth:`sample_noise` to multiple chunks preserves these
    common random numbers over an arbitrarily large population.
    """

    def __init__(
        self,
        cost: nn.Module,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        *,
        compile_kernel: bool | None = None,
        compile_mode: str = 'reduce-overhead',
        compile_backend: str | Callable[..., Any] | None = None,
        compile_fallback: bool = True,
    ) -> None:
        if not isinstance(cost, nn.Module) or not hasattr(cost, 'prepare'):
            raise TypeError(
                'cost must be an nn.Module exposing prepare(...) and forward(...)'
            )
        super().__init__(
            cost=cost,
            batch_size=batch_size,
            num_samples=num_samples,
            var_scale=var_scale,
            n_steps=n_steps,
            topk=topk,
            device=device,
            seed=seed,
            callbacks=None,
        )
        self.compile_kernel = compile_kernel
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.compile_fallback = compile_fallback
        self._kernel = _PopulationCEMTensorKernel(
            cost, n_steps=n_steps, topk=topk
        )
        self._compiled_kernel: Any | None = None
        self._compile_failed = False
        self._compile_error: str | None = None

    @property
    def compilation_enabled(self) -> bool:
        if self.compile_kernel is not None:
            return self.compile_kernel
        return torch.device(self.device).type == 'cuda'

    @property
    def compile_error(self) -> str | None:
        return self._compile_error

    def sample_noise(self, task_batch_size: int) -> torch.Tensor:
        """Draw one device-resident CEM stream shared by all model chunks."""
        return torch.randn(
            self.n_steps,
            task_batch_size,
            self.num_samples,
            self.horizon,
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
            dtype=self.dtype,
        )

    def _initial_distribution(
        self,
        population: int,
        task_batch_size: int,
        init_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (
            population,
            task_batch_size,
            self.horizon,
            self.action_dim,
        )
        if init_action is None:
            mean = torch.zeros(shape, device=self.device, dtype=self.dtype)
        else:
            init_action = init_action.to(device=self.device, dtype=self.dtype)
            if init_action.ndim == 3:
                init_action = init_action.unsqueeze(0).expand(
                    population, -1, -1, -1
                )
            if init_action.ndim != 4 or init_action.shape[:2] != (
                population,
                task_batch_size,
            ):
                raise ValueError(
                    'init_action must have shape (batch, time, action_dim) or '
                    '(population, batch, time, action_dim)'
                )
            if init_action.size(-1) != self.action_dim:
                raise ValueError('init_action has the wrong action dimension')
            if init_action.size(2) > self.horizon:
                init_action = init_action[:, :, : self.horizon]
            remaining = self.horizon - init_action.size(2)
            mean = torch.cat(
                [
                    init_action,
                    init_action.new_zeros(
                        population,
                        task_batch_size,
                        remaining,
                        self.action_dim,
                    ),
                ],
                dim=2,
            )
        var = torch.full_like(mean, self.var_scale)
        return mean, var

    def _run_kernel(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        predictor_parameters: tuple[torch.Tensor, ...],
        prepared: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if not self.compilation_enabled or self._compile_failed:
            result = self._kernel.forward_eager(
                mean, var, noise, predictor_parameters, *prepared
            )
            return (*result, False)

        try:
            if self._compiled_kernel is None:
                options: dict[str, Any] = {
                    'fullgraph': True,
                    'dynamic': False,
                    'mode': self.compile_mode,
                }
                if self.compile_backend is not None:
                    options['backend'] = self.compile_backend
                self._compiled_kernel = torch.compile(self._kernel, **options)
            result = self._compiled_kernel(
                mean, var, noise, predictor_parameters, *prepared
            )
            return (*result, True)
        except Exception as exc:
            if not self.compile_fallback:
                raise
            self._compile_failed = True
            self._compile_error = f'{type(exc).__name__}: {exc}'
            logging.warning(
                'Population CEM compilation failed; using eager tensor path: '
                f'{self._compile_error}'
            )
            result = self._kernel.forward_eager(
                mean, var, noise, predictor_parameters, *prepared
            )
            return (*result, False)

    @staticmethod
    def _validate_parameters(
        predictor_parameters: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> int:
        if not predictor_parameters:
            raise ValueError('predictor_parameters cannot be empty')
        if any(not torch.is_tensor(value) for value in predictor_parameters):
            raise TypeError('predictor_parameters must be a tuple of tensors')
        population = predictor_parameters[0].size(0)
        if population < 1 or any(
            value.size(0) != population for value in predictor_parameters
        ):
            raise ValueError(
                'all predictor parameter tensors need one common population axis'
            )
        wrong_device = [
            value.device
            for value in predictor_parameters
            if not _same_device(value.device, device)
        ]
        if wrong_device:
            raise ValueError(
                'candidate parameters must already reside on the solver device; '
                f'expected {device}, got {wrong_device[0]}'
            )
        return population

    @torch.inference_mode()
    def solve_population_tensors(
        self,
        predictor_parameters: tuple[torch.Tensor, ...],
        prepared: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve from prepared embeddings and return accelerator tensors."""
        device = torch.device(self.device)
        population = self._validate_parameters(predictor_parameters, device)
        if not prepared or any(
            not torch.is_tensor(value) for value in prepared
        ):
            raise TypeError('prepared must be a non-empty tuple of tensors')
        task_batch_size = prepared[0].size(0)
        if noise is None:
            noise = self.sample_noise(task_batch_size)
        expected_noise = (
            self.n_steps,
            task_batch_size,
            self.num_samples,
            self.horizon,
            self.action_dim,
        )
        if tuple(noise.shape) != expected_noise:
            raise ValueError(
                f'noise must have shape {expected_noise}, got {tuple(noise.shape)}'
            )
        if not _same_device(noise.device, device):
            raise ValueError('noise must already reside on the solver device')

        mean, var = self._initial_distribution(
            population, task_batch_size, init_action
        )
        mean, var, costs, compiled = self._run_kernel(
            mean,
            var,
            noise,
            predictor_parameters,
            prepared,
        )
        return {
            'actions': mean,
            'costs': costs,
            'var': var,
            'compiled': compiled,
            'compile_error': self._compile_error,
        }

    @torch.inference_mode()
    def solve_population(
        self,
        info_dict: dict[str, Any],
        predictor_parameters: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Prepare task observations once, then solve the model population."""
        prepared = self.cost.prepare(
            info_dict,
            device=self.device,
            dtype=self.dtype,
            action_dim=self.action_dim,
        )
        return self.solve_population_tensors(
            predictor_parameters,
            prepared,
            noise=noise,
            init_action=init_action,
        )

    def solve(self, info_dict: dict, init_action=None, **kwargs) -> dict:
        """Solver-protocol adapter; requires explicit predictor parameters."""
        predictor_parameters = kwargs.pop('predictor_parameters', None)
        if predictor_parameters is None:
            raise TypeError(
                'PopulationAcceleratedCEMSolver requires '
                'predictor_parameters=...'
            )
        return self.solve_population(
            info_dict,
            predictor_parameters,
            init_action=init_action,
            **kwargs,
        )


__all__ = ['PopulationAcceleratedCEMSolver']
