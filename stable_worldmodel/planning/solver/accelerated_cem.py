"""Device-resident CEM for pure-tensor planning costs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger as logging
from torch import nn

from .cem import CEMSolver
from .utils import prepare_init_action


class _CEMLoop(nn.Module):
    """The fixed-shape portion captured by ``torch.compile``."""

    def __init__(self, cost: nn.Module, n_steps: int, topk: int) -> None:
        super().__init__()
        self.cost, self.n_steps, self.topk = cost, n_steps, topk

    def _step(self, mean, std, noise, prepared):
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
        cost = mean.new_zeros(mean.size(0))
        for step_noise in noise:
            mean, std, cost = self._step(mean, std, step_noise, prepared)
        return mean, std, cost

    def forward(self, mean, std, noise, *prepared):
        step = torch.zeros((), dtype=torch.int64, device=mean.device)
        cost = mean.new_zeros(mean.size(0))

        def cond(step, _mean, _std, _cost):
            return step < self.n_steps

        def body(step, mean, std, _cost):
            step_noise = noise.index_select(0, step[None]).squeeze(0)
            mean, std, cost = self._step(mean, std, step_noise, prepared)
            return step + 1, mean, std, cost

        _, mean, std, cost = torch.while_loop(
            cond, body, (step, mean, std, cost)
        )
        return mean, std, cost


class AcceleratedCEMSolver(CEMSolver):
    """CEM with one compiled, device-resident refinement loop.

    ``cost.prepare(info, ...)`` runs once per planning decision and
    ``cost(candidates, *prepared)`` must be a pure tensor operation. The
    regular :meth:`solve` returns the usual CPU result; :meth:`solve_tensors`
    leaves its result on the planning device.
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
        callbacks: list[Any] | None = None,
        *,
        compile_kernel: bool | None = None,
        compile_mode: str = 'reduce-overhead',
        compile_backend: str | Callable[..., Any] | None = None,
        compile_fallback: bool = True,
    ) -> None:
        if n_steps < 1 or num_samples < 2 or not 2 <= topk <= num_samples:
            raise ValueError('invalid CEM steps, samples, or top-k')
        if callbacks:
            raise ValueError('accelerated CEM does not expose loop callbacks')
        super().__init__(
            cost,
            batch_size,
            num_samples,
            var_scale,
            n_steps,
            topk,
            device,
            seed,
        )
        self.compile_kernel = compile_kernel
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.compile_fallback = compile_fallback
        self._loop = _CEMLoop(cost, n_steps, topk)
        self._compiled_loop = None
        self._compile_error = None

    @property
    def compilation_enabled(self) -> bool:
        return (
            self.compile_kernel
            if self.compile_kernel is not None
            else torch.device(self.device).type == 'cuda'
        )

    @property
    def compile_error(self) -> str | None:
        return self._compile_error

    def prepare(self, info: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        """Encode inputs once and move them to the planning device."""
        return self.cost.prepare(
            info,
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

    def _run(self, mean, std, noise, prepared):
        if not self.compilation_enabled or self._compile_error is not None:
            return (*self._loop.eager(mean, std, noise, *prepared), False)
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
            return (*self._compiled_loop(mean, std, noise, *prepared), True)
        except Exception as exc:
            if not self.compile_fallback:
                raise
            self._compile_error = f'{type(exc).__name__}: {exc}'
            logging.warning(
                f'accelerated CEM compilation failed; using eager: '
                f'{self._compile_error}'
            )
            return (*self._loop.eager(mean, std, noise, *prepared), False)

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


__all__ = ['AcceleratedCEMSolver']
