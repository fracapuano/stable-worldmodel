"""PyTorch-only accelerated CEM for pure-tensor planning costs.

Unlike :class:`CEMSolver`, this opt-in solver prepares observations once,
pre-generates all random samples on the target device, and performs the whole
optimization without copying iteration results to the host.  On CUDA the
fixed-shape tensor loop is compiled by default.  MPS uses the same
device-resident path eagerly because compilation is not consistently faster
there; callers can override that choice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from loguru import logger as logging
from torch import nn

from .cem import CEMSolver
from .utils import prepare_init_action


class _CEMTensorKernel(nn.Module):
    """Fixed-shape CEM arithmetic and cost evaluation on one device."""

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
        prepared: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = noise * var.unsqueeze(1) + mean.unsqueeze(1)
        # Keep the current mean as sample zero without an in-place graph break.
        candidates = torch.cat([mean.unsqueeze(1), sampled[:, 1:]], dim=1)
        costs = self.cost(candidates, *prepared)
        topk_vals, topk_inds = torch.topk(
            costs, k=self.topk, dim=1, largest=False
        )
        gather_index = topk_inds[:, :, None, None].expand(
            -1, -1, candidates.size(2), candidates.size(3)
        )
        elites = torch.gather(candidates, 1, gather_index)
        return (
            elites.mean(dim=1),
            elites.std(dim=1),
            topk_vals.mean(dim=1),
        )

    def forward_eager(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        *prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference implementation used on MPS and as compile fallback."""
        final_cost = mean.new_zeros(mean.size(0))
        for step in range(self.n_steps):
            mean, var, final_cost = self._step(
                mean, var, noise[step], prepared
            )
        return mean, var, final_cost

    def forward(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        *prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Structured device loop captured as one compiled graph."""
        initial_step = torch.zeros((), dtype=torch.int64, device=mean.device)
        initial_cost = mean.new_zeros(mean.size(0))

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
                loop_mean, loop_var, step_noise, prepared
            )
            return step + 1, next_mean, next_var, next_cost

        _, mean, var, final_cost = torch.while_loop(
            cond, body, (initial_step, mean, var, initial_cost)
        )
        return mean, var, final_cost


class AcceleratedCEMSolver(CEMSolver):
    """CEM fast path for a cost exposing ``prepare`` plus tensor ``forward``.

    The optimization is numerically equivalent to :class:`CEMSolver`, while
    changing only its execution strategy.  The cost must be an ``nn.Module``
    with this narrow interface::

        prepared = cost.prepare(info, device=..., dtype=..., action_dim=...)
        costs = cost(candidates, *prepared)

    Args:
        compile_kernel: ``None`` compiles on CUDA and stays eager elsewhere.
            ``True`` requests compilation on any device; failure falls back to
            eager execution unless ``compile_fallback`` is false.
        compile_mode: Mode passed to :func:`torch.compile`.
        compile_backend: Optional compiler backend override.  The default lets
            PyTorch select Inductor; ``'aot_eager'`` is useful for graph-capture
            validation in tests.
        compile_fallback: Whether compilation/runtime compiler failures should
            transparently use the eager tensor kernel.

    CEM iteration callbacks are intentionally unsupported: inspecting every
    iteration requires materializing graph intermediates and defeats this
    solver's execution contract.  ``WorldModelPolicy.on_plan`` remains fully
    supported because it runs once after a solve.
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
        if callbacks:
            raise ValueError(
                'AcceleratedCEMSolver does not support per-iteration callbacks'
            )
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
        self._kernel = _CEMTensorKernel(cost, n_steps=n_steps, topk=topk)
        self._compiled_kernel: Any | None = None
        self._compile_failed = False
        self._compile_error: str | None = None

    @property
    def compilation_enabled(self) -> bool:
        """Whether this device/configuration requests the compiled kernel."""
        if self.compile_kernel is not None:
            return self.compile_kernel
        return torch.device(self.device).type == 'cuda'

    @property
    def compile_error(self) -> str | None:
        """Most recent compiler failure, if eager fallback was activated."""
        return self._compile_error

    @staticmethod
    def _slice_info(
        info_dict: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value) or isinstance(
                value, (np.ndarray, list, tuple)
            ):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _scalar_at(value: Any, index: int) -> int:
        item = value[index]
        if torch.is_tensor(item) or isinstance(item, np.ndarray):
            item = item.reshape(-1)[0].item()
        return int(item)

    def _sample_noise(
        self,
        current_bs: int,
        *,
        start_idx: int,
        controller_seeds: Any | None,
        decision_indices: Any | None,
    ) -> torch.Tensor:
        """Generate the reference CEM streams before entering the graph."""
        iterations = []
        for step in range(self.n_steps):
            if controller_seeds is None:
                sample = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                rows = []
                for local_idx in range(current_bs):
                    global_idx = start_idx + local_idx
                    seed = self._scalar_at(controller_seeds, global_idx)
                    decision = (
                        0
                        if decision_indices is None
                        else self._scalar_at(decision_indices, global_idx)
                    )
                    stream_seed = (
                        seed + 1_000_003 * decision + 10_007 * step
                    ) % (2**63 - 1)
                    generator = torch.Generator(
                        device=self.device
                    ).manual_seed(stream_seed)
                    rows.append(
                        torch.randn(
                            self.num_samples,
                            self.horizon,
                            self.action_dim,
                            generator=generator,
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )
                sample = torch.stack(rows)
            iterations.append(sample)
        return torch.stack(iterations)

    def _run_kernel(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        noise: torch.Tensor,
        prepared: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
        compile_seconds = 0.0
        if not self.compilation_enabled or self._compile_failed:
            result = self._kernel.forward_eager(mean, var, noise, *prepared)
            return (*result, False, compile_seconds)

        try:
            compile_start: float | None = None
            if self._compiled_kernel is None:
                compile_start = time.perf_counter()
                compile_options: dict[str, Any] = {
                    'fullgraph': True,
                    'dynamic': False,
                    'mode': self.compile_mode,
                }
                if self.compile_backend is not None:
                    compile_options['backend'] = self.compile_backend
                self._compiled_kernel = torch.compile(
                    self._kernel, **compile_options
                )
                # Compilation is lazy; the first invocation below is included.
            result = self._compiled_kernel(mean, var, noise, *prepared)
            if compile_start is not None:
                # The outputs stay asynchronous.  Their eventual host transfer
                # in ``solve`` synchronizes and makes this an upper bound that
                # includes first-graph compilation plus launch.
                compile_seconds = time.perf_counter() - compile_start
            return (*result, True, compile_seconds)
        except Exception as exc:
            if not self.compile_fallback:
                raise
            self._compile_failed = True
            self._compile_error = f'{type(exc).__name__}: {exc}'
            logging.warning(
                'Accelerated CEM compilation failed; using eager tensor path: '
                f'{self._compile_error}'
            )
            result = self._kernel.forward_eager(mean, var, noise, *prepared)
            return (*result, False, compile_seconds)

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Optimize actions with one host transfer after each solver batch."""
        start_time = time.perf_counter()
        controller_seeds = info_dict.get('controller_seed')
        decision_indices = info_dict.get('step_idx')
        total_envs = len(next(iter(info_dict.values())))

        init_action = prepare_init_action(
            self.cost,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        final_cost = mean.new_empty(total_envs)
        any_compiled = False
        compile_seconds = 0.0

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_info = self._slice_info(info_dict, start_idx, end_idx)
            prepared = self.cost.prepare(
                batch_info,
                device=self.device,
                dtype=self.dtype,
                action_dim=self.action_dim,
            )
            if not isinstance(prepared, tuple) or not all(
                torch.is_tensor(value) for value in prepared
            ):
                raise TypeError('cost.prepare must return a tuple of tensors')

            noise = self._sample_noise(
                current_bs,
                start_idx=start_idx,
                controller_seeds=controller_seeds,
                decision_indices=decision_indices,
            )
            batch_mean, batch_var, batch_cost, compiled, compile_time = (
                self._run_kernel(
                    mean[start_idx:end_idx],
                    var[start_idx:end_idx],
                    noise,
                    prepared,
                )
            )
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            final_cost[start_idx:end_idx] = batch_cost
            any_compiled = any_compiled or compiled
            compile_seconds += compile_time

        # These are the only accelerator-to-host transfers in the solve.
        actions_cpu = mean.detach().cpu()
        var_cpu = var.detach().cpu()
        costs_cpu = final_cost.detach().cpu().tolist()
        solve_seconds = time.perf_counter() - start_time
        outputs = {
            'costs': costs_cpu,
            'actions': actions_cpu,
            'mean': [actions_cpu],
            'var': [var_cpu],
            'solve_time_seconds': solve_seconds,
            'compiled': any_compiled,
            'compile_seconds': compile_seconds,
            'compile_error': self._compile_error,
        }
        print(f'Accelerated CEM solve time: {solve_seconds:.4f} seconds')
        return outputs


__all__ = ['AcceleratedCEMSolver']
