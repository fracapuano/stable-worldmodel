"""Fast CEM with a device-resident refinement loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from stable_worldmodel.planning.evaluator import (
    ShootingCostEvaluator,
    default_goal_encode,
)
from stable_worldmodel.planning.objective import GoalMSE
from stable_worldmodel.planning.tensor_cost import (
    _LeWMModelPopulationCost,
    _LeWMTerminalCost,
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
        if self.cost.backend == 'fused_predictor':
            return super().forward(mean, std, noise, *prepared)
        # ``functional_call``/``vmap`` cannot currently be nested inside a
        # ``torch.while_loop`` capture. n_steps is static, so Dynamo safely
        # unrolls the generic full-state path into one graph.
        return self.eager(mean, std, noise, *prepared)


class FastCEMSolver(CEMSolver):
    """Drop-in CEM with an automatic device-resident fast path.

    Standard costs, including ``ShootingCostEvaluator(model, GoalMSE())``, use
    the same constructor and :meth:`solve` contract as :class:`CEMSolver`.
    Compatible terminal LeWM costs are adapted internally to a pure-tensor
    kernel and can use :meth:`solve_population` with independent model
    instances. Model state and CEM tensors carry a population axis, so all
    candidate slices are ranked by one population model call. The generic
    backend stacks complete state; the fused backend stacks only differing
    predictor weights.
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
        self._population_models: tuple[nn.Module, ...] = ()
        self._compiled_loop = None
        self._compiled_population_loop = None
        self._compile_error = None
        self._population_compile_error = None

    @staticmethod
    def _adapt_cost(cost):
        if (
            isinstance(cost, ShootingCostEvaluator)
            and type(cost.objective) is GoalMSE
            and cost.objective.pred_key == 'predicted_emb'
            and cost.objective.goal_key == 'goal_emb'
            and cost.encode_goal is default_goal_encode
            and _LeWMTerminalCost.supports(cost.model)
        ):
            return _LeWMTerminalCost(cost.model)
        if (
            isinstance(cost, nn.Module)
            and callable(getattr(cost, 'prepare', None))
            and type(cost).forward is not nn.Module.forward
        ):
            return cost
        raise TypeError(
            'FastCEMSolver requires ShootingCostEvaluator(model, GoalMSE()) '
            'with compatible LeWM inference operations, or a tensor cost '
            'implementing prepare(...) and forward(...); use CEMSolver for '
            'other costs'
        )

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
        """Active population model execution backend, if models are bound."""
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

    def _bind_population_models(
        self, models: Sequence[nn.Module]
    ) -> _LeWMModelPopulationCost:
        """Bind independent models to one population-batched CEM graph."""
        models = tuple(models)
        if not models:
            raise ValueError('model population cannot be empty')
        if not isinstance(self._tensor_cost, _LeWMTerminalCost):
            raise TypeError(
                'model populations require a compatible LeWM terminal cost'
            )
        if models[0] is not self._tensor_cost.model:
            raise ValueError(
                'the first population model must be the solver cost model'
            )
        if len(models) == len(self._population_models) and all(
            model is bound
            for model, bound in zip(
                models, self._population_models, strict=True
            )
        ):
            assert self._population_loop is not None
            return self._population_loop.cost

        expected_device = torch.device(self.device)
        for index, model in enumerate(models):
            if not isinstance(model, nn.Module):
                raise TypeError('models must contain only torch modules')
            if model.training:
                raise ValueError(
                    f'population model {index} must be in evaluation mode'
                )
            wrong_device = next(
                (
                    value.device
                    for value in (
                        *tuple(model.parameters()),
                        *tuple(model.buffers()),
                    )
                    if not _same_device(value.device, expected_device)
                ),
                None,
            )
            if wrong_device is not None:
                raise ValueError(
                    f'population model {index} must reside on '
                    f'{expected_device}, got {wrong_device}'
                )

        population_cost = _LeWMModelPopulationCost(
            models, history_size=self._tensor_cost.history_size
        )
        self._population_loop = _PopulationCEMLoop(
            population_cost, self.n_steps, self.topk
        )
        self._population_models = models
        self._compiled_population_loop = None
        self._population_compile_error = None
        return population_cost

    def prepare_population(
        self,
        info: dict[str, Any],
        models: Sequence[nn.Module],
    ) -> tuple[torch.Tensor, ...]:
        """Stack model state and prepare the population input batch."""
        cost = self._bind_population_models(models)
        state = cost.stack_state(models)
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
            raise RuntimeError('population models have not been bound')
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
        models: Sequence[nn.Module],
        prepared: tuple[torch.Tensor, ...],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve a model population without transferring results to the CPU."""
        device = torch.device(self.device)
        models = tuple(models)
        population = len(models)
        self._bind_population_models(models)
        if not prepared or any(
            not torch.is_tensor(value) for value in prepared
        ):
            raise TypeError('prepared must be a non-empty tuple of tensors')
        if any(value.size(0) != population for value in prepared):
            raise ValueError(
                'prepared inputs and models disagree on population'
            )
        tasks = prepared[0].size(1)
        if any(
            value.ndim < 2 or value.size(1) != tasks for value in prepared[:3]
        ):
            raise ValueError(
                'observation, goal, and history must share population and '
                'task axes'
            )
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
        models: Sequence[nn.Module],
        *,
        noise: torch.Tensor | None = None,
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve one population using this solver's existing configuration."""
        return self.solve_population_tensors(
            models,
            self.prepare_population(info, models),
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
