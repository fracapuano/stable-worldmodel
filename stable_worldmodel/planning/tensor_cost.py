"""Pure-tensor terminal cost for LeWM planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

_CUDA_SDPA_GRID_Y_LIMIT = 65_535


class _LeWMTerminalCost(nn.Module):
    """Internal FastCEM adapter for terminal LeWM goal distance."""

    def __init__(
        self, model: nn.Module, *, history_size: int | None = None
    ) -> None:
        super().__init__()
        self.model = model
        self.history_size = history_size

    @staticmethod
    def supports(model: nn.Module) -> bool:
        """Whether the model exposes the LeWM inference operations we need."""
        return callable(getattr(model, 'rollout_from_embeddings', None)) or (
            all(
                callable(getattr(model, name, None))
                for name in ('encode', 'predict', 'action_encoder')
            )
            and hasattr(model, 'predictor')
        )

    def prepare(
        self,
        info: dict[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare candidate-independent observation, goal, and history."""

        def tensor(key):
            return torch.as_tensor(info[key], device=device, dtype=dtype)

        current = (
            tensor('emb')
            if 'emb' in info
            else self.model.encode({'pixels': tensor('pixels')})['emb']
        )
        goal = (
            tensor('goal_emb')[:, -1]
            if 'goal_emb' in info
            else self.model.encode({'pixels': tensor('goal')[:, -1:]})['emb'][
                :, 0
            ]
        )
        history = (
            tensor('action_history')
            if 'action_history' in info
            else current.new_zeros(
                current.size(0), current.size(1) - 1, action_dim
            )
        )
        expected = (current.size(0), current.size(1) - 1, action_dim)
        if history.shape != expected:
            raise ValueError(f'expected action_history shape {expected}')
        return current.detach(), goal.detach(), history.detach()

    def _rollout_terminal(self, current, candidates, history):
        model = self.model
        batch, samples, horizon = candidates.shape[:3]
        context_len = current.size(1)
        history_size = self.history_size
        if history_size is None:
            history_size = getattr(model.predictor, 'num_frames', 3)

        frames = list(
            current[:, None]
            .expand(batch, samples, -1, -1)
            .flatten(0, 1)
            .unbind(1)
        )
        actions = candidates.flatten(0, 1)
        if context_len > 1:
            actions = torch.cat(
                [
                    history[:, None]
                    .expand(batch, samples, -1, -1)
                    .flatten(0, 1),
                    actions,
                ],
                dim=1,
            )
        action_emb = model.action_encoder(actions)

        for step in range(horizon):
            lo = max(0, context_len + step - history_size)
            frames.append(
                model.predict(
                    torch.stack(frames[-history_size:], dim=1),
                    action_emb[:, lo : context_len + step],
                )[:, -1]
            )
            frames = frames[-history_size:]

        return frames[-1].unflatten(0, (batch, samples))

    def forward(self, candidates, current, goal, history):
        rollout = getattr(self.model, 'rollout_from_embeddings', None)
        terminal = (
            rollout(
                current,
                candidates,
                action_history=history,
                history_size=self.history_size,
                terminal_only=True,
            )
            if callable(rollout)
            else self._rollout_terminal(current, candidates, history)
        )
        return (terminal - goal[:, None]).square().sum(-1)


class _LeWMEncoder(nn.Module):
    """Functional-call wrapper for one model observation encoding."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor):
        return self.model.encode({'pixels': pixels})['emb']


class _LeWMModelPopulationCost(nn.Module):
    """Score all independent LeWMs in one population-batched tensor call.

    Fully independent model parameters and buffers are stacked on a leading
    population axis for ``torch.func.vmap``, avoiding a Python model loop
    inside the CEM refinement kernel.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        *,
        history_size: int | None = None,
    ) -> None:
        super().__init__()
        models = tuple(models)
        if not models:
            raise ValueError('model population cannot be empty')
        if not all(_LeWMTerminalCost.supports(model) for model in models):
            raise TypeError(
                'every population member must support LeWM tensor rollouts'
            )
        self.template = _LeWMTerminalCost(models[0], history_size=history_size)
        self.encoder = _LeWMEncoder(models[0])
        self._population_size = len(models)
        self._parameter_names: tuple[str, ...] = ()
        self._buffer_names: tuple[str, ...] = ()

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def backend(self) -> str:
        return 'functional_vmap'

    def stack_state(
        self, models: Sequence[nn.Module]
    ) -> tuple[torch.Tensor, ...]:
        """Stack all model parameters and buffers on a population axis."""
        costs = tuple(
            _LeWMTerminalCost(model, history_size=self.template.history_size)
            for model in models
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
        expected = len(self._parameter_names) + len(self._buffer_names)
        if len(state) != expected:
            raise ValueError(
                f'expected {expected} population state tensors, got '
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
        """Encode all population observations in one functional batch."""

        def tensor(key):
            return torch.as_tensor(info[key], device=device, dtype=dtype)

        parameters, buffers = self._unpack_state(state)
        if 'emb' in info:
            current = tensor('emb')
        else:
            current = self._call_population(
                self.encoder, parameters, buffers, tensor('pixels')
            )
        if 'goal_emb' in info:
            goal = tensor('goal_emb')[:, :, -1]
        else:
            goal = self._call_population(
                self.encoder,
                parameters,
                buffers,
                tensor('goal')[:, :, -1:],
            )[:, :, 0]

        history = (
            tensor('action_history')
            if 'action_history' in info
            else current.new_zeros(
                current.size(0),
                current.size(1),
                current.size(2) - 1,
                action_dim,
            )
        )
        expected = (
            current.size(0),
            current.size(1),
            current.size(2) - 1,
            action_dim,
        )
        if tuple(history.shape) != expected:
            raise ValueError(f'expected action_history shape {expected}')
        return (
            current.detach(),
            goal.detach(),
            history.detach(),
            *state,
        )

    def forward(self, candidates, current, goal, history, *state):
        parameters, buffers = self._unpack_state(state)
        try:
            return self._call_population(
                self.template,
                parameters,
                buffers,
                candidates,
                current,
                goal,
                history,
            )
        except RuntimeError as error:
            if 'CUDA error: invalid argument' not in str(error):
                raise
            population, tasks, samples = candidates.shape[:3]
            effective_batch = population * tasks * samples
            max_population = _CUDA_SDPA_GRID_Y_LIMIT // (tasks * samples)
            max_samples = _CUDA_SDPA_GRID_Y_LIMIT // (population * tasks)
            raise RuntimeError(
                'CUDA rejected the unsplit population FastCEM model-scoring '
                'launch. This is commonly caused by scaled-dot-product '
                'attention exceeding CUDA grid-y limits: '
                f'population={population}, tasks={tasks}, samples={samples}, '
                f'effective_attention_batch={effective_batch}. No automatic '
                'tiling or fallback is performed. For the usual CUDA grid-y '
                f'limit of {_CUDA_SDPA_GRID_Y_LIMIT}, use population <= '
                f'{max_population} with the current tasks/samples, or samples '
                f'<= {max_samples} with the current population/tasks.'
            ) from error


__all__: list[str] = []
