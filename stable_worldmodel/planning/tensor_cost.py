"""Pure-tensor terminal cost for LeWM planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


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
        return (
            callable(getattr(model, 'rollout_from_embeddings', None))
            or all(
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
        """Run LeWM's existing rollout semantics without retaining its output."""
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
    """Functional-call wrapper for one model's current and goal encodings."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor, goal: torch.Tensor):
        current = self.model.encode({'pixels': pixels})['emb']
        goal_emb = self.model.encode({'pixels': goal})['emb'][:, 0]
        return current, goal_emb


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
            pixels = tensor('pixels')
            goals = tensor('goal')[:, :, -1:]
            current, goal = self._call_population(
                self.encoder, parameters, buffers, pixels, goals
            )
        if 'goal_emb' in info:
            goal = tensor('goal_emb')[:, :, -1]

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
        return self._call_population(
            self.template,
            parameters,
            buffers,
            candidates,
            current,
            goal,
            history,
        )


class _LeWMFactorizedPopulationCost(nn.Module):
    """Score a population represented by batched model-state overrides.

    The base model is shared. Only the supplied factor tensors carry a
    population axis, so this path never constructs model copies or dense
    population parameters. Each factor name is the name of a registered
    buffer on the base model, as accepted by ``torch.func.functional_call``.
    """

    def __init__(
        self,
        model: nn.Module,
        factor_names: Sequence[str],
        *,
        history_size: int | None = None,
    ) -> None:
        super().__init__()
        factor_names = tuple(factor_names)
        if not factor_names:
            raise ValueError('factor state cannot be empty')
        if len(set(factor_names)) != len(factor_names):
            raise ValueError('factor state names must be unique')
        available = dict(model.named_buffers())
        missing = tuple(name for name in factor_names if name not in available)
        if missing:
            raise KeyError(
                f'factor state is absent from the model: {missing[0]}'
            )
        if not _LeWMTerminalCost.supports(model):
            raise TypeError('factorized population requires a compatible LeWM')

        self.template = _LeWMTerminalCost(model, history_size=history_size)
        self.encoder = _LeWMEncoder(model)
        self.factor_names = factor_names
        self.factor_shapes = {
            name: tuple(available[name].shape) for name in factor_names
        }

    @property
    def backend(self) -> str:
        return 'factorized_vmap'

    def pack_state(
        self, factor_state: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Return factor tensors in the stable schema used by the graph."""
        if tuple(factor_state) != self.factor_names:
            raise ValueError('factor state schema changed')
        state = tuple(factor_state.values())
        if any(not torch.is_tensor(value) for value in state):
            raise TypeError('factor state values must be tensors')
        if any(value.ndim < 1 for value in state):
            raise ValueError('factor state tensors need a population axis')
        for name, value in factor_state.items():
            if tuple(value.shape[1:]) != self.factor_shapes[name]:
                raise ValueError(
                    f'factor state {name!r} must have trailing shape '
                    f'{self.factor_shapes[name]}'
                )
        population = state[0].size(0)
        if population < 1 or any(
            value.size(0) != population for value in state
        ):
            raise ValueError(
                'factor state tensors must share a population axis'
            )
        return tuple(value.detach() for value in state)

    def _overrides(
        self, state: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        if len(state) != len(self.factor_names):
            raise ValueError(
                f'expected {len(self.factor_names)} factor tensors, got '
                f'{len(state)}'
            )
        return {
            f'model.{name}': value
            for name, value in zip(self.factor_names, state, strict=True)
        }

    @staticmethod
    def _call_population(module, overrides, *args):
        def call_member(member_overrides, *member_args):
            return functional_call(
                module,
                member_overrides,
                member_args,
                strict=False,
            )

        return vmap(
            call_member,
            in_dims=(0, *(0 for _ in args)),
            randomness='different',
        )(overrides, *args)

    def prepare(
        self,
        info: dict[str, Any],
        state: tuple[torch.Tensor, ...],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, ...]:
        """Encode population inputs with one base model and batched factors."""

        def tensor(key):
            return torch.as_tensor(info[key], device=device, dtype=dtype)

        overrides = self._overrides(state)
        if 'emb' in info:
            current = tensor('emb')
        else:
            pixels = tensor('pixels')
            goals = tensor('goal')[:, :, -1:]
            current, goal = self._call_population(
                self.encoder, overrides, pixels, goals
            )
        if 'goal_emb' in info:
            goal = tensor('goal_emb')[:, :, -1]

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
        return current.detach(), goal.detach(), history.detach(), *state

    def forward(self, candidates, current, goal, history, *state):
        return self._call_population(
            self.template,
            self._overrides(state),
            candidates,
            current,
            goal,
            history,
        )


__all__: list[str] = []
