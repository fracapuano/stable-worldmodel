"""LeWM-specific tensor planning adapter.

This module is the only accelerated-planning layer that knows LeWM's input
keys, encoder API, latent rollout contract, history semantics, or terminal
goal-distance objective.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stable_worldmodel.planning.tensor_cost import (
    PopulationCall,
    TensorPlanningCost,
)


class _LeWMEncoder(nn.Module):
    """Expose one LeWM's observation encoder as a tensor-only module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor):
        return self.model.encode({'pixels': pixels})['emb']


class LeWMGoalMSETensorCost(TensorPlanningCost):
    """Fast terminal latent-distance planning cost for LeWM-like models."""

    def __init__(
        self, model: nn.Module, *, history_size: int | None = None
    ) -> None:
        super().__init__()
        if not self.supports(model):
            raise TypeError(
                'LeWMGoalMSETensorCost requires a latent rollout primitive or '
                'the LeWM encode/predict/action_encoder interface'
            )
        self.model = model
        self.history_size = history_size

    @staticmethod
    def supports(model: nn.Module) -> bool:
        """Whether ``model`` provides the required latent rollout primitive."""
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
        """Prepare current, goal, and action history for one model."""

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
        if tuple(history.shape) != expected:
            raise ValueError(f'expected action_history shape {expected}')
        return current.detach(), goal.detach(), history.detach()

    def prepare_population(
        self,
        info: dict[str, Any],
        *,
        call_population: PopulationCall,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare LeWM inputs while batching parameter-dependent encoders."""

        def tensor(key):
            return torch.as_tensor(info[key], device=device, dtype=dtype)

        if 'emb' in info:
            current = tensor('emb')
        else:
            current = call_population(
                _LeWMEncoder(self.model), tensor('pixels')
            )
        if 'goal_emb' in info:
            goal = tensor('goal_emb')[:, :, -1]
        else:
            goal = call_population(
                _LeWMEncoder(self.model), tensor('goal')[:, :, -1:]
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
        return current.detach(), goal.detach(), history.detach()

    def _rollout_terminal(self, current, candidates, history):
        """Preserve the LeWM rollout when no tensor primitive is exposed."""
        batch, samples, horizon = candidates.shape[:3]
        context_len = current.size(1)
        history_size = self.history_size
        if history_size is None:
            history_size = getattr(self.model.predictor, 'num_frames', 3)

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
        action_emb = self.model.action_encoder(actions)

        for step in range(horizon):
            lo = max(0, context_len + step - history_size)
            frames.append(
                self.model.predict(
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


__all__ = ['LeWMGoalMSETensorCost']
