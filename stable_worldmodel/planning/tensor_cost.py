"""Pure-tensor planning costs for accelerator-resident optimization.

The regular :class:`~stable_worldmodel.planning.ShootingCostEvaluator` keeps a
flexible dictionary interface that works with every SWM dynamics backend.  A
compiled optimizer needs a narrower contract: observations are encoded once,
outside the optimization loop, and every CEM iteration receives only tensors.
This module provides that opt-in fast path for LeWM-style latent dynamics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LatentGoalCost(nn.Module):
    """Terminal latent error with one-time observation preparation.

    ``prepare`` encodes the current observation context and goal once per MPC
    decision.  ``forward`` is then a pure tensor function suitable for
    ``torch.compile`` and device-side structured control flow.

    The returned cost is exactly the terminal term produced by
    ``ShootingCostEvaluator(model, GoalMSE())``: a sum of squared errors over
    the final embedding dimension.

    Args:
        model: LeWM or a backend exposing ``encode`` and
            ``rollout_from_embeddings``.
        history_size: Optional maximum predictor context.  ``None`` preserves
            the model's configured default.
    """

    def __init__(
        self, model: nn.Module, *, history_size: int | None = None
    ) -> None:
        super().__init__()
        if not hasattr(model, 'encode'):
            raise TypeError('model must expose encode')
        if not hasattr(model, 'rollout_from_embeddings'):
            raise TypeError('model must expose rollout_from_embeddings')
        self.model = model
        self.history_size = history_size

    def prepare(
        self,
        info_dict: dict[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode candidate-independent inputs once for one solver batch."""
        if 'emb' in info_dict:
            current_emb = torch.as_tensor(info_dict['emb']).to(
                device=device, dtype=dtype
            )
        else:
            if 'pixels' not in info_dict:
                raise KeyError("LatentGoalCost requires 'pixels' or 'emb'")
            pixels = torch.as_tensor(info_dict['pixels']).to(
                device=device, dtype=dtype
            )
            current_emb = self.model.encode({'pixels': pixels})['emb']

        if 'goal_emb' in info_dict:
            goal_emb = torch.as_tensor(info_dict['goal_emb']).to(
                device=device, dtype=dtype
            )
        else:
            if 'goal' not in info_dict:
                raise KeyError("LatentGoalCost requires 'goal' or 'goal_emb'")
            goal = torch.as_tensor(info_dict['goal']).to(
                device=device, dtype=dtype
            )
            goal_emb = self.model.encode({'pixels': goal})['emb']

        history_len = current_emb.size(1)
        action_history = info_dict.get('action_history')
        if action_history is None:
            action_history = current_emb.new_zeros(
                current_emb.size(0), max(0, history_len - 1), action_dim
            )
        else:
            action_history = torch.as_tensor(action_history).to(
                device=device, dtype=dtype
            )

        if action_history.ndim != 3:
            raise ValueError(
                'action_history must have shape (batch, history - 1, action_dim)'
            )
        expected = (current_emb.size(0), history_len - 1, action_dim)
        if tuple(action_history.shape) != expected:
            raise ValueError(
                f'action_history must have shape {expected}, got '
                f'{tuple(action_history.shape)}'
            )

        # Planning is inference-only.  Detaching here also prevents the
        # one-time encoder graph from being retained across CEM iterations.
        return (
            current_emb.detach(),
            goal_emb.detach(),
            action_history.detach(),
        )

    def forward(
        self,
        action_candidates: torch.Tensor,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``(B, S, horizon, action_dim)`` candidate plans."""
        predicted = self.model.rollout_from_embeddings(
            current_emb,
            action_candidates,
            action_history=action_history,
            history_size=self.history_size,
        )
        terminal = predicted[:, :, -1, :]
        goal = goal_emb[:, None, -1, :]
        return (terminal - goal).square().sum(dim=-1)


class PopulationLatentGoalCost(LatentGoalCost):
    """Terminal latent error for independently parameterized LeWM predictors.

    Observation and goal embeddings are prepared once and shared across the
    model population.  The population dimension lives on the candidate action
    tensor and on every predictor parameter tensor; no resident module weights
    are mutated between candidates.
    """

    def __init__(
        self, model: nn.Module, *, history_size: int | None = None
    ) -> None:
        super().__init__(model, history_size=history_size)
        if not hasattr(model, 'rollout_population_from_embeddings'):
            raise TypeError(
                'model must expose rollout_population_from_embeddings'
            )
        if not hasattr(model, 'population_predictor_parameter_names'):
            raise TypeError(
                'model must expose population_predictor_parameter_names'
            )

    @property
    def predictor_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.model.population_predictor_parameter_names)

    def forward(
        self,
        action_candidates: torch.Tensor,
        predictor_parameters: tuple[torch.Tensor, ...],
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Score plans shaped ``(population, batch, samples, horizon, A)``."""
        predicted = self.model.rollout_population_from_embeddings(
            current_emb,
            action_candidates,
            predictor_parameters,
            action_history=action_history,
            history_size=self.history_size,
        )
        terminal = predicted[:, :, :, -1, :]
        goal = goal_emb[None, :, None, -1, :]
        return (terminal - goal).square().sum(dim=-1)


__all__ = ['LatentGoalCost', 'PopulationLatentGoalCost']
