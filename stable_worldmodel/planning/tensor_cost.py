"""Pure-tensor terminal cost for LeWM planning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


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


class PopulationLatentGoalCost(_LeWMTerminalCost):
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
        terminal = self.model.rollout_population_from_embeddings(
            current_emb,
            action_candidates,
            predictor_parameters,
            action_history=action_history,
            history_size=self.history_size,
            terminal_only=True,
        )
        goal = goal_emb[None, :, None, :]
        return (terminal - goal).square().sum(dim=-1)


__all__ = ['PopulationLatentGoalCost']
