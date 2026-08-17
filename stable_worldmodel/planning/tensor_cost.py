"""Pure-tensor terminal cost for LeWM planning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LatentGoalCost(nn.Module):
    """Encode fixed inputs once, then score terminal latent distance."""

    def __init__(
        self, model: nn.Module, *, history_size: int | None = None
    ) -> None:
        super().__init__()
        self.model = model
        self.history_size = history_size

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

    def forward(self, candidates, current, goal, history):
        terminal = self.model.rollout_from_embeddings(
            current,
            candidates,
            action_history=history,
            history_size=self.history_size,
            terminal_only=True,
        )
        return (terminal - goal[:, None]).square().sum(-1)


__all__ = ['LatentGoalCost']
