"""Composable adapters for planning dynamics."""

from __future__ import annotations

import torch
from torch import nn

from stable_worldmodel.protocols import Dynamics


class DecodedDynamics(nn.Module):
    """Project learned rollout embeddings into a task-state representation.

    The wrapped dynamics still performs the prediction. The decoder is only
    applied to its rollout output, making this adapter useful for evaluating a
    learned model under the same state-space objective as simulator dynamics.

    Args:
        model: Dynamics that populates ``rollout_key`` during ``rollout``.
        decoder: Pointwise module mapping predicted embeddings to task state.
        state_key: Privileged goal-state key consumed by ``encode``.
        embedding_key: Goal key expected by planning objectives.
        rollout_key: Learned rollout key decoded in place.
        freeze_decoder: Put the decoder in eval mode and disable gradients.
    """

    def __init__(
        self,
        model: Dynamics,
        decoder: nn.Module,
        *,
        state_key: str = 'state',
        embedding_key: str = 'emb',
        rollout_key: str = 'predicted_emb',
        freeze_decoder: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(model, Dynamics):
            raise TypeError('model must implement the SWM Dynamics protocol')
        self.model = model
        self.decoder = decoder
        self.state_key = state_key
        self.embedding_key = embedding_key
        self.rollout_key = rollout_key
        if freeze_decoder:
            self.decoder.eval()
            self.decoder.requires_grad_(False)

    def encode(self, info_dict: dict) -> dict:
        """Expose privileged task state under the objective embedding key."""
        if self.state_key not in info_dict:
            raise KeyError(
                f'decoded goal encoding requires {self.state_key!r}; '
                f'available keys: {sorted(info_dict)}'
            )
        state = torch.as_tensor(info_dict[self.state_key])
        if not state.is_floating_point():
            state = state.to(dtype=torch.float32)
        info_dict[self.embedding_key] = state
        return info_dict

    def rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Run learned dynamics, then decode every predicted embedding."""
        info_dict = self.model.rollout(info_dict, action_candidates)
        if self.rollout_key not in info_dict:
            raise KeyError(
                f'learned rollout did not populate {self.rollout_key!r}'
            )
        info_dict[self.rollout_key] = self.decoder(
            info_dict[self.rollout_key]
        )
        return info_dict


__all__ = ['DecodedDynamics']
