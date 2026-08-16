"""Simulator-backed dynamics for model-predictive control."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class Simulatable(Protocol):
    """Minimal contract for an environment with queryable dynamics.

    ``get_next_state`` must be side-effect free and support arbitrary leading
    batch dimensions. The simulator may use its fixed layout or dynamics
    parameters, but it must not read or mutate its current dynamic state.
    """

    action_space: Any

    def get_next_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Return states reached by applying environment actions."""
        ...


class SimulatorDynamics(nn.Module):
    """Expose a simulator's exact transition function as SWM dynamics.

    The adapter implements the existing ``Dynamics`` protocol used by
    ``ShootingCostEvaluator``. ``predict`` is the one-step primitive;
    ``rollout`` repeatedly applies it to every candidate action sequence.

    A single adapter represents one fixed simulator configuration. Candidate
    and batch dimensions all share that simulator's layout and dynamics.

    Args:
        simulator: Object implementing ``Simulatable`` (``get_next_state``).
            Wrapped Gymnasium environments are automatically unwrapped.
        state_key: Privileged current-state key in the planning info dict.
        goal_state_key: Privileged goal-state key used by ``encode_goal``.
        embedding_key: Key populated by ``encode``.
        rollout_key: Key populated by ``rollout``.
        decode_action: Optional conversion from planner action space to the
            simulator's action space, such as a normalizer's
            ``inverse_transform`` method.
    """

    def __init__(
        self,
        simulator: Simulatable,
        *,
        state_key: str = 'state',
        goal_state_key: str = 'goal_state',
        embedding_key: str = 'emb',
        rollout_key: str = 'predicted_emb',
        decode_action: Callable[[torch.Tensor], torch.Tensor | np.ndarray]
        | None = None,
    ) -> None:
        super().__init__()
        simulator = getattr(simulator, 'unwrapped', simulator)
        if not isinstance(simulator, Simulatable):
            raise TypeError(
                f'{type(simulator).__name__} does not implement the '
                'Simulatable contract'
            )
        self.simulator = simulator
        self.state_key = state_key
        self.goal_state_key = goal_state_key
        self.embedding_key = embedding_key
        self.rollout_key = rollout_key
        self.decode_action = decode_action

    def predict(
        self,
        state: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Predict one exact, side-effect-free simulator transition."""
        state = torch.as_tensor(state)
        if not state.is_floating_point():
            state = state.to(dtype=torch.float32)
        action = torch.as_tensor(
            action, device=state.device, dtype=state.dtype
        )
        if self.decode_action is not None:
            action = torch.as_tensor(
                self.decode_action(action),
                device=state.device,
                dtype=state.dtype,
            )
        next_state = self.simulator.get_next_state(state, action)
        return torch.as_tensor(
            next_state, device=state.device, dtype=state.dtype
        )

    def encode(self, info_dict: dict) -> dict:
        """Use privileged simulator state as its planning embedding."""
        if self.state_key not in info_dict:
            raise KeyError(
                f'simulator encoding requires {self.state_key!r}; '
                f'available keys: {sorted(info_dict)}'
            )
        state = torch.as_tensor(info_dict[self.state_key])
        if not state.is_floating_point():
            state = state.to(dtype=torch.float32)
        info_dict[self.embedding_key] = state
        return info_dict

    def encode_goal(self, info_dict: dict) -> torch.Tensor:
        """Return the privileged goal state in objective-compatible shape."""
        if self.goal_state_key not in info_dict:
            raise KeyError(
                f'simulator goal encoding requires {self.goal_state_key!r}; '
                f'available keys: {sorted(info_dict)}'
            )
        goal = torch.as_tensor(info_dict[self.goal_state_key])
        if not goal.is_floating_point():
            goal = goal.to(dtype=torch.float32)
        if goal.ndim == 2:
            goal = goal[:, None]
        elif goal.ndim == 4:
            goal = goal[:, 0]
        if goal.ndim != 3:
            raise ValueError(
                f'{self.goal_state_key!r} must have shape (B, D), '
                '(B, T, D), or (B, S, T, D)'
            )
        return goal

    def rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Roll all action candidates forward from ``info_dict[state_key]``.

        ``action_candidates`` has shape ``(B, S, H, blocked_dim)``. A blocked
        action dimension may contain multiple flattened environment actions;
        each is applied before emitting the state for that planning step.
        The returned trajectory includes the initial state.
        """
        if action_candidates.ndim != 4:
            raise ValueError(
                'action_candidates must have shape (B, S, H, action_dim)'
            )
        if self.state_key not in info_dict:
            raise KeyError(
                f'simulator rollout requires {self.state_key!r}; '
                f'available keys: {sorted(info_dict)}'
            )

        batch_size, samples, horizon, blocked_dim = action_candidates.shape
        state = torch.as_tensor(
            info_dict[self.state_key],
            device=action_candidates.device,
            dtype=action_candidates.dtype,
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] != batch_size:
            raise ValueError(
                f'{self.state_key!r} batch size {state.shape[0]} does not '
                f'match action batch size {batch_size}'
            )
        if state.ndim == 2:
            state = state[:, None].expand(batch_size, samples, -1)
        elif state.ndim == 3:
            state = state[:, -1, None].expand(batch_size, samples, -1)
        elif state.ndim == 4:
            if state.shape[1] != samples:
                raise ValueError(
                    f'{self.state_key!r} sample size {state.shape[1]} does '
                    f'not match candidate sample size {samples}'
                )
            state = state[:, :, -1]
        else:
            raise ValueError(
                f'{self.state_key!r} must have shape (B, D), (B, T, D), '
                'or (B, S, T, D)'
            )
        state = state.clone()

        action_shape = getattr(self.simulator.action_space, 'shape', None)
        if action_shape is None:
            raise TypeError('simulator action space must define a shape')
        action_dim = int(np.prod(action_shape))
        if blocked_dim % action_dim:
            raise ValueError(
                f'candidate action dimension {blocked_dim} is not a multiple '
                f'of simulator action dimension {action_dim}'
            )
        action_block = blocked_dim // action_dim
        actions = action_candidates.reshape(
            batch_size, samples, horizon, action_block, action_dim
        )

        states = [state]
        for plan_step in range(horizon):
            for block_step in range(action_block):
                state = self.predict(
                    state, actions[:, :, plan_step, block_step]
                )
            states.append(state)

        info_dict[self.rollout_key] = torch.stack(states, dim=2)
        return info_dict


def simulator_goal_encode(
    model: SimulatorDynamics, info_dict: dict
) -> torch.Tensor:
    """Goal encoder for ``ShootingCostEvaluator`` with simulator dynamics."""
    if not isinstance(model, SimulatorDynamics):
        raise TypeError('simulator_goal_encode requires SimulatorDynamics')
    return model.encode_goal(info_dict)


__all__ = ['Simulatable', 'SimulatorDynamics', 'simulator_goal_encode']
