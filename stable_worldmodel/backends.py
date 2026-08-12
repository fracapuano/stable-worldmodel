"""Interchangeable learned and exact simulator model backends.

The classes in this module deliberately implement the existing
:class:`stable_worldmodel.protocols.Dynamics` surface.  They can therefore be
composed with :class:`~stable_worldmodel.planning.ShootingCostEvaluator` and
the normal SWM solvers without a second planning or environment loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from stable_worldmodel.protocols import Dynamics


@runtime_checkable
class ExactSimulator(Protocol):
    """Minimal opt-in contract for simulator-backed planning.

    The transition query must be side-effect free and accept tensors with any
    leading batch dimensions.  It returns the exact next privileged state for
    one *environment* action, not one action block.
    """

    action_space: Any

    def get_oracle_state(self) -> torch.Tensor:
        """Return a copy of the current privileged Markov state."""
        ...

    def set_oracle_state(self, state: torch.Tensor | np.ndarray) -> None:
        """Restore a state returned by :meth:`get_oracle_state`."""
        ...

    def oracle_transition(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Apply the exact transition without mutating the simulator."""
        ...


class LearnedModelBackend(nn.Module):
    """Frozen ``Dynamics`` facade for an SWM pretrained world model.

    ``from_pretrained`` intentionally delegates checkpoint resolution and
    reconstruction to :func:`stable_worldmodel.wm.utils.load_pretrained`.
    The wrapper adds no prediction behavior; its purpose is to give evaluation
    code an explicit, backend-neutral type and to make freezing the default.
    """

    def __init__(self, model: Dynamics, *, freeze: bool = True) -> None:
        super().__init__()
        if not isinstance(model, Dynamics):
            raise TypeError('model must implement the SWM Dynamics protocol')
        self.model = model
        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        cache_dir: str | None = None,
        freeze: bool = True,
        map_location: str | torch.device | None = None,
    ) -> 'LearnedModelBackend':
        """Load an SWM checkpoint through the package checkpoint utility."""
        from stable_worldmodel.wm.utils import load_pretrained

        model = load_pretrained(checkpoint, cache_dir=cache_dir)
        if map_location is not None:
            model = model.to(map_location)
        return cls(model, freeze=freeze)

    def encode(self, x: dict) -> dict:
        return self.model.encode(x)

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        return self.model.rollout(info_dict, action_candidates)


class OracleModelBackend(nn.Module):
    """Exact simulator adapter implementing SWM's ``Dynamics`` protocol.

    The backend is bound to the environments by ``WorldModelPolicy.set_env``.
    Planning inputs contain only a stable ``env_index``; privileged state and
    geometry are read directly from the corresponding simulator and never
    exposed to the controller.  Candidate queries are side-effect free.

    Args:
        state_key: Key used by :meth:`encode` when encoding a goal state.
        embedding_key: Key populated by :meth:`encode`.
        rollout_key: Key populated by :meth:`rollout`.  The defaults match
            :class:`stable_worldmodel.planning.GoalMSE`.
        decode_action: Optional conversion from controller/model action space
            to simulator action space (for example, an inverse normalizer).
    """

    def __init__(
        self,
        *,
        state_key: str = 'state',
        embedding_key: str = 'emb',
        rollout_key: str = 'predicted_emb',
        decode_action: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.state_key = state_key
        self.embedding_key = embedding_key
        self.rollout_key = rollout_key
        self.decode_action = decode_action
        self._simulators: list[ExactSimulator] = []

    def bind_envs(self, envs: Any) -> None:
        """Bind an SWM ``EnvPool`` (or iterable of Gym environments)."""
        candidates = getattr(envs, 'envs', envs)
        simulators = [getattr(env, 'unwrapped', env) for env in candidates]
        for simulator in simulators:
            if not isinstance(simulator, ExactSimulator):
                raise TypeError(
                    f'{type(simulator).__name__} does not implement the '
                    'ExactSimulator contract'
                )
        self._simulators = simulators

    def encode(self, x: dict) -> dict:
        """Use the privileged goal state as the oracle embedding."""
        if self.state_key not in x:
            raise KeyError(
                f'oracle goal encoding requires {self.state_key!r}; '
                f'available keys: {sorted(x)}'
            )
        state = x[self.state_key]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        x[self.embedding_key] = state.to(dtype=torch.float32)
        return x

    @staticmethod
    def _indices(info_dict: dict, batch_size: int) -> list[int]:
        if 'env_index' not in info_dict:
            return list(range(batch_size))
        value = info_dict['env_index']
        value = torch.as_tensor(value) if not torch.is_tensor(value) else value
        value = value.reshape(batch_size, -1)[:, 0]
        return [int(item) for item in value.detach().cpu().tolist()]

    def rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Roll candidates through the exact bound simulators.

        A candidate's last dimension may contain an action block flattened by
        ``PlanConfig.action_block``.  Every physical action in the block is
        applied, and one state is emitted per planning step, matching learned
        model time semantics.
        """
        if not self._simulators:
            raise RuntimeError(
                'OracleModelBackend is not bound to simulators; attach its '
                'WorldModelPolicy to a World before planning'
            )
        if action_candidates.ndim != 4:
            raise ValueError(
                'action_candidates must have shape (B, S, H, action_dim)'
            )

        batch_size, samples, horizon, blocked_dim = action_candidates.shape
        indices = self._indices(info_dict, batch_size)
        device = action_candidates.device
        dtype = action_candidates.dtype
        per_env_rollouts = []

        for batch_idx, env_idx in enumerate(indices):
            try:
                simulator = self._simulators[env_idx]
            except IndexError as exc:
                raise IndexError(
                    f'env_index {env_idx} is outside the bound simulator pool'
                ) from exc

            action_dim = int(np.prod(simulator.action_space.shape))
            if blocked_dim % action_dim:
                raise ValueError(
                    f'candidate action dimension {blocked_dim} is not a '
                    f'multiple of simulator action dimension {action_dim}'
                )
            action_block = blocked_dim // action_dim
            actions = action_candidates[batch_idx].reshape(
                samples, horizon, action_block, action_dim
            )
            if self.decode_action is not None:
                actions = self.decode_action(actions)
                if not torch.is_tensor(actions):
                    actions = torch.as_tensor(actions, device=device, dtype=dtype)

            initial = simulator.get_oracle_state().to(device=device, dtype=dtype)
            state = initial.expand(samples, *initial.shape).clone()
            states = [state]
            for plan_step in range(horizon):
                for block_step in range(action_block):
                    state = simulator.oracle_transition(
                        state, actions[:, plan_step, block_step]
                    )
                states.append(state)
            per_env_rollouts.append(torch.stack(states, dim=1))

        info_dict[self.rollout_key] = torch.stack(per_env_rollouts, dim=0)
        return info_dict


def replay_simulator(
    env: Any, actions: torch.Tensor | np.ndarray
) -> torch.Tensor:
    """Replay actions in the real simulator and restore its original state.

    This is an oracle-correctness check, not another control method.  The
    returned tensor contains the initial state followed by every realized
    next state.
    """
    simulator = getattr(env, 'unwrapped', env)
    if not isinstance(simulator, ExactSimulator):
        raise TypeError('env does not implement the ExactSimulator contract')
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    original = simulator.get_oracle_state()
    states = [original.clone()]
    try:
        for action in action_tensor.reshape(-1, action_tensor.shape[-1]):
            simulator.step(action.detach().cpu().numpy())
            states.append(simulator.get_oracle_state())
    finally:
        simulator.set_oracle_state(original)
    return torch.stack(states)


__all__ = [
    'ExactSimulator',
    'LearnedModelBackend',
    'OracleModelBackend',
    'replay_simulator',
]
