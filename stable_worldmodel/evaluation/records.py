"""Structured, serializable closed-loop evaluation records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jsonable(value: Any) -> Any:
    """Recursively convert tensors and NumPy values to JSON-safe values."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def recordable(value: Any) -> Any:
    """Detach/copy nested values while preserving compact array storage."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(k): recordable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [recordable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(recordable(v) for v in value)
    return value


@dataclass(frozen=True)
class StepRecord:
    """One realized controller/environment transition."""

    decision: int
    observation: dict[str, Any]
    action: Any
    next_observation: dict[str, Any]
    reward: float
    cost: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeResult:
    """Per-task outcome plus optional full transition trace."""

    task_key: str
    environment_seed: int
    controller_seed: int
    success: bool
    episode_return: float
    length: int
    path_cost: float
    control_cost: float
    collisions: int
    constraint_violations: int
    steps: tuple[StepRecord, ...] = ()


@dataclass(frozen=True)
class EvaluationResults:
    """Backend-labelled ordered episode results returned by ``World``."""

    backend: str
    manifest_digest: str
    episodes: tuple[EpisodeResult, ...]
    model_queries: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return 100.0 * sum(ep.success for ep in self.episodes) / len(self.episodes)

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(ep.task_key for ep in self.episodes)

    @property
    def controller_seeds(self) -> tuple[int, ...]:
        return tuple(ep.controller_seed for ep in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['success_rate'] = self.success_rate
        return jsonable(value)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')
        return path

    def write_binary(self, path: str | Path) -> Path:
        """Write a compact tensor/array-preserving trace with ``torch.save``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)
        return path


__all__ = [
    'EpisodeResult',
    'EvaluationResults',
    'StepRecord',
    'jsonable',
    'recordable',
]
