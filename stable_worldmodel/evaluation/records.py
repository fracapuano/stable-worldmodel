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
    """Copy nested values while preserving tensors for compact storage."""
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
    """Per-task outcome plus an optional transition trace."""

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


def episodes_from_rollout(
    protocol, outcome
) -> tuple[tuple[EpisodeResult, ...], ...]:
    """Convert population/task rollout metrics once at the reporting boundary."""
    successes, returns, lengths, paths, controls, collisions = (
        value.detach().cpu().numpy()
        for value in (
            outcome.successes,
            outcome.returns,
            outcome.lengths,
            outcome.path_costs,
            outcome.control_costs,
            outcome.collisions,
        )
    )
    return tuple(
        tuple(
            EpisodeResult(
                task_key=task.key,
                environment_seed=task.environment_seed,
                controller_seed=task.controller_seed,
                success=bool(successes[p, t]),
                episode_return=float(returns[p, t]),
                length=int(lengths[p, t]),
                path_cost=float(paths[p, t]),
                control_cost=float(controls[p, t]),
                collisions=int(collisions[p, t]),
                constraint_violations=0,
                steps=(outcome.records[p][t] if outcome.records else ()),
            )
            for t, task in enumerate(protocol.tasks)
        )
        for p in range(len(successes))
    )


def rollout_records(infos, actions, lengths, successes, observations, rewards):
    """Materialize optional traces once, after the batched GPU execution.

    Observations are time-major next observations; initial/goal observations
    come from the existing task reset. No simulator is stepped during export.
    """
    infos, actions, lengths, successes, observations, rewards = recordable(
        (infos, actions, lengths, successes, observations, rewards)
    )
    population = []
    for p in range(actions.shape[0]):
        tasks = []
        for t in range(actions.shape[1]):
            previous = {
                key: value[t]
                for key, value in infos.items()
                if not key.startswith('_')
            }
            steps = []
            for h in range(int(lengths[p, t])):
                current = dict(previous)
                updates = {
                    key: value[h, p, t] for key, value in observations.items()
                }
                updates.update(
                    action=actions[p, t, h], step_idx=torch.tensor(h + 1)
                )
                for key, value in updates.items():
                    value = torch.as_tensor(value)
                    before = previous.get(key)
                    if before is not None:
                        before = torch.as_tensor(before)
                        value = value.reshape(1, *before.shape[1:])
                        current[key] = torch.cat((before, value))[
                            -len(before) :
                        ]
                    else:
                        current[key] = value[None]
                terminal = h + 1 == int(lengths[p, t])
                hit = terminal and bool(successes[p, t])
                current['terminated'] = hit
                reward = float(rewards[h, p, t])
                steps.append(
                    StepRecord(
                        decision=h,
                        observation=previous,
                        action=actions[p, t, h],
                        next_observation=current,
                        reward=reward,
                        cost=-reward,
                        terminated=hit,
                        truncated=terminal and not hit,
                    )
                )
                previous = current
            tasks.append(tuple(steps))
        population.append(tuple(tasks))
    return tuple(population)


@dataclass(frozen=True)
class EvaluationResults:
    """Condition-labelled, ordered results returned by ``World``."""

    backend: str
    protocol_digest: str
    episodes: tuple[EpisodeResult, ...]
    model_queries: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    backend_type: str = ''

    @property
    def success_rate(self) -> float:
        return (
            100.0
            * sum(episode.success for episode in self.episodes)
            / len(self.episodes)
        )

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(episode.task_key for episode in self.episodes)

    @property
    def controller_seeds(self) -> tuple[int, ...]:
        return tuple(episode.controller_seed for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['success_rate'] = self.success_rate
        return jsonable(value)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'
        )
        return path

    def write_binary(self, path: str | Path) -> Path:
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
