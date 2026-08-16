"""Immutable specifications for reproducible control evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class EvaluationTask:
    """One fully specified environment/controller evaluation task."""

    environment_seed: int
    controller_seed: int
    layout_seed: int
    start: tuple[float, ...]
    goal: tuple[float, ...]
    observation_noise_seed: int
    dynamics_parameters: tuple[tuple[str, Any], ...] = ()
    options: tuple[tuple[str, Any], ...] = ()
    name: str = ''
    _legacy_key: str | None = field(
        default=None, repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, 'start', tuple(self.start))
        object.__setattr__(self, 'goal', tuple(self.goal))
        object.__setattr__(
            self,
            'dynamics_parameters',
            tuple(sorted((str(k), v) for k, v in self.dynamics_parameters)),
        )
        object.__setattr__(
            self,
            'options',
            tuple(sorted((str(k), v) for k, v in self.options)),
        )

    @property
    def key(self) -> str:
        """Environment-task identity, independent of RNG and display name.

        Protocols read from older files retain their historical full-task
        keys. New tasks use the corrected environment-only identity.
        """
        if self._legacy_key == self._full_content_key:
            return self._legacy_key
        payload = self._content_dict()
        payload.pop('controller_seed')
        payload.pop('name')
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    @property
    def _full_content_key(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._content_dict()).encode()
        ).hexdigest()

    def _content_dict(self) -> dict[str, Any]:
        return {
            'environment_seed': self.environment_seed,
            'controller_seed': self.controller_seed,
            'layout_seed': self.layout_seed,
            'start': self.start,
            'goal': self.goal,
            'observation_noise_seed': self.observation_noise_seed,
            'dynamics_parameters': self.dynamics_parameters,
            'options': self.options,
            'name': self.name,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._content_dict()
        value['dynamics_parameters'] = dict(self.dynamics_parameters)
        value['options'] = dict(self.options)
        value['task_key'] = self.key
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationTask:
        value = dict(value)
        try:
            expected_key = value.pop('task_key')
        except KeyError as error:
            raise ValueError(
                'serialized task is missing required task_key'
            ) from error
        value['dynamics_parameters'] = tuple(
            value.get('dynamics_parameters', {}).items()
        )
        value['options'] = tuple(value.get('options', {}).items())
        task = cls(**value)
        if expected_key not in {task.key, task._full_content_key}:
            raise ValueError('task_key does not match the task contents')
        if expected_key == task._full_content_key:
            object.__setattr__(task, '_legacy_key', expected_key)
        return task


@dataclass(frozen=True)
class EvaluationProtocol:
    """Versioned, ordered contract shared by evaluation conditions."""

    CURRENT_SCHEMA: ClassVar[int] = 1

    split: str
    environment: str
    tasks: tuple[EvaluationTask, ...]
    version: str = '1.0.0'
    schema_version: int = CURRENT_SCHEMA
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA:
            raise ValueError(
                f'unsupported protocol schema_version {self.schema_version}; '
                f'expected {self.CURRENT_SCHEMA}'
            )
        if not self.split:
            raise ValueError('protocol split must be a non-empty string')
        if not self.environment:
            raise ValueError('protocol environment must be a non-empty string')
        object.__setattr__(self, 'tasks', tuple(self.tasks))
        object.__setattr__(
            self,
            'metadata',
            tuple(sorted((str(k), v) for k, v in self.metadata)),
        )
        if not self.tasks:
            raise ValueError('a protocol must contain at least one task')
        if not all(isinstance(task, EvaluationTask) for task in self.tasks):
            raise TypeError('protocol tasks must be EvaluationTask instances')
        if len(self.task_keys) != len(set(self.task_keys)):
            raise ValueError('a protocol may not contain duplicate task keys')

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(task.key for task in self.tasks)

    @property
    def controller_seeds(self) -> tuple[int, ...]:
        return tuple(task.controller_seed for task in self.tasks)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_dict(include_digest=False)).encode()
        ).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {
            'schema_version': self.schema_version,
            'version': self.version,
            'split': self.split,
            'environment': self.environment,
            'metadata': dict(self.metadata),
            'tasks': [task.to_dict() for task in self.tasks],
        }
        if include_digest:
            value['digest'] = self.digest
        return value

    def write(self, path: str | Path) -> Path:
        """Write once, refusing to replace a different frozen protocol."""
        path = Path(path)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'
        if path.exists():
            if path.read_text() != payload:
                raise FileExistsError(
                    f'refusing to overwrite immutable protocol {path}'
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    @classmethod
    def read(cls, path: str | Path) -> EvaluationProtocol:
        value = json.loads(Path(path).read_text())
        try:
            expected_digest = value.pop('digest')
        except KeyError as error:
            raise ValueError(
                f'protocol is missing required digest: {path}'
            ) from error
        tasks = tuple(
            EvaluationTask.from_dict(item) for item in value.pop('tasks')
        )
        value['metadata'] = tuple(value.get('metadata', {}).items())
        protocol = cls(tasks=tasks, **value)
        if protocol.digest != expected_digest:
            raise ValueError(f'protocol digest mismatch for {path}')
        return protocol


def assert_paired(left: EvaluationProtocol, right: EvaluationProtocol) -> None:
    """Require two conditions to use the same ordered tasks and RNG ledger."""
    if left.task_keys != right.task_keys:
        raise ValueError('paired protocols have different ordered task keys')
    if left.controller_seeds != right.controller_seeds:
        raise ValueError('paired protocols have different controller seeds')


__all__ = [
    'EvaluationProtocol',
    'EvaluationTask',
    'assert_paired',
]
