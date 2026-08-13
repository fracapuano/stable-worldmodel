"""Immutable task manifests and paired controller seed ledgers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar


SPLITS = (
    'fitness',
    'validation',
    'test_id',
    'test_layout',
    'test_noise',
    'test_dynamics',
    'prediction_validation',
    'prediction_test',
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class TaskKey:
    """A complete, stable key for one control-evaluation task."""

    environment_seed: int
    controller_seed: int
    layout_seed: int
    start: tuple[float, ...]
    goal: tuple[float, ...]
    observation_noise_seed: int
    dynamics_parameters: tuple[tuple[str, Any], ...] = ()
    options: tuple[tuple[str, Any], ...] = ()
    name: str = ''

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
        """Content-addressed task key (independent of split and ordering)."""
        payload = asdict(self)
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['dynamics_parameters'] = dict(self.dynamics_parameters)
        value['options'] = dict(self.options)
        value['task_key'] = self.key
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskKey:
        value = dict(value)
        if 'task_key' not in value:
            raise ValueError('serialized task is missing required task_key')
        expected_key = value.pop('task_key')
        value['dynamics_parameters'] = tuple(
            value.get('dynamics_parameters', {}).items()
        )
        value['options'] = tuple(value.get('options', {}).items())
        task = cls(**value)
        if task.key != expected_key:
            raise ValueError('task_key does not match the task contents')
        return task


@dataclass(frozen=True)
class EvaluationManifest:
    """Versioned, ordered, immutable collection of evaluation tasks."""

    CURRENT_SCHEMA: ClassVar[int] = 1

    split: str
    environment: str
    tasks: tuple[TaskKey, ...]
    version: str = '1.0.0'
    schema_version: int = CURRENT_SCHEMA
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA:
            raise ValueError(
                f'unsupported manifest schema_version {self.schema_version}; '
                f'expected {self.CURRENT_SCHEMA}'
            )
        if self.split not in SPLITS:
            raise ValueError(
                f'unknown split {self.split!r}; expected one of {SPLITS}'
            )
        object.__setattr__(self, 'tasks', tuple(self.tasks))
        object.__setattr__(
            self,
            'metadata',
            tuple(sorted((str(k), v) for k, v in self.metadata)),
        )
        if not self.tasks:
            raise ValueError('a manifest must contain at least one task')
        keys = self.task_keys
        if len(keys) != len(set(keys)):
            raise ValueError('a manifest may not contain duplicate task keys')

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
        """Write once; refuse to silently replace different manifest data."""
        path = Path(path)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'
        if path.exists():
            existing = path.read_text()
            if existing != payload:
                raise FileExistsError(
                    f'refusing to overwrite immutable manifest {path}'
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    @classmethod
    def read(cls, path: str | Path) -> EvaluationManifest:
        value = json.loads(Path(path).read_text())
        if 'digest' not in value:
            raise ValueError(f'manifest is missing required digest: {path}')
        expected_digest = value.pop('digest')
        tasks = tuple(TaskKey.from_dict(item) for item in value.pop('tasks'))
        value['metadata'] = tuple(value.get('metadata', {}).items())
        manifest = cls(tasks=tasks, **value)
        if manifest.digest != expected_digest:
            raise ValueError(f'manifest digest mismatch for {path}')
        return manifest


def validate_manifest_suite(manifests: list[EvaluationManifest]) -> None:
    """Validate complete split coverage and cross-split task disjointness."""
    by_split = {manifest.split: manifest for manifest in manifests}
    missing = set(SPLITS) - set(by_split)
    if missing:
        raise ValueError(f'missing required manifests: {sorted(missing)}')
    if len(by_split) != len(manifests):
        raise ValueError('only one manifest version may be active per split')

    seen: dict[str, str] = {}
    for manifest in manifests:
        for key in manifest.task_keys:
            if key in seen:
                raise ValueError(
                    f'task leakage between {seen[key]!r} and '
                    f'{manifest.split!r}: {key}'
                )
            seen[key] = manifest.split


def assert_paired(left: EvaluationManifest, right: EvaluationManifest) -> None:
    """Require exact ordered task and controller-RNG pairing."""
    if left.task_keys != right.task_keys:
        raise ValueError('paired manifests have different ordered task keys')
    if left.controller_seeds != right.controller_seeds:
        raise ValueError('paired manifests have different controller seeds')


__all__ = [
    'SPLITS',
    'TaskKey',
    'EvaluationManifest',
    'assert_paired',
    'validate_manifest_suite',
]
