"""Controller hashing and machine-readable Gate G0 invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_worldmodel.evaluation.records import EvaluationResults, jsonable


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def controller_hash(config: Any, *sources: Any) -> str:
    """Hash canonical controller config together with implementation source."""
    source_text = []
    for source in sources:
        try:
            source_text.append(inspect.getsource(source))
        except (OSError, TypeError):
            source_text.append(repr(source))
    payload = {
        'config': _normalize(config),
        'source': source_text,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Invariant:
    name: str
    passed: bool
    expected: Any
    actual: Any
    tolerance: float | None = None
    detail: str = ''


@dataclass(frozen=True)
class G0Audit:
    """Machine-readable pass/fail report for the oracle-gap harness."""

    schema_version: int
    swm_revision: str
    spt_revision: str
    invariants: tuple[Invariant, ...]
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.invariants)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(
            {
                'schema_version': self.schema_version,
                'gate': 'G0',
                'passed': self.passed,
                'swm_revision': self.swm_revision,
                'spt_revision': self.spt_revision,
                'invariants': [asdict(item) for item in self.invariants],
                'metadata': self.metadata,
            }
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')
        return path


def build_g0_audit(
    learned: EvaluationResults,
    oracle: EvaluationResults,
    *,
    learned_controller_hash: str,
    oracle_controller_hash: str,
    one_step_error: float,
    horizon_error: float,
    replay_tolerance: float,
    learned_query_budget: Any,
    oracle_query_budget: Any,
    learned_stopping_rules: Any,
    oracle_stopping_rules: Any,
    deterministic_pairs: list[tuple[EvaluationResults, EvaluationResults]],
    swm_revision: str,
    spt_revision: str,
    expected_episodes: int,
) -> G0Audit:
    """Evaluate every mandatory Experiment-0 invariant."""
    def initial_candidate(result):
        try:
            return result.model_queries[0]['solver_output']['callbacks'][
                'model_queries'
            ][0][0]['candidates']
        except (IndexError, KeyError, TypeError):
            return None

    def tensor_digest(value):
        if value is None:
            return None
        array = (
            value.detach().cpu().contiguous().numpy()
            if torch.is_tensor(value)
            else np.ascontiguousarray(value)
        )
        return {
            'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
        }

    learned_candidates = tensor_digest(initial_candidate(learned))
    oracle_candidates = tensor_digest(initial_candidate(oracle))
    invariants = [
        Invariant(
            'oracle_one_step_replay',
            one_step_error <= replay_tolerance,
            f'<= {replay_tolerance}',
            one_step_error,
            replay_tolerance,
        ),
        Invariant(
            'oracle_horizon_replay',
            horizon_error <= replay_tolerance,
            f'<= {replay_tolerance}',
            horizon_error,
            replay_tolerance,
        ),
        Invariant(
            'controller_hash_equality',
            learned_controller_hash == oracle_controller_hash,
            learned_controller_hash,
            oracle_controller_hash,
        ),
        Invariant(
            'ordered_task_pairing',
            learned.task_keys == oracle.task_keys,
            learned.task_keys,
            oracle.task_keys,
        ),
        Invariant(
            'manifest_digest_equality',
            learned.manifest_digest == oracle.manifest_digest,
            learned.manifest_digest,
            oracle.manifest_digest,
        ),
        Invariant(
            'backend_identity',
            bool(learned.backend)
            and bool(oracle.backend)
            and learned.backend != oracle.backend,
            'two distinct, non-empty backend labels',
            {'learned': learned.backend, 'oracle': oracle.backend},
            detail='Guards against silent backend fallback.',
        ),
        Invariant(
            'controller_rng_pairing',
            learned.controller_seeds == oracle.controller_seeds,
            learned.controller_seeds,
            oracle.controller_seeds,
        ),
        Invariant(
            'initial_candidate_pairing',
            learned_candidates is not None
            and learned_candidates == oracle_candidates,
            learned_candidates,
            oracle_candidates,
            detail='Candidates before the first backend query must match.',
        ),
        Invariant(
            'query_budget_equality',
            learned_query_budget == oracle_query_budget,
            learned_query_budget,
            oracle_query_budget,
        ),
        Invariant(
            'stopping_rule_equality',
            learned_stopping_rules == oracle_stopping_rules,
            learned_stopping_rules,
            oracle_stopping_rules,
        ),
        Invariant(
            'complete_manifest',
            len(learned.episodes) == len(oracle.episodes) == expected_episodes,
            expected_episodes,
            {'learned': len(learned.episodes), 'oracle': len(oracle.episodes)},
        ),
    ]

    finite = all(
        math.isfinite(value)
        for result in (learned, oracle)
        for episode in result.episodes
        for value in (
            episode.episode_return,
            episode.path_cost,
            episode.control_cost,
        )
    )
    invariants.append(Invariant('no_nans', finite, True, finite))

    repeat_equal = True
    repeat_detail = []
    for first, second in deterministic_pairs:
        first_summary = [
            (episode.task_key, episode.episode_return, [s.action for s in episode.steps])
            for episode in first.episodes
        ]
        second_summary = [
            (episode.task_key, episode.episode_return, [s.action for s in episode.steps])
            for episode in second.episodes
        ]
        equal = jsonable(first_summary) == jsonable(second_summary)
        repeat_equal &= equal
        repeat_detail.append(equal)
    invariants.append(
        Invariant(
            'deterministic_repeats',
            repeat_equal and bool(deterministic_pairs),
            True,
            repeat_detail,
        )
    )

    return G0Audit(
        schema_version=1,
        swm_revision=swm_revision,
        spt_revision=spt_revision,
        invariants=tuple(invariants),
        metadata={
            'learned_backend': learned.backend,
            'oracle_backend': oracle.backend,
            'manifest_digest': learned.manifest_digest,
        },
    )


__all__ = ['G0Audit', 'Invariant', 'build_g0_audit', 'controller_hash']
