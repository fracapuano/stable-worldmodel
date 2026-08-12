"""Paired, manifest-driven control evaluation infrastructure."""

from stable_worldmodel.evaluation.audit import (
    G0Audit,
    Invariant,
    build_g0_audit,
    controller_hash,
)
from stable_worldmodel.evaluation.manifest import (
    SPLITS,
    EvaluationManifest,
    TaskKey,
    assert_paired,
    validate_manifest_suite,
)
from stable_worldmodel.evaluation.records import (
    EpisodeResult,
    EvaluationResults,
    StepRecord,
)


__all__ = [
    'SPLITS',
    'EpisodeResult',
    'EvaluationManifest',
    'EvaluationResults',
    'G0Audit',
    'Invariant',
    'StepRecord',
    'TaskKey',
    'assert_paired',
    'build_g0_audit',
    'controller_hash',
    'validate_manifest_suite',
]
