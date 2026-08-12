import json

import pytest

from stable_worldmodel.evaluation import (
    SPLITS,
    EvaluationManifest,
    TaskKey,
    assert_paired,
    validate_manifest_suite,
)


def make_manifest(split, offset=0):
    return EvaluationManifest(
        split=split,
        environment='swm/TwoRoom-v1',
        tasks=(
            TaskKey(
                environment_seed=10 + offset,
                controller_seed=20 + offset,
                layout_seed=30 + offset,
                start=(40.0 + offset, 50.0),
                goal=(170.0, 180.0),
                observation_noise_seed=40 + offset,
            ),
        ),
    )


def test_manifest_round_trip_and_immutable_write(tmp_path):
    manifest = make_manifest('validation')
    path = manifest.write(tmp_path / 'validation.json')
    assert EvaluationManifest.read(path) == manifest
    assert manifest.write(path) == path

    data = json.loads(path.read_text())
    data['tasks'][0]['goal'][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='task_key'):
        EvaluationManifest.read(path)


def test_manifest_suite_is_complete_and_disjoint():
    suite = [make_manifest(split, i) for i, split in enumerate(SPLITS)]
    validate_manifest_suite(suite)

    leaked = list(suite)
    leaked[-1] = EvaluationManifest(
        split=SPLITS[-1],
        environment='swm/TwoRoom-v1',
        tasks=suite[0].tasks,
    )
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest_suite(leaked)


def test_pairing_is_order_sensitive():
    first = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(make_manifest('validation', 1).tasks[0], make_manifest('validation', 2).tasks[0]),
    )
    same = EvaluationManifest(
        split='validation', environment='swm/TwoRoom-v1', tasks=first.tasks
    )
    assert_paired(first, same)

    reversed_manifest = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=tuple(reversed(first.tasks)),
    )
    with pytest.raises(ValueError, match='ordered'):
        assert_paired(first, reversed_manifest)
