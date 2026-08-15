import json
from dataclasses import replace

import pytest

from stable_worldmodel.evaluation import (
    EvaluationManifest,
    TaskKey,
    assert_paired,
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


def test_manifest_read_requires_integrity_fields(tmp_path):
    manifest = make_manifest('validation')

    missing_digest = manifest.to_dict()
    missing_digest.pop('digest')
    digest_path = tmp_path / 'missing-digest.json'
    digest_path.write_text(json.dumps(missing_digest))
    with pytest.raises(ValueError, match='required digest'):
        EvaluationManifest.read(digest_path)

    missing_task_key = manifest.to_dict()
    missing_task_key['tasks'][0].pop('task_key')
    task_path = tmp_path / 'missing-task-key.json'
    task_path.write_text(json.dumps(missing_task_key))
    with pytest.raises(ValueError, match='required task_key'):
        EvaluationManifest.read(task_path)


def test_manifest_rejects_unknown_schema_version():
    with pytest.raises(
        ValueError, match='unsupported manifest schema_version'
    ):
        EvaluationManifest(
            split='validation',
            environment='swm/TwoRoom-v1',
            tasks=make_manifest('validation').tasks,
            schema_version=EvaluationManifest.CURRENT_SCHEMA + 1,
        )


def test_task_identity_excludes_controller_seed_and_name():
    task = make_manifest('validation').tasks[0]

    assert task.key == replace(task, controller_seed=999, name='renamed').key
    assert task.key != replace(task, environment_seed=999).key


def test_manifest_allows_project_defined_split_names():
    assert make_manifest('custom_holdout').split == 'custom_holdout'


def test_pairing_is_order_sensitive():
    first = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(
            make_manifest('validation', 1).tasks[0],
            make_manifest('validation', 2).tasks[0],
        ),
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

    different_controller = EvaluationManifest(
        split='validation',
        environment='swm/TwoRoom-v1',
        tasks=(
            replace(first.tasks[0], controller_seed=999),
            first.tasks[1],
        ),
    )
    with pytest.raises(ValueError, match='controller seeds'):
        assert_paired(first, different_controller)
