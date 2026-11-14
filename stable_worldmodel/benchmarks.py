"""Load and run frozen benchmarks hosted as Hugging Face dataset files."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from stable_worldmodel.evaluation import EvaluationProtocol, EvaluationResults
from stable_worldmodel.policy import Policy


@dataclass(frozen=True)
class Benchmark:
    protocol: EvaluationProtocol
    eval_budget: int
    image_shape: tuple[int, int] = (224, 224)

    def evaluate(
        self, policy: Policy, *, device: str | None = None
    ) -> EvaluationResults:
        """Score the full suite and return the evaluator's structured results."""
        return self._run(self.protocol, policy, device=device)

    def record_rollouts(
        self,
        policy: Policy,
        directory: str | Path,
        task_indices: Iterable[int],
        *,
        device: str | None = None,
    ) -> EvaluationResults:
        """Run a diagnostic evaluation and save MP4s for selected tasks."""
        indices = tuple(task_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(
                'provide at least one task index, without duplicates'
            )
        if any(not 0 <= index < len(self.protocol.tasks) for index in indices):
            raise ValueError('task index is outside the benchmark suite')

        tasks = tuple(self.protocol.tasks[index] for index in indices)
        labels = tuple(
            task.name or f'task-{i}' for i, task in enumerate(tasks)
        )
        if len(set(labels)) != len(labels) or any(
            '/' in label or '\\' in label for label in labels
        ):
            raise ValueError('rollout task names must be unique filenames')
        protocol = replace(self.protocol, tasks=tasks)
        return self._run(protocol, policy, device=device, video=directory)

    def _run(
        self, protocol, policy, *, device, video=None
    ) -> EvaluationResults:
        import torch

        from stable_worldmodel import World

        world = World(
            protocol.environment,
            num_envs=len(protocol.tasks),
            image_shape=self.image_shape,
            max_episode_steps=self.eval_budget,
            device=device,
        )
        try:
            world.set_policy(policy)
            with torch.inference_mode():
                result = world.evaluate(
                    protocol=protocol,
                    eval_budget=self.eval_budget,
                    video=video,
                )
            if 'final_distances' not in result.metadata:
                distances = world.infos.get('distance_to_target')
                if distances is not None:
                    distances = torch.as_tensor(distances).flatten().tolist()
                    result = replace(
                        result,
                        metadata={
                            **result.metadata,
                            'final_distances': distances,
                            'mean_final_distance': sum(distances)
                            / len(distances),
                        },
                    )
            return result
        finally:
            world.close()


def load_benchmark(
    source: str | Path,
    *,
    revision: str | None = None,
    split: str = 'validation',
) -> Benchmark:
    """Load a local JSON manifest or an HF dataset's `<split>.json` file.

    Pass a Hugging Face commit SHA as `revision` for reproducible downloads.
    The manifest supplies the evaluation budget via `eval_budget` or `steps`
    and can specify an `image_shape` in its metadata.
    """
    path = Path(source)
    if isinstance(source, Path) or path.is_file():
        if revision is not None:
            raise ValueError('revision applies only to Hugging Face datasets')
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=source,
            repo_type='dataset',
            filename=f'{split}.json',
            revision=revision,
        )

    protocol = EvaluationProtocol.read(path)
    metadata = dict(protocol.metadata)
    budget = metadata.get('eval_budget', metadata.get('steps'))
    if type(budget) is not int or budget < 1:
        raise ValueError(
            'benchmark metadata must specify a positive evaluation budget'
        )
    image_shape = tuple(metadata.get('image_shape', (224, 224)))
    if len(image_shape) != 2 or any(
        type(size) is not int or size < 1 for size in image_shape
    ):
        raise ValueError(
            'benchmark image_shape must contain two positive integers'
        )
    return Benchmark(protocol, budget, image_shape)
