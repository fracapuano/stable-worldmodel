"""Shared TwoRoom checkpoint, protocol, and transforms for the examples."""

import tempfile
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.evaluation import (
    EvaluationProtocol,
    EvaluationTask,
    assert_paired,
)

CHECKPOINT = 'quentinll/lewm-tworooms'
ENV_ID = 'swm/TwoRoom-v1'
IMAGE_SHAPE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EVAL_BUDGET = 50
DEVICE = torch.accelerator.current_accelerator(check_available=True) or 'cpu'

# Standard Planning config: 5 chunks (5 actions/chunk), execute 1 chunk, replan.
PLAN_CONFIG = swm.PlanConfig(
    horizon=5,
    receding_horizon=1,
    history_len=3,
    action_block=5,
)

# Training-set stats for CHECKPOINT: CEM searches normalized actions
ACTION_PROCESS = {
    'action': swm.data.ZScoreScaler(
        mean=[[0.00309341, -0.05298233]],
        std=[[0.8674758, 0.86776555]],
    )
}

# Shared layout - starts are in the left room, goals in the right.
TASK_STARTS_GOALS = (
    ((40.0, 50.0), (170.0, 180.0)),
    ((50.0, 80.0), (180.0, 40.0)),
    ((45.0, 120.0), (165.0, 60.0)),
    ((55.0, 160.0), (175.0, 140.0)),
    ((35.0, 70.0), (185.0, 110.0)),
    ((60.0, 40.0), (160.0, 170.0)),
    ((42.0, 100.0), (178.0, 90.0)),
    ((48.0, 140.0), (172.0, 50.0)),
    ((38.0, 175.0), (168.0, 155.0)),
    ((52.0, 55.0), (182.0, 125.0)),
)

EXAMPLES_DIR = Path(__file__).resolve().parent


def video_dir(name: str) -> Path:
    return EXAMPLES_DIR / 'videos' / name


def make_protocol() -> EvaluationProtocol:
    layout_seed = 0
    tasks = tuple(
        EvaluationTask(
            environment_seed=layout_seed,
            controller_seed=1_000 + i,
            layout_seed=layout_seed,
            start=start,
            goal=goal,
            observation_noise_seed=2_000 + i,
            name=f'task-{i}',
        )
        for i, (start, goal) in enumerate(TASK_STARTS_GOALS)
    )
    return EvaluationProtocol(
        split='validation',
        environment=ENV_ID,
        tasks=tasks,
        metadata=(('suite', 'tworoom'),),
    )


def freeze_protocol(protocol: EvaluationProtocol) -> EvaluationProtocol:
    """Round-trip through the immutable on-disk format."""
    with tempfile.TemporaryDirectory() as tmp:
        path = protocol.write(Path(tmp) / 'protocol.json')
        loaded = EvaluationProtocol.read(path)
    assert_paired(protocol, loaded)
    if loaded.digest != protocol.digest:
        raise ValueError('reloaded protocol digest does not match')
    return loaded


def load_protocol() -> EvaluationProtocol:
    return freeze_protocol(make_protocol())


def pixel_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=IMAGE_SHAPE[0]),
        ]
    )


def pixel_transforms():
    return {'pixels': pixel_transform(), 'goal': pixel_transform()}


def make_world(protocol):
    return swm.World(
        ENV_ID,
        num_envs=len(protocol.tasks),
        image_shape=IMAGE_SHAPE,
        max_episode_steps=EVAL_BUDGET,
    )


def load_lewm(device=DEVICE):
    model = swm.wm.utils.load_pretrained(CHECKPOINT).to(device).eval()
    model.requires_grad_(False)
    return model
