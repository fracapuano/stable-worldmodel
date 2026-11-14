"""Run one published default suite without importing evolving-wm."""

import argparse
from pathlib import Path

import stable_worldmodel as swm
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.benchmarks import load_benchmark
from stable_worldmodel.planning import FastCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_worldmodel.policy import RandomPolicy, WorldModelPolicy
from torchvision.transforms import v2

REPO = "fracapuano/evolving-wm-benchmarks"
REVISION = "564e09249409358d2fd2b8023a049e999d57e8bb"

CHECKPOINT = "quentinll/lewm-tworooms"
CHECKPOINT_REVISION = "77adaae0bc31deab21c93740d1f8bb947cd0bdec"


def make_tworoom_policy(device):
    torch.manual_seed(42)
    if device == "cpu":
        torch.set_num_threads(1)
    checkpoint = snapshot_download(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        allow_patterns=["config.json", "weights.pt"],
    )
    model = (
        swm.wm.utils.load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
    )
    model.interpolate_pos_encoding = True
    pixels = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            v2.Resize(224),
        ]
    )
    return WorldModelPolicy(
        solver=FastCEMSolver(
            ShootingCostEvaluator(model, GoalMSE()),
            batch_size=1,
            num_samples=300,
            n_steps=30,
            topk=30,
            var_scale=1.0,
            device=device,
            seed=42,
            compile_kernel=False,
        ),
        config=swm.PlanConfig(
            horizon=5, receding_horizon=5, action_block=5, history_len=1
        ),
        process={
            "action": swm.data.ZScoreScaler(
                mean=[[0.0030934159554243212, -0.05298253984909538]],
                std=[[0.867547357428919, 0.8686383074906906]],
            )
        },
        transform={"pixels": pixels, "goal": pixels},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=("tworoom", "pusht", "reacher", "cube"))
    parser.add_argument(
        "--manifest", type=Path, help="Use a previously downloaded manifest"
    )
    parser.add_argument("--device", help="Override the simulator device, e.g. cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    benchmark = load_benchmark(
        args.manifest or REPO,
        revision=None if args.manifest else REVISION,
        environment=args.environment,
    )
    device = args.device or (
        "cuda" if args.environment in ("reacher", "cube") else "cpu"
    )
    learned = args.environment == "tworoom"
    if learned and args.device is None and torch.cuda.is_available():
        device = "cuda"
    policy = make_tworoom_policy(device) if learned else RandomPolicy(seed=42)
    result = benchmark.evaluate(
        policy, policy_environment=args.environment, device=device
    )
    if len(result.episodes) != 60:
        raise RuntimeError(f"Expected 60 tasks, got {len(result.episodes)}")
    filename = "tworoom-lewm-standalone.json" if learned else f"{args.environment}.json"
    path = args.output_dir / filename
    result.write(path)
    print(f"{args.environment}: {result.success_rate:.2f}% success across 60 tasks")
    print(f"Results: {path.resolve()}")


if __name__ == "__main__":
    main()
