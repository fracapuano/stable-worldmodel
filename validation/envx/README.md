# envX integration evidence — 22 October 2026

All implementation changes are in stable-worldmodel. The implementation was
validated on `328528c`, then rebased onto fork main `c1f63de` (identical trees).
`source.json` records the 22 October implementation hashes. No evolving-wm or envX source
was modified. Remote work used isolated `/tmp/capuano-swm-envx-20261022` on
FLAIR node 1; existing training runtimes were not changed.

## Public API revision — 23 October

Frozen protocol evaluation now selects envX internally, without an
`execution_backend` constructor argument. Gym creation is deferred for these
three environments so attaching a policy does not construct an unused CPU
simulator. Ordinary episodic/dataset evaluation and collection keep the
existing default tasks, including when called after a frozen evaluation.
Both owned pools are closed idempotently.

`final-api/` records checks of this revision:

- Full local suite: 1,244 passed, 36 skipped, 1 xpassed. One DataLoader stress
  check failed because the sandbox denied `torch_shm_manager`; that same
  check passed with access to local shared memory (`local-stress.log`).
- CPU envX checks: 17 passed, 6 skipped, including routing back to ordinary
  task evaluation and a 25-control blocked plan with one planning call.
- Ruff/pre-commit, wheel/sdist build and strict documentation build passed.
- Frozen fixtures are under `tests/data/envx_tasks/`. The rename prevents
  pytest from mistaking a fixture directory for the optional `envx` package.
  Frozen task JSON contents and planner settings are unchanged.
- The GPU evidence below covers the earlier implementation. Final-revision
  GPU reruns are pending authorization to copy source to the existing FLAIR
  runtime; these checks must not be described as final-revision GPU evidence.

## Checks

- `gpu-tests.log`: **29 passed**, including the existing 12 TwoRoom checks;
  complete state restoration, all three goal renderers, population/task
  ordering, distinct plans, partial batches, changing batch sizes, first
  success, control repeats, scores, World MPC/records and idempotent cleanup.
- `local-tests.log`: **157 passed, 15 skipped** on Python 3.13. The skipped
  envX integrations require its supported Python 3.11/3.12 runtime.
- `cpu-envx-tests.log`: **11 passed, 6 skipped** on Python 3.11, covering
  actual CPU PushT execution plus contract/unit checks. Reacher and Cube
  integration checks are run on CUDA. An earlier macOS CPU Reacher check
  stalled and was stopped; this does not certify MJX CPU execution.
- Ruff lint/format and `git diff --check` passed on the changed Python files.

## Real checkpoint evaluations

Each environment uses two ordered frozen source tasks, three witness replays,
three ordinary World evaluations and three P=2 ManyWorlds evaluations. Search
settings remain horizon=5, action_block=5, receding_horizon=5, history_len=3,
64 samples, 3 CEM iterations, topk=8, seed=42, 224px and 25 actions.
Pretrained models and simulator boundaries use float32. Torch compilation uses
`compile_mode='default'` to share JAX-owned buffers without CUDA graph capture.

| Environment | Witness success, each of 3 replays | Released World success | Released population success |
| --- | --- | --- | --- |
| PushT | 2/2 | 0/2 | 0/4 |
| Reacher | 2/2 | 0/2 | 0/4 |
| Cube | 2/2 | 1/2 | 2/4 |

The released-model failures are retained unchanged. Two population members
use separate instances of the same released weights; distinct-plan independence
is additionally checked using real physics and different explicit controls.
This is a smoke test, not a complete task-suite, training or P128 benchmark.

The first runs are `pusht.json`, `reacher.json`, `cube.json`. The final-source
verification runs are `*-final.json`. `summary.json` combines the first-call
and final warm measurements. All runs use NVIDIA A40s; see `runtime.json` for
the exact Python, Torch, JAX, MJX, Warp, CUDA and cuDNN versions.

| Environment | First reset/render, s | First physics scan, s | First World planning, s | First population planning, s | Warm population planning, s | Warm population physics, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PushT | 1.588 | 1.182 | 26.044 | 84.858 | 0.114 | 0.028 |
| Reacher | 11.704 | 1.623 | 21.751 | 82.783 | 0.114 | 0.036 |
| Cube | 196.510 | 8.107 | 21.515 | 83.834 | 0.114 | 0.843 |

Timings synchronize actual GPU completion. First calls include tracing,
compilation/cache loading and execution, not pure compiler time. Cube's first
reset additionally compiled Warp collision kernels. Warm values are the median
of the final two P=2/T=2 runs, and exclude reset/rendering. No throughput
projection to a larger population is made. GPU Cube witness distances vary
slightly between repeats; comparisons use tolerances, not bitwise equality.

Checkpoint SHA256s are recorded in each JSON:

- PushT release `22b330c28c27ead4bfd1888615af1340e3fe9052`.
- Reacher release `62adae4b71dc474ddf8f794c476ebfe737a743ca`.
- Cube release `b0747c5002e86d2ce8f3cd8178004b97524c587d`.

Frozen fixture provenance is in each protocol under `tests/data/envx_tasks/`.
The source branches inspected were PushT `717d778`, Reacher `5b8b74e`, and
Cube `13d0eeb`. Start/goal states and witness actions were preserved. Full
Cube goal snapshots and witness controls were transported into existing task
reset options; source protocol/suite/task identities remain in metadata.
No simulator-dependent task framework is required by consumers.

## Failures investigated and remaining gaps

- The first GPU attempt loaded Torch's cuDNN 9.5 against JAX's ≥9.8
  requirement. The isolated runtime now explicitly loads cuDNN 9.27; both
  CUDA major and cuDNN compatibility matter. Production runtimes are unchanged.
- `gpu-tests-initial-chunk-abort.log` and
  `cube-chunks-before-alignment-fix.log` preserve the Cube native crashes.
  Contiguous five-dimensional Torch plan slices can have unaligned storage
  offsets. Copying only unaligned views on-device fixes XLA import; the
  previous failing case passes in `cube-chunks.log` and the full GPU suite.
  Multi-chunk output concatenation also synchronizes before releasing
  JAX-owned temporary storage.
- The pinned envX revision lacks general restore/observe/close APIs for MJX
  and unclipped relative PushT stepping. The small explicit compatibility
  boundary is documented in `docs/api/envx.md`. No physics was reimplemented.
- Cube declarations containing only goal XYZ/quaternion need the original
  full goal qpos/qvel snapshot to render that exact arm configuration.
  Missing snapshots produce errors, not inferred/reset goal arms.
- Full task suites, large populations, training throughput, MJX CPU execution,
  arbitrary Gym wrappers/variations/noise, and per-step ManyWorlds recording
  are not certified by these checks. Ordinary World MPC/records are covered.

Reproduce with a compatible CUDA runtime and the released local checkpoint:

```sh
python scripts/examples/envx_smoke.py \
  --environment pusht --gpu GPU-... \
  --checkpoint-dir /path/to/released-checkpoint \
  --output /tmp/pusht-smoke.json

SWM_ENVX_TEST_DEVICE=cuda:0 python -m pytest \
  tests/test_envx_world.py tests/test_jax_two_rooms.py -q
```

The smoke runner checks that the selected GPU is idle and acquires the same
per-GPU advisory lock as the workspace's existing evaluation scripts.
