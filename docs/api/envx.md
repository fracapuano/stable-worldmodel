# Frozen task execution with envX

The dependency direction is `evolving-wm → stable-worldmodel → envX`.
Applications declare `EvaluationTask` objects, attach policies, and call
`World.evaluate` or `ManyWorlds.evaluate`. They need no simulator, renderer,
MJX, JAX, population replay, or resource ownership code.

```python
import stable_worldmodel as swm

world = swm.World(
    protocol.environment,
    num_envs=len(protocol.tasks),
    image_shape=(224, 224),
    execution_backend='envx',
    device='cuda:0',
    max_episode_steps=25,
)
world.set_policy(policy)
try:
    result = world.evaluate(protocol=protocol, eval_budget=25)
finally:
    world.close()
```

Supported IDs are `swm/PushT-v1`, `swm/ReacherDMControl-v0`, and
`swm/OGBCube-v0` (single Cube). The default backend remains Gym. TwoRoom's
existing Gym reset and envX population execution are unchanged.

For populations use the same public constructors as TwoRoom:

```python
many = swm.ManyWorlds(worlds, model_names=model_names)
try:
    result = many.evaluate(protocol, solver=worlds[0].policy.solver,
                           eval_budget=25)
finally:
    many.close()  # closes its Worlds, idempotently
```

`ManyWorlds.from_population(world, population_size=P)` also accepts the
existing `PopulationModel` interface and `population=` evaluation argument.
Worlds must share environment, device, simulator settings, image settings,
preprocessing, action space and planner configuration. Each model/plan has
its own population/task simulator state. Pixels are rendered for the task
batch once, and population planning retains its existing tensor backend.

`result.task_final_distances`, `task_successes`, `environment_actions`, and
`scores` stay on the solver device, in population-major/task-minor order.
For these three task contracts, metadata `score='distance'` means **negative
mean task distance**; `score='success'` means mean task success. TwoRoom
retains its existing diagonal-normalized distance score. World results
retain `EvaluationResults` and ordered `EpisodeResult` objects.

## Reset options and task contracts

Use the existing `EvaluationProtocol` / `EvaluationTask` serialization.
`task.start` and `task.goal` remain the authoritative task coordinates.
The adapter checks that complete simulator coordinates agree with them.
Seeds and task ordering are preserved. Reset input values use float32,
matching envX; GPU replay comparisons must allow numerical tolerance.

| Environment | `task.start` / `task.goal` | Existing `task.options` |
| --- | --- | --- |
| PushT | state7: pusher XY, T body-origin XY, angle, pusher velocity XY | `goal_state` (state7), `variation=[]` |
| Reacher | two raw joint angles | `reacher_task` containing `start_qpos`, `start_qvel`, `goal_qpos`, `goal_qvel`; alternatively `start_qvel` / `goal_qvel` alongside the task coordinates |
| Cube | cube XYZ | `cube_qpos` (21), `cube_qvel` (20), `cube_goal_qpos` (21), `cube_goal_qvel` (20) |

Cube goal quaternion comes from the full goal qpos. A task that only stores
goal XYZ/quaternion cannot reconstruct a particular goal arm configuration.
Supply the frozen complete goal snapshot in reset options; the adapter
raises an error rather than inventing a goal arm or loading a dataset.

For portable full simulator inputs, `options['simulator_state']` may contain
`start` and `goal` dictionaries, each with `qpos` and `qvel`. These override
the legacy complete-state fields but must agree with `task.start/goal`.
MJX snapshots can additionally restore `time`, `act`, `ctrl`, `history`,
`qacc_warmstart`, `plugin_state`, `qfrc_applied`, `xfrc_applied`, `eq_active`,
`mocap_pos`, `mocap_quat`, and `userdata`; fields must have the model's shapes.
Cube target mocap is explicitly assigned the frozen goal pose. Other
unspecified MJX input fields come from the zero/template state, never a
random reset. Derived kinematics/contact buffers are rebuilt by MJX forward.

For PushT, `qpos` is state7 and `qvel` contains T linear XY and angular
velocity. Missing T velocity means zero, matching the existing state7 reset.
Optional `time`, `last_action`, and `n_contacts` restore the remaining envX
state fields. Geometry stays the pinned environment geometry.

These explicit task contracts apply to this opt-in frozen-task backend:

| Environment | Distance / fitness | Success | Control |
| --- | --- | --- | --- |
| Cube | cube XYZ L2, metres | distance ≤ 0.04 | OGBench delta XYZ/yaw/gripper |
| PushT | **raw state7 L2**, including velocity and raw angle | combined pusher/T position L2 < 20 px **and** wrapped T angle < π/9 | relative pusher target, scale 100, no absolute-target clipping |
| Reacher | maximum absolute **unwrapped** joint-angle error | error < 0.05 rad | two native envX physics steps per action; success checked after each repeat |

`options['task_contract']` can override `threshold` (Cube/Reacher),
`position_threshold` and `angle_threshold` (PushT), `repeat`, `relative`,
`action_scale`, and `clip_target` (PushT). Thresholds/scales must be finite
and positive and repeats must be positive integers. No task-suite values
or success rules replace the Gym environments' default tasks.
All tasks in a batch must share the same physics repeat count.

Outcomes freeze on the first successful transition, including success on
the first Reacher repeat. Otherwise they stop at the smaller of the action
budget and World episode limit. Initial success is checked after the first
action, matching the existing task runners. GPU batch rows may continue
through physics after completion; their reported state vector, distance,
return, length, path and control costs stay frozen. This avoids Python
candidate stepping and divergent host control flow.

## Planning, batching and ownership

The 25-action example uses `PlanConfig(horizon=5, receding_horizon=5,
action_block=5, history_len=3)`: one planning call and one compiled physics
scan. The implementation contains no fixed 25-action restriction. Shorter
budgets and different horizons work; ordinary World also supports MPC,
records and videos through its regular loop. ManyWorlds retains its
existing one-complete-plan contract and rejects budgets longer than the
generated plan and per-step recording.

`physics_batch_size` defaults to 16 population members. This limits Cube's
MJX-Warp contact storage without changing population planning or replaying
any plan. Each chunk simulates `chunk_population × tasks` states together.
No replay cache or plan deduplication is used. Task inputs and constants
are transferred at reset and expanded on the selected device. Torch plans
cross to JAX with DLPack; reduced results cross back with DLPack. Only
reporting, optional traces, and the ordinary MPC completion mask use host
values. `FastCEMSolver.solve(return_tensors=True)` is the opt-in device
return used by World; its default return contract is preserved.
Contiguous Torch views with unaligned storage offsets require a small
device-to-device copy before DLPack import (notably five-dimensional Cube
plans). Multi-chunk concatenation finishes on the consuming Torch stream
before JAX-owned temporary chunk buffers are released.

Use device-capable policy transforms/scalers. SWM scalers and sklearn
StandardScaler's affine parameters are supported. Custom action scalers
must return Torch tensors on the input device. Image transforms at World
construction receive batched HWC Torch pixels; policy transforms retain
their usual CHW convention. Native envX image sizes are used directly.

`envx_options` forwards static supported envX factory options. Cube contact
capacity defaults to `naconmax=max(512, 32 * simulated_worlds)` and
`njmax=1024` per world. Increase contact capacity explicitly if required;
there is no silent retry or CPU fallback. Close evaluators in `finally`.
Closing drops owned states, render contexts, and compiled closures without
clearing global JAX caches or other evaluators' resources.

## Pinned dependency and remaining upstream gaps

Install on Python 3.11 or 3.12:

```sh
uv pip install -e '.[envx-cuda12]'
# Select envx-cuda13 instead only for a compatible CUDA 13 Torch runtime.
```

The pin is envX `d26c817c1124229c819be574f137cd68bda4380a`; MJX/MuJoCo
3.11 and JAX < 0.11 are required. Torch and JAX must load compatible CUDA
and cuDNN libraries. Merely matching CUDA major versions is insufficient
when Torch loads an older cuDNN than JAX requires. See the smoke evidence
for the exact validated runtime and its library search path.

All private envX access is isolated in `world/_envx_compat.py`:

- Cube/Reacher have no public complete-state restoration API. The bridge
  uses Cube's `_mjx_model`/`_data_template`/target mocap IDs and Reacher's
  `model`/`_make_data`, then public `mjx.forward`. Public envX `step` executes
  the physics; no control or simulator implementation is copied.
- Cube/Reacher lack render-from-restored-state APIs. The bridge uses
  `_observe_batch` / `_observe` with task-sized native render contexts.
- Reacher's CUDA availability probe references a symbol removed by MuJoCo
  3.11. The bridge replaces only that probe with Warp's public
  `is_cuda_available` when the symbol is absent.
- PushT's public step clips targets. Its declared SWM action contract uses
  envX's public `simulate_control_step` directly, preserving unclipped
  state-relative targets. Public dataset restoration and rendering are reused.
- `rollout_plans` reports the last transition, and has neither custom task
  metrics nor first-success reduction. SWM scans public envX physics and
  reduces the requested metrics on device.
- envX has no public `close`. The bridge drops its owned render references.

Upstream public restore/observe/close APIs and unclipped relative PushT
control would remove most of this compatibility module. Arbitrary Gym
wrappers, nonempty Gym variation settings, observation noise, independent
layout RNG, and a generic random/episodic envX task generator are not
implemented. Unsupported requests raise instead of changing execution.
Use the existing Gym backend for those default environment workflows.
Collision/constraint counters are not instrumented for these three envX
contracts (the legacy result fields are zero).

## Validation

`tests/test_envx_world.py` covers contracts, restored states, goal images,
independent plans, ordering, limits, first success, scores, MPC and cleanup.
`tests/test_jax_two_rooms.py` retains the TwoRoom regression suite.

`scripts/examples/envx_smoke.py` runs the released checkpoints with the
frozen planner settings in `tests/data/envx/planner.json`, replays known
witnesses three times, and runs three ordinary and population evaluations.
The fixtures preserve the first two ordered source tasks' physical states,
goals and witness controls; complete Cube goal snapshots are transported
through reset options. Source protocol, suite and task identities are
recorded alongside the new complete reset declarations. This is test data,
not another task framework. Witness failures are reported unchanged.

Timing instrumentation synchronizes GPU completion and separates reset /
rendering, planning, and physics. First calls include cold compilation;
later calls show warm execution. See `validation/envx/` for logs and JSON.
