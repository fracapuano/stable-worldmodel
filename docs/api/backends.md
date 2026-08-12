---
title: Exact and learned backends
summary: Interchangeable Dynamics adapters for paired controller evaluation
---

`LearnedModelBackend` and `OracleModelBackend` implement the same public
[`Dynamics`][stable_worldmodel.planning.Dynamics] rollout surface. Both are
composed through `ShootingCostEvaluator`, an unchanged registered solver,
`WorldModelPolicy`, and `World.evaluate`; the oracle is an exact model for the
same controller, not an optimal policy.

```python
import stable_worldmodel as swm
from stable_worldmodel.planning import CEMSolver, GoalMSE, ShootingCostEvaluator

backend = swm.LearnedModelBackend.from_pretrained('org/tworoom-lewm')
# Or: backend = swm.OracleModelBackend()

cost = ShootingCostEvaluator(backend, GoalMSE())
solver = CEMSolver(cost=cost, seed=1234)
policy = swm.policy.WorldModelPolicy(
    solver=solver,
    config=swm.PlanConfig(horizon=5, receding_horizon=1),
)
world = swm.World('swm/TwoRoom-v1', num_envs=100, image_shape=(224, 224))
world.set_policy(policy)  # binds the oracle privately when one is used
results = world.evaluate(
    manifest=swm.evaluation.EvaluationManifest.read('validation.json'),
    eval_budget=50,
    backend=type(backend).__name__,
    record=True,
)
```

For environments that support an exact model, the unwrapped Gym environment
opts into `ExactSimulator` with three methods: `get_oracle_state`,
`set_oracle_state`, and a side-effect-free, batch-capable `oracle_transition`.
Privileged state stays inside the adapter. `World` supplies only `env_index`, a
non-privileged pool slot identifier.

The candidate last dimension may contain an action block flattened by
`PlanConfig.action_block`. The oracle applies every physical action and emits
one privileged-state embedding per planning step, so it has the same temporal
contract as LeWM.

To audit an exact adapter, `replay_simulator` executes a candidate in the real
simulator, captures the realized states, and restores the starting state. Use
it for both one-step and full-horizon parity tests.

## Manifests and G0 audits

`EvaluationManifest` stores ordered, content-addressed task keys including
environment/layout, observation-noise, dynamics, and controller seeds. Writes
are immutable, digests are verified on read, and `validate_manifest_suite`
checks required split coverage and leakage. Manifest-driven `World.evaluate`
returns ordered `EvaluationResults` with per-episode metrics and optional full
transition/model-query traces.

`controller_hash` hashes canonical configuration plus controller source.
`build_g0_audit` compares paired results and emits a machine-readable report
that fails on replay error, controller/task/RNG/budget/stopping mismatches,
nondeterminism, incomplete manifests, or non-finite metrics.

Run the maintained, low-cost TwoRoom example with:

```bash
python scripts/examples/oracle_backend.py
```

::: stable_worldmodel.backends.LearnedModelBackend
::: stable_worldmodel.backends.OracleModelBackend
::: stable_worldmodel.backends.ExactSimulator
::: stable_worldmodel.evaluation.EvaluationManifest
::: stable_worldmodel.evaluation.EvaluationResults
::: stable_worldmodel.evaluation.build_g0_audit
