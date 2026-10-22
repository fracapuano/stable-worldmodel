"""World's complete-plan fast path; MPC uses the ordinary World loop."""

from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel.evaluation import EpisodeResult, EvaluationResults
from stable_worldmodel.policy import WorldModelPolicy

from .envx import decode_actions


def episodes_from_outcome(protocol, outcome, population_index=0):
    arrays = [
        value[population_index].detach().cpu().numpy()
        for value in (
            outcome.successes,
            outcome.returns,
            outcome.lengths,
            outcome.path_costs,
            outcome.control_costs,
            outcome.collisions,
        )
    ]
    successes, returns, lengths, paths, controls, collisions = arrays
    return tuple(
        EpisodeResult(
            task_key=task.key,
            environment_seed=task.environment_seed,
            controller_seed=task.controller_seed,
            success=bool(successes[i]),
            episode_return=float(returns[i]),
            length=int(lengths[i]),
            path_cost=float(paths[i]),
            control_cost=float(controls[i]),
            collisions=int(collisions[i]),
            constraint_violations=0,
        )
        for i, task in enumerate(protocol.tasks)
    )


@torch.inference_mode()
def evaluate_world(world, protocol, budget, video, record, backend):
    policy = world.policy
    if not isinstance(policy, WorldModelPolicy) or video or record:
        return None
    if budget is None or budget < 1:
        raise ValueError('protocol evaluation requires eval_budget >= 1')
    if budget > policy.flatten_receding_horizon:
        return None
    world._reset_from_protocol(protocol, budget)
    events = []
    original = policy.on_plan

    def capture(event):
        events.append(event)
        if original:
            original(event)

    policy.on_plan = capture
    try:
        policy.get_action(dict(world.infos), return_tensor=True)
    finally:
        policy.on_plan = original
    if len(events) != 1:
        raise RuntimeError('expected one initial planning call')
    selected = events[0]['selected_plan']
    actions = selected.reshape(
        world.num_envs, policy.flatten_receding_horizon, -1
    )
    actions = decode_actions(policy, actions[:, :budget]).contiguous()
    outcome = world.envs.rollout(actions[None])
    successes = outcome.successes[0].cpu().numpy()
    lengths = outcome.lengths[0].cpu().numpy()
    distances = outcome.final_distances[0].cpu().numpy()
    world.terminateds = successes.copy()
    world.truncateds = ~successes
    world.rewards = outcome.returns[0].cpu().numpy()
    world.actions = actions
    world.infos.update(
        state=outcome.final_observations[0, :, None],
        distance_to_target=outcome.final_distances[0, :, None],
        terminated=successes.copy(),
        step_idx=lengths[:, None],
    )
    return EvaluationResults(
        backend=backend,
        backend_type=world._protocol_backend_type(),
        protocol_digest=protocol.digest,
        episodes=episodes_from_outcome(protocol, outcome),
        metadata={
            'split': protocol.split,
            'environment': protocol.environment,
            'eval_budget': budget,
            'simulator_backend': 'envx',
            'planning_calls': 1,
            'final_distances': distances.tolist(),
            'mean_final_distance': float(np.mean(distances)),
        },
    )
