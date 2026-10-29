"""World's complete-plan fast path; MPC uses the ordinary World loop."""

from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel.evaluation import EvaluationResults
from stable_worldmodel.evaluation.records import episodes_from_rollout
from stable_worldmodel.planning import FastCEMSolver
from stable_worldmodel.policy import WorldModelPolicy


@torch.inference_mode()
def evaluate_world(world, protocol, budget, video, record, backend):
    policy = world.policy
    two_room = world.env_name == 'swm/TwoRoom-v1'
    if (
        not isinstance(policy, WorldModelPolicy)
        or not isinstance(policy.solver, FastCEMSolver)
        or video
        or record
    ):
        return None
    if budget is None or budget < 1:
        raise ValueError('protocol evaluation requires eval_budget >= 1')
    if budget > policy.flatten_receding_horizon:
        return None
    world._reset_from_protocol(protocol, budget)
    actions = policy._get_action_plan(dict(world.infos), budget)
    outcome = world.envs.rollout(actions[None])
    episodes = episodes_from_rollout(protocol, outcome)[0]
    successes = np.asarray([episode.success for episode in episodes], bool)
    lengths = np.asarray([episode.length for episode in episodes], np.int32)
    distances = outcome.final_distances[0].cpu().numpy()
    world.terminateds = successes.copy()
    world.truncateds = ~successes
    task_index = torch.arange(world.num_envs, device=actions.device)
    final_actions = actions[task_index, outcome.lengths[0].long() - 1]
    active = outcome.lengths[0] == outcome.lengths[0].max()
    world.rewards = (
        torch.where(active, outcome.final_rewards[0], 0).cpu().numpy()
    )
    world.actions = torch.where(active[:, None], final_actions, float('nan'))
    state = outcome.final_observations[0, :, None]
    world.infos.update(
        state=state[..., :2] if two_room else state,
        distance_to_target=outcome.final_distances[0, :, None],
        terminated=successes.copy(),
        step_idx=lengths[:, None],
        action=final_actions[:, None],
    )
    if two_room:
        world.infos['proprio'] = state[..., :2]
    return EvaluationResults(
        backend=backend,
        backend_type=world._protocol_backend_type(),
        protocol_digest=protocol.digest,
        episodes=episodes,
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
