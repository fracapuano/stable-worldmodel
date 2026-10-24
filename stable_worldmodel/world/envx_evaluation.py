"""World's complete-plan fast path; MPC uses the ordinary World loop."""

from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel.evaluation import EvaluationResults
from stable_worldmodel.evaluation.records import (
    episodes_from_rollout,
    recordable,
)
from stable_worldmodel.planning import FastCEMSolver
from stable_worldmodel.policy import WorldModelPolicy


@torch.inference_mode()
def evaluate_world(world, protocol, budget, video, record, backend):
    from .jax_two_rooms import TwoRoomTaskPool

    policy = world.policy
    two_room = isinstance(world.envs, TwoRoomTaskPool)
    if not isinstance(policy, WorldModelPolicy) or not isinstance(
        policy.solver, FastCEMSolver
    ):
        if two_room:
            raise NotImplementedError(
                'TwoRoom World envX execution requires a complete FastCEM '
                'protocol plan'
            )
        return None
    if budget is None or budget < 1:
        raise ValueError('protocol evaluation requires eval_budget >= 1')
    if budget > policy.flatten_receding_horizon:
        if two_room:
            raise NotImplementedError(
                'TwoRoom World envX execution requires eval_budget within '
                'the selected complete plan'
            )
        return None
    world._reset_from_protocol(protocol, budget)
    model_queries = []
    original_on_plan = policy.on_plan

    def on_plan(event):
        model_queries.append(recordable(event))
        if original_on_plan is not None:
            original_on_plan(event)

    if record:
        policy.on_plan = on_plan
    try:
        actions = policy._get_action_plan(dict(world.infos), budget)
    finally:
        policy.on_plan = original_on_plan
    outcome = world.envs.rollout(actions[None], record=bool(record or video))
    episodes = episodes_from_rollout(protocol, outcome)[0]
    if video:
        from pathlib import Path

        from stable_worldmodel.plot import save_video

        directory = Path(video) / backend if backend else Path(video)
        for i, (task, episode) in enumerate(zip(protocol.tasks, episodes)):
            observations = [episode.steps[0].observation] + [
                step.next_observation for step in episode.steps
            ]
            frames = []
            for observation in observations:
                if 'pixels' not in observation:
                    raise ValueError(
                        'protocol video requires World image_shape'
                    )
                frame = np.asarray(observation['pixels'][-1])
                if 'goal' in observation:
                    frame = np.concatenate(
                        (frame, np.asarray(observation['goal'][-1])), axis=1
                    )
                frames.append(frame)
            save_video(directory / f'{task.name or f"task-{i}"}.mp4', frames)
        if not record:
            from dataclasses import replace

            episodes = tuple(
                replace(episode, steps=()) for episode in episodes
            )
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
    return EvaluationResults(
        backend=backend,
        backend_type=world._protocol_backend_type(),
        protocol_digest=protocol.digest,
        episodes=episodes,
        model_queries=tuple(model_queries),
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
