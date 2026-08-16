"""Runs a policy against a pool of vectorized environments.

``World`` bundles three things:

1. A batched simulator (``EnvPool``) that steps N envs in parallel and
   can skip terminated envs via a mask.
2. A preprocessing pipeline (``MegaWrapper``) that resizes pixels,
   lifts everything into the info dict, and applies optional transforms.
3. A rollout loop that drives ``policy.get_action(infos)`` and handles
   resets, per-env termination, and episode accounting.

Quick start::

    import stable_worldmodel as swm

    world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
    world.set_policy(policy)

    # Record expert episodes to disk.
    world.collect('data.lance', episodes=500, seed=0)

    # Evaluate a policy over a fixed number of episodes.
    results = world.evaluate(episodes=100, seed=42)

    # Evaluate from dataset-defined start/goal states (one env per episode).
    results = world.evaluate(
        dataset=ds,
        episodes_idx=[0, 1, 2, 3],
        start_steps=[0, 10, 20, 30],
        goal_offset=30,
        eval_budget=50,
        video='videos/',
    )
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from stable_worldmodel.plot import save_panel_videos, save_video
from stable_worldmodel.policy import Policy
from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.wrapper import MegaWrapper

RESET_MODES = ('auto', 'wait')


def _make_env(
    env_name, max_episode_steps, wrappers, add_pixels=True, **kwargs
):
    if add_pixels:
        kwargs.setdefault('render_mode', 'rgb_array')
    env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


class World:
    """Drive a policy through a pool of preprocessed envs.

    After construction, ``world.envs`` is an ``EnvPool`` of ``num_envs``
    environments, each wrapped by ``MegaWrapper`` (and any ``pre_wrappers`` /
    ``extra_wrappers`` you pass). Attach a policy with ``set_policy(...)`` and then call
    ``collect()`` or ``evaluate()`` to run rollouts.

    Attributes populated during a run:
        infos: Stacked info dict from the last reset/step. Tensor/array
            values have shape ``(num_envs, 1, ...)``.
        rewards, terminateds, truncateds: Per-env step outputs from the
            last ``step()``. Shape ``(num_envs,)``.

    Args:
        env_name: Gymnasium id registered for the target env
            (e.g. ``'swm/PushT-v1'``).
        num_envs: Number of parallel envs in the pool.
        image_shape: ``(H, W)`` that pixels/goal are resized to.
            Required unless ``add_pixels=False``.
        max_episode_steps: Per-env step cap before truncation.
        goal_conditioned: If True, the goal key is kept separate from
            regular observations (controls ``MegaWrapper.separate_goal``).
        pre_wrappers: ``gym.Wrapper`` factories applied *before*
            ``MegaWrapper`` (closer to the raw env). Use for env-level
            modifiers (action repeat, reward shaping, obs injection) whose
            output ``MegaWrapper`` should then standardize and validate.
        extra_wrappers: ``gym.Wrapper`` factories applied *after*
            ``MegaWrapper``. Use for transforms that consume the canonical
            observation (frame stacking, normalization).
        image_transform: Optional callable applied to pixels inside
            ``MegaWrapper``.
        goal_transform: Optional callable applied to the goal inside
            ``MegaWrapper``.
        image_resample: PIL resample mode for pixel/goal resizing
            (``'nearest'``, ``'bilinear'``, ...). Defaults to bilinear;
            use ``'nearest'`` for crisp pixel-art envs (e.g. Craftax).
        add_pixels: If True (default), render each env and add a resized
            ``pixels`` observation; goal images are resized too. Set False
            for envs without pixels (e.g. audio): ``image_shape`` may then
            be omitted and the raw observation is lifted into info as-is.
        **kwargs: Forwarded to ``gym.make`` (e.g. ``render_mode``).
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple[int, int] | None = None,
        max_episode_steps: int = 100,
        goal_conditioned: bool = True,
        pre_wrappers: list | None = None,
        extra_wrappers: list | None = None,
        image_transform: Callable | None = None,
        goal_transform: Callable | None = None,
        image_resample: str | int | None = None,
        add_pixels: bool = True,
        **kwargs: Any,
    ):
        if add_pixels and image_shape is None:
            raise ValueError('image_shape is required when add_pixels=True.')
        wrappers = [
            *(pre_wrappers or []),
            partial(
                MegaWrapper,
                image_shape=image_shape,
                pixels_transform=image_transform,
                goal_transform=goal_transform,
                separate_goal=goal_conditioned,
                image_resample=image_resample,
                add_pixels=add_pixels,
            ),
            *(extra_wrappers or []),
        ]
        env_fn = partial(
            _make_env,
            env_name,
            max_episode_steps,
            wrappers,
            add_pixels=add_pixels,
            **kwargs,
        )
        self.envs = EnvPool([env_fn] * num_envs)
        self.policy: Policy | None = None
        self.infos: dict = {}
        self.rewards: np.ndarray | None = None
        self.terminateds: np.ndarray | None = None
        self.truncateds: np.ndarray | None = None
        self.actions: np.ndarray | None = None

    @property
    def num_envs(self) -> int:
        """Number of envs in the pool."""
        return self.envs.num_envs

    def close(self) -> None:
        """Close all envs and release their resources."""
        self.envs.close()

    def set_policy(self, policy: Policy) -> None:
        """Attach a policy and configure it for this world's envs.

        Calls ``policy.set_env(self.envs)``. If the policy exposes a
        ``seed`` attribute and ``_set_seed`` method, the seed is applied.
        """
        self.policy = policy
        self.policy.set_env(self.envs)
        if hasattr(self.policy, '_set_seed') and self.policy.seed is not None:
            self.policy._set_seed(self.policy.seed)

    def reset(self, seed=None, options=None) -> None:
        """Reset every env and refresh ``self.infos``.

        Clears ``terminateds``/``truncateds`` back to all-False.
        """
        _, self.infos = self.envs.reset(seed=seed, options=options)
        if self.policy is not None and hasattr(self.policy, 'reset_state'):
            self.policy.reset_state()
        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

    def evaluate(
        self,
        episodes: int | None = None,
        seed: int | None = None,
        options: dict | None = None,
        video: str | Path | None = None,
        reset_mode: str | None = None,
        dataset: Any = None,
        episodes_idx: list[int] | None = None,
        start_steps: list[int] | None = None,
        goal_offset: int | None = None,
        eval_budget: int | None = None,
        callables: list[dict] | None = None,
        protocol: Any = None,
        record: bool = False,
        backend: str = '',
    ) -> dict:
        """Run the attached policy and return aggregated metrics.

        Two modes of operation:

        * **Episodic (default)**: set ``episodes`` to the number of
          episodes to roll out. Terminated envs are auto-reset until the
          target count is reached.

        * **Dataset-driven**: pass ``dataset`` with ``episodes_idx`` /
          ``start_steps`` / ``goal_offset`` / ``eval_budget``. Each env
          is seeded from one dataset episode, starts at
          ``start_steps[i]`` and targets the state at
          ``start_steps[i] + goal_offset``. Run length is capped at
          ``eval_budget`` steps. Requires ``num_envs == len(episodes_idx)``.

        Args:
            episodes: Total episodes to roll out (episodic mode).
            seed: Base seed. Per-env seeds are derived by offsetting it.
            options: Reset options forwarded to ``envs.reset``.
            video: Directory to write one mp4 per episode/env (optional).
            reset_mode: ``'auto'`` (reset terminated envs) or ``'wait'``
                (freeze terminated envs and stop when all are done).
                Defaults to ``'auto'`` for episodic eval and ``'wait'``
                for dataset eval.
            dataset: Source dataset for dataset-driven eval.
            episodes_idx: Dataset episode indices, one per env.
            start_steps: Starting step within each dataset episode.
            goal_offset: Offset from each start step that defines the goal.
            eval_budget: Max env steps per episode in dataset mode.
            callables: Per-env setup calls applied on the unwrapped env
                after reset. Each spec is
                ``{'method': name, 'args': {arg_name: {'value': ...,
                'in_dataset': bool}}}``; if ``in_dataset`` is True, the
                ``value`` names a key in the sliced dataset state and the
                per-env value is deep-copied in.
            protocol: Immutable ``EvaluationProtocol`` whose ordered tasks
                define resets and controller RNG streams.
            record: Include full transitions and planning queries in
                protocol-driven results.
            backend: Condition label stored in protocol-driven results.

        Returns:
            A dict with ``'success_rate'`` (percent), ``'episode_successes'``
            (per-episode bool/uint array), and ``'seeds'`` used for reset.
        """
        if protocol is not None:
            if dataset is not None or episodes is not None:
                raise ValueError(
                    'protocol evaluation is mutually exclusive with dataset '
                    'and episodic evaluation arguments'
                )
            return self._evaluate_protocol(
                protocol, eval_budget, video, record, backend
            )
        if dataset is not None:
            mode = reset_mode or 'wait'
            return self._evaluate_from_dataset(
                dataset,
                episodes_idx,
                start_steps,
                goal_offset,
                eval_budget,
                callables,
                video,
                mode,
            )
        mode = reset_mode or 'auto'
        return self._evaluate(episodes, seed, options, video, mode)

    def collect(
        self,
        path: str | Path | None = None,
        episodes: int = 0,
        seed: int | None = None,
        options: dict | None = None,
        format: str = 'lance',
        writer: Any = None,
        progress: bool = True,
    ) -> None:
        """Roll out ``episodes`` and dump their trajectories.

        Pass either ``path`` (a registered format writer is constructed for
        you) **or** ``writer`` (a pre-built object implementing the
        :class:`~stable_worldmodel.data.Writer` protocol — for example a
        :class:`~stable_worldmodel.data.ReplayBuffer` to fill in-memory).

        Each info key becomes a column. Leading length-1 time dims are
        squeezed. Columns starting with ``_`` (e.g. ``_needs_flush``)
        are skipped.

        Envs exposing ``get_episode_data()`` on the unwrapped env (a
        ``{name: str | bytes | np.ndarray}`` mapping snapshotted at reset —
        values constant within an episode, e.g. a scene XML) have it
        attached to each finished episode under
        :data:`~stable_worldmodel.data.EPISODE_DATA_KEY`. It bypasses the
        batched info dict entirely, so blobs never ride ``world.infos``.
        Episode data requires a destination that supports it (the
        ``'lance'`` / ``'lance_video'`` formats or a
        :class:`~stable_worldmodel.data.ReplayBuffer`); other writers are
        not episode-data-aware.

        Args:
            path: Output path (file or directory, depending on the format).
                Parent dirs are auto-created. Mutually exclusive with
                ``writer``.
            episodes: Number of episodes to record.
            seed: Base seed for env resets.
            options: Reset options forwarded to ``envs.reset``.
            format: Registered format name (default ``'lance'``); ignored
                when ``writer`` is provided. See
                :func:`stable_worldmodel.data.list_formats` for available
                writers; new formats can be added via
                :func:`stable_worldmodel.data.register_format`.
            writer: A pre-built writer (e.g. ``ReplayBuffer``) to fill
                directly. Mutually exclusive with ``path``.
            progress: Whether to show the ``Recording`` progress bar.
        """
        from tqdm import tqdm

        from stable_worldmodel.data.format import EPISODE_DATA_KEY, get_format

        if (path is None) == (writer is None):
            raise ValueError(
                'World.collect: pass exactly one of `path` or `writer`.'
            )

        if writer is None:
            writer_cm = get_format(format).open_writer(path)
        else:
            writer_cm = writer

        buffers = [defaultdict(list) for _ in range(self.num_envs)]

        def on_step(world, mask):
            for col, data in world.infos.items():
                if col.startswith('_'):
                    continue
                if not isinstance(data, (np.ndarray, torch.Tensor)):
                    continue
                if data.ndim > 1 and data.shape[1] == 1:
                    if isinstance(data, torch.Tensor):
                        data = data.squeeze(1)
                    else:
                        data = np.squeeze(data, axis=1)
                for i in np.where(mask)[0]:
                    val = data[i]
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    elif isinstance(val, np.ndarray):
                        val = val.copy()
                    buffers[i][col].append(val)

        with (
            writer_cm as w,
            tqdm(
                total=episodes, desc='Recording', disable=not progress
            ) as pbar,
        ):

            def episode_iter():
                for env_idx, _ in self._run_iter(
                    episodes=episodes,
                    seed=seed,
                    options=options,
                    mode='auto',
                    on_step=on_step,
                ):
                    ep = {k: list(v) for k, v in buffers[env_idx].items()}
                    buffers[env_idx].clear()
                    if 'action' in ep:
                        ep['action'].append(ep['action'].pop(0))
                    # `_run_iter` yields done envs before the auto-reset, so
                    # the env still holds the snapshot of the episode that
                    # just finished.
                    ep_data_fn = getattr(
                        self.envs.envs[env_idx].unwrapped,
                        'get_episode_data',
                        None,
                    )
                    if ep_data_fn is not None:
                        ep_extra = ep_data_fn()
                        if ep_extra:
                            ep[EPISODE_DATA_KEY] = dict(ep_extra)
                    pbar.update(1)
                    yield ep

            w.write_episodes(episode_iter())

    def _run(
        self,
        episodes: int | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
        options: dict | None = None,
        mode: str = 'auto',
        on_step=None,
        on_done=None,
    ) -> None:
        """Drive the policy. Thin wrapper around :meth:`_run_iter` that
        invokes ``on_done(env_idx, ep_idx, world)`` for each completion."""
        for env_idx, ep_count in self._run_iter(
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            options=options,
            mode=mode,
            on_step=on_step,
        ):
            if on_done:
                on_done(env_idx, ep_count, self)

    def _run_iter(
        self,
        episodes: int | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
        options: dict | None = None,
        mode: str = 'auto',
        on_step=None,
    ):
        """Drive the policy and yield ``(env_idx, ep_count)`` on each
        episode completion. Letting callers consume completions as a
        generator is what makes streaming writes possible without threading.

        ``on_step(world, mask)`` fires after every step and after every
        reset (initial and per-env auto-reset), so callers see the reset
        observation as well as stepped ones. ``mask`` marks which envs the
        call reflects — all envs for a real step, just the reset ones for
        an auto-reset.
        """
        assert mode in RESET_MODES, f'reset_mode must be one of {RESET_MODES}'

        if self.policy is None:
            raise RuntimeError('No policy set.')
        if episodes is None and max_steps is None:
            raise ValueError('Provide episodes or max_steps (or both).')

        if seed is not None or options is not None:
            self.reset(seed=seed, options=options)
            if on_step:
                on_step(self, mask=np.ones(self.num_envs, dtype=bool))

        alive = np.ones(self.num_envs, dtype=bool)
        next_seed = seed + self.num_envs if seed is not None else None
        ep_count = 0

        for t in range(max_steps if max_steps is not None else 2**63):
            actions = self._get_actions()
            self.actions = np.asarray(actions).copy()

            mask = alive if not alive.all() else None
            _, self.rewards, self.terminateds, self.truncateds, self.infos = (
                self.envs.step(actions, mask=mask)
            )

            if on_step:
                on_step(self, mask=mask if mask is not None else alive)

            done = alive & (self.terminateds | self.truncateds)
            if not done.any():
                continue

            budget_reached = False
            for i in np.where(done)[0]:
                yield int(i), ep_count
                ep_count += 1
                if episodes is not None and ep_count >= episodes:
                    budget_reached = True
                    break

            # Always reset the done envs before stopping. Returning straight
            # from the loop above would leave the env that completed the final
            # episode in its terminal state, so the next _run_iter/collect call
            # steps a dead env and records a spurious length-1 episode.
            if mode == 'auto':
                seeds = [None] * self.num_envs
                if next_seed is not None:
                    base = ep_count - int(done.sum())
                    for rank, i in enumerate(np.where(done)[0]):
                        seeds[i] = next_seed + base + rank
                _, self.infos = self.envs.reset(
                    seed=seeds, options=options, mask=done
                )
                self.terminateds[done] = False
                self.truncateds[done] = False
                self.infos['_needs_flush'] = done
                if on_step:
                    on_step(self, mask=done)
            elif mode == 'wait':
                alive[done] = False

            if budget_reached or (mode == 'wait' and not alive.any()):
                return

    def _get_actions(self) -> np.ndarray:
        return self.policy.get_action(self.infos)

    def _evaluate(self, episodes, seed, options, video, mode) -> dict:
        results = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(episodes),
            'seeds': np.zeros(episodes, dtype=np.int64),
        }
        frames: dict[int, list] = defaultdict(list) if video else None

        def on_step(world, mask):
            if frames is not None:
                for i in range(world.num_envs):
                    f = world.infos['pixels'][i]
                    frame = f[-1] if f.ndim > 3 else f
                    frames[i].append(np.asarray(frame).copy())

        def on_done(env_idx, ep_idx, world):
            results['episode_successes'][ep_idx] = world.terminateds[env_idx]
            results['seeds'][ep_idx] = world.envs.seeds[env_idx]
            if frames is not None:
                save_video(
                    Path(video) / f'episode_{ep_idx}.mp4',
                    frames.pop(env_idx, []),
                )

        self._run(
            episodes=episodes,
            seed=seed,
            options=options,
            mode=mode,
            on_step=on_step,
            on_done=on_done,
        )

        results['success_rate'] = (
            float(results['episode_successes'].sum()) / episodes * 100.0
        )
        if frames:
            for env_idx, f in frames.items():
                save_video(Path(video) / f'episode_remaining_{env_idx}.mp4', f)
        return results

    def _evaluate_protocol(
        self, protocol, eval_budget, video, record, backend
    ):
        """Run every task in an ``EvaluationProtocol`` exactly once."""
        from stable_worldmodel.evaluation import (
            EpisodeResult,
            EvaluationProtocol,
            EvaluationResults,
            StepRecord,
        )
        from stable_worldmodel.evaluation.records import recordable

        if not isinstance(protocol, EvaluationProtocol):
            raise TypeError('protocol must be an EvaluationProtocol')
        first_env = self.envs.envs[0]
        spec = getattr(first_env, 'spec', None)
        env_id = getattr(spec, 'id', None)
        accepted_names = {
            env_id,
            getattr(first_env.unwrapped, 'env_name', None),
        }
        if protocol.environment not in accepted_names:
            raise ValueError(
                f'protocol environment {protocol.environment!r} does not '
                f'match World environment {env_id!r}'
            )
        if len(protocol.tasks) != self.num_envs:
            raise ValueError(
                'protocol evaluation requires one World env per task: '
                f'{len(protocol.tasks)} tasks != {self.num_envs} envs'
            )
        if eval_budget is None or eval_budget < 1:
            raise ValueError('protocol evaluation requires eval_budget >= 1')

        seeds = [task.environment_seed for task in protocol.tasks]
        options = [self._task_reset_options(task) for task in protocol.tasks]
        self.reset(seed=seeds, options=options)
        self.infos['controller_seed'] = np.asarray(
            protocol.controller_seeds, dtype=np.int64
        )[:, None]
        self.infos['task_key'] = [[key] for key in protocol.task_keys]

        def info_at(key: str, index: int, default: Any = None) -> Any:
            value = self.infos.get(key)
            if value is None:
                return default
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return value[index]
            if isinstance(value, (list, tuple)):
                return value[index]
            return value

        def pixels_frame(index: int) -> np.ndarray:
            value = info_at('pixels', index)
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            frame = np.asarray(value)
            if frame.ndim > 3:
                frame = frame[-1]
            return np.asarray(frame).copy()

        def snapshot(index: int) -> dict[str, Any]:
            result = {}
            for key, value in self.infos.items():
                if key.startswith('_'):
                    continue
                if torch.is_tensor(value):
                    result[key] = value[index].detach().cpu().clone()
                elif isinstance(value, np.ndarray):
                    result[key] = value[index].copy()
                elif isinstance(value, list):
                    result[key] = deepcopy(value[index])
                else:
                    result[key] = deepcopy(value)
            return result

        def state_at(index: int) -> Any:
            value = info_at('state', index)
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            return deepcopy(value)

        previous_states = [state_at(i) for i in range(self.num_envs)]
        previous_records = (
            [snapshot(i) for i in range(self.num_envs)] if record else None
        )
        step_records: list[list[StepRecord]] = [
            [] for _ in range(self.num_envs)
        ]
        returns = np.zeros(self.num_envs, dtype=np.float64)
        lengths = np.zeros(self.num_envs, dtype=np.int64)
        path_costs = np.zeros(self.num_envs, dtype=np.float64)
        control_costs = np.zeros(self.num_envs, dtype=np.float64)
        collisions = np.zeros(self.num_envs, dtype=np.int64)
        violations = np.zeros(self.num_envs, dtype=np.int64)
        successes = np.zeros(self.num_envs, dtype=bool)

        model_queries: list[dict[str, Any]] = []
        original_on_plan = getattr(self.policy, 'on_plan', None)

        def on_plan(event):
            model_queries.append(recordable(event))
            if original_on_plan is not None:
                original_on_plan(event)

        if record and hasattr(self.policy, 'on_plan'):
            self.policy.on_plan = on_plan

        frames: dict[int, list] | None = None
        goal_frames: list[np.ndarray] | None = None
        if video:
            if 'pixels' not in self.infos:
                raise ValueError(
                    'protocol video requires pixels in World infos; '
                    'construct World with image_shape set'
                )
            frames = defaultdict(list)
            for i in range(self.num_envs):
                frames[i].append(pixels_frame(i))
            if 'goal' in self.infos:
                goal_frames = []
                for i in range(self.num_envs):
                    value = info_at('goal', i)
                    if torch.is_tensor(value):
                        value = value.detach().cpu().numpy()
                    frame = np.asarray(value)
                    if frame.ndim > 3:
                        frame = frame[-1]
                    goal_frames.append(np.asarray(frame).copy())

        def on_step(world, mask):
            active = np.asarray(mask, dtype=bool)
            for i in np.where(active)[0]:
                current_state = state_at(i)
                action = np.asarray(world.actions[i]).copy()
                reward = float(world.rewards[i])
                returns[i] += reward
                lengths[i] += 1
                successes[i] |= bool(world.terminateds[i])
                control_costs[i] += float(np.square(action).sum())

                before_state = previous_states[i]
                if before_state is not None and current_state is not None:
                    delta = np.asarray(current_state) - np.asarray(
                        before_state
                    )
                    path_costs[i] += float(np.linalg.norm(delta))
                collisions[i] += int(
                    np.asarray(info_at('collision', i, False)).any()
                )
                violations[i] += int(
                    np.asarray(info_at('constraint_violation', i, False)).any()
                )

                if record:
                    current_record = snapshot(i)
                    step_records[i].append(
                        StepRecord(
                            decision=int(lengths[i] - 1),
                            observation=previous_records[i],
                            action=action,
                            next_observation=current_record,
                            reward=reward,
                            cost=-reward,
                            terminated=bool(world.terminateds[i]),
                            truncated=bool(world.truncateds[i]),
                        )
                    )
                    previous_records[i] = current_record
                previous_states[i] = current_state
                if frames is not None:
                    frames[i].append(pixels_frame(i))

        try:
            self._run(max_steps=eval_budget, mode='wait', on_step=on_step)
        finally:
            if record and hasattr(self.policy, 'on_plan'):
                self.policy.on_plan = original_on_plan

        episodes = tuple(
            EpisodeResult(
                task_key=task.key,
                environment_seed=task.environment_seed,
                controller_seed=task.controller_seed,
                success=bool(successes[i]),
                episode_return=float(returns[i]),
                length=int(lengths[i]),
                path_cost=float(path_costs[i]),
                control_cost=float(control_costs[i]),
                collisions=int(collisions[i]),
                constraint_violations=int(violations[i]),
                steps=tuple(step_records[i]),
            )
            for i, task in enumerate(protocol.tasks)
        )
        if frames is not None:
            out_dir = Path(video)
            if backend:
                out_dir = out_dir / backend
            for i, task in enumerate(protocol.tasks):
                label = task.name or f'task-{i}'
                clip = frames[i]
                if goal_frames is not None:
                    goal = goal_frames[i]
                    clip = [
                        np.concatenate([frame, goal], axis=1) for frame in clip
                    ]
                save_video(out_dir / f'{label}.mp4', clip)
        return EvaluationResults(
            backend=backend,
            backend_type=self._protocol_backend_type(),
            protocol_digest=protocol.digest,
            episodes=episodes,
            model_queries=tuple(model_queries),
            metadata={
                'split': protocol.split,
                'environment': protocol.environment,
                'eval_budget': eval_budget,
            },
        )

    @staticmethod
    def _task_reset_options(task) -> dict[str, Any]:
        options = dict(task.options)
        if task.layout_seed != task.environment_seed:
            raise ValueError(
                'this environment does not expose an independent layout '
                'seed; layout_seed must equal environment_seed'
            )

        dynamics = dict(task.dynamics_parameters)
        noise_std = float(dynamics.pop('observation_noise_std', 0.0))
        if noise_std != 0.0:
            raise NotImplementedError(
                'protocol observation noise requires a seeded observation '
                'adapter on the World'
            )
        variation_values = options.get('variation_values', {})
        missing = sorted(set(dynamics) - set(variation_values))
        if missing:
            raise ValueError(
                'protocol dynamics_parameters must be present in reset '
                f'option variation_values; missing {missing}'
            )
        for key, expected in dynamics.items():
            actual = variation_values[key]
            try:
                matches = np.allclose(
                    np.asarray(actual).reshape(-1),
                    np.asarray(expected).reshape(-1),
                )
            except (TypeError, ValueError):
                matches = actual == expected
            if not bool(matches):
                raise ValueError(
                    'protocol dynamics_parameters disagree with reset '
                    f'option variation_values for {key!r}'
                )
        options['state'] = np.asarray(task.start, dtype=np.float32)
        options['target_state'] = np.asarray(task.goal, dtype=np.float32)
        return options

    def _protocol_backend_type(self) -> str:
        solver = getattr(self.policy, 'solver', None)
        cost = getattr(solver, 'cost', None)
        backend = getattr(cost, 'model', cost)
        if backend is None:
            return f'{type(self.policy).__module__}.{type(self.policy).__qualname__}'
        backend_cls = type(backend)
        return f'{backend_cls.__module__}.{backend_cls.__qualname__}'

    def _evaluate_from_dataset(
        self,
        dataset,
        episodes_idx,
        start_steps,
        goal_offset,
        eval_budget,
        callables,
        video,
        mode,
    ) -> dict:
        n = len(episodes_idx)
        assert n == self.num_envs

        init_rows, goal_rows, dataset_videos = _extract_init_goal(
            dataset,
            episodes_idx,
            start_steps,
            goal_offset,
        )
        episode_cols = set(
            getattr(dataset, 'episode_column_names', None) or []
        )

        seeds = None
        if init_rows and 'seed' in init_rows[0]:
            seeds = [
                int(np.asarray(row['seed']).reshape(-1)[0])
                for row in init_rows
            ]

        # Prefer the env's own dataset->reset-options converter when present;
        # it returns the per-env `options` (e.g. `{'state': ...}`) that reset()
        # consumes, so a single batched reset restores every env. Envs without
        # the method fall back to the legacy `callables` config: a plain reset
        # followed by per-env method calls on the unwrapped env.
        has_method = hasattr(
            self.envs.envs[0].unwrapped, 'reset_options_from_dataset'
        )
        if has_method:
            opts_list = [
                self.envs.envs[i].unwrapped.reset_options_from_dataset(
                    init_rows[i], goal_rows[i]
                )
                for i in range(n)
            ]
            self.reset(seed=seeds, options=opts_list)
        else:
            self.reset(seed=seeds)
            if callables:
                for i in range(n):
                    env_init = {**init_rows[i], **goal_rows[i]}
                    _apply_callables(
                        self.envs.envs[i].unwrapped, callables, env_init
                    )

        # Inject dataset values into the batched infos: columns whose
        # per-episode values share one shape are stacked to (N, ...) and
        # broadcast over the time dim. Episode-scoped columns are excluded
        # by name and columns with per-episode shapes are skipped — those
        # exist for the env resets, not for the planner infos.
        goal_keys = set(goal_rows[0]) if goal_rows else set()
        shape_prefix = self.infos['pixels'].shape[:2]
        for rows in (init_rows, goal_rows):
            if not rows:
                continue
            for key in rows[0]:
                if key == 'action' or key in episode_cols:
                    continue
                if key not in self.infos and key not in goal_keys:
                    continue
                vals = [row[key] for row in rows]
                if not all(isinstance(v, np.ndarray) for v in vals):
                    continue
                if len({v.shape for v in vals}) > 1:
                    continue
                stacked = np.stack(vals)
                self.infos[key] = np.broadcast_to(
                    stacked[:, None, ...], shape_prefix + stacked.shape[1:]
                ).copy()

        goal_snapshot = {
            k: self.infos[k].copy() for k in goal_keys if k in self.infos
        }

        results = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(n, dtype=bool),
            'seeds': seeds,
        }
        frames: dict[int, list] = defaultdict(list) if video else None

        def on_step(world, mask):
            world.infos.update(deepcopy(goal_snapshot))
            results['episode_successes'] |= world.terminateds
            if frames is not None:
                for i in range(world.num_envs):
                    f = world.infos['pixels'][i]
                    frame = f[-1] if f.ndim > 3 else f
                    frames[i].append(np.asarray(frame).copy())

        self._run(max_steps=eval_budget, mode=mode, on_step=on_step)

        results['success_rate'] = (
            float(results['episode_successes'].sum()) / n * 100.0
        )
        if frames:
            save_panel_videos(
                Path(video),
                {
                    'agent': frames,
                    'dataset': dataset_videos,
                    'goal': [row['goal'] for row in goal_rows],
                },
            )
        return results


def _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
    """Build per-episode init/goal rows for a dataset-driven evaluation.

    Returns ``(init_rows, goal_rows, dataset_videos)``:

    - ``init_rows[i]``: the requested episode's first-step value per
      per-step column, plus the episode-scoped columns (constants like a
      scene XML — reported by ``dataset.episode_column_names``).
    - ``goal_rows[i]``: the last chunk step per per-step column, remapped to
      ``'goal'`` for ``pixels`` and ``'goal_<col>'`` otherwise.
    - ``dataset_videos[i]``: the ``pixels`` window, for the eval panel video.
    """
    ep_idx_arr = np.array(episodes_idx)
    start_arr = np.array(start_steps)
    data = dataset.load_chunk(
        ep_idx_arr, start_arr, start_arr + goal_offset + 1
    )

    episode_cols = list(getattr(dataset, 'episode_column_names', None) or [])
    episode_data = (
        dataset.get_episode_data(episodes_idx) if episode_cols else {}
    )

    init_rows: list[dict] = []
    goal_rows: list[dict] = []
    dataset_videos: list = []

    for i, ep in enumerate(data):
        init_row: dict = {}
        goal_row: dict = {}
        for col in dataset.column_names:
            if col.startswith('goal'):
                continue
            if col.startswith('pixels'):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            val = ep[col]
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                continue
            arr = val.numpy() if isinstance(val, torch.Tensor) else val
            init_row[col] = arr[0]
            goal_row['goal' if col == 'pixels' else f'goal_{col}'] = arr[-1]
            if col == 'pixels':
                dataset_videos.append(arr)
        for col in episode_cols:
            init_row[col] = episode_data[col][i]
        init_rows.append(init_row)
        goal_rows.append(goal_row)

    return init_rows, goal_rows, dataset_videos


def _apply_callables(env, callables, init_state):
    for spec in callables:
        method = spec['method']
        if not hasattr(env, method):
            continue
        prepared = {}
        for name, data in spec.get('args', {}).items():
            if data.get('in_dataset', True):
                key = data.get('value')
                if key in init_state:
                    prepared[name] = deepcopy(init_state[key])
            else:
                prepared[name] = data.get('value')
        getattr(env, method)(**prepared)
