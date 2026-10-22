"""Compatibility boundary for envX d26c817 (not a replacement simulator).

Missing upstream APIs: batched restore/observe/close for MJX environments,
and an unclipped relative PushT step. Keep all private envX access here.
Physics is always envX's implementation. See docs/api/envx.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_env(name, **kwargs):
    import envx

    if name == 'reacher' and kwargs.get('observation_type') == 'pixels':
        import envx.reacher.env as module
        from mujoco.mjx.warp import io

        if not hasattr(io, 'has_cuda_gpu_device'):
            # MuJoCo 3.11 removed envX's CUDA availability probe. This narrow
            # version compatibility fix changes no physics or rendering.
            import warp

            module._has_warp_cuda = warp.is_cuda_available
    return envx.make(name, **kwargs)


class StateAdapter:
    def __init__(self, env, params, name):
        self.env, self.params, self.name = env, params, name
        self.raw = env.unwrapped
        if name == 'cube':
            self.model = self.raw._mjx_model
            self.template = self.raw._data_template
        elif name == 'reacher':
            self.model = self.raw.model
            self.template = self.raw._make_data()

    def restore(self, inputs):
        if self.name == 'pusht':
            # Public dataset restore preserves body-origin coordinates. The
            # dataset state7 omits block velocity; optional full-state fields
            # below also support moving-block snapshots.
            def one(value, block_velocity, extras):
                _, state = self.raw.reset_from_state(value[:5], self.params)
                extras = {
                    key: jnp.asarray(item, dtype=getattr(state, key).dtype)
                    for key, item in extras.items()
                }
                for key, item in extras.items():
                    if item.shape != getattr(state, key).shape:
                        raise ValueError(
                            f'simulator field {key} has the wrong shape'
                        )
                return state.replace(
                    agent_vel=value[5:7],
                    block_vel=block_velocity[:2],
                    block_angular_vel=block_velocity[2],
                    **extras,
                )

            return jax.vmap(one)(
                inputs['qpos'], inputs['qvel'], inputs['extras']
            )
        from mujoco import mjx

        def one(qpos, qvel, goal, extras):
            extras = {
                key: jnp.asarray(
                    value, dtype=getattr(self.template, key).dtype
                )
                for key, value in extras.items()
            }
            for key, value in extras.items():
                if value.shape != getattr(self.template, key).shape:
                    raise ValueError(
                        f'simulator field {key} has the wrong shape'
                    )
            data = self.template.replace(qpos=qpos, qvel=qvel, **extras)
            if self.name == 'cube':
                ids = self.raw._cube_target_mocap_ids
                data = data.replace(
                    mocap_pos=data.mocap_pos.at[ids].set(goal[None, :3]),
                    mocap_quat=data.mocap_quat.at[ids].set(goal[None, 3:]),
                )
            return mjx.forward(self.model, data)

        return jax.vmap(one)(
            inputs['qpos'], inputs['qvel'], inputs['goal'], inputs['extras']
        )

    def advance(self, state, action, contract):
        if self.name == 'pusht':
            # env.step clips absolute targets. The declared SWM relative
            # control deliberately does not. Call envX's public physics kernel.
            from envx.pusht.physics import simulate_control_step

            def one(current, control, relative, scale, clip):
                target = jnp.where(
                    relative, current.agent_pos + control * scale, control
                )
                target = jnp.where(clip, jnp.clip(target, 0, 512), target)
                return simulate_control_step(
                    current, target, self.params
                ).replace(last_action=target, time=current.time + 1)

            return jax.vmap(one)(
                state,
                action,
                contract['relative'],
                contract['action_scale'],
                contract['clip_target'],
            )
        # Public state-only step: no renderer and no candidate Python loop.
        n = action.shape[0]
        zeros = jnp.zeros(n, jnp.float32)
        false = jnp.zeros(n, jnp.bool_)
        count = jnp.zeros(n, jnp.int32)
        if self.name == 'cube':
            from envx.cube.environment import CubeJaxState

            wrapped = CubeJaxState(
                state, jnp.zeros((n, 1)), count, count, false
            )
        else:
            from envx.reacher.env import ReacherState

            wrapped = ReacherState(
                state,
                jnp.zeros((n, 6)),
                zeros,
                jnp.ones(n),
                false,
                false,
                count,
                zeros,
                false,
            )
        return self.env.step(jax.random.key(0), wrapped, action, self.params)[
            1
        ].data

    def vector(self, state):
        if self.name == 'cube':
            return state.qpos[:, 14:17]
        if self.name == 'reacher':
            return state.qpos
        return jnp.concatenate(
            (
                state.agent_pos,
                state.block_pos,
                jnp.mod(state.block_angle, 2 * jnp.pi)[:, None],
                state.agent_vel,
            ),
            axis=-1,
        )

    def render(self, state):
        if self.name == 'pusht':
            return self.env.render(state, self.params)
        if self.name == 'cube':
            return self.raw._observe_batch(state)[0]
        return self.raw._observe(state)[1]

    def close(self):
        # envX has no close API at this revision. Drop only owned references;
        # never clear process-wide JAX caches or another evaluator's buffers.
        for name in ('_render_context', '_render_context_pytree'):
            if hasattr(self.raw, name):
                setattr(self.raw, name, None)
        self.env = self.raw = self.template = self.model = None
        self.params = None
