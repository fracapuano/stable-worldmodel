"""Tests for the LeWM Dynamics surface + ShootingCostEvaluator planning seam.

LeWM no longer exposes ``get_cost``; planning goes through
``stable_worldmodel.planning.ShootingCostEvaluator(model, GoalMSE())``. These tests
guard that LeWM still satisfies the ``Dynamics`` protocol and that goal encoding
happens once (as ``(B, T, D)``) and is reused when pre-injected. The bit-for-bit
parity of ``GoalMSE`` with the old ``criterion`` lives in
``tests/planning/test_evaluator.py``.
"""

import pytest
import torch
from torch import nn
from torch.func import functional_call

from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor

# CEM-like dimensions
B, S, T, D, H, A = 2, 3, 2, 5, 4, 2


def _make_info_dict():
    """Return a CEM-expanded info_dict mimicking what a solver passes to get_cost."""
    return {
        'pixels': torch.randn(B, S, T, 3, 8, 8),
        'goal': torch.randn(B, S, T, 3, 8, 8),
        'action': torch.randn(B, S, H, A),
    }


def _make_action_candidates():
    return torch.randn(B, S, H, A)


def _bare_model():
    """Bypass LeWM.__init__; we only need the encode/rollout surface."""
    return object.__new__(LeWM)


def test_lewm_satisfies_dynamics_protocol():
    """LeWM exposes encode/rollout, so it is a Dynamics a ShootingCostEvaluator can wrap."""
    assert isinstance(_bare_model(), Dynamics)


def test_lewm_no_longer_exposes_get_cost():
    """Cost now lives in the ShootingCostEvaluator seam, not on the model."""
    assert not hasattr(_bare_model(), 'get_cost')


def test_cost_evaluator_encodes_goal_once_as_3d():
    """ShootingCostEvaluator(LeWM, GoalMSE) encodes the goal once and stores it (B, T, D).

    get_cost strips the S axis with v[:, 0] before encoding; GoalMSE then
    re-inserts it via goal_emb[:, None, -1:, :].expand_as(pred_emb).
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    action_candidates = _make_action_candidates()

    encode_calls = []

    def mock_encode(goal_dict):
        encode_calls.append(1)
        assert goal_dict['pixels'].shape == (B, T, 3, 8, 8), (
            f'encode received wrong pixels shape: {goal_dict["pixels"].shape}'
        )
        return {'emb': torch.randn(B, T, D)}

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = mock_encode
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )

    assert len(encode_calls) == 1, 'encode should be called exactly once'
    assert cost.shape == (B, S), (
        f'cost shape: expected ({B},{S}), got {cost.shape}'
    )


def test_cost_evaluator_respects_preinjected_goal_emb():
    """A pre-populated goal_emb is reused; encode is not called again."""
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    info_dict['goal_emb'] = torch.randn(B, T, D)
    action_candidates = _make_action_candidates()

    def encode_must_not_be_called(_):
        raise AssertionError(
            'encode must not be called when goal_emb is already in info_dict'
        )

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = encode_must_not_be_called
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )
    assert cost.shape == (B, S)


###############################################
## Real rollout: history / candidate pairing ##
###############################################

# Small dims for the real-rollout tests. The toy predictor adds action
# embeddings to frame embeddings, so action dim == embedding dim.
RB, RS, RD = 2, 3, 4


class _CumsumPredictor(nn.Module):
    """Causal toy predictor: output[t] = sum_{k<=t} (emb[k] + act[k]).

    The last output depends on the whole window, so predictions change
    whenever any past action changes. Records every (emb, act) call.
    """

    def __init__(self, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.calls = []

    def forward(self, emb, act_emb):
        self.calls.append((emb.detach().clone(), act_emb.detach().clone()))
        return (emb + act_emb).cumsum(dim=1)


class _RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return x


def _toy_model(num_frames=3):
    return LeWM(
        encoder=nn.Identity(),  # unused: tests pre-populate info['emb']
        predictor=_CumsumPredictor(num_frames=num_frames),
        action_encoder=_RecordingIdentity(),
    )


def _rollout_info(hist_len, emb=None):
    info = {'pixels': torch.randn(RB, RS, hist_len, 3, 8, 8)}
    info['emb'] = emb if emb is not None else torch.randn(RB, RS, hist_len, RD)
    return info


def _reference_rollout_legacy(emb_init, act_emb_seq, HS):
    """Hand-rolled pre-change rollout semantics (H frames, [H, T-H] split)."""
    H = emb_init.size(1)
    n_steps = act_emb_seq.size(1) - H
    emb_list = list(emb_init.unbind(dim=1))
    for t in range(n_steps + 1):
        lo = max(0, H + t - HS)
        emb_trunc = torch.stack(emb_list[lo:], dim=1)
        act_trunc = act_emb_seq[:, lo : H + t]
        emb_list.append((emb_trunc + act_trunc).cumsum(dim=1)[:, -1])
    return torch.stack(emb_list, dim=1)


def test_rollout_h1_matches_legacy_semantics():
    """With a single context frame the new contract is bit-identical to the
    old one (first candidate = action paired with the current frame)."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(hist_len=1)
    candidates = torch.randn(RB, RS, 5, RD)

    out = model.rollout(dict(info), candidates)

    emb_flat = info['emb'].reshape(RB * RS, 1, RD)
    cand_flat = candidates.reshape(RB * RS, 5, RD)
    expected = _reference_rollout_legacy(emb_flat, cand_flat, HS=3)
    torch.testing.assert_close(
        out['predicted_emb'].reshape(RB * RS, -1, RD), expected
    )
    assert 'action_history' not in out
    torch.testing.assert_close(out['action'], candidates[:, :, :1])


def test_rollout_consumes_past_actions_in_order():
    """action_encoder must receive cat([action_history, candidates])."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    past = torch.randn(RB, RS, 2, RD)
    candidates = torch.randn(RB, RS, 4, RD)
    info['action_history'] = past

    model.rollout(info, candidates)

    seq = model.action_encoder.inputs[-1]  # (BS, H-1+T, A)
    expected = torch.cat([past, candidates], dim=2).reshape(RB * RS, 6, RD)
    torch.testing.assert_close(seq, expected)
    torch.testing.assert_close(
        info['action'], torch.cat([past, candidates[:, :, :1]], dim=2)
    )


def test_rollout_output_length_and_past_sensitivity():
    """Output holds H context + T predicted frames, and predictions react
    to the executed past actions (they are inputs, not dead weight)."""
    torch.manual_seed(0)
    model = _toy_model()
    emb = torch.randn(RB, RS, 3, RD)
    candidates = torch.randn(RB, RS, 4, RD)

    info_a = _rollout_info(hist_len=3, emb=emb)
    info_a['action_history'] = torch.zeros(RB, RS, 2, RD)
    out_a = model.rollout(info_a, candidates)['predicted_emb']

    info_b = _rollout_info(hist_len=3, emb=emb)
    info_b['action_history'] = torch.ones(RB, RS, 2, RD)
    out_b = model.rollout(info_b, candidates)['predicted_emb']

    assert out_a.shape == (RB, RS, 3 + 4, RD)
    torch.testing.assert_close(out_a[:, :, :3], emb)  # context passthrough
    assert not torch.allclose(out_a[:, :, 3:], out_b[:, :, 3:])


def test_rollout_window_arithmetic():
    """First prediction sees the full H-frame context with the past blocks
    plus the first candidate; the window then slides, capped at HS."""
    torch.manual_seed(0)
    model = _toy_model(num_frames=3)
    info = _rollout_info(hist_len=3)
    past = torch.randn(RB, RS, 2, RD)
    candidates = torch.randn(RB, RS, 4, RD)
    info['action_history'] = past

    model.rollout(info, candidates)

    calls = model.predictor.calls
    assert len(calls) == 4  # one prediction per candidate
    emb0, act0 = calls[0]
    assert emb0.shape[1] == 3 and act0.shape[1] == 3
    expected_first = torch.cat([past, candidates[:, :, :1]], dim=2).reshape(
        RB * RS, 3, RD
    )
    torch.testing.assert_close(act0, expected_first)
    for emb_t, act_t in calls[1:]:  # window capped at HS = 3
        assert emb_t.shape[1] == 3 and act_t.shape[1] == 3


def test_rollout_rejects_mismatched_action_history():
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    info['action_history'] = torch.randn(RB, RS, 1, RD)  # needs H-1 = 2
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))


def test_rollout_rejects_multiframe_pixels_without_action_history():
    """H > 1 pixels without executed past actions must fail loudly rather
    than silently pairing context frames with optimizer candidates."""
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))


def _population_predictor():
    return Predictor(
        num_frames=3,
        depth=2,
        heads=2,
        mlp_dim=8,
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        dim_head=2,
        dropout=0.0,
        emb_dropout=0.0,
    ).eval()


def _stacked_predictor_parameters(predictor, population=2):
    generator = torch.Generator().manual_seed(123)
    values = []
    for _, parameter in predictor.named_parameters():
        perturbation = torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype
        )
        candidates = [parameter]
        candidates.extend(
            parameter + (index + 1) * 1e-3 * perturbation
            for index in range(population - 1)
        )
        values.append(torch.stack(candidates))
    return tuple(values)


def test_predictor_population_forward_matches_serial_functional_calls():
    torch.manual_seed(0)
    predictor = _population_predictor()
    parameters = _stacked_predictor_parameters(predictor, population=3)
    x = torch.randn(3, 5, 3, 4)
    conditioning = torch.randn_like(x)

    actual = predictor.forward_population(x, conditioning, parameters)
    expected = torch.stack(
        [
            functional_call(
                predictor,
                {
                    name: parameters[index][candidate]
                    for index, name in enumerate(
                        predictor.population_parameter_names
                    )
                },
                (x[candidate], conditioning[candidate]),
            )
            for candidate in range(3)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_population_latent_rollout_matches_serial_predictor_models():
    torch.manual_seed(0)
    predictor = _population_predictor()
    model = LeWM(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
    ).eval()
    parameters = _stacked_predictor_parameters(predictor, population=2)
    emb = torch.randn(2, 1, 4)
    actions = torch.randn(2, 2, 3, 4, 4)

    actual = model.rollout_population_from_embeddings(emb, actions, parameters)
    expected = []
    original = {
        name: value.detach().clone()
        for name, value in predictor.named_parameters()
    }
    with torch.no_grad():
        for candidate in range(2):
            for index, name in enumerate(predictor.population_parameter_names):
                dict(predictor.named_parameters())[name].copy_(
                    parameters[index][candidate]
                )
            expected.append(
                model.rollout_from_embeddings(emb, actions[candidate])
            )
        for name, value in predictor.named_parameters():
            value.copy_(original[name])

    torch.testing.assert_close(
        actual, torch.stack(expected), rtol=3e-5, atol=3e-6
    )
