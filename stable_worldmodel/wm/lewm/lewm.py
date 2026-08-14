import torch
from einops import rearrange
from torch import nn


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout_from_embeddings(
        self,
        emb: torch.Tensor,
        action_sequence: torch.Tensor,
        action_history: torch.Tensor | None = None,
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Pure-tensor latent rollout used by accelerated planners.

        Args:
            emb: Encoded context frames with shape ``(B, H, D)`` or an
                already sample-expanded ``(B, S, H, D)`` tensor.
            action_sequence: Strictly-future action blocks with shape
                ``(B, S, T, action_dim)``.
            action_history: Executed blocks between context frames, with
                shape ``(B, H - 1, action_dim)`` or
                ``(B, S, H - 1, action_dim)``.
            history_size: Maximum predictor context. Defaults to the
                predictor's configured ``num_frames``.

        Returns:
            Context plus predicted embeddings with shape
            ``(B, S, H + T, D)``.

        This method has no dictionary mutation or observation encoding. Its
        fixed tensor contract is suitable for ``torch.compile`` and structured
        device-side control flow.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        B, S, T = action_sequence.shape[:3]
        if emb.ndim == 3:
            emb = emb.unsqueeze(1).expand(B, S, -1, -1)
        if emb.ndim != 4 or emb.shape[:2] != (B, S):
            raise ValueError('emb must have shape (B, H, D) or (B, S, H, D)')

        H = emb.size(2)
        if action_history is None:
            action_history = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        elif action_history.ndim == 3:
            action_history = action_history.unsqueeze(1).expand(B, S, -1, -1)
        if action_history.ndim != 4 or action_history.shape[:2] != (B, S):
            raise ValueError(
                'action_history must have shape (B, H-1, A) or (B, S, H-1, A)'
            )
        assert action_history.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {action_history.size(2)}'
        )

        emb_init = rearrange(emb, 'b s ... -> (b s) ...')
        act_past_flat = rearrange(action_history, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )

        emb_list = list(emb_init.unbind(dim=1))
        for t in range(T):
            lo = max(0, H + t - history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        predicted = torch.stack(emb_list, dim=1)
        return rearrange(predicted, '(b s) ... -> b s ...', b=B, s=S)

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, h, w) — H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S = action_sequence.shape[:2]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        info['predicted_emb'] = self.rollout_from_embeddings(
            info['emb'],
            action_sequence,
            action_history=act_past,
            history_size=history_size,
        )

        return info


__all__ = ['LeWM']
