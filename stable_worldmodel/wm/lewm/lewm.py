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

    def _predict_next(self, emb, act_emb):
        """Project only the final predictor token used by rollouts."""
        return self.pred_proj(self.predictor(emb, act_emb)[:, -1])

    ####################
    ## Inference only ##
    ####################

    def rollout_from_embeddings(
        self,
        emb: torch.Tensor,
        action_sequence: torch.Tensor,
        action_history: torch.Tensor | None = None,
        history_size: int | None = None,
        *,
        terminal_only: bool = False,
    ) -> torch.Tensor:
        """Roll out pre-encoded context without dictionary mutation.

        Set ``terminal_only`` for terminal costs to keep only the active
        predictor window and return ``(B, S, D)`` instead of the full
        ``(B, S, H + T, D)`` sequence.
        """
        history_size = history_size or getattr(self.predictor, 'num_frames', 3)
        B, S, T = action_sequence.shape[:3]
        if emb.ndim == 3:
            emb = emb[:, None].expand(B, S, -1, -1)
        H = emb.size(2)

        if action_history is None:
            action_history = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        elif action_history.ndim == 3:
            action_history = action_history[:, None].expand(B, S, -1, -1)
        assert action_history.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks'
        )

        emb = rearrange(emb, 'b s ... -> (b s) ...')
        actions = rearrange(action_sequence, 'b s ... -> (b s) ...')
        if H > 1:
            actions = torch.cat(
                [
                    rearrange(action_history, 'b s ... -> (b s) ...'),
                    actions,
                ],
                dim=1,
            )
        act_emb = self.action_encoder(actions)

        frames = list(emb.unbind(dim=1))
        for t in range(T):
            lo = max(0, H + t - history_size)
            frames.append(
                self._predict_next(
                    torch.stack(frames[-history_size:], dim=1),
                    act_emb[:, lo : H + t],
                )
            )
            if terminal_only:
                frames = frames[-history_size:]

        if terminal_only:
            return rearrange(frames[-1], '(b s) d -> b s d', b=B, s=S)
        return rearrange(
            torch.stack(frames, dim=1), '(b s) ... -> b s ...', b=B, s=S
        )

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Roll out strictly-future candidates from observation context."""
        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S = action_sequence.shape[:2]
        action_history = info.get('action_history')
        if action_history is None:
            action_history = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert action_history.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks'
        )
        info['action'] = torch.cat(
            [action_history, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
            }
            info['emb'] = (
                self.encode(initial)['emb']
                .detach()
                .unsqueeze(1)
                .expand(B, S, -1, -1)
            )

        info['predicted_emb'] = self.rollout_from_embeddings(
            info['emb'],
            action_sequence,
            action_history=action_history,
            history_size=history_size,
        )
        return info


__all__ = ['LeWM']
