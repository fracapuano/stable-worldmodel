import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


def _population_affine(x, value):
    """Broadcast one affine vector per population member over ``x``."""
    shape = (value.size(0),) + (1,) * (x.ndim - 2) + (value.size(-1),)
    return value.reshape(shape)


def _population_linear(x, weight, bias=None):
    """Linear projection with weights shaped ``(population, out, in)``."""
    shape = x.shape
    output = torch.bmm(
        x.reshape(shape[0], -1, shape[-1]), weight.transpose(1, 2)
    ).reshape(*shape[:-1], weight.size(1))
    if bias is not None:
        output = output + _population_affine(output, bias)
    return output


def _population_layer_norm(x, weight=None, bias=None, *, eps=1e-5):
    """Layer normalization with optional candidate-specific affine terms."""
    mean = x.mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(
        (x - mean).square().mean(dim=-1, keepdim=True) + eps
    )
    if weight is not None:
        normalized = normalized * _population_affine(normalized, weight)
    if bias is not None:
        normalized = normalized + _population_affine(normalized, bias)
    return normalized


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x

    @property
    def population_parameter_names(self):
        """Stable parameter order consumed by :meth:`forward_population`."""
        return tuple(name for name, _ in self.named_parameters())

    @staticmethod
    def _project_population(x, module, state, prefix):
        if isinstance(module, nn.Identity):
            return x
        if not isinstance(module, nn.Linear):
            raise TypeError(
                'population predictor projections must be Linear or Identity, '
                f'got {type(module).__name__}'
            )
        bias = state.get(f'{prefix}.bias')
        return _population_linear(x, state[f'{prefix}.weight'], bias)

    def _attention_population(self, x, attention, state, prefix):
        norm_prefix = f'{prefix}.norm'
        x = _population_layer_norm(
            x,
            state[f'{norm_prefix}.weight'],
            state[f'{norm_prefix}.bias'],
            eps=attention.norm.eps,
        )
        qkv = _population_linear(x, state[f'{prefix}.to_qkv.weight'])
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, 'p b t (h d) -> p b h t d', h=attention.heads)
            for t in (q, k, v)
        )
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        output = rearrange(output, 'p b h t d -> p b t (h d)')
        return self._project_population(
            output,
            attention.to_out[0]
            if isinstance(attention.to_out, nn.Sequential)
            else attention.to_out,
            state,
            f'{prefix}.to_out.0'
            if isinstance(attention.to_out, nn.Sequential)
            else f'{prefix}.to_out',
        )

    def _feedforward_population(self, x, feedforward, state, prefix):
        norm = feedforward.net[0]
        x = _population_layer_norm(
            x,
            state[f'{prefix}.net.0.weight'],
            state[f'{prefix}.net.0.bias'],
            eps=norm.eps,
        )
        x = _population_linear(
            x,
            state[f'{prefix}.net.1.weight'],
            state[f'{prefix}.net.1.bias'],
        )
        activation = feedforward.net[2]
        if not isinstance(activation, nn.GELU):
            raise TypeError(
                'population predictor feed-forward activation must be GELU, '
                f'got {type(activation).__name__}'
            )
        x = F.gelu(x, approximate=activation.approximate)
        return _population_linear(
            x,
            state[f'{prefix}.net.4.weight'],
            state[f'{prefix}.net.4.bias'],
        )

    def forward_population(self, x, c, parameters):
        """Evaluate independently parameterized predictors in one tensor graph.

        Args:
            x: Embeddings shaped ``(population, batch, time, dim)``.
            c: Conditioning embeddings with the same leading dimensions.
            parameters: Tuple of tensors ordered like
                :attr:`population_parameter_names`; every tensor has a leading
                population dimension followed by the corresponding parameter
                shape.

        This inference-only path applies candidate-specific linear and
        normalization weights explicitly.  Attention sees the population as
        an ordinary batch dimension, avoiding ``vmap`` fallbacks around scaled
        dot-product attention while retaining fused CUDA attention kernels.
        """
        if self.training:
            raise RuntimeError(
                'forward_population is inference-only; call eval()'
            )
        names = self.population_parameter_names
        if len(parameters) != len(names):
            raise ValueError(
                f'expected {len(names)} predictor parameter tensors, '
                f'got {len(parameters)}'
            )
        state = dict(zip(names, parameters, strict=True))
        population = x.size(0)
        if c.size(0) != population or any(
            value.size(0) != population for value in parameters
        ):
            raise ValueError(
                'all population inputs must share their leading size'
            )

        time = x.size(2)
        x = x + state['pos_embedding'][:, :, :time]
        transformer = self.transformer
        x = self._project_population(
            x, transformer.input_proj, state, 'transformer.input_proj'
        )
        c = self._project_population(
            c, transformer.cond_proj, state, 'transformer.cond_proj'
        )

        for index, block in enumerate(transformer.layers):
            if not isinstance(block, ConditionalBlock):
                raise TypeError(
                    'population predictor requires ConditionalBlock layers, '
                    f'got {type(block).__name__}'
                )
            prefix = f'transformer.layers.{index}'
            modulation = _population_linear(
                F.silu(c),
                state[f'{prefix}.adaLN_modulation.1.weight'],
                state[f'{prefix}.adaLN_modulation.1.bias'],
            )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = modulation.chunk(6, dim=-1)
            attn_input = modulate(
                _population_layer_norm(x, eps=block.norm1.eps),
                shift_msa,
                scale_msa,
            )
            x = x + gate_msa * self._attention_population(
                attn_input, block.attn, state, f'{prefix}.attn'
            )
            mlp_input = modulate(
                _population_layer_norm(x, eps=block.norm2.eps),
                shift_mlp,
                scale_mlp,
            )
            x = x + gate_mlp * self._feedforward_population(
                mlp_input, block.mlp, state, f'{prefix}.mlp'
            )

        x = _population_layer_norm(
            x,
            state['transformer.norm.weight'],
            state['transformer.norm.bias'],
            eps=transformer.norm.eps,
        )
        return self._project_population(
            x, transformer.output_proj, state, 'transformer.output_proj'
        )
