from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_BETA_MIN = 0.5
_BETA_MAX = 32.0


def _resolve_num_heads(dim: int, num_heads: int | None, target_head_dim: int = 32) -> int:
    if num_heads is not None:
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        return num_heads
    for hd in (target_head_dim, 64, 16, 8, 4, 2):
        if dim >= hd and dim % hd == 0:
            return dim // hd
    return 1


class HopfieldFFN(nn.Module):
    """Modern Continuous Hopfield Network (MCHN) FFN for audio / 2D grids.

    Key design choices over a naive tied-key MCHN:
      - Unit-sphere fixed-point retrieval with `beta` initialized at
        `sqrt(head_dim)` (clamped to `[0.5, 32.0]`) so cosine-similarity logits
        start with well-scaled variance instead of collapsing to uniform weights.
      - Untied `values` table read out on the final step so the FFN can produce
        hetero-associative residual updates rather than projections onto keys.
      - Optional null/sink pattern (`attn_sink=True`) with zero value so silent
        or unremarkable tokens can attend to "nothing".
      - Input-driven multiplicative output gate (`sigmoid(out_gate)`) so radial
        energy/amplitude discarded by L2 normalization is preserved.

    Accepts any `(*leading, dim)` tensor (e.g. `[B, T, F, D]`) and returns a
    residual update of the same shape.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        num_patterns: int = 256,
        num_steps: int = 2,
        step_size: float = 1.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        attn_sink: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if num_patterns <= 0:
            raise ValueError(f"num_patterns must be positive, got {num_patterns}.")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if not 0.0 < step_size <= 2.0:
            raise ValueError(f"step_size must be in (0, 2], got {step_size}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        resolved_heads = _resolve_num_heads(dim, num_heads, target_head_dim=32)
        self.dim = dim
        self.num_heads = resolved_heads
        self.head_dim = dim // resolved_heads
        self.num_patterns = num_patterns
        self.num_steps = num_steps
        self.step_size = float(step_size)

        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.keys = nn.Parameter(
            torch.randn(resolved_heads, num_patterns, self.head_dim)
            * (self.head_dim ** -0.5)
        )
        self.values = nn.Parameter(
            torch.randn(resolved_heads, num_patterns, self.head_dim)
            * (self.head_dim ** -0.5)
        )
        init_beta = max(_BETA_MIN, min(_BETA_MAX, math.sqrt(self.head_dim)))
        self.log_beta = nn.Parameter(
            torch.full((resolved_heads, 1, 1), math.log(init_beta))
        )
        if attn_sink:
            self.sink_logit: nn.Parameter | None = nn.Parameter(
                torch.zeros(resolved_heads, 1, 1)
            )
        else:
            self.register_parameter("sink_logit", None)

        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights(residual_scale)

    def _init_weights(self, residual_scale: float) -> None:
        nn.init.normal_(self.in_proj.weight, std=1.0 / math.sqrt(self.dim))
        nn.init.normal_(
            self.out_proj.weight,
            std=(1.0 / math.sqrt(self.dim)) * residual_scale,
        )

    def _attention_weights(
        self,
        xi_norm: torch.Tensor,
        keys_t: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.matmul(xi_norm, keys_t).float() * beta
        if self.sink_logit is None:
            return F.softmax(logits, dim=-1)
        sink = self.sink_logit.float().expand(logits.shape[0], logits.shape[1], 1)
        return F.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        xi_raw, out_gate = self.in_proj(x).chunk(2, dim=-1)
        xi = (
            xi_raw.reshape(-1, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )  # [H, N, d_h]
        xi = F.normalize(xi.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)

        keys = F.normalize(self.keys.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)
        keys_t = keys.transpose(-2, -1)  # [H, d_h, M]
        beta = (
            self.log_beta.float()
            .clamp(math.log(_BETA_MIN), math.log(_BETA_MAX))
            .exp()
        )

        weights: torch.Tensor | None = None
        for step in range(self.num_steps):
            xi_norm = (
                xi
                if step == 0
                else F.normalize(xi.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)
            )
            weights = self._attention_weights(xi_norm, keys_t, beta).to(dtype=x.dtype)
            if step < self.num_steps - 1:
                retrieved = torch.matmul(weights, keys)
                xi = xi_norm + self.step_size * (retrieved - xi_norm)

        assert weights is not None
        out = torch.matmul(self.drop(weights), self.values.to(dtype=x.dtype))
        out = out.transpose(0, 1).contiguous().reshape(*lead, self.dim)
        out = self.out_proj(self.out_norm(out)) * torch.sigmoid(out_gate)
        return self.drop(out)
