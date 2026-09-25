"""Decoder-only transformer with swappable FFN (dense MLP or MoE).

Shared by:
  - Part 1: all 5 FFN variants (variant 1 = dense MLP, variants 2-5 = MoE).
  - Part 2: dense MLP config (ffn_variant=1), different optimizer + dataset.

Attention uses F.scaled_dot_product_attention (explicitly permitted by the
assignment). Positional embeddings are learned; sequences must be left-packed
(no padding) since attention is purely causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    d_model: int = 384
    n_ctx: int = 512
    n_vocab: int = 32_000
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1536  # dense FFN width (variants 1, and reference for 5)
    tie_embeddings: bool = True
    dropout: float = 0.0

    # FFN type: 1 = dense MLP, 2-5 = MoE variants from the assignment
    ffn_variant: int = 1
    n_experts: int = 4
    n_active_experts: int = 1  # routed experts active per token
    n_shared_experts: int = 0  # variant 4: 1 shared expert, always active
    expert_d_ff: Optional[int] = None  # per-expert FFN width; set via param matching


class MLP(nn.Module):
    """Two-layer dense FFN (variant 1, and the per-expert unit inside MoE)."""

    def __init__(self, config: TransformerConfig, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff if d_ff is not None else config.d_ff
        self.fc1 = nn.Linear(config.d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, config.d_model, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class MoE(nn.Module):
    """Mixture-of-Experts FFN for variants 2-5.

    Routing, dispatch, and combine are deliberately left as TODOs: they are the
    core of the assignment. The module tracks routed-expert usage counts for the
    language-specialization heatmap.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_routed = config.n_experts - config.n_shared_experts
        self.n_active = config.n_active_experts
        self.n_shared = config.n_shared_experts
        d_ff = config.expert_d_ff if config.expert_d_ff is not None else config.d_ff // config.n_experts

        self.experts = nn.ModuleList([MLP(config, d_ff=d_ff) for _ in range(self.n_routed)])
        self.shared = nn.ModuleList([MLP(config, d_ff=d_ff) for _ in range(self.n_shared)])
        self.gate = nn.Linear(config.d_model, self.n_routed, bias=False)

        self.register_buffer("usage_counts", torch.zeros(self.n_routed, dtype=torch.long))
        self.language_usage: dict[str, torch.Tensor] = {}

    def reset_usage(self) -> None:
        self.usage_counts.zero_()
        self.language_usage = {}

    def record_usage(self, language: str) -> None:
        """Called by the training loop with the batch's language; feeds the heatmap."""
        # TODO(6): accumulate this batch's per-expert token counts into
        #          self.language_usage[language] (a LongTensor over routed experts).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T, d_model). Output: (B, T, d_model).
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        # TODO(1): Router: gate logits (linear over flat), softmax, take top-k
        #          expert ids and their (re-normalized) weights for each token.
        # TODO(2): Dispatch: group token indices by chosen expert id; run each
        #          expert only on its own tokens; combine outputs as the
        #          gate-weighted sum over the active experts per token.
        # TODO(3): Variant 4: add the shared expert output (always active) to
        #          the routed combination.
        # TODO(4): Track usage: bump self.usage_counts per routed token so the
        #          heatmap can be built after evaluation.
        # TODO(5): Handle batches where some experts receive zero tokens.
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = MoE(config) if config.ffn_variant > 1 else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.n_vocab, config.d_model)
        self.pos_emb = nn.Embedding(config.n_ctx, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T) token ids. Output: (B, T, n_vocab) logits.
        B, T = x.shape
        assert T <= self.config.n_ctx
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.ln_f(h))


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: nn.Module) -> int:
    """Per-token active parameter count (the quantity variant 5 must match to variant 1).

    TODO(7): compute it analytically:
      - embedding + attention params are identical across variants; decide whether
        they are included in the "active" budget (they are active for every token,
        so including them is defensible; what matters is consistency).
      - per FFN layer, dense MLP: 2 * d_model * d_ff.
      - per FFN layer, MoE: 2 * d_model * (n_active_experts + n_shared_experts) * expert_d_ff,
        plus the (tiny) gate projection.
    """
    raise NotImplementedError


def ffn_variant_config(variant: int, d_ff: int, n_experts: int = 4) -> dict:
    """Config knobs for a given FFN variant, with expert widths chosen by param matching.

    Target: variants 1-4 have equal TOTAL params; variant 5 has equal ACTIVE params
    (matched to variant 1).

    Math to verify (per FFN layer, ignoring LayerNorms):
      - dense MLP params        ~ 2 * d_model * d_ff
      - MoE total params        ~ 2 * d_model * n_experts * expert_d_ff
      - MoE active params       ~ 2 * d_model * (n_active + n_shared) * expert_d_ff

    TODO(8): fill in the expert_d_ff (and shared-expert width, variant 4) per variant:
      - v2 (4 experts, top-1):  total = dense  ->  expert_d_ff = d_ff / 4
      - v3 (4 experts, top-2):  total = dense  ->  same widths as v2
      - v4 (1 shared + 3 routed, 2/4 active): pick widths so TOTAL still matches dense
        (shared expert and routed experts need not have the same width)
      - v5 (4 experts, top-2):  active = dense ->  expert_d_ff = d_ff / 2
    """
    raise NotImplementedError