"""Decoder-only transformer with swappable FFN (dense MLP or MoE).

Shared by:
  - Part 1: all 5 FFN variants (variant 1 = dense MLP, variants 2-5 = MoE).
  - Part 2: dense MLP config (ffn_variant=1), different optimizer + dataset.

Attention uses F.scaled_dot_product_attention (explicitly permitted by the
assignment). Positional embeddings are learned. forward() accepts an optional
attention_mask (B, T) with 1 = real token, 0 = pad, combined with the causal
mask inside the attention layer; without one, attention is purely causal.
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
        self.reset_usage()

    def reset_usage(self) -> None:
        self.usage_counts.zero_()
        self.language_usage = {}
        self._prev_usage = self.usage_counts.clone()

    def record_usage(self, language: str) -> None:
        """Called by the train/eval loop once per batch; buckets routed counts per language."""
        delta = self.usage_counts - self._prev_usage  # routed since the last call
        self._prev_usage = self.usage_counts.clone()
        if language not in self.language_usage:
            self.language_usage[language] = torch.zeros_like(self.usage_counts)
        self.language_usage[language] += delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T, d_model). Output: (B, T, d_model).
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        # TODO(1): Router: gate logits (linear over flat), softmax, take top-k
        #          expert ids and their (re-normalized) weights for each token.
        scores = self.gate(flat)
        gate_probs = torch.softmax(scores, dim=-1)
        topk_w, topk_ids = torch.topk(gate_probs, k=self.n_active, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)  # renormalize over selected

        # TODO(2)+(5): Dispatch: run each expert only on the tokens that picked it,
        #              weighted-combine into a zeroed buffer; skip empty experts.
        out = torch.zeros_like(flat)
        for e in range(self.n_routed):
            coords = (topk_ids == e).nonzero()  # (num_slots, 2): [token_idx, slot_idx]
            if coords.numel() == 0:  # no token routed to this expert
                continue
            tokens = coords[:, 0]
            w = topk_w[tokens, coords[:, 1]]  # weight of expert e per token
            out[tokens] += w[:, None] * self.experts[e](flat[tokens])

        # Variant 4: shared expert is always active; no routing, no gate weights.
        if self.n_shared > 0:
            out += self.shared[0](flat)

        # Usage tracking: one count per activation slot (top-2 tokens count twice).
        self.usage_counts += torch.bincount(topk_ids.flatten(), minlength=self.n_routed)

        return out.view(B, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attn_mask: (B, T), 1 = real token, 0 = pad; combined with the causal mask.
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # One bool mask over (B, 1, T, T): False = do not attend.
            causal = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
            padded = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = ~(causal | padded)  # broadcasts to (B, 1, T, T); True = attend
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = MoE(config) if config.ffn_variant > 1 else MLP(config)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input: (B, T) token ids. Output: (B, T, n_vocab) logits.
        B, T = x.shape
        assert T <= self.config.n_ctx
        if attn_mask is not None:
            assert attn_mask.shape == x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, attn_mask)
        return self.lm_head(self.ln_f(h))

    def record_usage(self, language: str) -> None:
        """Ask every MoE block to accumulate routing counts for `language`."""
        for block in self.blocks:
            if isinstance(block.ffn, MoE):
                block.ffn.record_usage(language)

    def reset_usage(self) -> None:
        for block in self.blocks:
            if isinstance(block.ffn, MoE):
                block.ffn.reset_usage()


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: nn.Module) -> int:
    """Per-token ACTIVE parameters (variant 5's matching quantity vs variant 1).

    Counts FFN layers only: embeddings + attention are identical across variants,
    so they cancel out of any variant-vs-variant comparison. Per layer:
      - dense MLP: 2 * d_model * d_ff
      - MoE: 2 * d_model * (n_active + n_shared) * expert_d_ff + gate params
    """
    config = model.config
    active = 0
    for block in model.blocks:
        if isinstance(block.ffn, MoE):
            m = block.ffn
            d_ff = m.experts[0].fc1.out_features  # uniform expert width
            active += 2 * config.d_model * (m.n_active + m.n_shared) * d_ff
            active += m.gate.weight.numel()
        else:
            active += 2 * config.d_model * config.d_ff
    return active


def ffn_variant_config(variant: int, d_ff: int, n_experts: int = 4) -> dict:
    """Config knobs for an FFN variant, expert widths chosen by param matching.

    Targets: variants 1-4 equal TOTAL FFN params; variant 5 equal ACTIVE FFN params
    (matched to variant 1). Math per layer (ignoring LayerNorms):
      - dense:        2 * d_model * d_ff
      - MoE total:    2 * d_model * n_experts * expert_d_ff
      - MoE active:   2 * d_model * (n_active + n_shared) * expert_d_ff
    The gate adds d_model * n_routed per layer (~0.1% of FFN params) --
    a report-level footnote, not worth compensating.
    """
    cfg = {"ffn_variant": variant, "n_experts": n_experts}
    if variant == 1:
        pass  # dense MLP; no expert knobs needed
    elif variant == 2:
        cfg.update(n_active_experts=1, n_shared_experts=0, expert_d_ff=d_ff // n_experts)
    elif variant == 3:
        cfg.update(n_active_experts=2, n_shared_experts=0, expert_d_ff=d_ff // n_experts)
    elif variant == 4:
        # 1 shared + 3 routed, uniform widths: total = 4 * 2d * (d_ff/4) = dense
        cfg.update(n_active_experts=1, n_shared_experts=1, expert_d_ff=d_ff // n_experts)
    elif variant == 5:
        # top-2 active = dense: expert_d_ff = d_ff / 2 (total becomes 2x dense)
        cfg.update(n_active_experts=2, n_shared_experts=0, expert_d_ff=d_ff // 2)
    else:
        raise ValueError(f"unknown FFN variant: {variant}")
    return cfg
