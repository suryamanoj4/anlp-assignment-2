"""Part 3 decoding strategies — built strictly on model.forward().

The assignment bans model.generate() and any third-party decoding; every
strategy here implements token selection itself. This module currently holds
the FIRST citizen only:

  greedy_decode  — batched argmax continuation (Part 2's BLEU eval consumes
                    it). Part 3 extends the module with top-k, top-p and
                    beam search (widths 1/2/4) on the same forward path.

Relationship to part 1: src/part1/evaluate.greedy_translate is the earlier
single-sequence, translation-flavored twin (bos + src + eos prompts, BLEU
for the vi/ja->en eval). It stays untouched; this module is the canonical,
batched home that part 3 grows. Both are hand-rolled per the assignment.

Decoding contract (shared by all Part 3 strategies):
  - model(x, attn_mask) -> (B, T, V) logits; mask: 1 = real token, 0 = pad;
  - sequences are extended one token at a time from the last logits row;
  - eos terminates a sequence; finished rows emit pad_id tokens so batches
    stay rectangular, and pad positions are masked out of attention.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_decode(
    model,
    prompt_ids: torch.Tensor,
    max_new: int,
    eos_id: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """Greedy continuation of batched prompts.

    prompt_ids: (B, T_prompt), already padded with pad_id.
    Returns (B, <=max_new) generated ids; rows stopped early at eos are
    right-padded with pad_id.

    NOTE (evaluation-grade simplicity): re-encodes the full sequence each
    step — no KV cache. Fine for the part 2 BLEU pass (26M model, ~380-token
    prompts, ~3k forwards per checkpoint); Part 3's beam-width-vs-time
    analysis will address caching explicitly.

    Caller contract: prompt_len + max_new must fit the model context
    (main.py enforces it via max_prompt = n_ctx - max_new); we verify it
    defensively when the model exposes config.n_ctx.
    """
    n_ctx = getattr(getattr(model, "config", None), "n_ctx", None)
    if n_ctx is not None and prompt_ids.shape[1] + max_new > n_ctx:
        raise ValueError(
            f"prompt({prompt_ids.shape[1]}) + max_new({max_new}) exceeds "
            f"model context n_ctx={n_ctx}"
        )
    ids = prompt_ids.clone()
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new):
        if bool(done.all()):
            break
        masks = (ids != pad_id).long()  # real + generated positions only
        logits = model(ids, masks)  # (B, T, V)
        nxt = logits[:, -1, :].argmax(dim=-1)  # (B,) greedy token selection
        nxt = torch.where(done, torch.tensor(pad_id, device=ids.device), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done |= nxt == eos_id
    return ids[:, prompt_ids.shape[1]:]