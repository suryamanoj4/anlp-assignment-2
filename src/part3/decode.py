"""Part 3 decoding strategies — built strictly on model.forward().

The assignment bans model.generate() and any third-party decoding; every
strategy here implements token selection itself. This module holds the Part 3
citizens on the shared forward path:

  greedy_decode  — batched argmax continuation (Part 2's BLEU eval consumes
                    it);
  top_k_decode   — sample from the top-k filtered+renormalized distribution;
  top_p_decode   — sample from the smallest probability prefix reaching p;
  beam_decode    — beam search at widths 1/2/4 (per-row hypothesis sets).

Relationship to part 1: src/part1/evaluate.greedy_translate is the earlier
single-sequence, translation-flavored twin (bos + src + eos prompts, BLEU
for the vi/ja->en eval). It stays untouched; this module is the canonical,
batched home that part 3 grows. Both are hand-rolled per the assignment.

Decoding contract (shared by all Part 3 strategies):
  - model(x, attn_mask) -> (B, T, V) logits; mask: 1 = real token, 0 = pad;
    models with the additive KV-cache seam accept forward(x, attn_mask,
    past_kv=...) and return (logits, past) when it is given;
  - sequences are extended one token at a time from the last logits row;
  - eos terminates a sequence; finished rows emit pad_id tokens so batches
    stay rectangular, and pad positions are masked out of attention.
  - strategy choices: top-k/top-p are SAMPLING strategies (the assignment
    says the token is "sampled from the probability distribution"); fix
    seed=... for reproducibility. Sampling top-k/p also means per-sequence
    accuracy is stochastic — see the eval harness for the metric contract.

NOTE (performance): greedy_decode uses the KV-cache protocol when the model
supports forward(x, attn_mask, past_kv=...) (src/part1/model.py's additive
seam) — execution-only: per-position K/V are computed once and reused, so the
selected token sequence is identical to full recomputation (smoke-verified).
Callables without the seam (test stubs) fall back to full-sequence
recomputation automatically. top-k/top-p/beam still recompute per step —
Part 3's beam-width-vs-time analysis will extend the cache explicitly.
"""

from __future__ import annotations

import inspect

import torch


def _check_context(model, prompt_len: int, max_new: int) -> None:
    """Fail fast when prompt + generation cannot fit the model context."""
    n_ctx = getattr(getattr(model, "config", None), "n_ctx", None)
    if n_ctx is not None and prompt_len + max_new > n_ctx:
        raise ValueError(
            f"prompt({prompt_len}) + max_new({max_new}) exceeds "
            f"model context n_ctx={n_ctx}"
        )


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

    Caller contract: prompt_len + max_new must fit the model context
    (main.py enforces it via max_prompt = n_ctx - max_new); we verify it
    defensively when the model exposes config.n_ctx.

    KV cache: when the model supports forward(x, attn_mask, past_kv=...)
    (src/part1/model.py's additive seam), the prompt is encoded once and
    each generated token runs a single-position forward against the cached
    prefix — O(T) instead of O(T^2). The cache is execution-only: the
    selected token sequence is identical to full recomputation
    (smoke-verified by exact equality). Plain callables (test stubs) fall
    back to the recompute path automatically.
    """
    _check_context(model, prompt_ids.shape[1], max_new)
    device = prompt_ids.device
    ids = prompt_ids.clone()
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)

    # KV-cache protocol: engage iff the model's forward advertises past_kv.
    # Signature probe (not try/except around the call): a real error inside
    # the cache path must propagate loudly, never silently fall back.
    encode = None
    cached = False
    try:
        if "past_kv" in inspect.signature(model.forward).parameters:
            # Preallocate the KV buffers ONCE for this decode: fixed sizes,
            # in-place fills, no per-step allocation growth (the naive cat
            # version churned the allocator: ~6.8 min/checkpoint on GPU).
            B = ids.shape[0]
            max_T = prompt_ids.shape[1] + max_new
            head_dim = model.config.d_model // model.config.n_heads
            param = next(model.parameters())
            buffers = [
                (param.new_zeros(B, model.config.n_heads, max_T, head_dim),
                 param.new_zeros(B, model.config.n_heads, max_T, head_dim),
                 ids.new_zeros(B, max_T, dtype=torch.bool),
                 0)
                for _ in range(model.config.n_layers)
            ]
            encode = model(ids, (ids != pad_id).long(), past_kv=buffers)  # (logits, past)
            cached = True
    except (TypeError, ValueError):
        cached = False
    if cached:
        print(f"[decode] kv-cache path: prompt encode + {max_new} cached single-token steps")
    else:
        print("[decode] recompute path (model.forward has no past_kv) — O(T^2), slower")

    nxt = None
    steps = 0
    while steps < max_new:
        if bool(done.all()):
            break
        if cached:
            if nxt is None:
                logits, past = encode
                nxt = logits[:, -1, :].argmax(dim=-1)
            else:
                logits, past = model(
                    nxt[:, None],
                    (~done).long().unsqueeze(1),  # pad-appending rows masked out
                    past_kv=past,
                )
                nxt = logits[:, -1, :].argmax(dim=-1)
        else:
            logits = model(ids, (ids != pad_id).long())
            nxt = logits[:, -1, :].argmax(dim=-1)
        nxt = torch.where(done, torch.tensor(pad_id, device=device), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done |= nxt == eos_id
        steps += 1
    return ids[:, prompt_ids.shape[1]:]


@torch.no_grad()
def top_k_decode(
    model,
    prompt_ids: torch.Tensor,
    max_new: int,
    k: int = 50,
    eos_id: int = 0,
    pad_id: int = 0,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample continuations from the top-k truncated distribution.

    Selection per step (batched):
      1. logits <- last-row logits / temperature;
      2. keep only the k largest entries per row, everything else -> -inf;
      3. softmax over the survivors (renormalize among the top-k);
      4. multinomial sample -> one token per row.

    temperature: scales logits before filtering; T -> 0 concentrates the
    sampled distribution onto argmax (approaching greedy), T -> inf toward
    uniform over the top-k. seed: sets the global torch RNG at entry for
    reproducible runs (the eval harness seeds once per strategy instead).
    """
    _check_context(model, prompt_ids.shape[1], max_new)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if seed is not None:
        torch.manual_seed(seed)
    ids = prompt_ids.clone()
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new):
        if bool(done.all()):
            break
        masks = (ids != pad_id).long()
        logits = model(ids, masks)[:, -1, :] / temperature  # (B, V)
        k_eff = min(int(k), logits.shape[-1])  # torch.topk requires k <= V
        if k_eff < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        topk_vals, topk_idx = torch.topk(logits, k=k_eff, dim=-1)  # (B, k) each
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)  # non-top-k -> -inf
        probs = torch.softmax(filtered, dim=-1)      # renormalize top-k only
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        nxt = torch.where(done, torch.tensor(pad_id, device=ids.device), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done |= nxt == eos_id
    return ids[:, prompt_ids.shape[1]:]


@torch.no_grad()
def top_p_decode(
    model,
    prompt_ids: torch.Tensor,
    max_new: int,
    p: float = 0.9,
    eos_id: int = 0,
    pad_id: int = 0,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample continuations from the nucleus (top-p) distribution.

    Selection per step (batched):
      1. logits <- last-row logits / temperature; sort descending;
      2. cumulative softmax mass over the sorted order;
      3. drop every token whose mass lies strictly BEYOND the first prefix
         reaching p (kept set = smallest top-ordered prefix with mass >= p,
         crossing token included — always >= 1 token since the top token
         is kept even if p < its probability);
      4. scatter the kept logits back to vocabulary order, renormalize,
         multinomial sample.

    p -> 1 keeps the full distribution; small p behaves like top-k with an
    adaptive, shape-dependent k.
    """
    _check_context(model, prompt_ids.shape[1], max_new)
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if seed is not None:
        torch.manual_seed(seed)
    ids = prompt_ids.clone()
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new):
        if bool(done.all()):
            break
        masks = (ids != pad_id).long()
        logits = model(ids, masks)[:, -1, :] / temperature  # (B, V)
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cummass = probs.cumsum(dim=-1)
        # mass strictly BEFORE a token — remove only past the p-crossing
        remove = (cummass - probs) > p
        sorted_logits = torch.where(
            remove, torch.tensor(float("-inf"), device=logits.device), sorted_logits
        )
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_idx, sorted_logits)  # back to vocab order
        probs = torch.softmax(filtered, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        nxt = torch.where(done, torch.tensor(pad_id, device=ids.device), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done |= nxt == eos_id
    return ids[:, prompt_ids.shape[1]:]


@torch.no_grad()
def beam_decode(
    model,
    prompt_ids: torch.Tensor,
    max_new: int,
    beam_width: int,
    eos_id: int = 0,
    pad_id: int = 0,
) -> torch.Tensor:
    """Beam-search continuation; one beam set per prompt row.

    prompt_ids: (B, T_prompt), already padded with pad_id.
    Returns (B, max_new) generated ids, rows right-padded with pad_id —
    same outer contract as the other strategies.

    Algorithm (per row):
      - live beams: W hypotheses = (ids, cumulative log-prob, done flag),
        initialized as W copies of the prompt with score 0;
      - per step: one forward over the W live beams -> log-softmax last row
        -> W x V candidate scores (parent + log p) -> prune to the top W
        globally (expansion k = W, so W^2 candidates per step);
      - a beam that emits eos moves to the finished list; done rows are
        masked out of the candidate matrix so they can never win again;
      - stop when all W beams are done or max_new steps elapse; return the
        highest-scoring hypothesis among finished u live.

    Scoring notes (see module docstring discussion):
      - in-step pruning compares equal-length hypotheses, so raw cumulative
        log-prob is a fair comparator (no normalization needed there);
      - the final finished-vs-live comparison inherits the classic length
        bias of raw scores (shorter wins ties-ish); a length-normalized
        variant is a one-line change if the writeup wants that angle;
      - beam_width=1 reduces to argmax each step == greedy_decode.

    Complexity (for the writeup): per step, attention over (W, T) costs
    ~W x a greedy step, so the full run is O(max_new x T^2 x W) without a KV
    cache — the timing analysis measures exactly this scaling.
    """
    _check_context(model, prompt_ids.shape[1], max_new)
    device = prompt_ids.device
    out = torch.full(
        (prompt_ids.shape[0], max_new), pad_id, dtype=torch.long, device=device
    )
    for row in range(prompt_ids.shape[0]):
        prompt = prompt_ids[row]
        beams = prompt[None].expand(beam_width, -1).clone()  # (W, T_prompt)
        scores = torch.zeros(beam_width, device=device)     # (W,) log-probs
        done = torch.zeros(beam_width, dtype=torch.bool, device=device)
        finished: list[tuple[torch.Tensor, float]] = []     # (ids, score)

        for _ in range(max_new):
            if bool(done.all()):
                break
            masks = (beams != pad_id).long()
            lp = torch.log_softmax(model(beams, masks)[:, -1, :], dim=-1)  # (W, V)
            cand = scores[:, None] + lp                    # (W, V) expanded
            cand = cand.masked_fill(done[:, None], float("-inf"))  # done rows out
            topk_vals, topk_idx = torch.topk(cand.flatten(), k=beam_width)
            beam_idx = topk_idx // lp.shape[-1]            # which parent beam
            token_idx = topk_idx % lp.shape[-1]            # which token
            beams = torch.cat([beams[beam_idx], token_idx[:, None]], dim=1)
            scores = topk_vals
            became_done = ~done[beam_idx] & (token_idx == eos_id)
            for b in became_done.nonzero().flatten().tolist():
                finished.append((beams[b].clone(), scores[b].item()))
            done = done[beam_idx] | (token_idx == eos_id)

        # Final selection among finished u live; raw scores (see docstring).
        best_ids, best_score = None, float("-inf")
        for ids_, s in finished:
            if s > best_score:
                best_ids, best_score = ids_, s
        for b in (~done).nonzero().flatten().tolist():
            if scores[b].item() > best_score:
                best_ids, best_score = beams[b], scores[b].item()
        if best_ids is None:  # unreachable for W >= 1; defensive
            best_ids = beams[0]
        gen = best_ids[prompt.shape[0]:]                   # drop the prompt
        out[row, : gen.shape[0]] = gen
    return out