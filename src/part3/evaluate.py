"""Part 3 eval harness — metrics, timing, raw-generation dumps.

Metric contract (agreed with the student):

  * gold perplexity — strategy-independent baseline: per-token NLL of the
    gold continuation under the model, given the prompt. Reported ONCE
    (it cannot discriminate strategies; it is the "model quality" number).
  * self perplexity — per-config: NLL of the strategy's OWN generated
    tokens under the model. This is the ppl that varies across configs and
    exposes degeneration (greedy repetition -> suspiciously low values).
  * first-token accuracy — share of runs whose first generated token equals
    the first gold token. EOS-as-first-token counts as a mismatch (an
    immediate stop is a real failure mode). Sampling configs are RNG-seeded
    once per config before decoding, so the numbers are reproducible.
  * timing — wall-clock seconds for the full decode pass (perf_counter
    around the decode loop) on the SAME test split, per config. The
    beam-widths 1/2/4 column feeds the time-complexity discussion.

Generations: one JSONL per config ({id, prompt, gold, generation}), which
is both the "raw generations" deliverable and the source table for the
report's generation samples.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from . import decode as D
from .data import tokenize_stories


def make_configs(
    k_values: tuple = (10, 50),
    p_values: tuple = (0.8, 0.95),
    beam_widths: tuple = (1, 2, 4),
) -> dict[str, tuple]:
    """{config_name: (decode_fn, kwargs)} — the "different generation
    parameters" the assignment asks to sweep (top-k sizes, top-p masses,
    beam widths; greedy sits in the list as the deterministic anchor)."""
    cfgs: dict[str, tuple] = {"greedy": (D.greedy_decode, {})}
    for k in k_values:
        cfgs[f"top-k-{k}"] = (D.top_k_decode, {"k": k})
    for p in p_values:
        cfgs[f"top-p-{p}"] = (D.top_p_decode, {"p": p})
    for w in beam_widths:
        cfgs[f"beam-{w}"] = (D.beam_decode, {"beam_width": w})
    return cfgs


@torch.no_grad()
def continuation_nll(
    model, prompts, conts, device: str, batch_size: int = 32
) -> tuple[float, int]:
    """Batched cumulative NLL of continuation tokens given prompts.

    prompts/conts: parallel lists of unpadded 1-D tensors (conts rows are
    trimmed to their real length). One forward per batch covers all
    positions: log p(c_t) comes from the last-row logits at position
    T_prompt - 1 + t, so no sequential decoding is needed here.
    Returns (nll_sum, n_tokens).
    """
    nll_sum, n_tokens = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        ps = prompts[i : i + batch_size]
        cs = conts[i : i + batch_size]
        seqs = [torch.cat([p, c]) if c.numel() else p for p, c in zip(ps, cs)]
        ids = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
        lp = torch.log_softmax(model(ids, (ids != 0).long()), dim=-1)  # (B, T, V)
        for p, c, row in zip(ps, cs, lp):
            if c.numel() == 0:
                continue  # empty continuation contributes nothing
            Tp = p.shape[0]
            # positions Tp-1 .. Tp+|c|-2 predict c[0..]; last gold token has
            # no successor needed — gather exactly |c| log-probs:
            logp = row[Tp - 1 : Tp + c.numel() - 1].gather(-1, c.unsqueeze(-1)).squeeze(-1)
            nll_sum += -logp.sum().item()
            n_tokens += c.numel()
    return nll_sum, n_tokens


def gold_perplexity(model, tokenized, device: str, batch_size: int = 32) -> float:
    """Strategy-independent baseline: exp(mean NLL) of gold continuations."""
    prompts = [p for p, _ in tokenized]
    golds = [g for _, g in tokenized]
    nll_sum, n_tokens = continuation_nll(model, prompts, golds, device, batch_size)
    return math.exp(nll_sum / n_tokens) if n_tokens else float("inf")


def self_perplexity(model, prompts, gens, device: str, batch_size: int = 32) -> float:
    """Per-config: exp(mean NLL) of the strategy's own generated tokens.

    gens: (B, max_new) padded rows from a decode call; trailing pads are
    trimmed back to each row's real length before scoring, so early-eos
    rows are scored over exactly what they generated.
    """
    lens = (gens != 0).sum(dim=1)
    conts = [gens[i, : lens[i]] for i in range(gens.shape[0])]
    nll_sum, n_tokens = continuation_nll(model, prompts, conts, device, batch_size)
    return math.exp(nll_sum / n_tokens) if n_tokens else float("inf")


def first_token_accuracy(gens: torch.Tensor, golds_first: torch.Tensor) -> float:
    """Share of runs whose first generated token equals the first gold token.

    gens: (B, max_new) padded; golds_first: (B,) first gold token per row.
    An EOS-as-first token (== pad_id 0) never equals a real gold token, so
    an immediate stop is scored as a miss — intended.
    """
    return (gens[:, 0] == golds_first).float().mean().item()


def run_part3_eval(
    model,
    tokenizer,
    stories: list[dict[str, str]],
    out_dir: str | Path,
    *,
    num_samples: int = 1000,
    max_new: int = 96,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cuda",
    configs: dict | None = None,
    max_prompt_tokens: int = 256,
    max_gold_tokens: int = 128,
) -> dict:
    """Full Part 3 eval pass; writes part3_metrics.json + generations/*.jsonl.

    Stories are subsampled deterministically (rng.sample with the seed), so
    a run with --seed 42 always evaluates the same 1000 stories.
    """
    out_dir = Path(out_dir)
    gen_dir = out_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    stories = rng.sample(stories, min(num_samples, len(stories)))
    tokenized = tokenize_stories(stories, tokenizer, max_prompt_tokens, max_gold_tokens)
    tokenized = [(p.to(device), g.to(device)) for p, g in tokenized]
    prompts = [p for p, _ in tokenized]
    golds_first = torch.stack([g[0] for _, g in tokenized])  # (B,) first gold tokens

    gold_ppl = gold_perplexity(model, tokenized, device, batch_size)

    cfgs = make_configs() if configs is None else configs
    results: dict[str, dict] = {}
    for name, (fn, kw) in cfgs.items():
        torch.manual_seed(seed)  # reproducible sampling top-k / top-p
        gens: list[torch.Tensor] = []
        t0 = time.perf_counter()
        for i in range(0, len(tokenized), batch_size):
            batch = tokenized[i : i + batch_size]
            prompts_b = pad_sequence(
                [p for p, _ in batch], batch_first=True, padding_value=0
            )
            gens.append(fn(model, prompts_b, max_new, eos_id=0, pad_id=0, **kw))
        dt = time.perf_counter() - t0
        gens = torch.cat(gens, dim=0)

        lens = (gens != 0).sum(dim=1)
        results[name] = {
            "self_ppl": self_perplexity(model, prompts, gens, device, batch_size),
            "first_token_accuracy": first_token_accuracy(gens, golds_first),
            "time_s": dt,
            "ms_per_sample": 1000.0 * dt / len(tokenized),
            "mean_gen_len": lens.float().mean().item(),
            "empty_frac": (lens == 0).float().mean().item(),
        }
        with open(gen_dir / f"{name}.jsonl", "w") as f:
            for i, (s, row) in enumerate(zip(stories, gens)):
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "prompt": s["prompt"],
                            "gold": s["gold"],
                            "generation": tokenizer.decode(row.tolist()),
                        }
                    )
                    + "\n"
                )

    metrics = {
        "num_samples": len(tokenized),
        "max_new": max_new,
        "seed": seed,
        "gold_ppl": gold_ppl,
        "configs": results,
    }
    with open(out_dir / "part3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # human-readable summary
    print(f"\ngold ppl (baseline, strategy-independent): {gold_ppl:.3f}")
    print(f"{'config':<12}{'self_ppl':>10}{'acc@1':>8}{'len':>7}{'ms/sample':>12}{'empty%':>8}")
    for name, r in results.items():
        print(
            f"{name:<12}{r['self_ppl']:>10.3f}{r['first_token_accuracy']:>8.3f}"
            f"{r['mean_gen_len']:>7.1f}{r['ms_per_sample']:>12.2f}"
            f"{100 * r['empty_frac']:>8.2f}"
        )
    return metrics