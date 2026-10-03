"""ROCStories loading + prompt/gold splitting for Part 3.

Dataset: hamishivi/ROCStories. The hub schema (as of 2026) is:

    prompt        — the first sentence
    continuation  — the remaining 4 sentences (the gold continuation)
    text          — full 5-sentence story

so the continuation task is: prompt (sentence 1) -> generate the rest
(.continuation). An older layout with explicit sentence1..sentence5 columns
is handled too: prompt = sentences 1..4, gold = sentence 5.

All decoding strategies share the same prompts, so any metric difference is
attributable to the strategy alone.

Design note: tokenization happens here but padding does not — the eval
harness batches with torch.nn.utils.rnn.pad_sequence(pad_value=pad_id),
which is exactly the right-padded, maskable form decode.py expects.
"""

from __future__ import annotations

import torch
from datasets import load_dataset

ROCDATASET_ID = "hamishivi/ROCStories"
PROMPT_SENTENCES = 4  # legacy schema: sentences 1..4 are the prompt, 5 is gold


def _to_prompt_gold(ex: dict) -> dict[str, str]:
    """Normalize one dataset row to {"prompt", "gold"} regardless of schema."""
    if "continuation" in ex:  # current schema: prompt = sentence 1
        return {"prompt": ex["prompt"], "gold": ex["continuation"]}
    if all(f"sentence{i}" in ex for i in range(1, PROMPT_SENTENCES + 2)):
        prompt = " ".join(ex[f"sentence{i}"] for i in range(1, PROMPT_SENTENCES + 1))
        return {"prompt": prompt, "gold": ex["sentence5"]}
    raise ValueError(
        f"ROCStories schema changed: need 'prompt'/'continuation' or "
        f"sentence1..5; have keys {list(ex)}"
    )


def load_stories(split: str = "test", limit: int | None = None) -> list[dict[str, str]]:
    """Return [{"prompt", "gold"}, ...].

    limit: cap the number of samples (the assignment requires >= 1000;
    default None = the whole split, ~20k rows on test — the eval harness
    typically passes an explicit --num-samples).
    """
    ds = load_dataset(ROCDATASET_ID, split=split)
    rows = [_to_prompt_gold(ex) for ex in ds]
    if limit is not None:
        rows = rows[:limit]
    return rows


def tokenize_stories(
    stories: list[dict[str, str]],
    tokenizer,
    max_prompt_tokens: int,
    max_gold_tokens: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize prompts and golds; truncate long ones, never pad.

    Returns [(prompt_ids, gold_ids), ...] as 1-D tensors; the harness pads
    them into batches. Truncation policy: prompts are capped so that
    prompt_len + max_new fits the model context; golds are capped at
    max_gold_tokens (the continuation is one sentence, ~60 tokens).
    """
    out = []
    for s in stories:
        p = tokenizer(s["prompt"], truncation=True, max_length=max_prompt_tokens)["input_ids"]
        g = tokenizer(s["gold"], truncation=True, max_length=max_gold_tokens)["input_ids"]
        out.append((torch.tensor(p), torch.tensor(g)))
    return out