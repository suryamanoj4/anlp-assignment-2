"""Pythia-160M loading for Part 3.

The decoding contract shared by every strategy in src/part3/decode.py is:

    model(x, attn_mask) -> (B, T, V) raw logits
    x: (B, T) id tensor, right-padded with pad_id; attn_mask: 1 = real, 0 = pad

GPTNeoXForCausalLM.forward matches the (input_ids, attention_mask) calling
convention but returns a CausalLMOutputWithPast (a ModelOutput), which does
not support the `[:, -1, :]` slicing decode.py does. The tiny Pythia adapter
below normalizes the return type to the contract — so strategies never touch
`.logits` themselves and stay model-agnostic.

Deliberately not used anywhere: model.generate() and all HF generation
helpers — the assignment bans them, so this module only ever calls forward.

Note: HF hub ids are lowercase ("EleutherAI/pythia-160m"); the assignment's
"Pythia-160M" is this checkpoint. The -deduped twin exists; swap the constant
if a run should compare the two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

PYTHIA_ID = "EleutherAI/pythia-160m"


class Pythia(nn.Module):
    """GPTNeoX wrapped to the decode.py contract: forward -> raw logits."""

    def __init__(self, core: GPTNeoXForCausalLM):
        super().__init__()
        self.core = core
        self.config = core.config  # expose n_ctx etc. to decode.py's checks

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.core(input_ids=input_ids, attention_mask=attention_mask).logits


def load_pythia(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Pythia, GPTNeoXTokenizerFast]:
    """Load EleutherAI/pythia-160m in half precision (inference only).

    The tokenizer has no pad token; pad_token = eos_token, so batched
    prompts padded with pad_id (decode.py's default 0 = "<|endoftext|>")
    are still terminated correctly by eos handling.
    """
    tok = GPTNeoXTokenizerFast.from_pretrained(PYTHIA_ID)
    tok.pad_token = tok.eos_token
    core = GPTNeoXForCausalLM.from_pretrained(PYTHIA_ID, torch_dtype=dtype)
    model = Pythia(core).to(device).eval()
    return model, tok