"""Data pipeline for part 1: vi/ja -> en translation.

Dataset: belumind/en-vi-ja-curated-500k-triplets.
Each row (vi, ja, en) produces TWO training samples: (vi -> en) and (ja -> en).
Every sample carries its source language label so the training loop can feed the
MoE per-expert usage heatmap.
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


@dataclass
class Part1Batch:
    input_ids: torch.Tensor  # (B, T) token ids
    labels: torch.Tensor  # (B, T) shifted targets, -100 where ignored by the loss
    attention_mask: torch.Tensor  # (B, T) 1 = real token, 0 = pad
    language: list[str]  # per-sample source language ("vi" / "ja")


def make_samples(hf_dataset, tokenizer: PreTrainedTokenizerFast, max_len: int):
    """Expand the HF dataset into (ids, labels, attention_mask, language) samples.

    Columns: en, vi, ja. Each row yields two samples: vi -> en and ja -> en,
    each tagged with its source language.
    """
    for row in hf_dataset:
        en_tokens = tokenizer(row["en"], add_special_tokens=False)["input_ids"]
        for lang in ("vi", "ja"):
            src_tokens = tokenizer(row[lang], add_special_tokens=False)["input_ids"]
            sequence = (
                [tokenizer.bos_token_id]
                + src_tokens
                + [tokenizer.eos_token_id]
                + en_tokens
                + [tokenizer.eos_token_id]
            )
            labels = sequence[1:] + [-100]
            n_prefix = 1 + len(src_tokens) + 1
            labels[:n_prefix] = [-100] * n_prefix
            yield {
                "ids": sequence,
                "labels": labels,
                "attention_mask": [1] * len(sequence),
                "language": lang,
            }

class TranslationDataset(Dataset):
    """PyTorch wrapper over make_samples output.
    """

    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizerFast, max_len: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        raw_samples = list(make_samples(hf_dataset, tokenizer, max_len))
        self.ids = np.array([s["ids"] for s in raw_samples], dtype=object)
        self.labels = np.array([s["labels"] for s in raw_samples], dtype=object)
        self.attention_masks = np.array([s["attention_mask"] for s in raw_samples], dtype=object)
        self.languages = np.array([s["language"] for s in raw_samples], dtype=object)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return {
            "ids": self.ids[idx][:self.max_len],
            "labels": self.labels[idx][:self.max_len],
            "attention_mask": self.attention_masks[idx][:self.max_len],
            "language": self.languages[idx],
        }


def collate_batch(samples, pad_id: int) -> Part1Batch:
    """Pad a batch to the longest sequence and build the attention mask."""
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s["ids"], dtype=torch.long) for s in samples],
        batch_first=True,
        padding_value=pad_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s["labels"], dtype=torch.long) for s in samples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (ids != pad_id).long()
    language = [s["language"] for s in samples]
    return Part1Batch(
        input_ids=ids,
        labels=labels,
        attention_mask=attention_mask,
        language=language,
    )


def make_dataloader(
    hf_dataset,
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = TranslationDataset(hf_dataset, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,  # same generator across variants => same shuffle stream
        collate_fn=functools.partial(collate_batch, pad_id=tokenizer.pad_token_id),
    )


def count_real_tokens(batch: Part1Batch) -> int:
    """Number of non-pad tokens in the batch (what the training-token budget counts)."""
    return int(batch.attention_mask.sum().item())
