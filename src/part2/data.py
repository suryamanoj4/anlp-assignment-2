"""Part 2 data pipeline: plain next-token prediction on browndw/human-ai-parallel-corpus.

PURELY ADDITIVE: imports nothing from src/part1/* and edits nothing there;
src/train.py keeps working on our batches by duck typing (batch.input_ids /
batch.labels / batch.attention_mask, count_real_tokens = attention_mask sum).

Corpus facts (spec: assignment Section 2):
  - single split, 66,320 rows, columns `doc_id`, `text` (English).
  - doc_id = '<text-type>_<n>@<author-model>': 8,290 docs x 8 author-variants
    (human chunk 1, human chunk 2, and 6 LLM continuations of chunk 1). The
    8 rows of a doc form one unit -> splits are grouped BY DOC so no unit
    ever straddles train/val/test (all 8 chunks must stay together).
  - rows are ~500-word chunks; with n_ctx=512 and ~1.3 tok/word most samples
    truncate at max_len, which the token budget must reflect (see
    LMDataset.total_real_tokens).

Samples: [bos] + tokenize(text) + [eos], truncated to max_len; labels are the
shifted ids (EVERY position is a target — no -100 prefixes as in part 1's
translation masking), and the final position is -100 (nothing follows the
last token). Empty texts degrade to [bos, eos] and stay safe.
"""

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Part2Batch:
    input_ids: torch.Tensor  # (B, T)
    labels: torch.Tensor  # (B, T) shifted targets, -100 on the final position
    attention_mask: torch.Tensor  # (B, T) 1 = real token, 0 = pad


def split_rows_by_doc(
    rows: list[dict],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic 90/5/5 split grouped by doc_id.

    Rows keep their (doc_id, text) fields; only the SAMPLE construction
    consumes them. Shuffling the DOC ids (never the rows) with a fixed seed
    makes the split reproducible and keeps the 8 rows of a doc together.
    """
    assert 0.0 < train_ratio + val_ratio < 1.0
    # The doc UNIT is the prefix before '@' — the corpus encodes the
    # author-variant in doc_id (e.g. 'acad_0001@gpt-4o-2024-08-06'), and all
    # 8 variants of a doc must stay in the same split.
    def doc_root(doc_id: str) -> str:
        return doc_id.split("@")[0]
    docs = sorted({doc_root(r["doc_id"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n = len(docs)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    buckets = [set(docs[:n_tr]), set(docs[n_tr:n_tr + n_va]), set(docs[n_tr + n_va:])]
    out = [[], [], []]
    for r in rows:
        for i, bucket in enumerate(buckets):
            if doc_root(r["doc_id"]) in bucket:
                out[i].append(r)
                break
    train, validation, test = out
    print(
        f"[data] doc-grouped split: {n} docs -> "
        f"{len(train):,} train rows / {len(validation):,} val / {len(test):,} test rows "
        f"(seed={seed}, ratios {train_ratio:.2f}/{val_ratio:.2f}/{1 - train_ratio - val_ratio:.2f})"
    )
    return train, validation, test


def make_lm_samples(rows: list[dict], tokenizer, max_len: int):
    """Per-row NTP sample: [bos] + text + [eos] truncated to max_len.

    labels = shifted ids (all positions are targets; -100 marks nothing-to-
    predict after the last real token). Yields plain dicts so the Dataset can
    materialize them into numpy arrays once.
    """
    for row in rows:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        seq = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        seq = seq[:max_len]
        yield {
            "ids": seq,
            "labels": seq[1:] + [-100],
            "attention_mask": [1] * len(seq),
        }


class LMDataset(Dataset):
    """PyTorch wrapper over make_lm_samples (materialized once, like part 1).

    total_real_tokens: sum of real (pre-padding) sample lengths — the token
    budget for training and the 0.1x dataset val cadence derive from it.
    """

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.max_len = max_len
        raw = list(make_lm_samples(rows, tokenizer, max_len))
        self.ids = np.array([s["ids"] for s in raw], dtype=object)
        self.labels = np.array([s["labels"] for s in raw], dtype=object)
        self.attention_masks = np.array([s["attention_mask"] for s in raw], dtype=object)
        self.total_real_tokens = int(sum(len(s["ids"]) for s in raw))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return {
            "ids": self.ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_masks[idx],
        }


def collate_lm_batch(samples, pad_id: int) -> Part2Batch:
    """Pad to the longest sequence in the batch; labels pad with -100."""
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
    return Part2Batch(input_ids=ids, labels=labels, attention_mask=attention_mask)


def count_real_tokens(batch: Part2Batch) -> int:
    """Number of non-pad tokens in the batch (must match src.train.count_real_tokens)."""
    return int(batch.attention_mask.sum().item())


def _loader_from_dataset(
    dataset: LMDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    generator: torch.Generator | None,
    pad_id: int,
    what: str,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        collate_fn=lambda s: collate_lm_batch(s, pad_id=pad_id),
    )
    print(
        f"[data] {what} ready: {len(dataset):,} samples -> {len(loader):,} batches "
        f"of {batch_size} ({'shuffled' if shuffle else 'sequential'}, drop_last={drop_last}) "
        f"| total_real_tokens={dataset.total_real_tokens:,}"
    )
    return loader


def make_dataloader(
    rows: list[dict],
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    generator: torch.Generator | None = None,
    what: str = "dataset",
) -> DataLoader:
    """Build a loader over the NTP samples; `what` names the stage in logs."""
    print(f"[data] building {what}: tokenizing {len(rows):,} rows -> samples ...")
    dataset = LMDataset(rows, tokenizer, max_len)
    return _loader_from_dataset(
        dataset, batch_size, shuffle, num_workers, drop_last, generator,
        tokenizer.pad_token_id, what,
    )


def make_lm_dataloaders(
    train_rows: list[dict],
    val_rows: list[dict],
    tokenizer,
    max_len: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Tokenize train + val ONCE and return (train_loader, val_loader, budget).

    budget = post-truncation real tokens of the train split: the 1x-dataset
    training budget, from which the 0.1x val cadence derives. The shuffling
    stream uses `generator` so every optimizer consumes the SAME stream
    (fairness constant, mirroring part 1).
    """
    print(
        f"[data] building train+validation: tokenizing "
        f"{len(train_rows):,} / {len(val_rows):,} rows -> samples ..."
    )
    train_ds = LMDataset(train_rows, tokenizer, max_len)
    val_ds = LMDataset(val_rows, tokenizer, max_len)
    pad_id = tokenizer.pad_token_id
    train_loader = _loader_from_dataset(
        train_ds, batch_size, shuffle=True, num_workers=0, drop_last=True,
        generator=generator, pad_id=pad_id, what="train split",
    )
    val_loader = _loader_from_dataset(
        val_ds, batch_size, shuffle=False, num_workers=0, drop_last=False,
        generator=None, pad_id=pad_id, what="validation split",
    )
    return train_loader, val_loader, train_ds.total_real_tokens