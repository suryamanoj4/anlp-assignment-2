# Advanced NLP — Assignment 2

Implementation of the three tasks of the Advanced NLP (Monsoon 2026) assignment 2:
Mixture-of-Experts FFN variants for translation, custom optimizer
implementations, and decoding strategies built on `model.forward()`.

All three parts share a single CLI entry point (`main.py`), the same
tokenizer utilities, and the same training loop in `src/train.py`.

## Repository layout

```
main.py                   CLI entry point (uv run python main.py part{N})
src/
  tokenizer.py            BPE tokenizer training/loading (parts 1-2)
  train.py                shared training loop + wandb init
  utils.py                env helpers
  part1/                  MoE FFN variants (v1 dense, v2-v5 MoE) + data + eval
  part2/                  optimizers (adamw, nadamw, lion, muon) + data + eval
  part3/                  decoding (greedy, top-k, top-p, beam 1/2/4) + eval
results/part3/generations/  raw part-3 generations (gzipped JSONL)
anlp_assignment2_report.pdf report
```

## Setup

```bash
uv sync            # installs from uv.lock (python >= 3.11)
```

Create a `.env` file with your credentials (the code reads it via
`src/utils.py`):

```
HF_TOKEN=...
WANDB_API_KEY=...
```

Training/eval artifacts are written under `outputs/` / `runs/` by default
(both gitignored); `--output <dir>` overrides the root.

## Usage

### Part 1 — Mixture of Experts

```bash
uv run python main.py part1                        # all 5 FFN variants back-to-back
uv run python main.py part1 --variant 2            # single variant
uv run python main.py part1 --max-tokens 30_000_000 --batch-size 32
uv run python main.py part1 --variant 5 --eval-only   # eval the saved best checkpoint
```

### Part 2 — Optimizers

```bash
uv run python main.py part2                        # all 4 optimizers
uv run python main.py part2 --optimizer lion       # one optimizer
uv run python main.py part2 --max-tokens 4_000_000   # token budget; 0 = 1x dataset tokens
uv run python main.py part2 --optimizer adamw --eval-only --skip-bleu   # eval reruns
```

Optimizer choices: `adamw`, `nadamw`, `lion`, `muon` (one per Table-1
category; all subclass `torch.optim.Optimizer` directly, no other
`torch.optim.*` modules).

### Part 3 — Decoding Strategies

```bash
uv run python main.py part3 --output runs/part3    # 1000 ROCStories samples, 8 configs
uv run python main.py part3 --num-samples 1000 --max-new 96
uv run python main.py part3 --top-k-values 10,50 --top-p-values 0.8,0.95 --beam-widths 1,2,4
```

Part 3 evaluates greedy, top-k, top-p, and beam search (widths 1/2/4) on
`EleutherAI/pythia-160m` (pretrained, public — nothing to upload). Every
strategy is implemented from scratch on `model.forward()`; `model.generate()`
is not used.

## Checkpoints and logs

- Part 1 trained models (variants v1–v5): https://huggingface.co/Surya4/anlp2-part1
- Part 2 trained models (4 optimizers):  https://huggingface.co/Surya4/anlp2-part2
- Training runs and curves (WandB):     https://forge.coreweave.com/wandb/suryamanojphy31-iiit-hyderabad/anlp-assignment2

## Datasets

- https://huggingface.co/datasets/belumind/en-vi-ja-curated-500k-triplets (part 1)
- https://huggingface.co/datasets/browndw/human-ai-parallel-corpus (part 2)
- https://huggingface.co/datasets/hamishivi/ROCStories (part 3)

## Notes

- Part-3 raw generations (1000 samples per configuration) are committed
  gzipped: `results/part3/generations/<config>.jsonl.gz` — read with
  `zcat results/part3/generations/greedy.jsonl.gz | head`.
- Model files, wandb logs, and large run dirs (`outputs/`, `runs/`,
  `anlp-assignment-2-output/`, `wandb/`) are gitignored and not part of the
  submission.