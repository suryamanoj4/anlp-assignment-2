"""Assignment 2 driver: one entry point per part.

Usage:
    uv run python main.py part1                          # run ALL 5 FFN variants back-to-back
    uv run python main.py part1 --variant 2 --batch-size 32 --max-tokens 30_000_000
    uv run python main.py part1 --output runs/part1      # ALL outputs under runs/part1/{checkpoints,assets}
    uv run python main.py part1 --variant 5 --eval-only  # rerun eval from saved best ckpt
    uv run python main.py part2 ...   (optimizers, coming soon)
    uv run python main.py part3 ...   (decoding strategies, coming soon)

Part 1: all variants share the same shuffle stream (fixed seed) and the same
token budget, so differences are attributable to the FFN variant alone.

Output layout (default root: outputs/):
    outputs/checkpoints/       model checkpoints (part1-v{N}_{best,tok...,final}.pt)
    outputs/assets/tokenizer.json   trained BPE tokenizer
    outputs/assets/eval/       metrics json, generations txt, heatmap png
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset

from src.part1.data import make_dataloader
from src.part1.evaluate import run_eval
from src.part1.model import Transformer, TransformerConfig, ffn_variant_config
from src.tokenizer import load_tokenizer, train_tokenizer_from_dataset
from src.train import TrainConfig, train_model
from src.utils import load_dotenv

DATASET_ID = "belumind/en-vi-ja-curated-500k-triplets"
VOCAB_SIZE = 32_000


def _out_paths(output: str) -> tuple[Path, Path, Path]:
    """Resolve the output root -> (tokenizer file, checkpoints dir, eval dir)."""
    root = Path(output)
    return root / "assets" / "tokenizer.json", root / "checkpoints", root / "assets" / "eval"


def _load_dataset(dataset_id: str):
    """Load the dataset, printing cache-vs-download status and split sizes."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    owner, name = dataset_id.split("/", 1)
    cached = (
        cache_root.is_dir()
        and any(p.name.startswith(f"{owner}___{name}") for p in cache_root.iterdir())
    )
    print(
        f"[main] loading dataset '{dataset_id}' "
        f"({'HF cache hit' if cached else 'not cached -> downloading (~98MB), progress bars below'}) ..."
    )
    ds = load_dataset(dataset_id)
    print(f"[main] dataset ready: " + ", ".join(f"{k}={len(v):,}" for k, v in ds.items()))
    return ds


def run_part1(args: argparse.Namespace) -> None:
    load_dotenv()
    tokenizer_path, _, _ = _out_paths(args.output)
    print(f"[main] outputs root: {Path(args.output)}/ | tokenizer -> {tokenizer_path}")
    generator = torch.Generator().manual_seed(args.seed)  # same stream for every variant

    # 1) Tokenizer: train once (cached on disk afterwards).
    ds = None
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"tokenizer loaded from {tokenizer_path}")
    else:
        ds = _load_dataset(DATASET_ID)
        print(f"[main] training BPE tokenizer (vocab {VOCAB_SIZE:,}) on the train split; takes a few minutes ...")
        tokenizer = train_tokenizer_from_dataset(
            ds["train"], ["en", "vi", "ja"], VOCAB_SIZE, tokenizer_path
        )
        print(f"tokenizer trained and saved to {tokenizer_path}")

    if ds is None:
        ds = _load_dataset(DATASET_ID)
    variants = [args.variant] if args.variant is not None else list(range(1, 6))
    for variant in variants:
        print(f"\n===== part1 | variant {variant} | run {run_name_of(variant)} =====")
        run_variant(variant, args, tokenizer, ds, generator)


def run_name_of(variant: int) -> str:
    return f"part1-v{variant}"


def run_variant(variant: int, args: argparse.Namespace, tokenizer, ds, generator) -> None:
    # 2) Model config: base transformer + variant FFN knobs (param-matched).
    base = TransformerConfig(
        n_vocab=tokenizer.vocab_size,
        n_ctx=args.max_len,
    )
    config = TransformerConfig(**{**asdict(base), **ffn_variant_config(variant, base.d_ff)})
    model = Transformer(config)
    print(f"v{variant}: {sum(p.numel() for p in model.parameters()):,} total params")

    run_name = run_name_of(variant)
    _, ckpt_dir, eval_dir = _out_paths(args.output)
    train_cfg = TrainConfig(
        max_tokens=args.max_tokens,
        run_name=run_name,
        ckpt_dir=str(ckpt_dir),  # outputs/<root>/checkpoints
    )

    if not args.eval_only:
        train_loader = make_dataloader(
            ds["train"], tokenizer, args.max_len, args.batch_size,
            shuffle=True, drop_last=True, generator=generator, what="train split",
        )
        val_loader = make_dataloader(
            ds["validation"], tokenizer, args.max_len, args.batch_size,
            shuffle=False, drop_last=False, what="validation split",
        )
        train_model(model, train_loader, val_loader, train_cfg, device=args.device)

    # 3) Evaluate the best-val checkpoint on the held-out test split.
    ckpt_path = f"{train_cfg.ckpt_dir}/{run_name}_best.pt"
    if not Path(ckpt_path).exists():
        raise SystemExit(f"{ckpt_path} not found; run without --eval-only first")
    run_eval(ckpt_path, ds["test"], tokenizer, device=args.device, out_dir=str(eval_dir))


def run_part2(args: argparse.Namespace) -> None:
    raise NotImplementedError("part 2 (custom optimizers) lands here once implemented")


def run_part3(args: argparse.Namespace) -> None:
    raise NotImplementedError("part 3 (decoding strategies) lands here once implemented")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced NLP Assignment 2 driver")
    sub = parser.add_subparsers(dest="part", required=True)

    p1 = sub.add_parser("part1", help="Mixture-of-Experts FFN variants: tokenize -> train -> eval")
    p1.add_argument("--variant", type=int, choices=range(1, 6),
                    help="single variant to run; omit to run all 5 back-to-back")
    p1.add_argument("--batch-size", type=int, default=32)
    p1.add_argument("--max-tokens", type=int, default=30_000_000)
    p1.add_argument("--max-len", type=int, default=512)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--output", default="outputs",
                    help="root dir for ALL outputs (checkpoints/, assets/tokenizer.json, assets/eval/ live under it); default: outputs")
    p1.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p1.add_argument("--eval-only", action="store_true", help="skip training; eval best ckpt")
    p1.set_defaults(func=run_part1)

    p2 = sub.add_parser("part2", help="custom optimizers (not implemented yet)")
    p2.set_defaults(func=run_part2)

    p3 = sub.add_parser("part3", help="decoding strategies (not implemented yet)")
    p3.set_defaults(func=run_part3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()