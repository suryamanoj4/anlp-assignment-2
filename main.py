"""Assignment 2 driver: one entry point per part.

Usage:
    uv run python main.py part1                          # run ALL 5 FFN variants back-to-back
    uv run python main.py part1 --variant 2 --batch-size 32 --max-tokens 30_000_000
    uv run python main.py part1 --output runs/part1      # ALL outputs under runs/part1/{checkpoints,assets}
    uv run python main.py part1 --variant 5 --eval-only  # rerun eval from saved best ckpt
    uv run python main.py part2 --optimizer lion --max-tokens 4_000_000  # part 2: one Table-1 optimizer (calibration-sized)
    uv run python main.py part2                # all 4 optimizers, 1x dataset budget each
    uv run python main.py part3 --num-samples 1000 --max-new 96   # all 8 configs on ROCStories test (GPU)
    uv run python main.py part3 --num-samples 1000 --output runs/part3

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
from src.part2.data import make_lm_dataloaders, split_rows_by_doc
from src.part2.data import make_dataloader as make_lm_dataloader
from src.part2.evaluate import plot_part2_curves, run_eval_pass
from src.part2.optimizers import PART2_OPTIMIZERS, make_optimizer as make_p2_optimizer
from src.part3.data import load_stories
from src.part3.evaluate import make_configs, run_part3_eval
from src.part3.model import load_pythia
from src.tokenizer import load_tokenizer, train_tokenizer_from_dataset
from src.train import TrainConfig, evaluate_ppl, train_model
from src.utils import load_dotenv

DATASET_ID = "belumind/en-vi-ja-curated-500k-triplets"
VOCAB_SIZE = 32_000
PART2_DATASET_ID = "browndw/human-ai-parallel-corpus"
# English-only corpus -> English-only tokenizer: part 1's 32k vocab was
# sized for en/vi/ja; this corpus is English-only, so 16k BPE suffices —
# and it halves the tied embedding/head, the tensor where our 32k-vocab
# sustained-lr churn lived.
PART2_VOCAB_SIZE = 16_000


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
    """One optimizer per Table-1 category, trained on the human-ai corpus.

    Flow: tokenizer (once) -> doc-grouped 90/5/5 split -> per-optimizer loop:
    model (seeded identically) + make_optimizer + train_model with budget =
    1x dataset tokens (or --max-tokens) and val cadence = 0.1x budget.
    Purely additive: part 1's entry point and data path are untouched.
    """
    load_dotenv()
    tokenizer_path, _, eval_dir = _out_paths(args.output)
    print(f"[main] outputs root: {Path(args.output)}/ | tokenizer -> {tokenizer_path}")
    generator = torch.Generator().manual_seed(args.seed)  # same stream for every optimizer

    ds = None
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"[main] tokenizer loaded from {tokenizer_path}")
    else:
        ds = _load_dataset(PART2_DATASET_ID)
        print(f"[main] training BPE tokenizer (vocab {PART2_VOCAB_SIZE:,}) on the corpus; a few minutes ...")
        tokenizer = train_tokenizer_from_dataset(
            ds["train"], ["text"], PART2_VOCAB_SIZE, tokenizer_path
        )
        print(f"[main] tokenizer trained and saved to {tokenizer_path}")

    if ds is None:
        ds = _load_dataset(PART2_DATASET_ID)
    print(f"[main] rows: train split = {len(ds['train']):,} (single split; carving 90/5/5 by doc)")
    train_rows, val_rows, test_rows = split_rows_by_doc(list(ds["train"]), seed=args.seed)

    names = [args.optimizer] if args.optimizer else list(PART2_OPTIMIZERS)
    _, ckpt_dir, _ = _out_paths(args.output)
    for name in names:
        print(f"\n===== part2 | optimizer {name} | run part2-{name} =====")
        if args.eval_only:
            # rerun the eval pass from saved checkpoints (no training)
            run_eval_pass(name, tokenizer, test_rows, eval_dir, str(ckpt_dir),
                          args.device, args.max_len, args.bleu_max_new, args.batch_size)
        else:
            run_optimizer(name, args, tokenizer, (train_rows, val_rows, test_rows), generator, eval_dir)
            if args.skip_bleu:
                print(f"[main] --skip-bleu: BLEU eval pass skipped for {name} — rerun later "
                      f"with `uv run python main.py part2 --optimizer {name} --eval-only` "
                      f"(same --output)")
            else:
                run_eval_pass(name, tokenizer, test_rows, eval_dir, str(ckpt_dir),
                              args.device, args.max_len, args.bleu_max_new, args.batch_size)

    # Part-2-owned plots: all optimizers on shared axes (separate from part 1).
    plot_part2_curves(eval_dir)


def run_optimizer(
    name: str,
    args: argparse.Namespace,
    tokenizer,
    splits: tuple,
    generator: torch.Generator,
    eval_dir: Path,
) -> None:
    """Train one optimizer on the shared corpus splits; log + write metrics."""
    import json

    train_rows, val_rows, test_rows = splits

    # Tokenize train+val once; the 1x-dataset budget and 0.1x cadence come
    # from real post-truncation tokens (docs ~650 tokens > n_ctx).
    train_loader, val_loader, train_tokens = make_lm_dataloaders(
        train_rows, val_rows, tokenizer, args.max_len, args.batch_size, generator
    )
    budget = args.max_tokens if args.max_tokens > 0 else train_tokens
    val_every = max(budget // 10, 1)  # deliverable: measurements every 0.1x dataset
    warmup = max(budget // 20, 1)     # same 5% ratio as part 1 (fairness constant)

    base = TransformerConfig(n_vocab=tokenizer.vocab_size, n_ctx=args.max_len)
    config = TransformerConfig(**{**asdict(base), **ffn_variant_config(1, base.d_ff)})
    torch.manual_seed(args.seed)  # identical init across optimizers
    model = Transformer(config)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} total params (dense v1)")

    run_name = f"part2-{name}"
    _, ckpt_dir, _ = _out_paths(args.output)
    train_cfg = TrainConfig(
        max_tokens=budget,
        val_every_tokens=val_every,
        warmup_tokens=warmup,
        lr=args.lr,  # adam family; lion/muon use their calibration constants
        weight_decay=args.wd,  # adam family + muon branches; lion uses LION_WD
        betas=(0.9, args.beta2),  # part-2 runs pass 0.95 (standard LLM pair)
        amp_dtype=args.amp_dtype,  # part-2 runs pass bf16 (the paper's precision, scaler-free)
        run_name=run_name,
        ckpt_dir=str(ckpt_dir),
    )
    opt = make_p2_optimizer(name, model, train_cfg)
    # Truth in logging: wandb's config + the banner show the ACTUAL optimizer
    # lrs (lion/muon ignore cfg.lr — their old runs displayed a false 8e-4).
    train_cfg.optimizer_name = name
    train_cfg.optimizer_lrs = {
        (g.get("branch") or "all"): g["lr"] for g in opt.param_groups
    }
    steps, tokens = train_model(model, train_loader, val_loader, train_cfg,
                                device=args.device, optimizer=opt)

    # Final held-out evaluation: test-split ppl (BLEU eval lands with part 3's
    # decoder; the metrics file carries a placeholder slot for it).
    test_loader = make_lm_dataloader(
        test_rows, tokenizer, args.max_len, args.batch_size,
        shuffle=False, drop_last=False, what="test split",
    )
    test_ppl = evaluate_ppl(model, test_loader, args.device, what="test split")
    n_ckpts = len(list(Path(ckpt_dir).glob(f"{run_name}_*.pt")))
    metrics = {
        "part": 2, "optimizer": name, "run_name": run_name,
        "budget_tokens": budget, "val_every_tokens": val_every,
        "trained_steps": steps, "trained_tokens": tokens,
        "val_ppl_points": n_ckpts, "test_ppl": test_ppl,
        "test_bleu": None,  # TODO: continuation-BLEU eval (needs the part 3 decoder)
        # base (initial) lrs, not the end-of-run decayed values
        "lrs": {(g.get("branch") or "all"): g.get("initial_lr", g["lr"])
                for g in opt.param_groups},
        "weight_decay": {(g.get("branch") or "all"): g["weight_decay"]
                         for g in opt.param_groups},
        "seed": args.seed,
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[main] {name} done: {steps} steps, {tokens:,} tokens, test ppl {test_ppl:.3f} "
          f"| metrics -> {eval_dir / f'{run_name}_metrics.json'}")


def run_part3(args: argparse.Namespace) -> None:
    """Decoding strategies on hamishivi/ROCStories with EleutherAI/pythia-160m.

    No training: load the pretrained checkpoint, then evaluate every
    strategy config (greedy, top-k, top-p, beam 1/2/4) on the same seeded
    subsample of the test split: self-perplexity, first-token accuracy,
    wall-clock timing, and raw generations (JSONL per config). Gold-ppl is
    computed once as the strategy-independent baseline.
    """
    dtype = torch.float32 if args.fp32 else torch.float16
    print(f"[main] loading EleutherAI/pythia-160m on {args.device} ({dtype}) ...")
    model, tokenizer = load_pythia(device=args.device, dtype=dtype)

    stories = load_stories(split="test")  # subsampled by seed inside the eval
    configs = make_configs(
        k_values=tuple(int(x) for x in args.top_k_values.split(",") if x),
        p_values=tuple(float(x) for x in args.top_p_values.split(",") if x),
        beam_widths=tuple(int(x) for x in args.beam_widths.split(",") if x),
    )
    _, _, eval_dir = _out_paths(args.output)
    out_dir = Path(eval_dir) / "part3"
    run_part3_eval(
        model, tokenizer, stories, out_dir,
        num_samples=args.num_samples,
        max_new=args.max_new,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        configs=configs,
    )
    print(f"[main] part 3 metrics -> {out_dir}/part3_metrics.json")
    print(f"[main] part 3 raw generations -> {out_dir}/generations/")


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

    p2 = sub.add_parser(
        "part2",
        help="optimizers: one per Table-1 category (AdamW/NadamW/Lion/Muon) on browndw/human-ai-parallel-corpus",
    )
    p2.add_argument("--optimizer", choices=sorted(PART2_OPTIMIZERS),
                    help="single optimizer to run; omit to run all 4 back-to-back")
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--max-tokens", type=int, default=0,
                    help="token budget; 0 = 1x dataset tokens (default). Use e.g. 4_000_000 for a 0.1x calibration run")
    p2.add_argument("--max-len", type=int, default=512)
    p2.add_argument("--lr", type=float, default=8e-4,
                    help="adam-family lr (lion/muon use their calibration constants in src/part2/optimizers.py)")
    p2.add_argument("--wd", type=float, default=0.01,
                    help="decoupled weight decay (adam family + muon branches; lion uses its own LION_WD); "
                         "default 0.01 = part 1; real part-2 runs pass 0.1 (the paper's tuned AdamW value)")
    p2.add_argument("--beta2", type=float, default=0.98,
                    help="second beta for the adam family + muon's adam branch (lion uses LION_BETAS); "
                         "default 0.98 = part 1; real part-2 runs pass 0.95 — the standard LLM-pretraining "
                         "pair (0.9, 0.95), faster v adaptation for our batch size")
    p2.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16",
                    help="autocast dtype (default fp16 = part 1); real part-2 runs pass bf16 — the paper's own "
                         "precision (fp32 params + bf16 activations) and scaler-free, unlike fp16")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--eval-only", action="store_true",
                    help="skip training; rerun the BLEU eval pass from saved checkpoints")
    p2.add_argument("--skip-bleu", action="store_true",
                    help="train only; skip the per-checkpoint BLEU eval pass (rerun later with --eval-only)")
    p2.add_argument("--bleu-max-new", type=int, default=128,
                    help="tokens generated per test doc for continuation BLEU (reference truncated to the same budget)")
    p2.add_argument("--output", default="outputs",
                    help="root dir for ALL outputs (checkpoints/, assets/tokenizer.json, assets/eval/ live under it); default: outputs")
    p2.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p2.set_defaults(func=run_part2)

    p3 = sub.add_parser(
        "part3",
        help="decoding strategies (greedy/top-k/top-p/beam 1-2-4) on ROCStories with Pythia-160M",
    )
    p3.add_argument("--num-samples", type=int, default=1000,
                    help="stories to evaluate on the test split (assignment minimum: 1000); "
                         "subsampled deterministically by --seed")
    p3.add_argument("--max-new", type=int, default=96,
                    help="token generation budget (gold continuations are ~60-80 tokens)")
    p3.add_argument("--batch-size", type=int, default=32)
    p3.add_argument("--seed", type=int, default=42,
                    help="subsample RNG + per-config sampling seed (reproducible runs)")
    p3.add_argument("--top-k-values", default="10,50",
                    help="comma list of top-k sizes to evaluate")
    p3.add_argument("--top-p-values", default="0.8,0.95",
                    help="comma list of top-p masses to evaluate")
    p3.add_argument("--beam-widths", default="1,2,4",
                    help="comma list of beam widths (deliverable: 1, 2 and 4)")
    p3.add_argument("--fp32", action="store_true",
                    help="fp32 inference (CPU smoke tests); default fp16")
    p3.add_argument("--output", default="outputs",
                    help="root dir for ALL outputs; part 3 writes assets/eval/part3/ under it")
    p3.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p3.set_defaults(func=run_part3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()