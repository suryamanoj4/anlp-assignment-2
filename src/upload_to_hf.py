"""Upload trained checkpoints to the HuggingFace Hub — one repo per assignment part.

Layout (repo per part, models as subfolders, tokenizer shared at the root):

    anlp2-part1/
      README.md  tokenizer.json  tokenizer_config.json  special_tokens_map.json
      v1-dense/        model.pt  config.json  eval_result.json  checkpoint_meta.json
      v2-top1-quarter/ ...
      ...
    anlp2-part2/
      README.md  tokenizer.json  tokenizer_config.json  special_tokens_map.json
      adamw/  model.pt  config.json  eval_result.json  checkpoint_meta.json
      nadamw/ muon/ lion/ ...

Each `model.pt` is an untouched copy of the original `*_best.pt` (self-contained:
state dict + config + tokens/step/val-ppl). The README model card carries the
dataset (+license), the training recipe, the evaluation protocol, and the full
results matrix for every model in the repo.

Reads HF_TOKEN from .env (via src.utils.load_dotenv). Namespace defaults to the
token's owner (whoami); override with --namespace.

Usage:
    uv run python -m src.upload_to_hf --part 1          # anlp2-part1 (v1..v5)
    uv run python -m src.upload_to_hf --part 2          # anlp2-part2 (4 optimizers)
    uv run python -m src.upload_to_hf --part 1 --dry-run
"""

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi

from src.utils import load_dotenv

ROOT_DEFAULT = "anlp-assignment-2-output"

# ---------------------------------------------------------------------------
# Per-model facts
# ---------------------------------------------------------------------------

PART1_VARIANTS = {
    "v1": {
        "slug": "v1-dense",
        "label": "dense MLP (d_ff=1536)",
        "structure": "dense MLP FFN, full width",
        "vi_ja_bleu": ("25.07", "17.40"),
        "empty_preds": "80 / 600 (13.3%)",
    },
    "v2": {
        "slug": "v2-top1-quarter",
        "label": "top-1 routing (¼ active FFN)",
        "structure": "4 experts × d_ff 384, top-1 per token; active FFN = ¼ dense",
        "vi_ja_bleu": ("25.10", "16.76"),
        "empty_preds": "2 / 600 (0.3%)",
    },
    "v3": {
        "slug": "v3-top2-half",
        "label": "top-2 routing (½ active FFN)",
        "structure": "4 experts × d_ff 384, top-2 per token; active FFN = ½ dense",
        "vi_ja_bleu": ("27.75", "19.01"),
        "empty_preds": "3 / 600 (0.5%)",
    },
    "v4": {
        "slug": "v4-shared-top1-half",
        "label": "shared + top-1 routing (½ active FFN)",
        "structure": "3 routed experts × d_ff 384 + 1 shared expert (always active), top-1 routing",
        "vi_ja_bleu": ("27.06", "19.75"),
        "empty_preds": "5 / 600 (0.8%)",
    },
    "v5": {
        "slug": "v5-top2-wide",
        "label": "top-2 wide experts (active ≈ dense)",
        "structure": "4 experts × d_ff 768, top-2 per token; active FFN ≈ dense, 2× stored FFN",
        "vi_ja_bleu": ("27.40", "16.63"),
        "empty_preds": "1 / 600 (0.2%)",
    },
}

PART2_OPTIMIZERS = {
    "adamw": {
        "label": "AdamW (baseline)",
        "recipe_line": "lr 5e-4, wd 0.1, betas (0.9, 0.95), state 2.00×",
        "note": "The Table-1 baseline: decoupled weight decay + bias-corrected first/second moments.",
    },
    "nadamw": {
        "label": "NadamW (variance-reduced AdamW)",
        "recipe_line": "lr 5e-4, wd 0.1, betas (0.9, 0.95), state 2.00×",
        "note": "Variance-reduced variant of AdamW from the assignment's reference paper.",
    },
    "muon": {
        "label": "Muon (matrix-based)",
        "recipe_line": "NS lr 0.03 (momentum 0.95) + AdamW branch lr 3e-4, wd 0.1, betas (0.9, 0.95)",
        "note": "Matrix-based: Newton–Schulz orthogonalization for 2-D matrices with an AdamW branch for vector parameters (assignment's reference paper); state ≈1.31×.",
    },
    "lion": {
        "label": "Lion (memory-efficient, sign updates)",
        "recipe_line": "lr 1.5e-4, wd 0.6, betas (0.9, 0.95), state 1.00×",
        "note": "Sign-of-interpolation updates with weight decay folded in — single momentum buffer (1.00× state), per the Lion paper.",
    },
}

# ---------------------------------------------------------------------------
# Per-part card knowledge
# ---------------------------------------------------------------------------

PART_META = {
    1: {
        "task": "vi→en and ja→en translation, decoder-only, next-token with task masking (2 samples per data row)",
        "dataset": {
            "id": "belumind/en-vi-ja-curated-500k-triplets",
            "license": "CC-BY-4.0",
            "note": ("EN-VI-JA translation triplets curated from OPUS parallel corpora "
                     "(train 446,252 / val 24,792 / test 24,792 rows)."),
        },
        "tokenizer_note": "byte-level BPE, vocab 32,000, special tokens: <pad> <bos> <eos> <unk>",
        "recipe": [
            "Optimizer: AdamW, lr 8e-4, betas (0.9, 0.98), weight decay 0.01, grad clip 1.0",
            "Precision: fp16 AMP (fp32 master params)",
            "Schedule: warmup 1.5M tokens + cosine decay to 10% (final lr 8.94e-5)",
            "Seed 42; identical shuffle stream, budget, and steps (23,382; ~30M tokens) for all five variants",
            "Val every 3M tokens (10 checkpoints); best checkpoint = final (monotone curves)",
        ],
        "eval_protocol": [
            "600 greedy translations (300 test rows × 2 languages) via a hand-rolled decoder (no model.generate)",
            "BLEU: sacrebleu corpus BLEU, empty predictions included",
            "test/ppl over the test split; val/ppl over the val split",
        ],
        "usage": (
            "Custom decoder-only architecture (course repository `src/part1/model.py`, "
            "`TransformerConfig` + `ffn_variant`): `ckpt = torch.load('<variant-folder>/model.pt', "
            "weights_only=False)` and rebuild the model with `ckpt['config']`. `tokenizer.json` "
            "is a `tokenizers` BPE (`Tokenizer.from_file` or "
            "`PreTrainedTokenizerFast(tokenizer_file=...)`)."
        ),
        "limitations": [
            "Small-scale: ~39M (v1–v4) / 48M (v5) params, ~30M tokens (~0.1× Chinchilla regime)",
            "v1 (dense) shows an empty-prediction degeneracy (13.3% of greedy translations are empty)",
            "Per-variant ppl-vs-BLEU trade-offs documented in the report",
        ],
        "repo_name": "anlp2-part1",
        "provenance": (
            "Part 1 of Advanced NLP Assignment 2. Training runs {runs} logged in the WandB "
            "project `suryamanojphy31-iiit-hyderabad/anlp-assignment2`."
        ),
    },
    2: {
        "task": "English next-token prediction / text continuation (language modeling)",
        "dataset": {
            "id": "browndw/human-ai-parallel-corpus",
            "license": "MIT",
            "note": ("HAP-E: human-authored English text (seed ~500-word chunks and their true "
                     "continuations). Used for language modeling: doc-grouped 90/5/5 splits by "
                     "document root; English side tokenized with a 16k byte-level BPE."),
        },
        "tokenizer_note": "byte-level BPE, vocab 16,000 (English-only), trained on the train split",
        "recipe": [
            "One epoch = 1× the dataset (29,511,235 tokens; 29,523,280 seen incl. spill batch)",
            "Steps: 1,866 (batch 32 × 512 tokens)",
            "Schedule: per-group warmup 5% + cosine decay to 10% lr; seed 42",
            "Precision: bf16 autocast (fp32 master params), grad clip 1.0",
            "Identical stream/budget/steps/schedule across all four optimizer runs",
        ],
        "eval_protocol": [
            "val/ppl: val split, every 2,951,123 tokens (10 checkpoints + final)",
            "test BLEU: 415 human chunk1→chunk2 pairs from the TEST documents, 128-token greedy continuations, single human reference",
            "Full per-checkpoint curves (tokens vs val ppl vs BLEU) in each model folder's `eval_result.json`",
        ],
        "usage": (
            "Custom decoder-only architecture (course repository `src/part1/model.py`, dense "
            "`ffn_variant=1`, `n_vocab=16000`): `ckpt = torch.load('<optimizer>/model.pt', weights_only=False)` "
            "and rebuild the model with `ckpt['config']`. `tokenizer.json` is a `tokenizers` BPE."
        ),
        "limitations": [
            "Tiny model (~20.5M params) at ~0.1× Chinchilla tokens — single pass, no repetition",
            "Continuation BLEU is a noisy, single-reference signal at this scale (~0.5–1.1); val/ppl is the primary metric",
            "ppl↔BLEU dissociate at this scale (best ppl does not imply best BLEU)",
        ],
        "repo_name": "anlp2-part2",
        "provenance": (
            "Part 2 of Advanced NLP Assignment 2: four optimizers from the assignment's "
            "reference paper (Appendix A), each sub-classing `torch.optim.Optimizer` "
            "directly. Training runs {runs} in the WandB project "
            "`suryamanojphy31-iiit-hyderabad/anlp-assignment2`."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_params(sd) -> int:
    """Parameter total, counting tied tensors (tok_emb/lm_head share storage)
    exactly once."""
    unique = {t.data_ptr(): t for t in sd.values()}
    return sum(t.numel() for t in unique.values())


def _tokenizer_files(tok_path: Path, n_ctx: int, out_dir: Path) -> None:
    """Copy tokenizer.json + write tokenizer_config.json & special_tokens_map.json."""
    shutil.copy(tok_path, out_dir / "tokenizer.json")
    raw = json.loads(tok_path.read_text())
    added = raw.get("added_tokens", [])
    specials = [t["content"] for t in added if t.get("special", False)]
    if not specials:  # fall back to all added tokens if no flags were stored
        specials = [t["content"] for t in added]
    roles = {}
    for t in sorted(specials):
        if "pad" in t:
            roles["pad_token"] = t
        elif "bos" in t:
            roles["bos_token"] = t
        elif "eos" in t:
            roles["eos_token"] = t
        elif "unk" in t:
            roles["unk_token"] = t
    (out_dir / "special_tokens_map.json").write_text(json.dumps(roles, indent=2) + "\n")
    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": n_ctx,
        "padding_side": "right",
        "truncation_side": "right",
    }
    tok_cfg.update(roles)
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2) + "\n")


def _model_inputs(part: int, name: str, root: Path):
    if part == 1:
        ckpt = root / "runs/part1/checkpoints" / f"part1-{name}_best.pt"
        tok = root / "runs/part1/assets/tokenizer.json"
        ev = root / "runs/part1/assets/eval" / f"metrics_part1-{name}_best.json"
    else:
        ckpt = root / "runs/part2/checkpoints" / f"part2-{name}_best.pt"
        tok = root / "runs/part2/assets/tokenizer.json"
        ev = root / "runs/part2/assets/eval" / f"part2-{name}_eval.json"
    for p in (ckpt, tok, ev):
        if not p.exists():
            raise SystemExit(f"missing file: {p}")
    return ckpt, tok, ev


def _collect_summaries(part: int, root: Path) -> list[dict]:
    """Load every model of the part; returns the facts + numbers used by the card."""
    meta = PART_META[part]
    names = sorted(PART1_VARIANTS if part == 1 else PART2_OPTIMIZERS)
    out = []
    for name in names:
        facts = dict(PART1_VARIANTS.get(name) or PART2_OPTIMIZERS[name])
        ckpt, _, ev = _model_inputs(part, name, root)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        config = ck["config"]
        if is_dataclass(config):
            config = asdict(config)
        facts.update(
            name=name,
            subdir=name if part == 2 else facts["slug"],
            tokens_seen=ck.get("tokens_seen"),
            step=ck.get("step"),
            ckpt_val_ppl=ck.get("val_ppl"),
            total_params=_unique_params(ck["model"]),
            config=config,
        )
        if part == 1:
            m = json.loads(ev.read_text())
            facts.update(
                variant=name,
                test_ppl=m["test_ppl"],
                bleu=m["bleu"],
                n_generations=m.get("n_generations", 600),
                eval=m,
            )
        else:
            evj = json.loads(ev.read_text())
            mj = json.loads(
                (root / "runs/part2/assets/eval" / f"part2-{name}_metrics.json").read_text()
            )
            pts = evj["points"]
            facts.update(
                final_ppl=pts[-1]["val_ppl"],
                final_bleu=pts[-1]["bleu"],
                best_bleu=max(p["bleu"] for p in pts),
                test_ppl=mj.get("test_ppl"),
                points=pts,
                eval=evj,
            )
        out.append(facts)
    return out


# ---------------------------------------------------------------------------
# Card assembly
# ---------------------------------------------------------------------------


def _front_matter(extra_tags: list[str]) -> list[str]:
    t = ["---", "tags:"]
    for tag in ["text-generation", "pytorch", "custom-architecture"] + extra_tags:
        t.append(f"  - {tag}")
    t.append("---")
    return t


def build_part1_card(repo_id: str, models: list[dict], meta: dict) -> str:
    t = _front_matter(["moe", "translation"])
    t += [
        "",
        f"# ANLP Assignment 2 · Part 1 — MoE FFN variants (v1–v5)",
        "",
        "Decoder-only transformer trained for **vi→en and ja→en translation** across five "
        "parameter-matched feed-forward variants: one dense MLP (v1) and four MoE variants "
        "(v2–v5) — top-1, top-2, shared+top-1, and top-2-wide expert layouts. All five share "
        "the same embeddings/attention, data stream, budget, and schedule; differences are "
        "attributable to the FFN variant alone.",
        "",
        "## Models in this repo",
        "",
        "| variant | folder | structure | params | val ppl | test ppl | BLEU | vi→en | ja→en | empty preds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        vi, ja = m["vi_ja_bleu"]
        t.append(
            f"| v{m['name'][1:]} | `{m['subdir']}/model.pt` | {m['structure']} | "
            f"{m['total_params']/1e6:.1f}M | {m['ckpt_val_ppl']:.2f} | {m['test_ppl']:.2f} | "
            f"**{m['bleu']:.2f}** | {vi} | {ja} | {m['empty_preds']} |"
        )
    m0 = models[0]["config"]
    t += [
        "",
        "Shared architecture: d_model 384, 8 layers, 6 heads, d_ff 1536 (dense reference), "
        "tied embeddings, learned positional embeddings, dropout 0, context 512. MoE variants: "
        f"experts of d_ff {m0.get('expert_d_ff')}, routed per token by a learned gate "
        "(n_active = 1 or 2); v4 adds 1 always-active shared expert.",
        "",
        "## Dataset",
        "",
        f"- **{meta['dataset']['id']}** (license: {meta['dataset']['license']}) — {meta['dataset']['note']}",
        f"- {meta['tokenizer_note']}",
        "",
        "## Training details",
        "",
    ]
    t += [f"- {b}" for b in meta["recipe"]]
    t += [
        "",
        "## Evaluation",
        "",
        "Protocol:",
    ]
    t += [f"- {b}" for b in meta["eval_protocol"]]
    t += [
        "",
        "BLEU = sacrebleu corpus BLEU over 600 greedy translations (300 test rows × 2 langs), "
        "empty predictions included; val/ppl at the best (== final) checkpoint. Results in the "
        "table above and, per model, in `eval_result.json`.",
        "",
        "## Files",
        "",
        "Root: `tokenizer.json` (+ configuration) shared by all variants. Per model: "
        "`<folder>/model.pt` (= copy of `part1-vN_best.pt`, self-contained: state dict + "
        "config + tokens/step/val-ppl), `config.json`, `eval_result.json`, "
        "`checkpoint_meta.json`.",
        "",
        "## Usage",
        "",
        meta["usage"],
        "",
        "## Limitations",
        "",
    ]
    t += [f"- {b}" for b in meta["limitations"]]
    t += [
        "",
        "## Provenance",
        "",
        meta["provenance"].format(runs=", ".join(f"`part1-{m['name']}`" for m in models)),
        "",
    ]
    return "\n".join(t)


def build_part2_card(repo_id: str, models: list[dict], meta: dict) -> str:
    t = _front_matter(["optimizer-benchmark", "language-modeling"])
    t += [
        "",
        f"# ANLP Assignment 2 · Part 2 — four optimizers on one epoch of HAP-E",
        "",
        "Decoder-only transformer (dense MLP FFN, ~20.5M params) trained for **English "
        "next-token prediction / text continuation** on the human-authored side of the "
        "HAP-E corpus, for exactly one epoch (1× the dataset = 29.5M tokens) — once with "
        "each of four optimizers from the assignment's reference paper (Appendix A): "
        "AdamW (baseline), NadamW (variance-reduced), Muon (matrix-based), Lion "
        "(memory-efficient). Identical stream, budget, steps, and schedule across runs; "
        "differences are attributable to the update rule + its tuned hyperparameters.",
        "",
        "## Models in this repo",
        "",
        "| run | folder | optimizer | lr / wd / betas | state | final val ppl | test ppl | final BLEU | best BLEU |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        t.append(
            f"| `part2-{m['name']}` | `{m['subdir']}/model.pt` | {m['label']} | "
            f"{m['recipe_line']} | {m['final_ppl']:.2f} | "
            f"{m['test_ppl']:.2f} | {m['final_bleu']:.3f} | {m['best_bleu']:.3f} |"
        )
    t += ["", "Per-optimizer notes:"]
    for m in models:
        t.append(f"- **{m['label']}** — {m['note']}")
    t += [
        "",
        "Shared architecture: d_model 384, 8 layers, 6 heads, d_ff 1536, tied embeddings, "
        "learned positional embeddings, dropout 0, context 512, vocab 16,000.",
        "",
        "## Full curves (tokens vs val ppl / test BLEU, 10 checkpoints + final)",
        "",
    ]
    # token grid is identical across runs (same stream)
    grid = [p["tokens"] for p in models[0]["points"]]
    grid = list(dict.fromkeys(grid))
    t.append("| tokens (×1e6) | AdamW | NadamW | Muon | Lion |")
    t.append("|---|---|---|---|---|")
    for tok in grid:
        row = [f"{tok/1e6:.2f}"]
        for m in models:
            row.append(f"{next(p['val_ppl'] for p in m['points'] if p['tokens'] == tok):.1f}")
        t.append("| " + " | ".join(row) + " |")
    t.append("")
    t.append("| tokens (×1e6) | AdamW | NadamW | Muon | Lion |")
    t.append("|---|---|---|---|---|")
    for tok in grid:
        row = [f"{tok/1e6:.2f}"]
        for m in models:
            row.append(f"{next(p['bleu'] for p in m['points'] if p['tokens'] == tok):.3f}")
        t.append("| " + " | ".join(row) + " |")
    t += [
        "",
        "## Dataset",
        "",
        f"- **{meta['dataset']['id']}** (license: {meta['dataset']['license']}) — {meta['dataset']['note']}",
        f"- {meta['tokenizer_note']}",
        "",
        "## Training details",
        "",
    ]
    t += [f"- {b}" for b in meta["recipe"]]
    t += [
        "| run | tokens seen | steps | checkpoint val ppl |",
        "|---|---|---|---|",
    ]
    for m in models:
        t.append(f"| `part2-{m['name']}` | {m['tokens_seen']:,} | {m['step']} | {m['ckpt_val_ppl']:.2f} |")
    t += [
        "",
        "## Evaluation",
        "",
        "Protocol:",
    ]
    t += [f"- {b}" for b in meta["eval_protocol"]]
    t += [
        "",
        "BLEU uses a single human reference (test split, 415 chunk1→chunk2 pairs, 128-token "
        "greedy continuations); val/ppl comes from the val split. Full 11-point data per run "
        "in each folder's `eval_result.json`.",
        "",
        "## Files",
        "",
        "Root: `tokenizer.json` (+ configuration) shared by all runs. Per run: "
        "`<optimizer>/model.pt` (= copy of `part2-<name>_best.pt`, self-contained), "
        "`config.json`, `eval_result.json`, `checkpoint_meta.json`.",
        "",
        "## Usage",
        "",
        meta["usage"],
        "",
        "## Limitations",
        "",
    ]
    t += [f"- {b}" for b in meta["limitations"]]
    t += [
        "",
        "## Provenance",
        "",
        meta["provenance"].format(runs=", ".join(f"`part2-{m['name']}`" for m in models)),
        "",
    ]
    return "\n".join(t)


# ---------------------------------------------------------------------------
# Build + upload
# ---------------------------------------------------------------------------


def build_repo(part: int, root: Path, dry: bool) -> tuple[Path, str]:
    meta = PART_META[part]
    repo_id = meta["repo_name"]
    models = _collect_summaries(part, root)
    out = (root / ".hf_dry" / repo_id if dry
           else Path(tempfile.mkdtemp(prefix=f"hf_{repo_id}_")))
    out.mkdir(parents=True, exist_ok=True)

    # shared tokenizer at the repo root
    _, tok, _ = _model_inputs(part, models[0]["name"], root)
    _tokenizer_files(tok, models[0]["config"].get("n_ctx", 512), out)

    for m in models:
        folder = out / m["subdir"]
        folder.mkdir(parents=True, exist_ok=True)
        ckpt, _, _ = _model_inputs(part, m["name"], root)
        shutil.copy(ckpt, folder / "model.pt")
        (folder / "config.json").write_text(json.dumps(m["config"], indent=2) + "\n")
        (folder / "eval_result.json").write_text(json.dumps(m["eval"], indent=2) + "\n")
        (folder / "checkpoint_meta.json").write_text(
            json.dumps(
                {
                    "part": part,
                    "name": m["name"],
                    "source_file": ckpt.name,
                    "tokens_seen": m["tokens_seen"],
                    "step": m["step"],
                    "val_ppl": m["ckpt_val_ppl"],
                    "total_params": m["total_params"],
                },
                indent=2,
            )
            + "\n"
        )

    readme = build_part1_card(repo_id, models, meta) if part == 1 \
        else build_part2_card(repo_id, models, meta)
    (out / "README.md").write_text(readme)
    return out, repo_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", type=int, choices=(1, 2), required=True,
                    help="1 = v1..v5 variants, 2 = four optimizers")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--namespace", default=None, help="HF namespace (default: token owner)")
    ap.add_argument("--private", action="store_true", help="create private repos")
    ap.add_argument("--dry-run", action="store_true", help="build repos locally, no upload")
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env (create a token on huggingface.co)")

    root = Path(args.root)
    api = HfApi(token=token) if not args.dry_run else None
    namespace = args.namespace
    if not args.dry_run and namespace is None:
        namespace = api.whoami()["name"]
        print(f"[upload] namespace: {namespace}")

    folder, repo_id = build_repo(args.part, root, dry=args.dry_run)
    if args.dry_run:
        print(f"\n===== DRY-RUN {repo_id} -> {folder} =====")
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(folder):34s} {f.stat().st_size/1e6:6.1f} MB")
        print(f"{folder}/README.md")
        return

    full = f"{namespace}/{repo_id}"
    print(f"[upload] {full}: creating repo + uploading folder ...")
    try:
        api.create_repo(repo_id=full, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(folder), repo_id=full, token=token)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(f"[upload] done -> https://huggingface.co/{full}")


if __name__ == "__main__":
    main()