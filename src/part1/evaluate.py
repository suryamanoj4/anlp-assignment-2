"""Post-training evaluation for part 1: test perplexity, BLEU, expert-usage heatmap.

All three run in inference mode from a saved checkpoint; no training involved.
Usage: called by main.py after training (or via --eval-only with a saved ckpt).
"""

import functools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import torch
from torch.utils.data import DataLoader, Dataset

from src.part1.data import TranslationDataset, collate_batch
from src.part1.model import Transformer
from src.train import evaluate_ppl
from src.utils import load_dotenv


def load_checkpoint_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Transformer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


class LanguageSubset(Dataset):
    """Filter a TranslationDataset to one source language (for the heatmap pass)."""

    def __init__(self, dataset: TranslationDataset, lang: str):
        self.dataset = dataset
        self.idx = [i for i in range(len(dataset)) if dataset[i]["language"] == lang]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return self.dataset[self.idx[i]]


def _make_loader(dataset: Dataset, tokenizer, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(collate_batch, pad_id=tokenizer.pad_token_id),
    )


@torch.no_grad()
def greedy_translate(model, tokenizer, src_text: str, device: str, max_new: int = 128) -> str:
    """Greedy decode: bos + src + eos, then argmax steps until eos / max_new."""
    prompt = torch.tensor(
        [tokenizer.bos_token_id]
        + tokenizer(src_text, add_special_tokens=False)["input_ids"]
        + [tokenizer.eos_token_id],
        device=device,
    ).unsqueeze(0)
    gen = []
    for _ in range(max_new):
        tok = int(model(prompt)[:, -1, :].argmax(dim=-1).item())
        if tok == tokenizer.eos_token_id:
            break
        gen.append(tok)
        prompt = torch.cat([prompt, torch.tensor([[tok]], device=device)], dim=1)
    return tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def collect_usage(model, loader, language: str, device: str):
    """One language pass: reset counters, run inference, bucket counts under `language`."""
    model.reset_usage()
    for batch in loader:
        model(
            batch.input_ids.to(device),
            batch.attention_mask.to(device),
        )
    model.record_usage(language)


def plot_usage_heatmap(block_usage, languages, save_path: str) -> None:
    """block_usage: dict layer_idx -> dict language -> counts tensor; sum over layers."""
    n_experts = next(iter(next(iter(block_usage.values())).values())).numel()
    arr = np.zeros((len(languages), n_experts))
    for layer, per_lang in block_usage.items():
        for r, lang in enumerate(languages):
            arr[r] += per_lang[lang].cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(3, n_experts), 3))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(n_experts), [f"E{e}" for e in range(n_experts)])
    ax.set_yticks(range(len(languages)), languages)
    ax.set_xlabel("expert")
    for r in range(len(languages)):
        for c in range(n_experts):
            ax.text(c, r, f"{int(arr[r, c]):,}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="tokens routed")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_eval(
    ckpt_path: str,
    hf_test,
    tokenizer,
    device: str,
    out_dir: str = "assets/eval",
    batch_size: int = 32,
    max_len: int = 512,
    max_new: int = 128,
    n_bleu_rows: int = 300,
) -> dict:
    """Evaluate one trained variant on the test split: ppl + BLEU + heatmap.

    Files always land in `out_dir` (metrics.json, generations.txt, heatmap PNG).
    If WANDB_API_KEY is available, a short eval run is opened in the same
    project: scalars as charts, the heatmap as an inline image, and the three
    files as an artifact -- all linked from the run URL.
    """
    load_dotenv()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    model.eval()
    print(f"[eval] checkpoint {ckpt_path} (variant {ckpt['config'].ffn_variant}, tokens_seen {ckpt['tokens_seen']:,})")

    run = None
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
        try:
            import wandb

            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "anlp-assignment2"),
                name=f"{Path(ckpt_path).stem}-eval",
                reinit=True,
                settings=wandb.Settings(init_timeout=120),
            )
        except Exception as exc:  # noqa: BLE001 - eval must not die on logging issues
            print(f"wandb init failed, continuing offline: {exc}")
            run = None

    # 1) Test-set perplexity (full test split).
    test_loader = _make_loader(TranslationDataset(hf_test, tokenizer, max_len), tokenizer, batch_size)
    test_ppl = float(evaluate_ppl(model, test_loader, device, what="test split"))
    model.eval()

    # 2) BLEU: greedy translate the first `n_bleu_rows` rows (vi->en and ja->en).
    preds, refs = [], []
    gen_path = out_dir / f"generations_{Path(ckpt_path).stem}.txt"
    n_bleu_rows = min(n_bleu_rows, len(hf_test))
    print(f"[eval] [2/3] BLEU: greedy-translating {n_bleu_rows} rows x2 langs (max_new={max_new}) -> {gen_path.name}")
    with open(gen_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(hf_test.select(range(n_bleu_rows))):
            for lang in ("vi", "ja"):
                src, ref = row[lang].strip(), row["en"].strip()
                if not src or not ref:
                    continue
                pred = greedy_translate(model, tokenizer, src, device, max_new)
                preds.append(pred)
                refs.append(ref)
                f.write(f"[{lang}] src: {src}\nref: {ref}\npred: {pred}\n\n")
            if (i + 1) % 50 == 0:
                print(f"[eval] [2/3] translated {i + 1}/{n_bleu_rows} rows")
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    print(f"[eval] [2/3] done: {len(preds):,} translations, corpus bleu {bleu:.2f}")

    # 3) Heatmap: one inference pass per language over the full test split.
    ds = TranslationDataset(hf_test, tokenizer, max_len)
    usage = {}
    for block_idx, block in enumerate(model.blocks):
        ffn = block.ffn
        if hasattr(ffn, "language_usage"):
            usage[block_idx] = ffn.language_usage
    print(f"[eval] [3/3] usage passes over the full test split (one pass per language)")
    for lang in ("vi", "ja"):
        loader = _make_loader(LanguageSubset(ds, lang), tokenizer, batch_size)
        print(f"[eval] [3/3] pass '{lang}': {len(loader):,} batches")
        collect_usage(model, loader, lang, device)
    heatmap_path = out_dir / f"heatmap_{Path(ckpt_path).stem}.png"
    plot_usage_heatmap(usage, ["vi", "ja"], heatmap_path)
    print(f"[eval] [3/3] heatmap written to {heatmap_path.name}")

    metrics = {
        "variant": ckpt["config"].ffn_variant,
        "tokens_seen": ckpt["tokens_seen"],
        "checkpoint_val_ppl": ckpt["val_ppl"],
        "test_ppl": test_ppl,
        "bleu": float(bleu),
        "n_generations": len(preds),
    }
    (out_dir / f"metrics_{Path(ckpt_path).stem}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[eval] files written to {out_dir}/ (metrics, generations, heatmap for stem '{Path(ckpt_path).stem}')")

    if run is not None:
        heatmap_path = out_dir / f"heatmap_{Path(ckpt_path).stem}.png"
        run.log({**metrics, "eval/heatmap": wandb.Image(str(heatmap_path))})
        artifact = wandb.Artifact(name=f"{Path(ckpt_path).stem}-eval", type="evaluation")
        for f in (heatmap_path, gen_path, out_dir / f"metrics_{Path(ckpt_path).stem}.json"):
            if f.exists():
                artifact.add_file(str(f))
        run.log_artifact(artifact)
        run.finish()
        print(f"[eval] wandb run {run.id} finished (see {run.url})")

    return metrics