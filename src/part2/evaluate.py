"""Part 2 evaluation: continuation-BLEU at every saved checkpoint + plots.

Definition (report-grade, chosen from the corpus structure):
  - ONLY the human rows of a test doc are references: prompt = human chunk 1
    (truncated to n_ctx - max_new tokens), greedy continuation (max_new
    tokens, src/part3.decode.greedy_decode) scored against human chunk 2
    (truncated to the same max_new budget) with sacrebleu at corpus level
    (default 13a tokenization). The 6 LLM rows of each doc are other models'
    outputs, not ground truth, and are excluded.
  - Val ppl per point is read from the saved checkpoint metadata, so the
    val-loss and BLEU curves share one tokens axis (0.1x dataset cadence).
  - Plots are part-2 owned: part2-* PNG names, written under the part 2
    output root — they never mix with part 1's assets.
  - Every point is also logged to WandB (run "part2-{name}-eval") when an
    API key is available.
"""

import json
import os
from pathlib import Path

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, which
# breaks `import matplotlib` inside the uv venv (same guard as part 1).
# UNCONDITIONAL assignment: setdefault would keep Colab's poisoned value.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu
import torch

from src.part1.model import Transformer
from src.part3.decode import greedy_decode
from src.train import TrainConfig, init_wandb


def _is_human_suffix(suffix: str) -> bool:
    """True for the human rows of a doc.

    Real corpus: the two human chunks are suffixed 'chunk_1' and 'chunk_2'
    (verified on browndw/human-ai-parallel-corpus; doc_ids like
    'acad_0001@chunk_1', LLM rows carry model names). The synthetic smoke
    corpus uses 'human_chunk1'/'human_chunk2' — covered by the 'human'
    fallback.
    """
    return suffix in ("chunk_1", "chunk_2") or "human" in suffix.lower()


def human_pairs(test_rows, tokenizer, max_prompt_tokens: int, max_new: int) -> list[dict]:
    """(doc, human chunk1) -> (chunk1 prompt ids, chunk2 reference text).

    Grouping is by the doc root (pre-'@' prefix); within a doc the two human
    chunks are identified by _is_human_suffix and ordered by suffix so
    chunk1 (the prompt) precedes chunk2 (the reference). Docs without both
    human chunks are skipped (counted in the log).
    """
    by_doc: dict[str, dict[str, str]] = {}
    hist: dict[str, int] = {}
    for r in test_rows:
        doc = r["doc_id"].split("@")[0]
        suffix = r["doc_id"].split("@")[-1]
        hist[suffix] = hist.get(suffix, 0) + 1
        if _is_human_suffix(suffix):
            by_doc.setdefault(doc, {})[suffix] = r["text"]
    pairs = []
    skipped = 0
    for doc, chunks in sorted(by_doc.items()):
        keys = sorted(chunks)
        if len(keys) < 2:
            skipped += 1
            continue
        prompt_ids = torch.tensor(
            tokenizer(chunks[keys[0]], add_special_tokens=False)["input_ids"][:max_prompt_tokens]
        )
        ref_ids = tokenizer(chunks[keys[1]], add_special_tokens=False)["input_ids"][:max_new]
        pairs.append({
            "doc": doc,
            "prompt_ids": prompt_ids,
            "reference": tokenizer.decode(ref_ids, skip_special_tokens=True),
        })
    print(
        f"[eval] human-pair suffixes: {sorted(h for h in hist if _is_human_suffix(h))} "
        f"| {len(pairs)} chunk1->chunk2 pairs from {len(by_doc)} docs "
        f"({skipped} docs skipped: <2 human chunks) | "
        f"max_prompt={max_prompt_tokens} max_new={max_new}"
    )
    return pairs


def continuation_bleu(pairs, model, tokenizer, device, max_new: int, batch_size: int = 16) -> float:
    """Greedy continuation of every prompt; corpus BLEU vs the human chunks 2."""
    import torch.nn as nn

    if not pairs:
        raise ValueError("continuation_bleu needs >= 1 human pair (run_eval_pass guards this)")
    hyps = []
    model.eval()
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        prompts = nn.utils.rnn.pad_sequence(
            [p["prompt_ids"] for p in chunk], batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)
        gen = greedy_decode(model, prompts, max_new,
                            tokenizer.eos_token_id, tokenizer.pad_token_id)
        for g in gen.tolist():
            # stop at the first eos OR pad (finished rows are pad-padded)
            cut = len(g)
            for j, tok in enumerate(g):
                if tok in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    cut = j
                    break
            hyps.append(tokenizer.decode(g[:cut], skip_special_tokens=True))
    model.train()
    refs = [[p["reference"] for p in pairs]]
    return float(sacrebleu.corpus_bleu(hyps, refs).score)


def eval_all_checkpoints(
    run_name: str,
    ckpt_dir,
    pairs,
    tokenizer,
    device: str,
    max_prompt_tokens: int,
    max_new: int,
    batch_size: int,
    wandb_run=None,
) -> list[dict]:
    """Continuation-BLEU on every saved checkpoint of one optimizer run.

    Loads the model once, swaps state_dict per checkpoint (fast; constant
    memory). Returns [{tokens, val_ppl, bleu}] in token order and logs each
    point to wandb_run when given.
    """
    ckpt_dir = Path(ckpt_dir)
    if not pairs:
        print("[eval] no human chunk1->chunk2 pairs available; skipping BLEU pass")
        return []
    paths = sorted(ckpt_dir.glob(f"{run_name}_tok*.pt"),
                   key=lambda p: int(p.stem.rsplit("tok", 1)[1]))
    final = ckpt_dir / f"{run_name}_final.pt"
    if final.exists():
        paths.append(final)
    if not paths:
        print(f"[eval] no checkpoints for {run_name}; skipping BLEU pass")
        return []
    print(f"[eval] {run_name}: {len(paths)} checkpoints x {len(pairs)} pairs "
          f"(max_prompt={max_prompt_tokens}, max_new={max_new}) ...")
    model = None
    points = []
    for ck in paths:
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        if model is None:
            model = Transformer(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        bleu = continuation_bleu(pairs, model, tokenizer, device, max_new, batch_size)
        point = {"tokens": int(ckpt["tokens_seen"]),
                 "val_ppl": float(ckpt["val_ppl"]), "bleu": bleu}
        points.append(point)
        print(f"[eval] {ck.name}: tokens {point['tokens']:,} | "
              f"val ppl {point['val_ppl']:.3f} | bleu {bleu:.2f}")
        if wandb_run is not None:
            wandb_run.log({"tokens": point["tokens"], "val/ppl": point["val_ppl"],
                           "bleu": point["bleu"]})
    return points


def plot_part2_curves(eval_dir, names=None) -> list[Path]:
    """One figure per metric (val ppl, BLEU) vs tokens, all optimizers.

    Reads part2-{name}_eval.json files; writes part2_val_ppl_vs_tokens.png /
    part2_bleu_vs_tokens.png under eval_dir (part-2 owned names — never
    mix with part 1's plots).
    """
    eval_dir = Path(eval_dir)
    if names is None:
        files = sorted(eval_dir.glob("part2-*_eval.json"))
    else:
        files = [eval_dir / f"part2-{n}_eval.json" for n in names]
    data = {}
    for f in files:
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        if j.get("points"):
            data[j["optimizer"]] = j["points"]
    if not data:
        print("[eval] no eval jsons found; skipping plots")
        return []
    print(f"[eval] plotting {len(data)} optimizers' curves ...")

    for metric, fname in (("val_ppl", "part2_val_ppl_vs_tokens.png"),
                          ("bleu", "part2_bleu_vs_tokens.png")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, pts in sorted(data.items()):
            xs = [p["tokens"] / 1e6 for p in pts]
            ys = [p[metric] for p in pts]
            ax.plot(xs, ys, marker="o", ms=3, label=name)
        ax.set_xlabel("tokens (millions)")
        ax.set_ylabel("val ppl" if metric == "val_ppl" else "test BLEU")
        ax.set_title(f"Part 2 — {'validation ppl' if metric == 'val_ppl' else 'continuation BLEU'} vs tokens")
        if metric == "val_ppl":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = eval_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] plot saved: {out}")
    return [eval_dir / f for _, f in (("val_ppl", "part2_val_ppl_vs_tokens.png"),
                                      ("bleu", "part2_bleu_vs_tokens.png"))]


def run_eval_pass(
    name: str,
    tokenizer,
    test_rows,
    eval_dir,
    ckpt_dir,
    device: str,
    max_len: int,
    bleu_max_new: int,
    batch_size: int,
) -> list[dict]:
    """Full per-optimizer eval pass: pairs -> per-ckpt BLEU -> wandb -> json.

    Returns the points list (also persisted as part2-{name}_eval.json).
    """
    max_prompt_tokens = max(max_len - bleu_max_new, 8)
    pairs = human_pairs(test_rows, tokenizer, max_prompt_tokens, bleu_max_new)
    if not pairs:
        print("[main] no human pairs -> BLEU eval skipped (training/eval pipeline "
              "continues; run after fixing the split or suffixes if this is unexpected)")
    run = init_wandb(TrainConfig(run_name=f"part2-{name}-eval"))  # None w/o key
    points = eval_all_checkpoints(
        f"part2-{name}", ckpt_dir, pairs, tokenizer, device,
        max_prompt_tokens, bleu_max_new, batch_size, wandb_run=run,
    )
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"part2-{name}_eval.json"
    out.write_text(json.dumps({"optimizer": name, "points": points}, indent=2))
    print(f"[main] {name} eval points -> {out}")
    if run is not None:
        if points:
            run.summary.update({"best_bleu": max(p["bleu"] for p in points),
                                "best_val_ppl": min(p["val_ppl"] for p in points)})
        run.finish()
    return points