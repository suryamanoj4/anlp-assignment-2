"""Per-language BLEU from eval generations — LOCAL files or WANDB artifacts.

Each generations_*.txt written by src/part1/evaluate.py contains blocks:

    [vi] src: <source sentence>
    ref: <reference translation>
    pred: <greedy translation>

    [ja] src: ...

The script reports corpus BLEU over ALL translations plus per source language
(vi / ja). Sources, in order of preference:
  1. --files <paths>                  explicit local files
  2. --root <dir> (or default)        scan a local eval dir
  3. --wandb (or fallback)            fetch the part1-v{N}_best-eval artifacts,
                                      download generations into a temp dir that
                                      is auto-deleted after computation

Usage:
    uv run python scripts/bleu_from_generations.py runs/part1/assets/eval/generations_part1-v3_best.txt
    uv run python scripts/bleu_from_generations.py --root runs/part1/assets/eval
    uv run python scripts/bleu_from_generations.py --wandb
    uv run python scripts/bleu_from_generations.py --wandb --json bleu.json
    uv run python scripts/bleu_from_generations.py --no-wandb   # local only, no fallback
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import sacrebleu

BLOCK = re.compile(r"^\[(vi|ja)\] src: (.*)$")  # block starts; ref/pred lines follow
DEFAULT_ROOT = "outputs/assets/eval"
DEFAULT_PROJECT = "suryamanojphy31-iiit-hyderabad/anlp-assignment2"


def parse_generations(path: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Parse a generations file -> {lang: (preds, refs)}, skipping empty rows.

    Line-based (NOT regex-block) parsing: decoded predictions may legitimately
    contain literal newlines (the byte-10 BPE token), which would break a
    single-line block regex and silently drop rows.
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    lang: str | None = None
    ref: list[str] = []
    pred: list[str] = []

    def flush() -> None:
        if lang is not None and ref:  # empty PREDs are kept: eval counted them too
            r, p = " ".join(ref).strip(), " ".join(pred).strip()
            if r:  # degenerate block only if no reference at all
                preds, refs = out.setdefault(lang, ([], []))
                preds.append(p)
                refs.append(r)

    for raw in path.read_text(encoding="utf-8").splitlines():
        m = BLOCK.match(raw)
        if m:  # new block: finalize the previous one
            flush()
            lang, ref, pred = m.group(1), [], []
            continue
        if lang is None:
            continue
        if raw.startswith("ref: "):
            ref.append(raw[5:])
        elif raw.startswith("pred: "):
            pred.append(raw[6:])
        elif ref and not pred:  # continuation of a multi-line reference
            ref.append(raw)
        elif pred:  # continuation of a multi-line prediction
            pred.append(raw)
    flush()
    return out


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_bleu(preds, [refs]).score if preds else float("nan")


def analyze(path: Path) -> dict:
    """One generations file -> one summary row (overall + per-language BLEU)."""
    parsed = parse_generations(path)
    all_preds = [p for lang in parsed.values() for p in lang[0]]
    all_refs = [r for lang in parsed.values() for r in lang[1]]
    row = {
        "file": path.name,
        "overall_bleu": round(corpus_bleu(all_preds, all_refs), 2),
        "n_total": len(all_preds),
    }
    for lang, (preds, refs) in sorted(parsed.items()):
        row[f"bleu_{lang}"] = round(corpus_bleu(preds, refs), 2)
        row[f"n_{lang}"] = len(preds)
    return row


def fetch_from_wandb(project: str, tmpdir: Path) -> list[Path]:
    """Download generations files from the -eval artifacts into tmpdir; caller cleans up."""
    import wandb  # only needed for this path
    from src.utils import load_dotenv

    load_dotenv()
    api = wandb.Api()
    files: list[Path] = []
    for run in api.runs(project, per_page=50):
        if not run.name.endswith("-eval") or run.state != "finished":
            continue
        for art in run.logged_artifacts():
            if art.name == f"{run.name}:v0":  # the eval artifact (skip history/events)
                d = Path(art.download(root=str(tmpdir / run.name)))
                files += [f for f in d.iterdir() if f.name.startswith("generations_") and f.suffix == ".txt"]
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="explicit generations_*.txt paths")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="local eval dir to scan")
    parser.add_argument("--wandb", action="store_true", help="fetch from wandb artifacts")
    parser.add_argument("--no-wandb", action="store_true", help="never fall back to wandb")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="wandb project (entity/name)")
    parser.add_argument("--json", metavar="OUT", help="also write results as JSON")
    args = parser.parse_args()

    source = ""
    files: list[Path] = []

    if args.files:
        files = [Path(f) for f in args.files]
        source = "explicit local files"
    elif (Path(args.root).is_dir() and list(Path(args.root).glob("generations_*.txt"))) or not args.wandb:
        files = sorted(Path(args.root).glob("generations_*.txt"))
        source = f"local dir {args.root}" if files else "local dir (nothing found)"
        if not files and args.wandb:
            pass  # fall through to wandb below
    if not files and (args.wandb or not args.no_wandb):
        with tempfile.TemporaryDirectory(prefix="bleu_wandb_") as td:
            files = fetch_from_wandb(args.project, Path(td))
            source = "wandb artifacts (temp, auto-cleaned)"
            if files:
                rows = [analyze(f) for f in files]
                print_table(rows)
                if args.json:
                    Path(args.json).write_text(json.dumps(rows, indent=2))
                    print(f"wrote {args.json}")
                return
        source = "wandb (nothing found)"

    if not files:
        sys.exit(f"no generations files found (tried: {args.files or args.root}, wandb={args.wandb or not args.no_wandb})")

    rows = [analyze(f) for f in files]
    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    print(f"(source: {source}, {len(files)} file(s))")


def print_table(rows: list[dict]) -> None:
    langs = sorted({k.split("_", 1)[1] for r in rows for k in r if k.startswith("bleu_")})
    header = f"{'file':38s} {'overall':>8s} " + " ".join(f"{f'bleu_{l}':>8s} {f'n_{l}':>4s}" for l in langs) + f" {'n_total':>7s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = f"{r['file']:38s} {r['overall_bleu']:>8.2f} "
        cells += " ".join(f"{r.get(f'bleu_{l}', float('nan')):>8.2f} {r.get(f'n_{l}', 0):>4d}" for l in langs)
        cells += f" {r['n_total']:>7d}"
        print(cells)


if __name__ == "__main__":
    main()