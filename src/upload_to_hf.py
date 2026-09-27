"""Upload a trained checkpoint to HuggingFace Hub.

Separate from training on purpose: run it once runs are done and results
finalized. Reads HF_TOKEN from .env (via src.utils.load_dotenv).

Usage:
    uv run python -m src.upload_to_hf \
        --ckpt checkpoints/v2_best.pt \
        --repo-id <your-username>/anlp2-part1-v2
"""

import argparse
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi

from src.utils import load_dotenv

README_TEMPLATE = """\
# {repo_id}

Checkpoint for **Advanced NLP Assignment 2, Part 1** (MoE variants).

- FFN variant: {ffn_variant}
- Number of tokens trained: {tokens_seen}
- Val perplexity (checkpoint time): {val_ppl:.3f}

See the assignment README + report for analysis and the WandB run links.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to a .pt checkpoint from src/train.py")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/anlp2-part1-v2")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env (create a token on huggingface.co)")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if is_dataclass(config):
        config = asdict(config)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Standard HF file layout: weights + config + card.
        from safetensors.torch import save_file

        save_file(ckpt["model"], tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_text(json.dumps(config, indent=2))
        readme = README_TEMPLATE.format(
            repo_id=args.repo_id,
            ffn_variant=config.get("ffn_variant", "unknown"),
            tokens_seen=ckpt.get("tokens_seen", "?"),
            val_ppl=ckpt.get("val_ppl", float("nan")),
        )
        (tmp_path / "README.md").write_text(readme)

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(tmp_path), repo_id=args.repo_id, token=token)
        print(f"uploaded {ckpt_path} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()