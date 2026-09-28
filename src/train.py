"""Shared training loop for parts 1 and 2.

Both parts train the same transformer from scratch with cross-entropy on
pre-shifted labels (labels[i] = token after position i), so the loop itself
is task-agnostic: masking differences already live in the labels (-100).

Seams for the parts:
  - optimizer: build_optimizer() -> AdamW now; part 2 registers its custom
    optimizers here.
  - on_batch hook: part 1 uses it to feed model.record_usage(language).
Part 3 has no training loop (decoding on a pretrained checkpoint).
"""

from dataclasses import dataclass, asdict
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from src.part1.data import count_real_tokens
from src.utils import load_dotenv


@dataclass
class TrainConfig:
    max_tokens: int = 30_000_000  # training-token budget (equal across variants/runs)
    val_every_tokens: int = 3_000_000  # val cadence; part 2 uses 0.1x dataset tokens
    batch_log_every: int = 50  # steps between wandb logs
    lr: float = 8e-4  # part 1 baseline; part 2 overrides per optimizer
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_tokens: int = 1_500_000  # ~5% of the 30M budget; schedule is a fairness constant
    lr_min_ratio: float = 0.1  # cosine floor
    amp: bool = True
    ckpt_dir: str = "checkpoints"
    run_name: str = "run"
    wandb_project: str = "anlp-assignment2"  # WANDB_API_KEY comes from .env


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1):
    """Linear warmup, then cosine decay to `min_ratio` (token-scaled A1 shape)."""
    if step < warmup:
        return (step + 1.0) / warmup
    prog = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


def init_wandb(cfg: TrainConfig):
    """Init wandb if a key is available (WANDB_API_KEY from .env) or offline mode set."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without logging")
        return None
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        print("WANDB_API_KEY not set; continuing without wandb logging")
        return None
    return wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))


def save_checkpoint(model, cfg: TrainConfig, tokens_seen: int, step: int, ppl: float, tag: str) -> Path:
    """Save model state + config; `tag` is e.g. 'best' or 'step_0123'."""
    out_dir = Path(cfg.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.run_name}_{tag}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "tokens_seen": tokens_seen,
            "step": step,
            "val_ppl": ppl,
        },
        path,
    )
    print(f"[train] saved checkpoint {path.name} (tokens {tokens_seen:,} | val ppl {ppl:.3f})")
    return path


def build_optimizer(
    model: torch.nn.Module,
    name: str = "adamw",
    cfg: TrainConfig | None = None,
):
    """Optimizer factory. Part 2 registers its custom optimizers here."""
    cfg = cfg or TrainConfig()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
        )
    raise NotImplementedError(f"unknown optimizer: {name}")


@torch.no_grad()
def max_tokens_per_step(train_loader) -> int:
    """Estimated real tokens per optimizer step (for token-scaled schedule math)."""
    for batch in train_loader:
        return count_real_tokens(batch)
    return 0


@torch.no_grad()
def evaluate_ppl(model, val_loader, device: str = "cuda", what: str = "data") -> float:
    """Mean target-position perplexity over the validation set."""
    print(f"[eval] computing ppl over {len(val_loader):,} batches ({what}) ...")
    model.eval()
    total_ce, total_tokens = 0.0, 0
    for batch in val_loader:
        ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        attn_mask = batch.attention_mask.to(device) if batch.attention_mask is not None else None
        logits = model(ids, attn_mask)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.n_vocab),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_ce += loss.item()
        total_tokens += (labels != -100).sum().item()
    model.train()
    return float(torch.exp(torch.tensor(total_ce / max(total_tokens, 1))))


def train_model(
    model,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    device: str = "cuda",
    optimizer=None,
    on_batch=None,
):
    """Train until the token budget is exhausted, validating at intervals."""
    load_dotenv()
    model.to(device)
    optimizer = optimizer or build_optimizer(model, cfg=cfg)
    tokens_per_step = max_tokens_per_step(train_loader)
    total_steps = max(cfg.max_tokens // max(tokens_per_step, 1), 1)
    warmup_steps = max(cfg.warmup_tokens // max(tokens_per_step, 1), 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, warmup_steps, total_steps, cfg.lr_min_ratio)
    )
    run = init_wandb(cfg)
    model.train()

    tokens_seen = 0
    step = 0
    next_val = cfg.val_every_tokens
    best_ppl = float("inf")
    use_amp = cfg.amp and device.startswith("cuda")

    print(
        f"[train] {cfg.run_name} | device={device} | amp={use_amp} | budget={cfg.max_tokens:,} tokens "
        f"| ~{tokens_per_step} tokens/step | {total_steps} steps | lr={cfg.lr} | "
        f"warmup ~{warmup_steps} steps, cos-decay to {cfg.lr_min_ratio}"
    )

    while tokens_seen < cfg.max_tokens:
        for batch in train_loader:
            ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            attn_mask = batch.attention_mask.to(device) if batch.attention_mask is not None else None

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(ids, attn_mask)
                    loss = F.cross_entropy(
                        logits.reshape(-1, model.config.n_vocab),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
            else:
                logits = model(ids, attn_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.config.n_vocab),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            tokens_seen += count_real_tokens(batch)
            if on_batch is not None:
                on_batch(model, batch)

            if step % cfg.batch_log_every == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "tokens": tokens_seen,
                    "lr": schedule.get_last_lr()[0],
                    "step": step,
                }
                if run is not None:
                    run.log(metrics)
                else:
                    print(f"step {step} | tokens {tokens_seen} | loss {loss.item():.4f}")

            if tokens_seen >= next_val:
                ppl = evaluate_ppl(model, val_loader, device)
                if run is not None:
                    run.log({"val/ppl": ppl, "val/tokens": tokens_seen, "step": step})
                print(f"VAL at {tokens_seen} tokens: ppl {ppl:.3f}")
                save_checkpoint(model, cfg, tokens_seen, step, ppl, f"tok{tokens_seen}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    save_checkpoint(model, cfg, tokens_seen, step, ppl, "best")
                next_val += cfg.val_every_tokens
                model.train()

            if tokens_seen >= cfg.max_tokens:
                break

    save_checkpoint(model, cfg, tokens_seen, step, best_ppl, "final")
    if run is not None:
        run.finish()
    print(f"[train] done: {step} steps, {tokens_seen:,} tokens, best val ppl {best_ppl:.3f}")
    return step, tokens_seen