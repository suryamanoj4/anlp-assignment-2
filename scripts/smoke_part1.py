"""Smoke test for Part 1 — tiny config, CPU only, no training runs, no network.

Covers, in order:
  A. tokenizer: train a tiny BPE on synthetic strings -> save -> load -> encode/decode
  B. data: synthetic HF dataset -> samples -> collate -> shapes, -100 masking, token counting
  C. model dims per FFN variant (1..5): forward shapes/finiteness, mask path,
     usage accounting (B*T*n_active per batch, routed-only), zero-token expert
     guard, gradient flow through the gate
  D. param matching: v1-v4 equal total FFN params (+gate footnote), v5 active == v1 (+gate)
  E. checkpoint round-trip: save -> load -> identical logits

Run:  uv run python scripts/smoke_part1.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import torch
import torch.nn.functional as F
from datasets import Dataset

from src.part1.data import count_real_tokens, make_dataloader
from src.part1.model import (
    MoE,
    Transformer,
    TransformerConfig,
    count_active_params,
    count_total_params,
    ffn_variant_config,
)
from src.tokenizer import load_tokenizer, train_bpe_tokenizer
from src.train import TrainConfig, save_checkpoint

# Tiny sizes so everything fits in a second on a laptop CPU.
D, FF, LAYERS, HEADS, CTX = 64, 128, 2, 4, 32
B, T = 3, 17          # batch, sequence length for forward checks
N_EXPERTS = 4
DEVICE = "cpu"

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(name)


def synthetic_texts() -> list[str]:
    en = ["the cat sits on the mat", "i like green tea", "the train leaves at noon"]
    vi = ["con meo ngoi tren tam tham", "toi thich tra xanh", "chuyen tau roi luc trua"]
    ja = ["ねこがマットの上にすわっている", "わたしは緑茶が好きです", "列車は正午に出発します"]
    return en + vi + ja


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- A. tokenizer
def test_tokenizer() -> None:
    section("A. Tokenizer (train tiny BPE -> save -> load -> encode/decode)")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tok.json"
        tok = train_bpe_tokenizer(synthetic_texts(), vocab_size=300, save_path=path)
        check("train + save", path.exists())
        loaded = load_tokenizer(path)
        vocab = loaded.get_vocab()
        check("reload + vocab", len(vocab) >= 200, f"vocab={len(vocab)} (BPE target 300; tiny corpus runs out of merges)")
        check("special ids", all(s in vocab for s in ("<pad>", "<bos>", "<eos>", "<unk>")),
              f"pad={loaded.pad_token_id} bos={loaded.bos_token_id} eos={loaded.eos_token_id} unk={loaded.unk_token_id}")
        enc = loaded("ねこがマットの上にすわっている", add_special_tokens=False)["input_ids"]
        check("encode", len(enc) > 0, f"{len(enc)} ids")
        dec = loaded.decode(enc, skip_special_tokens=True)
        check("decode round-trip", dec == "ねこがマットの上にすわっている", dec)
    print(f"  (section A: {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- B. data
def test_data(tokenizer) -> None:
    section("B. Data pipeline (synthetic rows -> samples -> collate)")
    t0 = time.time()
    texts = synthetic_texts()
    rows = [{"en": texts[i % 3], "vi": texts[i % 3 + 3], "ja": texts[i % 3 + 6]} for i in range(8)]
    hf = Dataset.from_dict({k: [r[k] for r in rows] for k in ("en", "vi", "ja")})
    loader = make_dataloader(hf, tokenizer, max_len=16, batch_size=4, shuffle=False, what="smoke")
    batch = next(iter(loader))
    B_ = len(batch.language)
    check("collate shapes", batch.input_ids.shape == batch.labels.shape == batch.attention_mask.shape,
          f"ids {tuple(batch.input_ids.shape)} (B={B_})")
    check("language labels", set(batch.language) == {"vi", "ja"}, str(batch.language))
    check("mask is binary", set(batch.attention_mask.unique().tolist()) <= {0, 1})
    check("token count == mask sum", count_real_tokens(batch) == int(batch.attention_mask.sum()),
          f"{count_real_tokens(batch)} tokens")
    # -100 prefix masking: first sample's first 1+src_len+1 positions must be -100.
    src_len = len(tokenizer(rows[0]["vi"], add_special_tokens=False)["input_ids"])
    prefix = 1 + src_len + 1
    check("prefix masked with -100", bool((batch.labels[0][:prefix] == -100).all()),
          f"{prefix} leading positions")
    n_real = int(batch.attention_mask[0].sum())  # real tokens of sample 0
    # Last real position holds the trailing shift-marker -100 (predict-nothing-
    # after-final-token, per data.py) — so real targets live in [prefix, n_real-1).
    check("targets after prefix are real", bool((batch.labels[0][prefix:n_real - 1] != -100).all()))
    print(f"  (section B: {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- C. model dims
def test_model_dims(tokenizer) -> None:
    section("C. Model dims per FFN variant (forward, mask, usage, gradients)")
    t0 = time.time()
    V = tokenizer.vocab_size
    for v in range(1, 6):
        cfg = TransformerConfig(
            d_model=D, n_ctx=CTX, n_vocab=V, n_layers=LAYERS, n_heads=HEADS, d_ff=FF,
            **ffn_variant_config(v, FF, N_EXPERTS),  # includes ffn_variant
        )
        model = Transformer(cfg).to(DEVICE).eval()
        x = torch.randint(0, V, (B, T))
        logits = model(x)
        check(f"v{v}: logits shape (B,T,V)", tuple(logits.shape) == (B, T, V), str(tuple(logits.shape)))
        check(f"v{v}: logits finite", bool(torch.isfinite(logits).all()))
        # Masked path: pad token 5 at the end of cols -> same output shape, finite.
        xm = x.clone(); xm[:, -3:] = 5
        mm = torch.ones_like(xm); mm[:, -3:] = 0
        out_m = model(xm, mm)
        check(f"v{v}: masked forward shape/finite", tuple(out_m.shape) == (B, T, V) and bool(torch.isfinite(out_m).all()))

        if v > 1:
            n_active = cfg.n_active_experts  # routed slots per token (shared excluded)
            n_rec, n_fwd = 2, 3  # record after batch 1&2; a 3rd batch is never recorded
            model.reset_usage()
            model(x); model.record_usage("vi")
            model(x); model.record_usage("vi")
            model(x)
            got = sum(model.blocks[b].ffn.language_usage["vi"].sum().item() for b in range(LAYERS))
            raw = sum(model.blocks[b].ffn.usage_counts.sum().item() for b in range(LAYERS))
            check(f"v{v}: usage counts = B*T*n_active per layer-batch",
                  got == LAYERS * n_rec * B * T * n_active and raw == LAYERS * n_fwd * B * T * n_active,
                  f"recorded {got} (expect {LAYERS * n_rec * B * T * n_active}), buffer {raw}")
            check(f"v{v}: usage per-expert dims", all(
                model.blocks[b].ffn.language_usage["vi"].numel() == (N_EXPERTS - cfg.n_shared_experts)
                for b in range(LAYERS)))
        # Gradient flow: loss must reach the router gate on MoE variants.
        lab = torch.randint(0, V, (B, T))
        loss = F.cross_entropy(logits.reshape(-1, V), lab.reshape(-1))
        loss.backward()
        g = model.blocks[0].ffn.gate.weight.grad if v > 1 else None
        check(f"v{v}: gate receives gradient", (v == 1) or (g is not None and bool(torch.isfinite(g).all())),
              "" if v == 1 else f"gate.grad max|.|={g.abs().max().item():.2e}" if g is not None else "no grad")
        check(f"v{v}: grads finite on all touched params",
              all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    # Deterministic zero-token-expert guard: a uniform row of -100 makes expert 0's
    # score = -100 * sum(x). With N(0,1) input that sum flips sign ~half the time
    # (score_0 then WINS -- gate scale must be read with input stats in mind!).
    # With strictly positive input (torch.rand), score_0 <= -3200 while every other
    # expert scores within +/-8, so expert 0 can never win top-1 -> the dispatch
    # loop must exercise its empty-expert `continue` branch.
    section("C2. Zero-token expert guard (routed counts, expert 0 starved)")
    cfg2 = TransformerConfig(d_model=D, n_ctx=CTX, n_vocab=V, n_layers=1, n_heads=4,
                             d_ff=FF, **ffn_variant_config(2, FF, N_EXPERTS))
    moe = MoE(cfg2).to(DEVICE).eval()
    with torch.no_grad():
        moe.gate.weight[0] = -100.0
        xr = torch.rand(B, T, D)  # strictly positive: sum(x) > 0 always
        y = moe(xr)
    check("zero-expert fwd shape", tuple(y.shape) == (B, T, D))
    check("starved expert got 0 tokens", moe.usage_counts[0].item() == 0, str(moe.usage_counts.tolist()))
    check("remaining slots still counted", int(moe.usage_counts.sum()) == B * T * 1, f"sum={moe.usage_counts.sum()}")
    print(f"  (sections C/C2: {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- D. param matching
def test_param_matching(tokenizer) -> None:
    section("D. Param matching (v1-4 total FFN equal; v5 active == v1 + gate)")
    t0 = time.time()
    V = tokenizer.vocab_size
    models = {}
    for v in range(1, 6):
        cfg = TransformerConfig(
            d_model=D, n_ctx=CTX, n_vocab=V, n_layers=LAYERS, n_heads=HEADS, d_ff=FF,
            **ffn_variant_config(v, FF, N_EXPERTS),
        )
        models[v] = Transformer(cfg)

    t = {v: count_total_params(models[v]) for v in range(1, 6)}
    a = {v: count_active_params(models[v]) for v in range(1, 6)}
    gate = {v: (LAYERS * D * (N_EXPERTS - models[v].blocks[0].ffn.n_shared)) if v > 1 else 0
            for v in range(1, 6)}  # v1 dense has no gate at all

    base = t[1] - 2 * D * FF * LAYERS          # embeddings + attn + norms (shared by all)
    inner = {v: t[v] - base for v in range(1, 6)}  # FFN params incl. gate
    dense_ffn = 2 * D * FF * LAYERS
    check("v1-4 total FFN params equal (modulo gate)", all(inner[v] - gate[v] == dense_ffn for v in (1, 2, 3, 4)),
          f"ffn totals: {[inner[v] - gate[v] for v in (1,2,3,4)]}")
    check("v5 FFN total = 2x dense", inner[5] - gate[5] == 2 * dense_ffn, f"{inner[5] - gate[5]} vs {2 * dense_ffn}")
    check("v5 active == v1 active (+gate)", a[5] == a[1] + gate[5], f"{a[5]} vs {a[1]} + {gate[5]}")
    check("v3 active = half dense (+gate)", a[3] == dense_ffn // 2 + gate[3], f"{a[3]} vs {dense_ffn // 2} + {gate[3]}")
    check("v4 active = half dense (+gate)", a[4] == dense_ffn // 2 + gate[4], f"{a[4]} vs {dense_ffn // 2} + {gate[4]}")
    check("total params ordered v5 > others", t[5] == t[1] + dense_ffn + gate[5], f"{t[5]} vs {t[1]} + {dense_ffn} + {gate[5]}")
    print(f"  totals: v1={t[1]:,} v2={t[2]:,} v3={t[3]:,} v4={t[4]:,} v5={t[5]:,} | "
          f"active: v1={a[1]:,} v3={a[3]:,} v4={a[4]:,} v5={a[5]:,}")
    print(f"  (section D: {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- E. checkpoint round-trip
def test_checkpoint_roundtrip(tokenizer) -> None:
    section("E. Checkpoint save -> load -> identical outputs")
    t0 = time.time()
    V = tokenizer.vocab_size
    cfg = TransformerConfig(
        d_model=D, n_ctx=CTX, n_vocab=V, n_layers=LAYERS, n_heads=HEADS, d_ff=FF,
        **ffn_variant_config(3, FF, N_EXPERTS),
    )
    model = Transformer(cfg).eval()
    x = torch.randint(0, V, (3, 17))
    with torch.no_grad():
        logits_a = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(model, TrainConfig(run_name="smoke", ckpt_dir=tmp), tokens_seen=1234, step=7, ppl=5.5, tag="best")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        loaded = Transformer(ckpt["config"]).eval()
        loaded.load_state_dict(ckpt["model"])
        with torch.no_grad():
            logits_b = loaded(x)
    check("ckpt metadata", ckpt["tokens_seen"] == 1234 and ckpt["step"] == 7 and ckpt["val_ppl"] == 5.5)
    check("identical logits after round-trip", bool(torch.equal(logits_a, logits_b)))
    print(f"  (section E: {time.time() - t0:.1f}s)")


def main() -> None:
    t_start = time.time()
    print(f"smoke (part 1) | device={DEVICE} | sizes: d_model={D} d_ff={FF} layers={LAYERS} ctx={CTX}")
    test_tokenizer()
    # The tiny tokenizer from section A is needed by B-E: train it once into a
    # dedicated temp dir (keeps the repo clean; the real 32k one trains via main.py).
    tok_path = Path(tempfile.mkdtemp()) / "tok.json"
    train_bpe_tokenizer(synthetic_texts(), vocab_size=300, save_path=tok_path)
    tokenizer = load_tokenizer(tok_path)
    test_data(tokenizer)
    test_model_dims(tokenizer)
    test_param_matching(tokenizer)
    test_checkpoint_roundtrip(tokenizer)

    print(f"\n=== SUMMARY: {'ALL CHECKS PASSED' if not _failures else f'{len(_failures)} FAILED'} "
          f"({time.time() - t_start:.1f}s total) ===")
    for name in _failures:
        print(f"  FAILED: {name}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()