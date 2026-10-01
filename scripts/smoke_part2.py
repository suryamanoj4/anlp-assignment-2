"""Smoke test for Part 2 — custom optimizers, CPU only, no network.

Covers, in order:
  A. shared helper _clamp_grad_norm: no-op at g_norm<=0, scales above, zero-grad guard
  B. AdamW (custom): hand-computed t=1 update (bias corrections cancel), then a
     300-step run whose parameters AND m/v state buffers are bit-identical to
     torch.optim.AdamW (fp32 op-order noise only), g_norm semantics (clamp
     enters the moments, NOT the first step's normalized direction), closure /
     frozen params / multi-group lr, invalid-hyperparameter guards
  C. NadamW (custom): hand-computed t=1 (m_hat=(1+b1)g) and t=2 full algebra,
     b1=0 degeneracy (bit-identical to our AdamW — lookahead collapses), the
     g_norm path into m_tilde, guards. NOTE: torch.optim.NAdam is NOT an
     oracle (momentum-decay variant; nesterov flag gone in 2.14)
  D. Lion (custom): t=1 and t=2 hand-computed (beta roles/ordering — sign from
     b1-mix of OLD state, state advanced with b2), single-buffer memory claim,
     g_norm invisibility at t=1 but divergence later; guards
  E. Muon + _newton_schulz (custom): NS invariants (u^T u ~ I, ~ polar factor,
     left-Gram branch, steps=0 normalization, zero-matrix safety), role
     partition (2D-muon / 1D+adam_ids-adam), hand-computed t=1 muon step with
     ns_steps=0, adam-branch bit-identical to our AdamW, end-to-end tiny
     transformer training with make_optimizer("muon", ...)
  F. factory: all four names -> our classes (adamw is NOT torch's), per-name
     lr constants, unknown name raises, subclass constraint

NOTE on the oracle: torch.optim.* appears in this file ONLY as a test
reference. src/part2/* never imports any torch.optim module other than
torch.optim.Optimizer (hard assignment constraint).

Run:  uv run python scripts/smoke_part2.py
"""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import torch
import torch.nn.functional as F

from src.part1.model import Transformer, TransformerConfig
from src.part2.data import (
    LMDataset,
    Part2Batch,
    collate_lm_batch,
    count_real_tokens,
    make_dataloader,
    make_lm_samples,
    split_rows_by_doc,
)
from src.part2.evaluate import continuation_bleu, human_pairs, plot_part2_curves, run_eval_pass
from src.part2.optimizers import (
    LION_LR,
    PART2_OPTIMIZERS,
    AdamW,
    Lion,
    Muon,
    NadamW,
    _clamp_grad_norm,
    _newton_schulz,
    make_optimizer,
)
from src.part3.decode import greedy_decode
from src.train import TrainConfig
from src.tokenizer import load_tokenizer, train_bpe_tokenizer

B1, B2, LR, EPS, WD = 0.9, 0.98, 8e-4, 1e-8, 0.01  # part 1 baseline hyperparams
DEVICE = "cpu"

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- A. helpers
def test_clamp_grad_norm() -> None:
    section("A. _clamp_grad_norm (paper Appendix A preamble)")
    g = torch.tensor([[3.0, -4.0]])  # ||g||_2 = 5
    check("g_norm=0 is identity", torch.equal(_clamp_grad_norm(g, 0.0), g))
    check("g_norm<0 is identity", torch.equal(_clamp_grad_norm(g, -1.0), g))
    out = _clamp_grad_norm(g, 10.0)  # scale = max(1, 10/5) = 2
    check("g_norm>||g|| scales up", torch.equal(out, g * 2.0), str(out.tolist()))
    check("g_norm<||g|| is identity", torch.equal(_clamp_grad_norm(g, 2.0), g))
    z = torch.zeros(4)
    check("zero gradient is safe (no nan)", bool(torch.isfinite(_clamp_grad_norm(z, 3.0)).all()))


# ---------------------------------------------------------------- B. AdamW
def test_adamw() -> None:
    section("B. Custom AdamW (paper Algorithm 1 / torch-identical)")

    # B1: hand-computed first step. At t=1 bias corrections cancel exactly:
    # m_hat = g, v_hat = g^2, so p1 = p0(1-lr*wd) - lr*g/(|g|+eps).
    p0 = torch.tensor([[1.0, -2.0, 0.5]])
    g = torch.tensor([[0.3, -0.7, 0.0]])
    p = p0.clone().requires_grad_()
    opt = AdamW([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    p.grad = g
    opt.step()
    expect = p0 * (1 - LR * WD) - LR * g / (g.abs() + EPS)
    check("t=1 update == hand-computed paper formula",
          bool(torch.allclose(p.detach(), expect, rtol=1e-6, atol=1e-12)),
          f"p={p.tolist()}")
    st = opt.state[p]
    check("state layout {step, exp_avg, exp_avg_sq}",
          set(st) == {"step", "exp_avg", "exp_avg_sq"} and st["step"].item() == 1.0,
          str(sorted(st)))

    # B2: 300 steps vs torch.optim.AdamW on identical inputs — params AND buffers.
    def run(cls, steps: int = 300):
        torch.manual_seed(1)
        p = torch.randn(6, 8).requires_grad_()
        o = cls([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
        traj, bufs = [], []
        for t in range(steps):
            p.grad = torch.randn_like(p) * (0.5 + 0.1 * (t % 5))
            o.step()
            traj.append(p.detach().clone())
            bufs.append((o.state[p]["exp_avg"].clone(), o.state[p]["exp_avg_sq"].clone()))
        return traj, bufs

    mine, b_mine = run(AdamW)
    ref, b_ref = run(torch.optim.AdamW)
    maxrel = max((a - b).norm().item() / b.norm().item() for a, b in zip(mine, ref))
    check("params track torch.optim.AdamW (fp noise only)", maxrel < 1e-7, f"max rel {maxrel:.2e}")
    check("m/v buffers bit-identical to torch over 300 steps",
          all(torch.equal(m, r) and torch.equal(v, rv)
              for (m, v), (r, rv) in zip(b_mine, b_ref)))
    # torch 2.14 source: denom = sqrt(v)/sqrt(1-b2^t)+eps, step = lr/(1-b1^t) —
    # i.e. torch ALREADY implements the paper's both-moments bias correction.

    # B3: g_norm semantics — enters the moments, not the t=1 normalized step.
    def run_gnorm(g_norm: float, steps: int = 30):
        torch.manual_seed(3)
        p = torch.randn(5, 5).requires_grad_()
        o = AdamW([p], lr=LR, g_norm=g_norm)
        traj, first = [], None
        for t in range(steps):
            p.grad = torch.randn_like(p) * 0.1  # ||g|| ~ 0.05-0.2 << g_norm
            o.step()
            if t == 0:
                first = o.state[p]["exp_avg"].clone()  # m right after t=1
            traj.append(p.detach().clone())
        return traj, first

    clamped, m1 = run_gnorm(10.0)
    plain, _ = run_gnorm(0.0)
    diff = max((a - b).norm().item() for a, b in zip(clamped, plain))
    check("g_norm >> ||g|| reshapes trajectory vs g_norm=0", diff > 1e-4, f"max diff {diff:.3e}")
    check("raw |m| after t=1 == (1-b1)*g_norm (g_hat ends up with norm g_norm)",
          abs(m1.norm().item() - (1 - B1) * 10.0) < 1e-3,
          f"|m|={m1.norm().item():.4f} expect {(1 - B1) * 10.0}")

    # B4: closure, frozen params, multi-group lr.
    pA = torch.randn(3).requires_grad_()
    pB = torch.randn(3).requires_grad_()
    opt3 = AdamW([{"params": [pA], "lr": 1e-3}, {"params": [pB], "lr": 2e-3}])
    pA.grad = torch.ones(3)
    pB.grad = None
    before = pB.clone()
    loss = opt3.step(lambda: torch.tensor(42.0))
    check("closure returns loss; frozen param untouched; per-group lr",
          loss.item() == 42.0 and torch.equal(pB, before))

    # B5: invalid hyperparameters raise ValueError.
    bad = [dict(lr=-1.0), dict(betas=(1.5, 0.9)), dict(betas=(0.9, 1.1)),
           dict(eps=-1.0), dict(weight_decay=-1.0), dict(g_norm=-0.1)]
    ok = all(_raises_valueerror(b) for b in bad)
    check("invalid hyperparameters raise ValueError", ok)


def _raises_valueerror(kwargs: dict) -> bool:
    try:
        AdamW([torch.randn(2)], **kwargs)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------- C. NadamW
def test_nadamw() -> None:
    section("C. Custom NadamW (paper Appendix A, Algorithm 2)")
    b1o, b2, lr, eps, wd = 0.9, B2, LR, EPS, WD

    # C1: hand-computed t=1: m = (1-b1)g, m_tilde = b1*m + (1-b1)g = (1-b1^2)g,
    #     m_hat = m_tilde/(1-b1) = (1+b1)g; v_hat = g^2 (corrections cancel).
    #     p1 = p0(1-lr*wd) - lr*(1+b1)*g/(|g|+eps).
    p0 = torch.tensor([[1.0, -2.0, 0.5]])
    g = torch.tensor([[0.3, -0.7, 0.0]])
    p = p0.clone().requires_grad_()
    opt = NadamW([p], lr=lr, betas=(b1o, b2), eps=eps, weight_decay=wd)
    p.grad = g
    opt.step()
    expect = p0 * (1 - lr * wd) - lr * (1 + b1o) * g / (g.abs() + eps)
    check("t=1 update == hand-computed paper formula (m_hat=(1+b1)g)",
          bool(torch.allclose(p.detach(), expect, rtol=1e-6, atol=1e-12)), f"p={p.tolist()}")
    check("state layout {step, exp_avg, exp_avg_sq}",
          set(opt.state[p]) == {"step", "exp_avg", "exp_avg_sq"} and opt.state[p]["step"].item() == 1.0)

    # C2: hand-computed t=2 from the paper's algebra (fp64 reference).
    g2 = torch.tensor([[0.4, -0.1, 0.2]])
    p_prev = p.detach().clone()  # t=1 parameters, before the 2nd step
    p.grad = g2
    opt.step()
    m = torch.tensor([[(1 - b1o) * 0.3, (1 - b1o) * -0.7, 0.0]])
    v = torch.tensor([[(1 - b2) * 0.09, (1 - b2) * 0.49, 0.0]])
    m = b1o * m + (1 - b1o) * g2           # EMA at t=2
    v = b2 * v + (1 - b2) * g2 ** 2
    m_tilde = b1o * m + (1 - b1o) * g2     # lookahead
    m_hat = m_tilde / (1 - b1o ** 2)
    v_hat = v / (1 - b2 ** 2)
    expect2 = p_prev * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps)
    check("t=2 update == hand-computed paper algebra",
          bool(torch.allclose(p.detach(), expect2, rtol=1e-6, atol=1e-10)),
          f"max|d|={(p.detach()-expect2).abs().max().item():.2e}")

    # C3: beta1=0 degeneracy — lookahead collapses to plain momentum, so
    #     NadamW(b1=0) must be bit-identical to our AdamW(b1=0).
    def run_zero_b1(cls):
        torch.manual_seed(7)
        p = torch.randn(4, 5).requires_grad_()
        o = cls([p], lr=lr, betas=(0.0, b2), eps=eps, weight_decay=wd)
        bufs = []
        for t in range(30):
            p.grad = torch.randn_like(p) * (0.5 + 0.1 * (t % 3))
            o.step()
            bufs.append((p.detach().clone(), o.state[p]["exp_avg"].clone(),
                         o.state[p]["exp_avg_sq"].clone()))
        return bufs
    nad, ada = run_zero_b1(NadamW), run_zero_b1(AdamW)
    check("b1=0: params+m/v bit-identical to our AdamW (lookahead degenerates)",
          all(torch.equal(n[0], a[0]) and torch.equal(n[1], a[1]) and torch.equal(n[2], a[2])
              for n, a in zip(nad, ada)))

    # C4: g_norm reaches the lookahead via g_hat (t=1 numerator = (1+b1)*g_hat).
    g3 = torch.tensor([[0.1, 0.1]])  # ||g|| ~ 0.141
    p3 = torch.tensor([[1.0, -2.0]]).requires_grad_()
    opt4 = NadamW([p3], lr=lr, betas=(b1o, b2), eps=eps, weight_decay=wd, g_norm=10.0)
    p3.grad = g3
    opt4.step()
    scale = max(1.0, 10.0 / g3.norm().item())
    g_hat = g3 * scale
    expect4 = p0[:, :2] * (1 - lr * wd) - lr * (1 + b1o) * g_hat / (g_hat.abs() + eps)
    check("g_norm feeds m_tilde through g_hat (t=1 formula with clamped g)",
          bool(torch.allclose(p3.detach(), expect4, rtol=1e-6, atol=1e-10)))

    # C5: guards + closure + frozen params.
    bad = [dict(lr=-1.0), dict(betas=(1.1, 0.98)), dict(betas=(0.9, 1.0)),
           dict(eps=-1.0), dict(weight_decay=-1.0), dict(g_norm=-1.0)]
    ok = all(_raises_valueerror_opt(NadamW, b) for b in bad)
    check("invalid hyperparameters raise ValueError", ok)
    pA = torch.randn(2).requires_grad_(); pB = torch.randn(2).requires_grad_()
    opt5 = NadamW([pA, pB])
    pA.grad = torch.ones(2); pB.grad = None
    before = pB.clone()
    check("closure/frozen params", opt5.step(lambda: torch.tensor(1.0)).item() == 1.0
          and torch.equal(pB, before))


def _raises_valueerror_opt(cls, kwargs: dict) -> bool:
    try:
        cls([torch.randn(2)], **kwargs)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------- D. Lion
def test_lion() -> None:
    section("D. Custom Lion (paper Appendix A, Algorithm 3)")
    b1o, b2, lr, wd = 0.9, 0.98, 3e-4, 0.1

    # D1: t=1. m=0 -> m_hat = (1-b1)*g; sign(m_hat)=sign(g); wd decoupled.
    #     p1 = p0(1-lr*wd) - lr*sign(g);  state m1 = (1-b2)*g.
    p0 = torch.tensor([[1.0, -2.0, 0.5]])
    g1 = torch.tensor([[0.3, -0.7, 0.0]])
    p = p0.clone().requires_grad_()
    opt = Lion([p], lr=lr, betas=(b1o, b2), weight_decay=wd)
    p.grad = g1
    opt.step()
    expect = p0 * (1 - lr * wd) - lr * torch.sign(g1)
    check("t=1: p = p0(1-lr*wd) - lr*sign(g); sign(0)=0 element gets decay only",
          bool(torch.allclose(p.detach(), expect, rtol=1e-6, atol=1e-12)), f"p={p.tolist()}")
    state = opt.state[p]
    check("state = {step, exp_avg} ONLY (memory claim: no v buffer)",
          set(state) == {"step", "exp_avg"}, str(sorted(state)))
    check("t=1: state m = (1-b2)*g", bool(torch.allclose(state["exp_avg"], (1 - b2) * g1, rtol=1e-6, atol=1e-12)))

    # D2: t=2 with a new gradient — the ordering+beta-roles check.
    #     m_hat2 = b1*(old m = 0.02 g1) + (1-b1)*g2   [b1 mixes the SIGN input]
    #     m2     = b2*(old m = 0.02 g1) + (1-b2)*g2   [b2 advances the STATE]
    g2 = torch.tensor([[0.4, -0.1, 0.2]])
    old_m = (1 - b2) * g1
    m_hat2 = b1o * old_m + (1 - b1o) * g2
    m2 = b2 * old_m + (1 - b2) * g2
    p_prev = p.detach().clone()
    p.grad = g2
    opt.step()
    expect2 = p_prev * (1 - lr * wd) - lr * torch.sign(m_hat2)
    check("t=2: sign input uses b1-mix of OLD state; state advanced with b2",
          bool(torch.allclose(p.detach(), expect2, rtol=1e-6, atol=1e-12)),
          f"p={p.tolist()}")
    check("t=2: state m2 = b2*old_m + (1-b2)*g2",
          bool(torch.allclose(state["exp_avg"], m2, rtol=1e-6, atol=1e-12)))
    # NOTE: swapping the beta roles or updating m before building m_hat would
    # fail both checks above — this is the classic Lion bug.

    # D3: g_norm — sign(g_hat)=sign(g) makes t=1 step g_norm-invariant, but
    #     the clamped gradient enters the state, so later steps diverge.
    def run_lion(g_norm: float, steps: int = 5):
        torch.manual_seed(11)
        p = torch.randn(3, 4).requires_grad_()
        o = Lion([p], lr=lr, betas=(b1o, b2), weight_decay=wd, g_norm=g_norm)
        traj = []
        for t in range(steps):
            p.grad = torch.randn_like(p) * 0.1  # ||g|| << g_norm=10
            o.step()
            traj.append(p.detach().clone())
        return traj
    clamped, plain = run_lion(10.0), run_lion(0.0)
    check("g_norm=10: t=1 step identical (sign(g_hat)=sign(g))",
          torch.equal(clamped[0], plain[0]))
    later = max((a - b).norm().item() for a, b in zip(clamped[1:], plain[1:]))
    check("  but later steps diverge (clamped grad reshapes the state)",
          later > 1e-6, f"max diff after t=1: {later:.3e}")

    # D4: guards + closure/frozen params.
    bad = [dict(lr=-1.0), dict(betas=(1.1, 0.98)), dict(betas=(0.9, 1.0)),
           dict(weight_decay=-1.0), dict(g_norm=-1.0)]
    ok = all(_raises_valueerror_opt(Lion, b) for b in bad)
    check("invalid hyperparameters raise ValueError", ok)
    pA = torch.randn(2).requires_grad_(); pB = torch.randn(2).requires_grad_()
    opt5 = Lion([pA, pB])
    pA.grad = torch.ones(2); pB.grad = None
    before = pB.clone()
    check("closure/frozen params", opt5.step(lambda: torch.tensor(1.0)).item() == 1.0
          and torch.equal(pB, before))


# ---------------------------------------------------------------- E. Muon
def test_muon() -> None:
    section("E. Newton-Schulz + custom Muon (paper Appendix A, Algorithm 8)")
    torch.manual_seed(21)

    # E1: Newton-Schulz helper invariants.
    u0 = torch.randn(6, 4)  # wide: right-Gram branch
    # steps=5 (paper's convention) is an APPROXIMATE polar factor: smallest
    # singular values only reach ~0.98-0.99 -> loose bounds here.
    u = _newton_schulz(u0, steps=5)
    gram_err5 = (u.T @ u - torch.eye(4)).norm().item()
    check("NS(5): u^T u ~ I (approx., per paper's 5-step convention)",
          gram_err5 < 5e-2, f"||u^T u - I||_F={gram_err5:.2e}")
    p = torch.linalg.svd(u0, full_matrices=False)
    polar = p.U @ p.Vh
    d5 = (u - polar).norm().item() / polar.norm().item()
    check("NS(5): u close to exact polar factor", d5 < 1e-2, f"rel diff {d5:.2e}")
    # With more steps the iteration converges quadratically (singular values
    # -> 1, i.e. exact polar): tight bounds prove the contraction claim.
    u20 = _newton_schulz(u0, steps=20)
    gram_err20 = (u20.T @ u20 - torch.eye(4)).norm().item()
    d20 = (u20 - polar).norm().item() / polar.norm().item()
    check("NS(20): converges to exact polar (quadratic contraction)",
          gram_err20 < 1e-4 and d20 < 1e-4, f"||u^T u - I||={gram_err20:.2e} rel {d20:.2e}")
    u1 = torch.randn(3, 7)  # tall: left-Gram branch
    u1ns = _newton_schulz(u1, steps=8)
    p1 = torch.linalg.svd(u1, full_matrices=False)
    d1 = (u1ns - p1.U @ p1.Vh).norm().item() / (p1.U @ p1.Vh).norm().item()
    check("NS(8) tall matrix ~ polar (left-Gram branch)", d1 < 2e-3, f"rel diff {d1:.2e}")
    z = torch.zeros(4, 3)
    check("NS on zero matrix is safe (no nan)",
          bool(torch.isfinite(_newton_schulz(z)).all()))
    ns0 = _newton_schulz(u0, steps=0)
    check("ns_steps=0 = pure Frobenius normalization",
          bool(torch.allclose(ns0, u0 / (u0.norm() + 1e-5), rtol=1e-6, atol=1e-8)))

    # E2: role partition — 2D -> muon; 1D and adam_ids-2D -> adam.
    wm = torch.randn(4, 3).requires_grad_()  # matrix -> muon branch
    we = torch.randn(3, 4).requires_grad_()  # 2D but embedding -> adam branch
    wn = torch.randn(2).requires_grad_()     # 1D (LayerNorm) -> adam branch
    opt = Muon([wm, we, wn], lr=8e-3, lr_adam=2.4e-3, adam_ids={id(we)})
    check("partition: [muon] vs [adam] groups with per-branch lrs",
          [g["branch"] for g in opt.param_groups] == ["muon", "adam"]
          and tuple(g["lr"] for g in opt.param_groups) == (8e-3, 2.4e-3)
          and opt.param_groups[0]["params"] == [wm]
          and opt.param_groups[1]["params"] == [we, wn],
          f"groups={[[p.shape for p in g['params']] for g in opt.param_groups]}")

    # E3: hand-computed muon-branch t=1 with ns_steps=0.
    #     m = g_hat;  u0 = (1+beta)*g_hat;  NS-0 -> u0/(||u0||+eps_muon);
    #     s = sqrt(max(1, rows/cols)) = 1 for (2,3);  p1 = p0(1-lr*wd) - lr*u*s.
    w = torch.randn(2, 3).requires_grad_()
    w_init = w.detach().clone()
    g = torch.tensor([[0.3, -0.4, 0.2], [-0.1, 0.5, 0.0]])
    optm = Muon([w], lr=8e-3, momentum=0.95, weight_decay=0.01, ns_steps=0)
    w.grad = g
    optm.step()
    u0 = 1.95 * g  # (1 + beta) * g_hat, g_norm=0 -> g_hat = g
    u_norm = u0 / (u0.norm() + 1e-5)
    s = math.sqrt(max(1.0, 2 / 3))
    expect = w_init * (1 - 8e-3 * 0.01) - 8e-3 * s * u_norm
    check("t=1 muon step == hand formula (m=g_hat, u=(1+b)g_hat, NS-0, s=1)",
          bool(torch.allclose(w.detach(), expect, rtol=1e-5, atol=1e-9)),
          f"max|d|={(w.detach()-expect).abs().max().item():.2e}")
    check("muon state: momentum buffer == g_hat (no (1-b) factor)",
          torch.equal(optm.state[w]["momentum"], g))
    check("state layout {step, momentum} on muon branch",
          set(optm.state[w]) == {"step", "momentum"})

    # E4: adam branch inside Muon must be bit-identical to our AdamW.
    def run_adam_branch():
        torch.manual_seed(31)
        pa = torch.randn(4, 5).requires_grad_()
        oa = Muon([pa], lr=8e-3, lr_adam=2.4e-3, adam_ids={id(pa)})  # all -> adam
        traj = []
        for t in range(30):
            pa.grad = torch.randn_like(pa) * (0.5 + 0.1 * (t % 3))
            oa.step()
            traj.append(pa.detach().clone())
        return traj

    def run_plain_adamw():
        torch.manual_seed(31)
        pb = torch.randn(4, 5).requires_grad_()
        ob = AdamW([pb], lr=2.4e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
        traj = []
        for t in range(30):
            pb.grad = torch.randn_like(pb) * (0.5 + 0.1 * (t % 3))
            ob.step()
            traj.append(pb.detach().clone())
        return traj

    m = run_adam_branch()
    a = run_plain_adamw()
    check("Muon-adam branch == our AdamW (bit-identical params, 30 steps)",
          all(torch.equal(x, y) for x, y in zip(m, a)))

    # E5: end-to-end tiny transformer with make_optimizer("muon", ...).
    V = 97
    cfgm = TransformerConfig(d_model=64, n_ctx=16, n_vocab=V, n_layers=2,
                             n_heads=4, d_ff=128, ffn_variant=1)
    model = Transformer(cfgm)
    optmuon = make_optimizer("muon", model, TrainConfig())
    n_muon = len(optmuon.param_groups[0]["params"])
    n_adam = len(optmuon.param_groups[1]["params"])
    check("factory split: 8 matrices muon / 12 adam (2 emb + 10 layernorm w/b)",
          n_muon == 8 and n_adam == 12, f"muon={n_muon} adam={n_adam}")
    check("muon group lrs from calibration constants",
          optmuon.param_groups[0]["lr"] == 8e-3 and optmuon.param_groups[1]["lr"] == 2.4e-3)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    losses = []
    for _ in range(12):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        losses.append(loss.item())
        loss.backward()
        optmuon.step()
        optmuon.zero_grad(set_to_none=True)
    check("trained 12 steps: loss decreased, grads finite",
          losses[-1] < losses[0] - 0.05
          and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    muon_p = optmuon.param_groups[0]["params"][0]
    adam_p = optmuon.param_groups[1]["params"][0]
    check("per-branch state present",
          "momentum" in optmuon.state[muon_p] and "exp_avg" in optmuon.state[adam_p])

    # E6: state_dict round-trip.
    sd = optmuon.state_dict()
    model2 = Transformer(cfgm)
    opt2 = make_optimizer("muon", model2, TrainConfig())
    opt2.load_state_dict(sd)
    x2, y2 = x.clone(), y.clone()
    logits = model2(x2)
    loss2 = F.cross_entropy(logits.reshape(-1, V), y2.reshape(-1))
    loss2.backward()
    opt2.step()
    check("state_dict round-trip: step runs after load", bool(torch.isfinite(logits).all()))


# ---------------------------------------------------------------- F. factory
def test_factory() -> None:
    section("F. make_optimizer factory + constraints")
    V = 61
    cfgm = TransformerConfig(d_model=32, n_ctx=8, n_vocab=V, n_layers=1,
                             n_heads=2, d_ff=48, ffn_variant=1)
    model = Transformer(cfgm)
    tcfg = TrainConfig()  # lr=8e-4, betas=(0.9, 0.98), weight_decay=0.01

    for name, cls in PART2_OPTIMIZERS.items():
        opt = make_optimizer(name, model, tcfg)
        check(f"'{name}' -> our {cls.__name__} (subclass of torch.optim.Optimizer)",
              isinstance(opt, cls) and issubclass(cls, torch.optim.Optimizer))
    check("'adamw' is OUR AdamW, not torch's", not isinstance(
        make_optimizer("adamw", model, tcfg), torch.optim.AdamW))
    check("adam family lr == cfg.lr (part 1 baseline)",
          make_optimizer("adamw", model, tcfg).param_groups[0]["lr"] == tcfg.lr
          and make_optimizer("nadamw", model, tcfg).param_groups[0]["lr"] == tcfg.lr)
    check("lion lr == LION_LR calibration constant",
          make_optimizer("lion", model, tcfg).param_groups[0]["lr"] == LION_LR)
    try:
        make_optimizer("sofia", model, tcfg)
    except ValueError:
        check("unknown name raises ValueError", True)
    else:
        check("unknown name raises ValueError", False)


# ---------------------------------------------------------------- G. plug-in
def test_train_model_plugin() -> None:
    """The real integration seam: train_model(optimizer=make_optimizer(...))
    with the manual per-group LR schedule — the exact path part 2 runs will
    take. Each of the four optimizers does a tiny train -> val -> ckpt cycle."""
    section("G. train_model plug-in (optimizer= + manual schedule)")
    import tempfile

    from torch.utils.data import DataLoader, Dataset

    from src.train import TrainConfig, train_model

    class TinyBatch:
        def __init__(self, input_ids, labels, attention_mask):
            self.input_ids = input_ids
            self.labels = labels
            self.attention_mask = attention_mask

    class TinyDS(Dataset):
        def __init__(self, n, V, T):
            self.n, self.V, self.T = n, V, T

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return dict(
                input_ids=torch.randint(0, self.V, (self.T,)),
                labels=torch.randint(0, self.V, (self.T,)),
                attention_mask=torch.ones(self.T),
            )

    def collate(samples):
        return TinyBatch(
            torch.stack([s["input_ids"] for s in samples]),
            torch.stack([s["labels"] for s in samples]),
            torch.stack([s["attention_mask"] for s in samples]),
        )

    V = 53
    cfgm = TransformerConfig(d_model=32, n_ctx=8, n_vocab=V, n_layers=1,
                             n_heads=2, d_ff=48, ffn_variant=1)
    train_loader = DataLoader(TinyDS(n=8, V=V, T=8), batch_size=4,
                              collate_fn=collate)
    val_loader = DataLoader(TinyDS(n=4, V=V, T=8), batch_size=4, collate_fn=collate)

    for i, name in enumerate(PART2_OPTIMIZERS):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(100 + i)
            model = Transformer(cfgm)
            opt = make_optimizer(name, model, TrainConfig())
            tcfg = TrainConfig(
                max_tokens=2048, val_every_tokens=1024, warmup_tokens=256,
                batch_log_every=3, ckpt_dir=f"{tmp}/ckpt", run_name=f"plug-{name}",
            )
            steps, tokens = train_model(model, train_loader, val_loader, tcfg,
                                        device="cpu", optimizer=opt)
            lrs = [g["lr"] for g in opt.param_groups]
            scales = [g["lr"] / g["initial_lr"] for g in opt.param_groups]
            finite = all(torch.isfinite(p).all() for p in model.parameters())
            check(f"'{name}': trained {steps} steps, {tokens} tokens, "
                  f"ckpt saved, params finite, lr scaled {len(set(scales)) == 1}",
                  steps > 0 and tokens == 2048 and finite and len(set(scales)) == 1
                  and all(s <= 1.0 + 1e-9 for s in scales),
                  f"lr={['%.2e' % s for s in scales]}")
            ok = any(Path(tcfg.ckpt_dir).glob("plug-*.pt"))
            check(f"'{name}': checkpoint files written", ok)


# ---------------------------------------------------------------- H. data
def test_part2_data(tokenizer) -> None:
    """Part 2 data pipeline (purely additive: nothing imported from part 1)."""
    section("H. part2 data pipeline (doc-grouped splits, NTP samples)")

    # synthetic corpus: 20 docs x 8 author-variants, incl. empty + long texts
    authors = ["human_chunk1", "human_chunk2"] + [f"llm_{i}" for i in range(6)]
    texts = ["the cat sits on the mat", "", "i like green tea and biscuits",
             "the train leaves at noon sharp every single day of the week"]
    rows = []
    for d in range(20):
        for a in authors:
            rows.append({"doc_id": f"acad_{d:04d}@{a}", "text": texts[(d + len(a)) % len(texts)]})
    check("corpus rows built", len(rows) == 160, f"{len(rows)}")

    # H1: doc-grouped split, deterministic, no straddling docs
    tr, va, te = split_rows_by_doc(rows, seed=42)
    n_docs = lambda rs: len({r["doc_id"].split("@")[0] for r in rs})
    check("split ratios 90/5/5 by DOC (18/1/1 docs)",
          n_docs(tr) == 18 and n_docs(va) == 1 and n_docs(te) == 1,
          f"docs {n_docs(tr)}/{n_docs(va)}/{n_docs(te)} rows {len(tr)}/{len(va)}/{len(te)}")
    tr2, va2, te2 = split_rows_by_doc(rows, seed=42)
    check("deterministic under same seed",
          [r["doc_id"] for r in tr] == [r["doc_id"] for r in tr2])
    union = {r["doc_id"] for r in tr} | {r["doc_id"] for r in va} | {r["doc_id"] for r in te}
    check("all rows accounted for", len(union) == len(rows) and len(tr) + len(va) + len(te) == 160)

    # H2: sample construction: [bos]+text+[eos], shifted labels, -100 only at end
    samples = list(make_lm_samples(tr, tokenizer, max_len=4))
    s = samples[0]
    check("label shift: seq[1:] + [-100]", s["labels"] == s["ids"][1:] + [-100])
    nonpad = [i for i, l in enumerate(s["labels"]) if l != -100]
    check("-100 appears ONLY on the final position",
          nonpad == list(range(len(s["ids"]) - 1)))
    ml = max(len(x["ids"]) for x in samples)
    check("truncation at max_len (never exceeds)", ml <= 4, f"max len {ml}")
    bos = tokenizer.bos_token_id
    check("every sample starts with <bos>", all(x["ids"][0] == bos for x in samples))

    # H3: dataset materialization + token budget
    ds = LMDataset(tr, tokenizer, max_len=4)
    raw_sum = sum(len(x["ids"]) for x in samples)
    check("total_real_tokens == sum of sample lengths",
          ds.total_real_tokens == raw_sum, f"{ds.total_real_tokens} vs {raw_sum}")
    check("dataset lengths align", len(ds) == len(samples))

    # H4: collate + token counting (duck-typed against src.train.count_real_tokens)
    batch = collate_lm_batch([ds[i] for i in range(6)], pad_id=tokenizer.pad_token_id)
    B = batch.input_ids.shape
    check("collate shapes (B, T) aligned",
          batch.input_ids.shape == batch.labels.shape == batch.attention_mask.shape
          and batch.input_ids.shape[0] == 6, str(B))
    check("mask binary", set(batch.attention_mask.unique().tolist()) <= {0, 1})
    check("count_real_tokens == mask sum",
          count_real_tokens(batch) == int(batch.attention_mask.sum()),
          f"{count_real_tokens(batch)}")
    from src.train import count_real_tokens as train_count
    check("matches src.train.count_real_tokens (duck-typed)",
          train_count(batch) == count_real_tokens(batch))

    # H5: loader end-to-end (shuffle stream via generator, like part 1)
    gen = torch.Generator().manual_seed(7)
    loader = make_dataloader(tr, tokenizer, max_len=4, batch_size=8, shuffle=True,
                             generator=gen, what="smoke-train")
    first = next(iter(loader))
    check("loader yields shuffled Part2Batch", isinstance(first, Part2Batch)
          and tuple(first.input_ids.shape)[0] == 8)  # drop_last=True
    gen2 = torch.Generator().manual_seed(7)
    loader2 = make_dataloader(tr, tokenizer, max_len=4, batch_size=8, shuffle=True,
                              generator=gen2, what="smoke-train")
    check("same generator => same shuffle stream (fairness across optimizers)",
          torch.equal(first.input_ids, next(iter(loader2)).input_ids))


# ---------------------------------------------------------------- J. eval
def test_decode_and_eval(tokenizer) -> None:
    """Part 3 greedy decoder (part-2 consumer) + continuation-BLEU pieces."""
    section("J. greedy decode + continuation BLEU (part 3 decoder)")

    class Stub(torch.nn.Module):
        """Deterministic logits: eos bias > 0 => always emit eos; < 0 => never."""

        def __init__(self, V: int, eos: int, eos_bias: float):
            super().__init__()
            self.V, self.eos, self.bias = V, eos, eos_bias

        def forward(self, x, attn_mask=None):
            B, T = x.shape
            logits = torch.zeros(B, T, self.V)
            logits[:, :, self.eos] = self.bias
            return logits

    pad, eos, V = 0, 7, 16
    prompts = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])  # (2, 5) padded
    no_eos = Stub(V, eos, -5.0)  # never terminates
    gen = greedy_decode(no_eos, prompts, max_new=4, eos_id=eos, pad_id=pad)
    check("stub-no-eos: generates exactly max_new tokens, batched",
          gen.shape == (2, 4) and bool(torch.isfinite(gen).all()), str(tuple(gen.shape)))
    # determinism: decoded ids == manual argmax rollout on the same model
    ids = prompts.clone()
    for _ in range(4):
        ids = torch.cat([ids, no_eos(ids, (ids != pad).long())[:, -1, :].argmax(-1)[:, None]], dim=1)
    check("greedy == manual argmax rollout (selection logic)",
          torch.equal(gen, ids[:, 5:]))
    always_eos = Stub(V, eos, 5.0)  # terminates after 1 step
    gen2 = greedy_decode(always_eos, prompts, max_new=8, eos_id=eos, pad_id=pad)
    check("stub-eos: stops at eos; finished rows pad-padded",
          gen2.shape == (2, 1) and bool((gen2 == eos).all()), str(gen2.tolist()))

    # human-pair extraction from the doc-grouped structure
    authors = ["human_chunk1", "human_chunk2"] + [f"llm_{i}" for i in range(6)]
    rows = []
    for d in range(20):
        for a in authors:
            rows.append({"doc_id": f"acad_{d:04d}@{a}",
                         "text": f"doc {d} chunk text {d} {a}"})
    tr, va, te = split_rows_by_doc(rows, seed=9)
    pairs = human_pairs(te, tokenizer, max_prompt_tokens=8, max_new=4)
    check("human_pairs: one chunk1->chunk2 pair per test doc (LLM rows excluded)",
          len(pairs) == 1 and pairs[0]["doc"] == te[0]["doc_id"].split("@")[0],
          f"{len(pairs)} pairs")
    check("prompts truncated <= max_prompt; references non-empty",
          all(len(p["prompt_ids"]) <= 8 and p["reference"] for p in pairs))

    # real-corpus naming (regression guard): suffixes are 'chunk_1'/'chunk_2',
    # not 'human_chunk1' — this exact mismatch broke the first real run.
    rows2 = []
    for d in range(20):
        for a in ["chunk_1", "chunk_2", "Meta-Llama-3-8B", "Meta-Llama-3-70B",
                  "Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct",
                  "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]:
            rows2.append({"doc_id": f"acad_{d:04d}@{a}", "text": f"doc {d} {a} text"})
    tr2, va2, te2 = split_rows_by_doc(rows2, seed=3)
    pairs3 = human_pairs(te2, tokenizer, max_prompt_tokens=8, max_new=4)
    check("real naming: chunk_1/chunk_2 suffixes resolve to pairs",
          len(pairs3) == 1 and pairs3[0]["doc"] == te2[0]["doc_id"].split("@")[0],
          f"{len(pairs3)} pairs")

    # continuation BLEU on a random tiny model: score must be a sane float
    V2 = 53
    cfgm = TransformerConfig(d_model=32, n_ctx=16, n_vocab=V2, n_layers=1,
                             n_heads=2, d_ff=48, ffn_variant=1)
    model = Transformer(cfgm).eval()
    pairs2 = human_pairs(te, tokenizer, max_prompt_tokens=8, max_new=4)
    bleu = continuation_bleu(pairs2, model, tokenizer, "cpu", max_new=4, batch_size=2)
    check("continuation_bleu returns a score in [0, 100]", 0.0 <= bleu <= 100.0,
          f"bleu={bleu:.2f}")


# ---------------------------------------------------------------- I. glue
def test_main_glue(tokenizer) -> None:
    """main.py::run_optimizer on a tiny offline corpus — the real part 2
    driver path (tokenize-once budget, per-optimizer train, metrics json,
    then the BLEU eval pass + part-2 plots)."""
    section("I. main.py::run_optimizer offline glue (part 2 driver)")
    from argparse import Namespace

    from main import run_optimizer

    authors = ["human_chunk1", "human_chunk2"] + [f"llm_{i}" for i in range(6)]
    texts = ["the cat sits on the mat", "i like green tea", "trains leave at noon"]
    rows = [
        {"doc_id": f"acad_{d:04d}@{a}", "text": texts[(d + len(a)) % len(texts)]}
        for d in range(20) for a in authors
    ]
    tr, va, te = split_rows_by_doc(rows, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        args = Namespace(batch_size=4, max_len=16, max_tokens=1024, seed=7,
                         lr=8e-4, device="cpu", output=f"{tmp}/out")
        gen = torch.Generator().manual_seed(7)
        run_optimizer("adamw", args, tokenizer, (tr, va, te), gen, Path(tmp) / "eval")
        ckpts = list((Path(tmp) / "out" / "checkpoints").glob("part2-adamw_*.pt"))
        check("run_optimizer: checkpoints written", len(ckpts) >= 2,
              f"{len(ckpts)} ckpts")
        mpath = Path(tmp) / "eval" / "part2-adamw_metrics.json"
        assert mpath.exists()
        m = json.loads(mpath.read_text())
        check("run_optimizer: metrics json (budget, test ppl, lrs)",
              m["optimizer"] == "adamw" and m["budget_tokens"] == 1024
              and m["test_ppl"] > 0 and m["test_bleu"] is None,
              f"test_ppl={m['test_ppl']:.2f}")

        # the new eval pass: per-ckpt continuation BLEU -> json -> plots
        run_eval_pass("adamw", tokenizer, te, Path(tmp) / "eval",
                      f"{tmp}/out/checkpoints", "cpu", max_len=8,
                      bleu_max_new=4, batch_size=4)
        ej = json.loads((Path(tmp) / "eval" / "part2-adamw_eval.json").read_text())
        check("eval pass: per-ckpt points (tokens, val_ppl, bleu)",
              len(ej["points"]) >= 2 and all(0.0 <= p["bleu"] <= 100.0 and p["val_ppl"] > 0
                                             for p in ej["points"]),
              f"{len(ej['points'])} points, bleu={[round(p['bleu'], 2) for p in ej['points']]}")
        plots = plot_part2_curves(Path(tmp) / "eval")
        check("part-2 plots written (separate files from part 1)",
              len(plots) == 2 and all(p.exists() for p in plots),
              [p.name for p in plots])


def main() -> None:
    t_start = time.time()
    print(f"smoke (part 2) | device={DEVICE} | betas=({B1},{B2}) lr={LR} eps={EPS} wd={WD}")
    tok_path = Path(tempfile.mkdtemp()) / "tok.json"
    train_bpe_tokenizer(
        ["the cat sits on the mat", "i like green tea", "ねこがマットの上にすわっている"],
        vocab_size=300, save_path=tok_path,
    )
    tokenizer = load_tokenizer(tok_path)
    test_clamp_grad_norm()
    test_adamw()
    test_nadamw()
    test_lion()
    test_muon()
    test_factory()
    test_train_model_plugin()
    test_part2_data(tokenizer)
    test_decode_and_eval(tokenizer)
    test_main_glue(tokenizer)

    print(f"\n=== SUMMARY: {'ALL CHECKS PASSED' if not _failures else f'{len(_failures)} FAILED'} "
          f"({time.time() - t_start:.1f}s total) ===")
    for name in _failures:
        print(f"  FAILED: {name}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()