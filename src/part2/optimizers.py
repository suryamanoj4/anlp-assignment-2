"""Part 2: one optimizer from each of the first 4 categories of Table 1 in
"Fantastic Pretraining Optimizers and Where to Find Them" (arXiv:2509.02046).

Category -> pick (paper appendix section for the update rule):
  1. AdamW (baseline)            -> AdamW         (Appendix A, Algorithm 1)
  2. Variance-reduced AdamW      -> NadamW        (Appendix A, Algorithm 2)
  3. Memory-efficient optimizers -> Lion          (Appendix A, Algorithm 3)
  4. Matrix-based optimizers     -> Muon          (Appendix A, Algorithm 8)

Hard constraint (assignment): every class here subclasses
torch.optim.Optimizer directly; no other torch.optim.* module may be used in
part 2 code (incl. no torcht.optim.AdamW, no torch.optim.lr_scheduler.*).

Paper-wide convention (Appendix A preamble): gradients enter every update as
  g_hat = g * max(1, g_norm / ||g||_2)      (per-tensor norm clamp)
with g_norm a hyperparameter. g_norm = 0 makes it a no-op (max(1, 0) = 1),
which is what AdamW must use to stay numerically identical to the part 1
baseline (torch.optim.AdamW). All classes below expose g_norm for the
remaining optimizers; the calibration runs decide whether to enable it.

All hyperparameter defaults below follow part 1's TrainConfig where the
mathematics is shared (betas (0.9, 0.98), eps 1e-8, wd 0.01); the paper's own
tuned values at 130M-1.2B scale are noted in comments as calibration
starting points, NOT transcribed defaults.
"""

from __future__ import annotations

import math

import torch


def _clamp_grad_norm(g: torch.Tensor, g_norm: float) -> torch.Tensor:
    """Paper Appendix A preamble: g_hat = g * max(1, g_norm / ||g||_2)."""
    # TODO(0): per-tensor 2-norm clamp; g_norm <= 0 must be a no-op (return g).
    raise NotImplementedError


def _newton_schulz(u: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration: orthogonalize u toward the nearest O with
    ||O||_op = 1 (paper Section 2 formula, Algorithm 8 calls it with steps=5).
    """
    # TODO(1): normalize u by its Frobenius norm, then iterate
    #          u <- u @ (a*I + b*(u^T u) + c*(u^T u)^2)  (a,b,c configurable),
    #          matching the coefficient family used in the paper (a=3/2,b=-1/2
    #          classic, or the Moon set (3.4445, -4.775, 2.0315) that
    #          torch.optim.Muon defaults to). Consider what epsilon does in
    #          this loop (paper's eps_muon) and how many steps are enough.
    raise NotImplementedError


class AdamW(torch.optim.Optimizer):
    """Category 1: AdamW with decoupled weight decay (Appendix A, Algorithm 1).

    Update rule to implement in step():
        g_hat  = clamp(grad, g_norm)             (disabled at g_norm=0)
        m      = b1*m + (1-b1)*g_hat
        v      = b2*v + (1-b2)*g_hat^2
        m_hat  = m / (1 - b1^t),  v_hat = v / (1 - b2^t)
        p      = p - lr*m_hat/(sqrt(v_hat)+eps) - lr*wd*p      (decoupled)
    """

    def __init__(
        self,
        params,
        lr: float = 8e-4,          # part 1 baseline; paper's tuned eta 2e-3..8e-3 at 130M-1.2B
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,         # paper uses 1e-10 at scale
        weight_decay: float = 0.01,
        g_norm: float = 0.0,       # 0 = off (must match torch.optim.AdamW exactly then)
    ):
        # TODO(2): defaults dict (torch.optim.Optimizer contract!), super().__init__,
        #          sanity checks (betas in [0,1), lr >= 0 ...).
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        # TODO(3): loss = closure() if closure is not None
        # TODO(4): per param_group, per param with grad:
        #          state keys "step" (scalar long tensor, 1-based), "exp_avg", "exp_avg_sq"
        #          (lazy init zeros_like on first touch; do NOT detach on update —
        #          in-place ops on state are fine).
        # TODO(5): m/v EMA + bias correction; wd applied DECOUPLED (on p, not
        #          inside the sqrt denominator). Verify <your AdamW> ==
        #          torch.optim.AdamW on identical inputs when g_norm=0.
        raise NotImplementedError


class NadamW(torch.optim.Optimizer):
    """Category 2: Nesterov (variance-reduced) AdamW — Appendix A, Algorithm 2.

    Identical to AdamW except the numerator uses the lookahead
        m_tilde = b1*m_t + (1-b1)*g_hat          (note: uses the UPDATED m_t)
        then bias-corrects m_tilde and v as usual and divides.
    No extra state beyond AdamW's (m, v, step). The update reduces gradient
    variance in the sense that the Nesterov term anticipates the momentum
    direction one step ahead (paper Section 2 shows the delta in red).
    """

    def __init__(self, params, lr: float = 8e-4,
                 betas: tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-8, weight_decay: float = 0.01,
                 g_norm: float = 0.0):
        # TODO(6): same init contract as AdamW; paper's tuned betas at scale
        #          are often (0.95-0.98, 0.98) — calibration decides.
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        # TODO(7): reuse AdamW's m/v machinery, but per step:
        #          1) update m from g_hat
        #          2) build m_tilde = b1*m + (1-b1)*g_hat
        #          3) bias-correct BOTH m_tilde and v with their 1-beta^t factors
        #          4) p -= lr*m_tilde_hat/(sqrt(v_hat)+eps) - lr*wd*p
        #          Think about whether your sqrt uses v_hat or raw v (the
        #          paper's Algorithm 2 bias-corrects v; Section 2's gallery
        #          short-hand does not — Appendix A is canonical).
        raise NotImplementedError


class Lion(torch.optim.Optimizer):
    """Category 3: memory-efficient Lion — Appendix A, Algorithm 3.

    Single momentum state (no v): sign-based update.
        m_hat   = b1*m_{t-1} + (1-b1)*g_hat      (input to the sign; b1 mixes)
        m_next  = b2*m_{t-1} + (1-b2)*g_hat      (state advances with b2)
        p       = p - lr*sign(m_hat) - lr*wd*p   (decoupled wd, same form as AdamW)
    Note the two betas play different roles — read the appendix ordering
    carefully before coding. Lion is scale-sensitive: expect lr ~1e-4..3e-4
    vs AdamW's 8e-4; paper found its optimal wd ~0.6 at scale with their lambda
    convention (calibration will settle ours).
    """

    def __init__(self, params, lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.98),
                 weight_decay: float = 0.1,
                 g_norm: float = 0.0):
        # TODO(8): init contract + defaults; no eps needed (sign is discontinuous).
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        # TODO(9): state key "exp_avg" (momentum) + "step".
        #          Order matters: sign from the b1-mix of the PRE-update state,
        #          then advance state with b2. sign(0)=0 -> those entries get
        #          no gradient-driven move (only wd) — that is intended.
        raise NotImplementedError


class Muon(torch.optim.Optimizer):
    """Category 4: matrix-based Muon — Appendix A, Algorithm 8.

    Split by parameter role, NOT by shape alone:
      - LM head / embeddings / LayerNorm params: AdamW update with lr_adam.
      - transformer-layer matrices: momentum m = b*m_prev + g_hat  (no 1-b!),
        Nesterov combo u = b*m + g_hat, NewtonSchulz(u, steps=5), aspect
        scale s = sqrt(max(1, rows/cols)), then p -= lr*u*s - lr*wd*p.
    """

    def __init__(
        self,
        params,
        lr: float = 8e-3,             # paper's tuned eta_muon 4e-3..8e-3 at 300-520M
        lr_adam: float = 2.4e-3,      # eta_adam for the AdamW branch
        betas: tuple[float, float] = (0.9, 0.98),   # AdamW branch betas
        eps_adam: float = 1e-8,
        momentum: float = 0.95,       # paper uses 0.98 at scale
        eps_muon: float = 1e-5,       # enters the Newton-Schulz normalization
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        g_norm: float = 0.0,
    ):
        # TODO(10): the optimizer must know WHICH params take the Muon branch.
        #           Options (pick one, justify in the report):
        #           (a) accept param-groups-as-dicts (torch convention) with a
        #               per-group flag, partition at the call site where the
        #               model is known (tok_emb/lm_head/pos_emb/LayerNorms ->
        #               adam; attention/FFN weight matrices -> muon);
        #           (b) accept an iterable + a set of param ids to treat as
        #               AdamW. Either way p.ndim>=2 is NOT sufficient by itself
        #               (embeddings are 2D but belong to the AdamW branch).
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        # TODO(11): per group, per param:
        #           muon branch: state key "momentum"; m = b*m + g_hat;
        #                        u = b*m + g_hat;  u = ns(u, ns_steps);
        #                        u *= sqrt(max(1, rows/cols)); p -= lr*u; p -= lr*wd*p.
        #           adam branch: same as AdamW with lr_adam/betas/eps_adam.
        #           Sanity invariants to hold: u orthogonal-ish (u^T u ~ I),
        #           ||u||_F ~ 1 after NS, update norm ~ lr.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

PART2_OPTIMIZERS = {
    "adamw": AdamW,
    "nadamw": NadamW,
    "lion": Lion,
    "muon": Muon,
}


def make_optimizer(name: str, model, cfg) -> torch.optim.Optimizer:
    """Part 2 factory: `name` in PART2_OPTIMIZERS, hyperparams from cfg.

    Separate from src/train.build_optimizer (that one serves part 1 and its
    'adamw' must remain torch.optim.AdamW). Part 2 runs go through here so
    'adamw' unambiguously means OUR AdamW.
    """
    # TODO(12): per-optimizer lr overrides live in TrainConfig or a small
    #           dataclass (part 2 lr defaults; cfg.lr = part-1 baseline).
    #           Muon: construct the role split here (model is known).
    raise NotImplementedError