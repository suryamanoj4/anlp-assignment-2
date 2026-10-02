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
baseline (torch.optim.AdamW). All classes below expose g_norm. Decision
(round-1 calibration + the paper's own tuned configs, whose g_norm column
reads 0): g_norm stays OFF everywhere; the global grad clip (cfg.grad_clip,
as in their "Max Grad Norm") remains the shared stabilizer.

All hyperparameter defaults below follow part 1's TrainConfig where the
mathematics is shared (betas (0.9, 0.98), eps 1e-8); wd comes from cfg
(part-2 real runs pass --wd 0.1, the paper's tuned AdamW value — 0.01 was
part 1's easy-task value and churned the 2e-4 real run back to the unigram
floor under sustained lr). The paper's own tuned values at 130M-1.2B scale
are noted in comments as calibration starting points, NOT transcribed
defaults.
"""

from __future__ import annotations

import math

import torch


def _clamp_grad_norm(g: torch.Tensor, g_norm: float) -> torch.Tensor:
    """Paper Appendix A preamble: g_hat = g * max(1, g_norm / ||g||_2)."""
    if g_norm <= 0.0:
        return g  # paper convention: g_norm=0 -> max(1, 0) = 1 -> no-op
    nrm = g.norm()  # per-tensor 2-norm
    if nrm.item() == 0.0:
        return g  # zero gradient (dead unit); avoid 0 * inf -> nan
    return g * max(1.0, g_norm / nrm.item())


def _newton_schulz(
    u: torch.Tensor,
    steps: int = 5,
    coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    eps: float = 1e-5,
) -> torch.Tensor:
    """Newton-Schulz: orthogonalize u toward the nearest O with ||O||_op = 1.

    Spec: paper Section 2's NS(M) = M(aM + bM^T M + c(M^T M)^2) family;
    Algorithm 8 hard-codes the Moon set (3.4445, -4.775, 2.0315) with
    steps=5 — that is the default here. Normalizing first by ||u||_F + eps
    (this is where eps_muon lives) guarantees ||u||_op <= 1, the convergence
    condition for the polynomial map on the singular values
        sigma -> sigma * (a + b*sigma^2 + c*sigma^4).
    The Moon set drives every sigma into a small 2-cycle AROUND 1 (~0.7 to
    ~1.1) rather than exactly onto it — cheap near-full whitening in 5
    steps, the accepted Muon behavior (paper: "NS^(5)(M) ~
    argmax_{||O||_op=1} Tr(O^T M)"): tiny sigmas are pushed up FAST
    (sigma=0.1 -> ~0.34 in one step). The polar set (3/2, -1/2, 0) is the
    exact alternative — its fixed point is exactly sigma=1 (the polar
    factor argmax_{||O||_op<=1} Tr(O^T u)) — but it converges slowly from
    below (sigma=0.1 only reaches ~0.66 in 5 steps), under-sharpening small
    directions. steps=0 returns the mere normalization (used by tests to
    isolate the surrounding Muon mechanics).

    Cost note (report material): the Gram matrix is formed on the MINOR
    dimension — u(u^T u) == (u u^T)u by associativity, so form u^T u
    (cols x cols) when rows >= cols and u u^T (rows x rows) otherwise,
    multiplying on the left. That keeps the per-step NS cost a few percent of
    the forward+backward for these d_model sizes (the paper's "under 10% with
    proper implementation" point).
    """
    a, b, c = coefficients
    rows, cols = u.shape
    u = u / (u.norm() + eps)
    for _ in range(steps):
        if rows >= cols:
            gram = u.T @ u  # (cols, cols): u(u^T u) == u @ polynomial
            eye = torch.eye(cols, device=u.device, dtype=u.dtype)
            u = u @ (a * eye + b * gram + c * (gram @ gram))
        else:
            gram = u @ u.T  # (rows, rows): polynomial (u u^T) == left factor
            eye = torch.eye(rows, device=u.device, dtype=u.dtype)
            u = (a * eye + b * gram + c * (gram @ gram)) @ u
    return u


class AdamW(torch.optim.Optimizer):
    """Category 1: AdamW with decoupled weight decay — Appendix A, Algorithm 1.

    Part 2 drop-in replacement following the paper EXACTLY (both moments
    bias-corrected):
        g_hat  = clamp(grad, g_norm)                 (no-op at g_norm=0)
        m      = b1*m + (1-b1)*g_hat
        v      = b2*v + (1-b2)*g_hat^2
        m_hat  = m / (1 - b1^t),  v_hat = v / (1 - b2^t)
        p      = p - lr*m_hat/(sqrt(v_hat)+eps) - lr*wd*p      (decoupled)

    NOTE: part 1 keeps torch.optim.AdamW (src/train.build_optimizer) untouched;
    this class is the part 2 equivalent. torch 2.14's AdamW bias-corrects BOTH
    moments (denominator sqrt(v)/sqrt(1-b2^t) + eps) — i.e. torch already
    implements the paper's Algorithm 1 form, and our class reproduces it to
    fp32 op-order noise (measured < 1e-7 rel in scripts/smoke_part2.py). The
    older torch 1.x convention (uncorrected denominator) is NOT what 2.14 does.
    """

    def __init__(
        self,
        params,
        lr: float = 8e-4,          # part 1 baseline; paper's tuned eta 2e-3..8e-3 at 130M-1.2B
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,         # paper uses 1e-10 at scale
        weight_decay: float = 0.01,
        g_norm: float = 0.0,       # 0 = off (paper convention: max(1, 0) = 1)
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {b2}")
        if not 0.0 <= g_norm:
            raise ValueError(f"Invalid g_norm value: {g_norm} (0 disables the clamp)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, g_norm=g_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2 = group["betas"]
            eps, g_norm = group["eps"], group["g_norm"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue  # frozen param: nothing to update this step
                state = self.state[p]
                if len(state) == 0:
                    # Lazy per-param initialization on first touch.
                    state["step"] = torch.zeros((), dtype=torch.long, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)     # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v
                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = int(state["step"].item()) + 1  # 1-based step counter
                state["step"].fill_(t)

                # Paper Appendix A preamble: gradient norm clamp.
                g_hat = _clamp_grad_norm(grad, g_norm)

                # Decoupled weight decay: applied to p directly; never enters
                # m/v nor the denominator (that is what 'decoupled' means here).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # First/second moment EMAs of the clamped gradient. lerp_ is the fused
                # form of the paper's EMA (m + (g-m)*t); as a side effect it is
                # bit-identical to torch 2.14's adam, confirming empirically
                # that torch implements the same convention (smoke_part2 B).
                m.lerp_(g_hat, 1.0 - b1)
                v.mul_(b2).addcmul_(g_hat, g_hat, value=1.0 - b2)

                # Bias correction (paper Algorithm 1: BOTH moments).
                m_hat = m / (1.0 - b1 ** t)
                v_hat = v / (1.0 - b2 ** t)
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss


class NadamW(torch.optim.Optimizer):
    """Category 2: Nesterov (variance-reduced) AdamW — Appendix A, Algorithm 2.

    Spec: the assignment paper (Appendix A, Algorithm 2). Identical machinery
    to our AdamW (Algorithm 1) with ONE delta: the numerator is the Nesterov
    lookahead built from the UPDATED momentum,
        m_tilde = b1*m_t + (1-b1)*g_hat
    then both m_tilde and v are bias-corrected and divided as usual:
        w <- w - lr * m_tilde_hat / (sqrt(v_hat) + eps) - lr * wd * w.
    No extra state beyond AdamW's (m, v, step). The lookahead anticipates the
    direction the momentum is already carrying, reducing per-step gradient
    noise (paper Section 2 shows the delta in red).

    NOTE for the report: torch.optim.NAdam is a DIFFERENT variant (Dozat's
    momentum-decay schedule mu = b1*(1 - 0.5*0.96^(t*momentum_decay)); its
    nesterov flag was removed in torch 2.14). We implement the paper's form.
    """

    def __init__(self, params, lr: float = 8e-4,
                 betas: tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-8, weight_decay: float = 0.01,
                 g_norm: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {b2}")
        if not 0.0 <= g_norm:
            raise ValueError(f"Invalid g_norm value: {g_norm} (0 disables the clamp)")
        # paper's tuned betas at scale are often (0.95-0.98, 0.98); calibration
        # decides; these defaults keep the part-2 fairness constants.
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, g_norm=g_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2 = group["betas"]
            eps, g_norm = group["eps"], group["g_norm"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # Same lazy layout as AdamW: m = exp_avg, v = exp_avg_sq.
                    state["step"] = torch.zeros((), dtype=torch.long, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = int(state["step"].item()) + 1
                state["step"].fill_(t)

                # Paper preamble: gradient norm clamp.
                g_hat = _clamp_grad_norm(grad, g_norm)

                # Decoupled weight decay (same as Algorithm 1).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment EMAs — identical to Algorithm 1.
                m.lerp_(g_hat, 1.0 - b1)
                v.mul_(b2).addcmul_(g_hat, g_hat, value=1.0 - b2)

                # THE delta (Algorithm 2): lookahead re-mix of the UPDATED m.
                # Must be non-mutating: m feeds next step's EMA.
                m_tilde = b1 * m + (1.0 - b1) * g_hat

                # Bias correction of BOTH the lookahead numerator and v
                # (Appendix A; Section 2's shorthand omits it — appendix wins).
                m_hat = m_tilde / (1.0 - b1 ** t)
                v_hat = v / (1.0 - b2 ** t)
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss


class Lion(torch.optim.Optimizer):
    """Category 3: memory-efficient Lion — Appendix A, Algorithm 3.

    Spec: the assignment paper (Appendix A, Algorithm 3). Single momentum
    state (NO v buffer) + sign-projected update:
        m_hat   = b1*m_{t-1} + (1-b1)*g_hat      (sign input; OLD state)
        m_next  = b2*m_{t-1} + (1-b2)*g_hat      (state advances with b2)
        p       = p - lr*sign(m_hat) - lr*wd*p   (decoupled wd, as in Alg 1)
    Both lines read the PRE-update state — the paper's ordering is the spec.
    sign(0)=0: those coordinates get decay only, which is intended.
    Memory: one buffer per param (vs two for Adam-class) — the category claim.
    No eps: sign() is discontinuous and needs no stability constant.
    Scale-sensitive: sign updates move EVERY coordinate +-lr per step (a
    bounded random walk of std lr*sqrt(t) on noise-dominated coordinates),
    so wd must be large enough to counteract (paper's headline: "Lion's
    optimal weight decay ~=0.6 vs. AdamW's ~=0.1"; their 130M sweep has
    beta2 0.95 best, 0.98 measurably worse). Locked for the 1x runs: lr
    5e-5 — the paper's tuned 130M lr (1-2e-3 at their 0.52M-token batches)
    rescaled two ways that agree: their Lion ~ AdamW/4-8 ratio (2e-4/4-8),
    and batch scaling (their 8e-3 AdamW / ~33 batch ratio ~= 2.4e-4, which
    independently lands on our AdamW-family lock, so Lion 1-2e-3/33).
    """

    def __init__(self, params, lr: float = 5e-5,
                 betas: tuple[float, float] = (0.9, 0.95),
                 weight_decay: float = 0.6,
                 g_norm: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {b2}")
        if not 0.0 <= g_norm:
            raise ValueError(f"Invalid g_norm value: {g_norm} (0 disables the clamp)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, g_norm=g_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2 = group["betas"]
            g_norm = group["g_norm"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # The category's memory claim: exp_avg is the ONLY buffer.
                    # "step" kept for uniform state layout across optimizers;
                    # Lion has no bias correction, so t never enters the math.
                    state["step"] = torch.zeros((), dtype=torch.long, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                t = int(state["step"].item()) + 1
                state["step"].fill_(t)

                # Paper preamble: gradient norm clamp.
                g_hat = _clamp_grad_norm(grad, g_norm)

                # Sign input from the PRE-update state (Alg 3 ordering).
                m_hat = b1 * m + (1.0 - b1) * g_hat

                # Decoupled weight decay, then the sign-projected step.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(m_hat), alpha=-lr)

                # State advances with the OTHER beta — also from the old state.
                # In-place is fine and intended: m_hat already consumed old m.
                m.lerp_(g_hat, 1.0 - b2)
        return loss


class Muon(torch.optim.Optimizer):
    """Category 4: matrix-based Muon — Appendix A, Algorithm 8.

    Spec: the assignment paper (Appendix A, Algorithm 8). Split by parameter
    ROLE, not shape alone: LM head / embeddings / LayerNorm params get the
    AdamW update (with lr_adam); transformer-layer matrices get:
        m     = beta*m_prev + g_hat        (NO 1-beta factor — paper line)
        u     = beta*m + g_hat             (Nesterov combo)
        u     = NewtonSchulz(u, steps=5)
        s     = sqrt(max(1, rows/cols))    (aspect-ratio gain)
        p     = p - lr*u*s - lr*wd*p       (decoupled wd, as in Alg 1)
    Partitioning: constructor takes a plain iterable of params plus an
    OPTIONAL adam_ids set (param ids that belong to the AdamW branch even
    when 2D — embeddings/LM head). Everything 1D, or in adam_ids, is routed
    to the AdamW branch; everything else 2D goes to the Muon branch. The set
    is computed by make_optimizer/_embedding_param_ids where the model is
    known; the class itself never inspects modules. Internally the branches
    become two torch param groups (lr for muon, lr_adam for adam), which the
    shared LR schedule scales by the same multiplier.
    """

    def __init__(
        self,
        params,
        lr: float = 6e-3,             # locked: best early 0.1x curve at 8e-3, -25% horizon trim (paper range 4-8e-3)
        lr_adam: float = 5e-5,        # locked: half the adam family's 1e-4 (round-2 anti-drift; see constants)
        betas: tuple[float, float] = (0.9, 0.98),   # AdamW branch betas
        eps_adam: float = 1e-8,
        momentum: float = 0.95,       # paper uses 0.98 at scale
        eps_muon: float = 1e-5,       # enters the Newton-Schulz normalization
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),  # paper Alg 8 (Moon set)
        weight_decay: float = 0.01,
        g_norm: float = 0.0,
        adam_ids: set[int] | None = None,  # ids of 2D params that take AdamW
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_adam:
            raise ValueError(f"Invalid lr_adam value: {lr_adam}")
        if not 0.0 <= eps_adam:
            raise ValueError(f"Invalid eps_adam value: {eps_adam}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {b2}")
        if not 0.0 <= g_norm:
            raise ValueError(f"Invalid g_norm value: {g_norm} (0 disables the clamp)")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if len(ns_coefficients) != 3:
            raise ValueError(f"ns_coefficients must be (a, b, c): {ns_coefficients}")

        adam_ids = adam_ids or set()
        muon_params, adam_params = [], []
        for p in params:
            if id(p) in adam_ids or p.ndim < 2:
                adam_params.append(p)  # LayerNorm/1D, embeddings, tied LM head
            else:
                muon_params.append(p)  # attention/FFN weight matrices

        defaults = dict(lr=lr, lr_adam=lr_adam, betas=betas, eps_adam=eps_adam,
                        momentum=momentum, eps_muon=eps_muon, ns_steps=ns_steps,
                        ns_coefficients=ns_coefficients,
                        weight_decay=weight_decay, g_norm=g_norm)
        super().__init__([
            {"params": muon_params, "branch": "muon", "lr": lr},
            {"params": adam_params, "branch": "adam", "lr": lr_adam},
        ], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            branch = group["branch"]
            lr, wd = group["lr"], group["weight_decay"]
            g_norm = group["g_norm"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.long, device=p.device)
                    if branch == "muon":
                        state["momentum"] = torch.zeros_like(p)  # paper State: m
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                t = int(state["step"].item()) + 1
                state["step"].fill_(t)

                # Paper preamble: gradient norm clamp (both branches).
                g_hat = _clamp_grad_norm(grad, g_norm)

                if branch == "muon":
                    beta = group["momentum"]
                    m = state["momentum"]
                    # Paper Alg 8: m = β*m + ĝ (no 1-β factor); Nesterov u = β*m + ĝ.
                    m.mul_(beta).add_(g_hat)
                    u = beta * m + g_hat  # non-mutating: m feeds the next step
                    u = _newton_schulz(
                        u, steps=group["ns_steps"],
                        coefficients=group["ns_coefficients"], eps=group["eps_muon"],
                    )
                    # Aspect-ratio gain (Alg 8: s = sqrt(max(1, rows/cols))).
                    rows, cols = u.shape
                    u = u * math.sqrt(max(1.0, rows / cols))
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-lr)
                else:
                    # "Same as AdamW Update Rule" — our Algorithm 1 body,
                    # with this group's lr (lr_adam) and betas/eps_adam.
                    b1, b2 = group["betas"]
                    eps = group["eps_adam"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    m.lerp_(g_hat, 1.0 - b1)
                    v.mul_(b2).addcmul_(g_hat, g_hat, value=1.0 - b2)
                    m_hat = m / (1.0 - b1 ** t)
                    v_hat = v / (1.0 - b2 ** t)
                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat, denom, value=-lr)
        return loss


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

PART2_OPTIMIZERS = {
    "adamw": AdamW,
    "nadamw": NadamW,
    "lion": Lion,
    "muon": Muon,
}


def _embedding_param_ids(model) -> set[int]:
    """Parameter ids that Algorithm 8 routes to the AdamW branch: embeddings
    (tok_emb, pos_emb) and LayerNorm params. The tied LM head shares its
    weight object with tok_emb, so it is covered; attention/FFN Linear
    weights are NOT matched here and go to the Muon branch.
    """
    from torch import nn

    ids = set()
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.LayerNorm)):
            for p in m.parameters():
                ids.add(id(p))
    return ids


# Locked per-optimizer hyperparameters for the 1x real runs. Anchors: our
# 0.1x calibration curves (local evidence), the round-2 real-run tripwire
# evidence, and the paper's tuned RATIOS (their absolute values are tuned
# at 130M-1.2B params with >=0.4M token batches, ~33x our batch — and their
# own thesis is that blind transfer is unfair). Adam family: launch with
# --lr 1e-4 --wd 0.1.
#
#   Lion 0.1x run (3e-4, wd 0.01) stalled at the unigram floor and crept UP
#   (10.3k -> 11.4k): sign updates random-walk every coordinate at +-lr, so
#   wd must counteract (paper: optimal wd ~= 0.6; we ran 60x below it, and
#   make_optimizer passed cfg's 0.01 over the class default). lr 5e-5 via
#   the paper's Lion~AdamW/4-8 ratio + batch scaling (Lion docstring).
#   Muon 0.1x run (8e-3 / 2.4e-3) exploded to 162k ppl: the ADAM branch
#   (2.4e-3 = 3x the lr that collapsed whole-model adamw at 8e-4) on the
#   tied 32k x 384 head. The NS branch had the BEST early curve of all four
#   optimizers (3160 @ 411k vs adamw's 6225) -> only a 25% horizon trim.
#   ROUND 2 (1x real run, adamw @ 2e-4 / wd 0.01): churned back to the
#   unigram floor under SUSTAINED lr — val 1464 @ 2.9M (best of any run so
#   far) -> 2678 -> 5148, train loss 7.29 -> 9.36, rising even as lr decayed
#   2e-4 -> 1.57e-4. The 0.1x calibration could not see this: its compressed
#   cosine decayed lr to 3e-5 by 4M tokens, so sustained-lr stability was
#   never tested. Fix: adam-family lr halved (1e-4) + wd -> 0.1 (the paper's
#   own tuned AdamW value, the anti-drift damper). Muon's adam branch gets
#   the same protection: 5e-5 (half the family's, as before) + wd 0.1 via
#   --wd (also the paper's tuned muon wd).
LION_LR = 5e-5
LION_WD = 0.6
LION_BETAS = (0.9, 0.95)
MUON_LR = 6e-3
MUON_ADAM_LR = 5e-5


def make_optimizer(name: str, model, cfg) -> torch.optim.Optimizer:
    """Part 2 factory: `name` in PART2_OPTIMIZERS, hyperparams from cfg.

    Separate from src/train.build_optimizer (that one serves part 1 and its
    'adamw' must remain torch.optim.AdamW). Part 2 runs go through here so
    'adamw' unambiguously means OUR AdamW.

    cfg is duck-typed TrainConfig (src.train): the fairness constants
    (schedule, budget, stream, seed, batch) live in cfg; betas/wd come from
    cfg for the Adam family and Muon, while Lion overrides wd/betas with its
    own tuned constants — per-optimizer tuning is the paper's own
    methodology ("Lion's optimal weight decay ~=0.6 vs. AdamW's ~=0.1"),
    not a fairness violation. cfg.weight_decay is set per launch (--wd):
    real part-2 runs pass 0.1; the 0.01 default is part 1's easy-task value.
    """
    if name not in PART2_OPTIMIZERS:
        raise ValueError(
            f"unknown part 2 optimizer: {name} (choose from {sorted(PART2_OPTIMIZERS)})"
        )
    betas = tuple(cfg.betas)
    wd = cfg.weight_decay
    if name == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, betas=betas, weight_decay=wd)
    if name == "nadamw":
        return NadamW(model.parameters(), lr=cfg.lr, betas=betas, weight_decay=wd)
    if name == "lion":
        return Lion(model.parameters(), lr=LION_LR, betas=LION_BETAS,
                    weight_decay=LION_WD)
    if name == "muon":
        return Muon(
            model.parameters(), lr=MUON_LR, lr_adam=MUON_ADAM_LR,
            betas=betas, weight_decay=wd, adam_ids=_embedding_param_ids(model),
        )
    raise AssertionError(f"unreachable: {name}")