"""
Muon optimizer (Keller Jordan) for RETRANS-X.

Muon = Momentum orthogonalized by Newton-Schulz. It runs SGD-momentum and then
replaces each 2D update with the nearest orthogonal matrix via a quintic
Newton-Schulz iteration, computed in bfloat16 for GPU efficiency.

It applies only to 2D matmul weights (nn.Linear). Embeddings, biases, norm
weights and non-Linear parameters (e.g. A_log, D) are trained by AdamW through
the MuonAdamW hybrid wrapper.

References:
  https://kellerjordan.github.io/posts/muon/
  modded-nanogpt train_gpt2.py (pinned commit 9730304)
"""

import inspect

import torch
import torch.nn as nn


_NS_STEPS_RECT = 5      # provably robust for every non-square matrix
_NS_STEPS_SQUARE = 14   # square matrices need ~14 steps (see docstring)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz quintic iteration for the zeroth power (orthogonal factor) of a 2D matrix.

    Runs in bfloat16. The matrix is transposed to rows <= cols so the polynomial
    approximation stays well-conditioned, then restored to the original orientation.

    Step count is shape-dependent (measured on this repo, 100+ random draws):
      * non-square (ratio >= 1.25, the dominant case: in_proj/out_proj are 2:1,
        x_proj 8:1, head wide): 5 steps lands singular values in ~[0.68, 1.13]
        every draw.
      * square (ratio 1.0): 5 steps can collapse the smallest singular value to
        ~0.002 (the smallest singular value of a square Gaussian tends to zero,
        and bf16 rounding amplifies it), silently distorting that gradient
        direction every step. 10 steps still lets the 512x512 (the real dt_proj
        size) tail dip to ~0.13 in worst-case bf16 and ~0.29 on real GPU draws.
        14 steps raises the worst-case floor to ~0.68 in emulation and is the
        number used here. bf16 square NS is not exact orthogonalization: the
        weakest direction can still carry ~0.2-0.5x weight on rare draws, which
        is benign for a gradient preconditioner (the guard is collapse, ~0.002).

    Stream's default config DOES hit the square case: dt_proj = Linear(hidden,
    hidden) is nn.Linear and therefore Muon-partitioned. Keep this step bump if
    model dims ever change (a future config where any two dims match re-enters
    the square path). An external caller may override steps via the parameter.

    The input scale is removed in fp32 BEFORE the bf16 cast. Quantizing first
    (G.bfloat16() then normalize) puts NS(G) and NS(3G) on different bf16 grids;
    the iteration then amplifies that mismatch to up to ~30% relative difference
    on small square matrices. Normalizing in fp32 first makes NS scale-invariant
    to ~1e-3 (bf16) instead of ~5e-2..3e-1, at no cost to the bf16 matmuls.
    """
    assert len(G.shape) == 2
    if G.size(0) == G.size(1):
        steps = max(steps, _NS_STEPS_SQUARE)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)  # remove input scale in fp32, then quantize
    X = X.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Momentum orthogonalized by Newton-Schulz.

    All parameters passed in must be 2D matrices. Do not pass embeddings, the
    final projection, or any 0/1-D parameter; those go through AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g.to(p.data.dtype, copy=False), alpha=-group['lr'])
        return loss


def partition_params(model, include_embeddings=False):
    """Split trainable parameters into (muon_params, adamw_params).

    Muon handles nn.Linear weight matrices; everything else (embeddings, biases,
    norm weights, non-Linear params such as A_log/D) goes to AdamW.
    """
    muon = []
    seen = set()
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight is not None and m.weight.requires_grad:
            muon.append(m.weight)
            seen.add(id(m.weight))
    if include_embeddings:
        for _, m in model.named_modules():
            if isinstance(m, nn.Embedding) and m.weight is not None and m.weight.requires_grad:
                muon.append(m.weight)
                seen.add(id(m.weight))
    adamw = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    return muon, adamw


class MuonAdamW:
    """Hybrid optimizer: Muon for nn.Linear weights, AdamW for everything else.

    Exposes the torch.optim.Optimizer surface (param_groups, step, zero_grad,
    state_dict, load_state_dict) so it can drop into existing training loops.

    NOTE: an external LR schedule that overwrites every param_group['lr'] will
    also overwrite the Muon lr; schedule Muon and AdamW lrs explicitly.
    """

    def __init__(self, model, muon_lr=0.02, adamw_lr=6e-4, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.1, betas=(0.9, 0.95), device_type='cpu',
                 include_embeddings=False):
        muon_params, adamw_params = partition_params(model, include_embeddings=include_embeddings)
        self.muon = None
        self.adamw = None
        self.optimizers = []
        if muon_params:
            self.muon = Muon(muon_params, lr=muon_lr, momentum=momentum,
                             nesterov=nesterov, ns_steps=ns_steps)
            self.optimizers.append(self.muon)
        groups = []
        decay = [p for p in adamw_params if p.dim() >= 2]
        nodecay = [p for p in adamw_params if p.dim() < 2]
        if decay:
            groups.append({'params': decay, 'weight_decay': weight_decay})
        if nodecay:
            groups.append({'params': nodecay, 'weight_decay': 0.0})
        if groups:
            fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters
                     and device_type == 'cuda')
            self.adamw = torch.optim.AdamW(groups, lr=adamw_lr, betas=betas, fused=fused)
            self.optimizers.append(self.adamw)
        if not self.optimizers:
            raise ValueError('no trainable parameters found')

    @property
    def param_groups(self):
        return [pg for opt in self.optimizers for pg in opt.param_groups]

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state):
        for opt, s in zip(self.optimizers, state):
            opt.load_state_dict(s)
