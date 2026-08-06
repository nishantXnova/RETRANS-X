"""
Stream: Continuous Byte-Level SSM
Token-free, position-free, O(n) language model.
Predicts next N bytes directly from raw bytes — no tokenizer, no PE, no gate, no MoE.

Architecture:
- Byte embedding (256 → D) — the only "vocabulary"
- Stacked SSM blocks — recurrence = position by construction
- Optional sparse-retrieval blocks (windowed attention + global tokens) for
  content-based recall that a fixed d_state recurrence cannot do, while staying
  O(n) memory (window, not full attention). Off by default (n_retrieval=0).
- Multi-byte head: predict next N bytes per position
- Single loss: next-byte CE summed over N future predictions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING


# -----------------------------------------------------------------------------
# SSM scan: JIT-compiled sequential recurrence.
# On CPU the sequential loop is optimal (Blelloch tree scan adds overhead from
# non-contiguous access). JIT eliminates Python loop overhead.
# -----------------------------------------------------------------------------

# ── JIT-compiled forward/backward scan loops ───────────────────────
# TorchScript fuses the per-step elementwise ops into a single CUDA kernel,
# eliminating the O(T) kernel-launch overhead from the Python loop.

@torch.jit.script
def _ssm_fwd(a_vec: torch.Tensor, b_vec: torch.Tensor, T_s: int) -> torch.Tensor:
    Bs, _, Hc, Nc = a_vec.shape
    h = torch.zeros(Bs, Hc, Nc, device=a_vec.device)
    out = torch.empty(Bs, T_s, Hc, Nc, device=a_vec.device)
    for t in range(T_s):
        h = h * a_vec[:, t] + b_vec[:, t]
        out[:, t] = h
    return out

@torch.jit.script
def _ssm_bwd(grad_output: torch.Tensor, a_vec: torch.Tensor,
             out: torch.Tensor) -> List[torch.Tensor]:
    Bs, T_s, Hc, Nc = a_vec.shape
    grad_a = torch.zeros_like(a_vec); grad_b = torch.zeros_like(a_vec)
    dh = torch.zeros(Bs, Hc, Nc, device=a_vec.device)
    for t in range(T_s - 1, -1, -1):
        dh_total = grad_output[:, t] + dh
        h_prev = out[:, t - 1] if t > 0 else torch.zeros(Bs, Hc, Nc, device=a_vec.device)
        grad_b[:, t] = dh_total; grad_a[:, t] = dh_total * h_prev
        dh = dh_total * a_vec[:, t]
    return [grad_a, grad_b]

class SSMScanFn(torch.autograd.Function):
    """
    Custom autograd Function wrapping JIT-compiled scan kernels.
    The JIT-compiled forward/backward loops are fused into single CUDA
    kernels, eliminating per-step Python overhead and most kernel-launch
    overhead. The custom backward avoids building the full O(T) autograd
    graph that PyTorch would construct from the loop.
    """
    @staticmethod
    def forward(ctx, a_vec, b_vec, T_s):
        out = _ssm_fwd(a_vec, b_vec, T_s)
        ctx.save_for_backward(a_vec, out)
        ctx.T_s = T_s
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a_vec, out = ctx.saved_tensors
        grad_a, grad_b = _ssm_bwd(grad_output, a_vec, out)
        return grad_a, grad_b, None


def _ssm_scan(a_vec, b_vec, T):
    """Wrapper that calls SSMScanFn.apply."""
    return SSMScanFn.apply(a_vec, b_vec, T)


def parallel_ssm_scan(u: torch.Tensor, dt: torch.Tensor,
                      A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                      D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSM scan: sequential recurrence h_{t+1} = a_t · h_t + b_t, h_0 = 0,
    with a_t and b_t being functions of the input.

    Uses a JIT-compiled loop over T to eliminate Python overhead.
    On CPU there is no O(log T) parallel advantage (tree scan adds non-contiguous
    access cost), but the JIT avoids O(T) Python-level iteration cost.

    Args:
      u:  (B, T, H)    input
      dt: (B, T, H)    step sizes
      A:  (H, N)       state matrix (negative = -exp(A_log))
      B:  (B, T, N)    input projection
      C:  (B, T, N)    output projection
      D:  (H,)         skip connection

    Returns:
      y:     (B, T, H)  output
      state: (B, H, N)  final hidden state (detached)
    """
    Bs, T, H = u.shape
    N = A.shape[-1]

    # Precompute transition a_t and input b_t
    a_vec = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, H, N)
    b_vec = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)        # (B, T, H, N)

    # Custom autograd scan — h[t] = state after processing input t
    h = _ssm_scan(a_vec, b_vec, T)  # (B, T, H, N)

    # Output: y[t] = (h[t] · C[t]).sum(-1) + D · u[t]
    y = (h * C.unsqueeze(2)).sum(-1) + D * u

    return y, (h[:, -1].detach(), None)


# -----------------------------------------------------------------------------
# SSM Block: selective state space (Mamba-style)
# -----------------------------------------------------------------------------
class SSMBlock(nn.Module):
    def __init__(self, n_embd: int, ssm_d_state: int = 16,
                 ssm_d_conv: int = 4, ssm_expand: int = 2, bias: bool = False):
        super().__init__()
        self.n_embd = n_embd
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        hidden = n_embd * ssm_expand

        self.in_proj = nn.Linear(n_embd, hidden * 2, bias=bias)
        self.conv1d = nn.Conv1d(hidden, hidden, kernel_size=ssm_d_conv,
                                padding=ssm_d_conv - 1, groups=hidden, bias=bias)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(hidden, ssm_d_state * 2, bias=bias)
        self.dt_proj = nn.Linear(hidden, hidden, bias=True)

        self.A_log = nn.Parameter(torch.zeros(hidden, ssm_d_state))
        self.D = nn.Parameter(torch.randn(hidden))
        self.out_proj = nn.Linear(hidden, n_embd, bias=bias)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H = self.n_embd * (D // self.n_embd) if D != self.n_embd else self.n_embd * 2

        x_proj = self.in_proj(x)
        x_main, gate = x_proj.chunk(2, dim=-1)
        x_main = self.act(x_main)
        gate = torch.sigmoid(gate)

        x_conv = self.conv1d(x_main.transpose(1, 2))[..., :T].transpose(1, 2)
        x_conv = self.act(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        B_param, C_param = self.x_proj(x_conv).chunk(2, dim=-1)
        A = -torch.exp(self.A_log.float())

        y, state = self._ssm_scan(x_conv, dt, A, B_param, C_param)
        y = y * gate
        out = self.out_proj(y)
        return self.ln(out + x), state

    def _ssm_scan(self, u, dt, A, B, C):
        return parallel_ssm_scan(u, dt, A, B, C, self.D)


# -----------------------------------------------------------------------------
# Sparse Retrieval Block: causal sliding-window attention + global tokens
# -----------------------------------------------------------------------------
class RetrievalBlock(nn.Module):
    """
    Gives the SSM backbone content-based recall which a fixed d_state recurrence
    cannot do, while keeping O(n) memory: each query attends to the last `window`
    positions (sliding window) plus a small set of global key/value tokens.

    - Window keys come from a left-padded unfold, so slice t covers positions
      [t-window+1, t] — causal by construction, no triangular mask needed.
    - Relative-position bias (translation-invariant) instead of absolute PE, so
      the model stays position-free like the SSM recurrence it sits on.
    - Global tokens are static learned memory in this v1 (sink-token style);
      content-derived global keys/values are the natural v2 upgrade.

    Output: post-norm residual like SSMBlock (ln(proj(y) + x), None).
    """
    def __init__(self, n_embd: int, n_head: int = 4, window: int = 128,
                 n_global: int = 16, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.window = window
        self.n_global = n_global
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.g_k = nn.Parameter(torch.randn(n_global, n_head, self.head_dim) * 0.02)
        self.g_v = nn.Parameter(torch.randn(n_global, n_head, self.head_dim) * 0.02)
        self.rel_bias = nn.Parameter(torch.zeros(2 * window - 1))
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        nh, hd, w, ng = self.n_head, self.head_dim, self.window, self.n_global

        q, k, v = self.qkv(x).chunk(3, dim=-1)            # (B, T, D)
        q = q.view(B, T, nh, hd).transpose(1, 2)          # (B, nh, T, hd)

        k_pad = F.pad(k, (0, 0, w - 1, 0))
        v_pad = F.pad(v, (0, 0, w - 1, 0))
        k_win = k_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)  # (B, nh, T, hd, w)
        v_win = v_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)

        lw = torch.einsum('bhtd,bhtdw->bhtw', q, k_win) * (hd ** -0.5)       # (B, nh, T, w)
        dist = (w - 1) - torch.arange(w, device=x.device)
        lw = lw + self.rel_bias[dist].view(1, 1, 1, w)

        lg = torch.einsum('bhtd,ghd->bhtg', q, self.g_k) * (hd ** -0.5)      # (B, nh, T, ng)

        att = torch.softmax(torch.cat([lg, lw], dim=-1), dim=-1)             # (B, nh, T, ng+w)
        ow = torch.einsum('bhtw,bhtdw->bhtd', att[..., ng:], v_win)
        og = torch.einsum('bhtg,ghd->bhtd', att[..., :ng], self.g_v)
        y = (ow + og).transpose(1, 2).reshape(B, T, D)

        return self.ln(self.proj(y) + x), None


# -----------------------------------------------------------------------------
# Stream Model
# -----------------------------------------------------------------------------
@dataclass
class StreamConfig:
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    ssm_d_state: int = 16
    n_predict: int = 4
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    # Sparse retrieval (off by default). Retrieval blocks replace the LAST
    # n_retrieval SSM blocks so the head directly sees retrieved content.
    n_retrieval: int = 0
    n_attn_head: int = 4
    window_size: int = 128
    n_global: int = 16


class Stream(nn.Module):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config

        self.byte_embed = nn.Embedding(config.vocab_size, config.n_embd)

        n_ssm = max(0, config.n_layer - config.n_retrieval)
        self.blocks = nn.ModuleList(
            [SSMBlock(config.n_embd, ssm_d_state=config.ssm_d_state, bias=config.bias)
             for _ in range(n_ssm)]
            + [RetrievalBlock(config.n_embd, n_head=config.n_attn_head,
                              window=config.window_size, n_global=config.n_global,
                              bias=config.bias)
               for _ in range(config.n_retrieval)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.n_predict * config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # out_proj / c_proj / retrieval proj get GPT-2-style residual scaling.
            # '.proj.weight' matches only the retrieval block's proj (dt_proj and
            # out_proj have an underscore before 'proj', so they don't match).
            if (pn.endswith('out_proj.weight') or pn.endswith('c_proj.weight')
                    or pn.endswith('.proj.weight')):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Stream parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                return_logits: bool = False,
                iter_num: int = 0):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.byte_embed(idx)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = self._compute_loss(logits, targets)
        else:
            loss = None

        if return_logits:
            return logits, loss
        return logits, loss

    def _compute_loss(self, logits, targets):
        B, T, _ = logits.shape
        np = self.config.n_predict
        vs = self.config.vocab_size
        logits = logits.view(B, T, np, vs)

        loss = 0.0
        for k in range(np):
            loss = loss + F.cross_entropy(
                logits[:, :T - k, k].reshape(-1, vs),
                targets[:, k:].reshape(-1),
                ignore_index=-1
            )
        return loss / np

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        np = self.config.n_predict
        vs = self.config.vocab_size
        generated = 0
        while generated < max_new_tokens:
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].view(1, np, vs) / temperature
            for k in range(np):
                if generated >= max_new_tokens:
                    break
                probs = F.softmax(logits[:, k], dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(probs, top_k)
                    probs[probs < v[:, [-1]]] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                generated += 1
        self.train()
        return idx
