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
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING

try:
    from delta_scan import delta_scan_fused
except Exception:            # delta_scan optional; eager loop is the fallback
    delta_scan_fused = None

try:
    from apple_scan import apple_ssm_scan, apple_fused_scan, AppleSSMScanFn, apple_mps_available
    HAS_APPLE_SCAN = True
except Exception:
    HAS_APPLE_SCAN = False
    apple_mps_available = lambda: False


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
    """Wrapper — Apple path only on MPS (Metal), JIT on CPU/CUDA (faster on CPU)."""
    # CPU: JIT loop is already optimal (43ms @ T=4096). Apple cumsum is MPS-only win (3µs via Metal parallel prefix).
    if HAS_APPLE_SCAN and apple_mps_available() and a_vec.device.type == 'mps':
        try:
            return AppleSSMScanFn.apply(a_vec, b_vec, T)
        except Exception:
            pass  # fallback to JIT
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

    # Apple Silicon fused path: avoid materializing (B,T,H,N) a_vec/b_vec — saves 2x unified memory, stays in M3 L2 (16MB)
    # Only on MPS where unified memory makes this 4x more valuable; CPU keeps materialized + JIT (faster on CPU)
    if HAS_APPLE_SCAN and apple_mps_available() and u.device.type == 'mps':
        try:
            h = apple_fused_scan(u, dt, A, B, T)  # (B,T,H,N) without ever forming a_vec
            y = (h * C.unsqueeze(2)).sum(-1) + D * u
            return y, (h[:, -1].detach(), None)
        except Exception:
            pass  # fallback to materialized path

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, 'SSMBlockState']:
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

        y, ssm_state = self._ssm_scan(x_conv, dt, A, B_param, C_param)
        # The historical scan wrappers returned `(state, aux)`; accept that
        # ABI while the stateful inference contract stores the tensor itself.
        if isinstance(ssm_state, tuple):
            ssm_state = ssm_state[0]
        y = y * gate
        out = self.out_proj(y)
        history_len = self.ssm_d_conv - 1
        if history_len:
            if x_main.shape[1] >= history_len:
                conv_history = x_main[:, -history_len:].detach()
            else:
                # pad left with zeros so prefill with small T still yields fixed-size state
                pad = history_len - x_main.shape[1]
                z = torch.zeros(x_main.shape[0], pad, x_main.shape[2], device=x_main.device, dtype=x_main.dtype)
                conv_history = torch.cat([z, x_main], dim=1).detach()
        else:
            conv_history = x_main[:, :0].detach()
        return self.ln(out + x), SSMBlockState(ssm=ssm_state, conv=conv_history)

    def _ssm_scan(self, u, dt, A, B, C):
        return parallel_ssm_scan(u, dt, A, B, C, self.D)

    @torch.no_grad()
    def step(self, x: torch.Tensor, state: Optional['SSMBlockState'] = None
             ) -> Tuple[torch.Tensor, 'SSMBlockState']:
        """Process exactly one byte position while carrying only recurrent state.

        This is intentionally separate from the training scan: it is the
        correctness reference for the eventual fused decode kernel.  `x` has
        shape `(batch, n_embd)` and the returned state is detached, making its
        memory independent of generated length.
        """
        if x.ndim != 2:
            raise ValueError(f"SSMBlock.step expects (B, D), got {tuple(x.shape)}")
        B, D = x.shape
        x_proj = self.in_proj(x)
        x_main, gate = x_proj.chunk(2, dim=-1)
        x_main = self.act(x_main)
        gate = torch.sigmoid(gate)

        history_len = self.ssm_d_conv - 1
        if state is None:
            conv_history = x_main.new_zeros(B, history_len, x_main.shape[-1])
            ssm_state = x_main.new_zeros(B, x_main.shape[-1], self.ssm_d_state)
        else:
            conv_history, ssm_state = state.conv, state.ssm
            expected_history = (B, history_len, x_main.shape[-1])
            expected_ssm = (B, x_main.shape[-1], self.ssm_d_state)
            if tuple(conv_history.shape) != expected_history or tuple(ssm_state.shape) != expected_ssm:
                raise ValueError("SSM state shape does not match this block or batch")

        conv_input = torch.cat((conv_history, x_main.unsqueeze(1)), dim=1)
        x_conv = F.conv1d(conv_input.transpose(1, 2), self.conv1d.weight,
                          self.conv1d.bias, groups=self.conv1d.groups).squeeze(-1)
        x_conv = self.act(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        B_param, C_param = self.x_proj(x_conv).chunk(2, dim=-1)
        A = -torch.exp(self.A_log.float()).to(dtype=x.dtype)
        a = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        b = dt.unsqueeze(-1) * B_param.unsqueeze(1) * x_conv.unsqueeze(-1)
        next_ssm = ssm_state * a + b
        y = (next_ssm * C_param.unsqueeze(1)).sum(-1) + self.D.to(x.dtype) * x_conv
        out = self.ln(self.out_proj(y * gate) + x)
        next_history = (conv_input[:, -history_len:].detach() if history_len else
                        conv_input[:, :0].detach())
        return out, SSMBlockState(ssm=next_ssm.detach(), conv=next_history)


@dataclass
class SSMBlockState:
    """Persistent state of one selective-SSM layer during inference."""
    ssm: torch.Tensor
    conv: torch.Tensor


# -----------------------------------------------------------------------------
# Sparse Retrieval Block v2: multi-pathway content recall, O(n) memory.
# -----------------------------------------------------------------------------
def _causal_window(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   w: int, rel_bias: torch.Tensor, per_head: bool,
                   chunk: Optional[int] = None) -> torch.Tensor:
    """Exact causal window attention with bounded peak memory.

    Same math as the full `unfold` formulation, but key/value windows are
    materialized per query-chunk so the (B, T, D, w) intermediate never
    exists in full (peak O(B*C*D*w) instead of O(B*T*D*w) -- at leg B this
    is the difference between ~4.3 GB of temporaries and ~67 MB).

    q: (B, nh, T, hd); k, v: raw (B, T, D); rel_bias shared (2w-1,) or
    per-head (nh, 2w-1). Returns (B, nh, T, hd). When C >= T this reduces
    to exactly one full-sequence chunk (identical op sequence as before).
    """
    B, nh, T, hd = q.shape
    D = k.shape[-1]
    if chunk is None:
        # keep each chunk's window temporaries <= ~2**24 elements
        chunk = max(64, min(T, int((2 ** 24) // max(1, B * D * w))))
    chunk = min(chunk, T)
    k_pad = F.pad(k, (0, 0, w - 1, 0))   # left-pad: query t sees keys t-w+1..t
    v_pad = F.pad(v, (0, 0, w - 1, 0))
    dist = (w - 1) - torch.arange(w, device=q.device)
    if per_head:
        bias = rel_bias[:, dist].view(1, nh, 1, w)
    else:
        bias = rel_bias[dist].view(1, 1, 1, w)
    outs = []
    for s0 in range(0, T, chunk):
        s1 = min(s0 + chunk, T)
        L = s1 - s0
        kw = k_pad[:, s0:s1 + w - 1].unfold(1, w, 1)          # (B, L, D, w)
        vw = v_pad[:, s0:s1 + w - 1].unfold(1, w, 1)
        kw = kw.view(B, L, nh, hd, w).transpose(1, 2)         # (B, nh, L, hd, w)
        vw = vw.view(B, L, nh, hd, w).transpose(1, 2)
        lw = torch.einsum('bhtd,bhtdw->bhtw', q[:, :, s0:s1], kw) * (hd ** -0.5)
        lw = lw + bias
        att_w = torch.softmax(lw, dim=-1)
        outs.append(torch.einsum('bhtw,bhtdw->bhtd', att_w, vw))
    return torch.cat(outs, dim=2)


class RetrievalBlock(nn.Module):
    """
    Gives the SSM backbone content-based recall which a fixed d_state recurrence
    cannot do, while keeping O(n) memory and position-free relative biases.

    Pathways (all causal, all translation-invariant, no absolute PE):
      1. Dense window  — exact recall over the last `window` positions (v1).
      2. Strided window — exact recall at a stride beyond the dense window
         (offsets w+1, w+1+s, w+1+2s, ...), reaching ~w + slots·s bytes back.
      3. Segment memory — content-derived: the sequence is split into
         `mem_seg`-byte segments and each segment's k/v are attention-pooled by a
         learned per-head query into one (k, v) pair; every position attends to
         the `mem_slots` most recent segments before its own. Unlike v1's static
         sink tokens, these keys/values ARE the pooled content, so distant bytes
         are actually retrievable.
      4. Global tokens  — a small set of static learned sinks (anchors).

    Fusion: each head learns per-pathway weights (softmax over active pathways),
    so different heads can specialize on local vs. far recall — the gating idea
    from the design doc, at zero per-position cost.

    When all v2 features are OFF (per_head_bias=False, stride=0, mem_slots=0,
    gated=False) this block reproduces v1 EXACTLY (shared rel_bias, joint
    softmax over [global, window]), so the pilot and CELL 13 numbers remain
    reproducible. Enabling any feature switches to the pathway-fusion path.

    Output: post-norm residual like SSMBlock (ln(proj(y) + x), None).
    """
    def __init__(self, n_embd: int, n_head: int = 4, window: int = 128,
                 n_global: int = 16, bias: bool = False,
                 per_head_bias: bool = False,
                 stride: int = 0, stride_slots: int = 16,
                 mem_slots: int = 0, mem_seg: int = 32,
                 gated: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.window = window
        self.n_global = n_global
        self.head_dim = n_embd // n_head
        self.per_head_bias = per_head_bias
        self.stride = stride
        self.stride_slots = stride_slots
        self.mem_slots = mem_slots
        self.mem_seg = mem_seg
        self.gated = gated

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        if n_global > 0:
            self.g_k = nn.Parameter(torch.randn(n_global, n_head, self.head_dim) * 0.02)
            self.g_v = nn.Parameter(torch.randn(n_global, n_head, self.head_dim) * 0.02)
        else:
            self.register_buffer('g_k', torch.zeros(0, n_head, self.head_dim))
            self.register_buffer('g_v', torch.zeros(0, n_head, self.head_dim))

        if per_head_bias:
            self.rel_bias = nn.Parameter(torch.zeros(n_head, 2 * window - 1))
        else:
            self.rel_bias = nn.Parameter(torch.zeros(2 * window - 1))

        if stride > 0:
            if per_head_bias:
                self.stride_bias = nn.Parameter(torch.zeros(n_head, stride_slots))
            else:
                self.stride_bias = nn.Parameter(torch.zeros(stride_slots))
        if mem_slots > 0:
            self.seg_q = nn.Parameter(torch.randn(n_head, self.head_dim) * 0.02)
            if per_head_bias:
                self.mem_bias = nn.Parameter(torch.zeros(n_head, mem_slots))
            else:
                self.mem_bias = nn.Parameter(torch.zeros(mem_slots))

        if gated:
            n_paths = (1 + (1 if stride > 0 else 0)
                       + (1 if mem_slots > 0 else 0)
                       + (1 if n_global > 0 else 0))
            self.path_logits = nn.Parameter(torch.zeros(n_paths, n_head))
        self.ln = nn.LayerNorm(n_embd)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """Softmax over `dim` with binary mask; invalid slots get ~0 weight (never NaN)."""
        fill = -1e4 if logits.dtype == torch.float16 else -1e9
        l = torch.where(mask, logits, torch.full_like(logits, fill))
        a = torch.softmax(l, dim=dim)
        a = a * mask
        return a / (a.sum(dim=dim, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        nh, hd, w, ng = self.n_head, self.head_dim, self.window, self.n_global

        q, k, v = self.qkv(x).chunk(3, dim=-1)            # (B, T, D)
        q = q.view(B, T, nh, hd).transpose(1, 2)          # (B, nh, T, hd)

        is_v2 = (self.per_head_bias or self.stride > 0
                 or self.mem_slots > 0 or self.gated)
        if not is_v2:
            # ---- exact v1 path (byte-identical to the original block) ----
            k_pad = F.pad(k, (0, 0, w - 1, 0))
            v_pad = F.pad(v, (0, 0, w - 1, 0))
            k_win = k_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)
            v_win = v_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)
            lw = torch.einsum('bhtd,bhtdw->bhtw', q, k_win) * (hd ** -0.5)
            dist = (w - 1) - torch.arange(w, device=x.device)
            lw = lw + self.rel_bias[dist].view(1, 1, 1, w)
            lg = torch.einsum('bhtd,ghd->bhtg', q, self.g_k) * (hd ** -0.5)
            att = torch.softmax(torch.cat([lg, lw], dim=-1), dim=-1)
            ow = torch.einsum('bhtw,bhtdw->bhtd', att[..., ng:], v_win)
            og = torch.einsum('bhtg,ghd->bhtd', att[..., :ng], self.g_v)
            y = (ow + og).transpose(1, 2).reshape(B, T, D)
            return self.ln(self.proj(y) + x), None

        k_h = k.view(B, T, nh, hd)
        v_h = v.view(B, T, nh, hd)
        outs = []

        # ---- pathway 1: dense window (v1 mechanism, optional per-head bias) ----
        outs.append(_causal_window(q, k, v, w, self.rel_bias, self.per_head_bias))

        # ---- pathway 2: strided far window (exact recall at distance) ----
        if self.stride > 0:
            Ks = self.stride_slots
            offsets = w + 1 + torch.arange(Ks, device=x.device) * self.stride   # (Ks,)
            pos_idx = torch.arange(T, device=x.device).unsqueeze(1) - offsets.unsqueeze(0)  # (T, Ks)
            valid_s = pos_idx >= 0
            idx_s = pos_idx.clamp(min=0)   # (T, Ks)
            k_str = k_h[:, idx_s, :, :]    # (B, T, Ks, nh, hd) advanced index on dim 1
            v_str = v_h[:, idx_s, :, :]
            k_str = k_str.permute(0, 3, 1, 2, 4).contiguous()   # (B, nh, T, Ks, hd)
            v_str = v_str.permute(0, 3, 1, 2, 4).contiguous()
            ls = torch.einsum('bhtd,bhtsd->bhts', q, k_str) * (hd ** -0.5)
            if self.per_head_bias:
                ls = ls + self.stride_bias.unsqueeze(0).unsqueeze(2)
            else:
                ls = ls + self.stride_bias.view(1, 1, 1, Ks)
            att_s = self._masked_softmax(ls, valid_s.unsqueeze(0).unsqueeze(0), dim=-1)   # (B, nh, T, Ks)
            os_ = torch.einsum('bhts,bhtsd->bhtd', att_s, v_str)
            outs.append(os_)

        # ---- pathway 3: content-derived segment memory ----
        if self.mem_slots > 0:
            ms, Km = self.mem_seg, self.mem_slots
            n_seg = (T + ms - 1) // ms
            padT = n_seg * ms - T
            # pad T on the left (positions before 0) with zeros; pooling rows for
            # padded tail tokens are masked out so no dummy content leaks in.
            k_seg = F.pad(k_h, (0, 0, 0, 0, padT, 0))      # (B, n_seg*ms, nh, hd)
            v_seg = F.pad(v_h, (0, 0, 0, 0, padT, 0))
            k_seg = k_seg.view(B, n_seg, ms, nh, hd)
            v_seg = v_seg.view(B, n_seg, ms, nh, hd)
            # learned per-head pooling query → content-derived summary per segment
            sp = torch.einsum('bsmhd,hd->bsmh', k_seg, self.seg_q) * (hd ** -0.5)
            pos_tok = torch.arange(n_seg * ms, device=x.device).view(1, n_seg, ms)
            seg_valid = pos_tok < T          # (1, n_seg, ms)
            att_pool = self._masked_softmax(sp, seg_valid.unsqueeze(-1).expand(B, n_seg, ms, nh), dim=2)
            mem_k = torch.einsum('bsmh,bsmhd->bshd', att_pool, k_seg)   # (B, n_seg, nh, hd)
            mem_v = torch.einsum('bsmh,bsmhd->bshd', att_pool, v_seg)
            # each position attends to the last Km segments strictly before its own
            seg_idx = (torch.arange(T, device=x.device) // ms)           # (T,)
            m_start = (seg_idx - Km).clamp(min=0)                        # (T,)
            m_idx = m_start.unsqueeze(1) + torch.arange(Km, device=x.device).unsqueeze(0)  # (T, Km)
            valid_m = m_idx < seg_idx.unsqueeze(1)                       # segment must precede own
            m_idx_c = m_idx.clamp(min=0, max=n_seg - 1)                  # (T, Km)
            gk = mem_k[:, m_idx_c].permute(0, 3, 1, 2, 4).contiguous()      # (B, nh, T, Km, hd)
            gv = mem_v[:, m_idx_c].permute(0, 3, 1, 2, 4).contiguous()
            lm = torch.einsum('bhtd,bhtkd->bhtk', q, gk) * (hd ** -0.5)
            rel_dist = seg_idx.unsqueeze(1) - m_idx                      # 1..Km (larger = older)
            if self.per_head_bias:
                lm = lm + self.mem_bias[:, (rel_dist - 1).clamp(min=0)].unsqueeze(0)
            else:
                lm = lm + self.mem_bias[(rel_dist - 1).clamp(min=0)].view(1, 1, T, Km)
            att_m = self._masked_softmax(lm, valid_m.unsqueeze(0).unsqueeze(0), dim=-1)
            om = torch.einsum('bhtk,bhtkd->bhtd', att_m, gv)
            outs.append(om)

        # ---- pathway 4: static global tokens ----
        if ng > 0:
            lg = torch.einsum('bhtd,ghd->bhtg', q, self.g_k) * (hd ** -0.5)
            att_g = torch.softmax(lg, dim=-1)
            og = torch.einsum('bhtg,ghd->bhtd', att_g, self.g_v)
            outs.append(og)

        # ---- fusion: learned per-head pathway weights (gating) ----
        if self.gated:
            wts = torch.softmax(self.path_logits, dim=0)   # (n_paths, nh)
            y = sum(wts[p].view(1, nh, 1, 1) * o for p, o in enumerate(outs))
        else:
            y = sum(outs) if len(outs) > 1 else outs[0]
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.ln(self.proj(y) + x), None


class RMSNorm(nn.Module):
    """True RMSNorm: no mean subtraction, learnable scale. Fixes LayerNorm bug in audit."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# -----------------------------------------------------------------------------
# Gated Delta Memory Block: pure-recurrence content-addressed retrieval.
# -----------------------------------------------------------------------------
class DeltaMemoryBlock(nn.Module):
    """
    Content-addressed memory via a per-head associative matrix updated with the
    delta rule — the pure-recurrence alternative to RetrievalBlock's bounded
    attention. No tokens, no positions, no PE, no O(T^2): the state is exactly
    nh small matrices W_h in R^(hd x hd), and every token does O(hd^2) work.

      read : o_t   = W_{t-1} @ q_t                       (content-addressed)
      write: pred  = W_{t-1} @ k_t                       (predict stored value)
             W_t   = lam_t * W_{t-1} + beta_t * (v_t - pred) otimes k_t

    lam_t (decay) and beta_t (write gain) are learned per head and input-driven:
    a head can act as a persistent associative store (high lam, low beta) or a
    volatile short-term buffer (low lam, high beta). k is RMSNorm-normalized per
    head (unit-length keys bound interference and keep the matrix stable in
    fp16). Reads use the state BEFORE the current token's write, so recall is
    strictly of the past — causal by construction, like the SSM.

    With delta_window > 0 the block adds an exact-recent dense window pathway
    (causal unfold, like RetrievalBlock) and fuses it with the matrix via a
    learned per-head softmax gate; delta_window=0 is pure recurrence.

    Output: post-norm residual like the other blocks (ln(proj(y) + x), None).
    """
    def __init__(self, n_embd: int, n_head: int = 4, window: int = 0,
                 bias: bool = False, per_head_bias: bool = True,
                 lam_init: float = 2.2, beta_init: float = -2.2):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.window = window
        self.head_dim = n_embd // n_head
        self.per_head_bias = per_head_bias
        self.lam_init = lam_init
        self.beta_init = beta_init
        self._use_fused = False   # set True by delta_scan.enable_delta_triton

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # per-head gate logits: [lambda_raw, beta_raw] -> (B, T, 2*nh)
        self.gate_logits = nn.Linear(n_embd, 2 * n_head, bias=True)
        self.reset_gate_bias()
        self.k_norm = RMSNorm(self.head_dim)  # true RMSNorm: no mean-subtraction, unit-length keys
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.ln = nn.LayerNorm(n_embd)

        if window > 0:
            if per_head_bias:
                self.rel_bias = nn.Parameter(torch.zeros(n_head, 2 * window - 1))
            else:
                self.rel_bias = nn.Parameter(torch.zeros(2 * window - 1))
            self.path_logits = nn.Parameter(torch.zeros(2, n_head))

    def reset_gate_bias(self):
        """Zero the gate weights and pin the lambda/beta logit biases.

        MUST be (re-)applied after any blanket nn.Linear re-initialization
        (e.g. Stream._init_weights via self.apply), which would otherwise
        wipe the pinned biases: sigmoid(0)=0.5 decay instead of ~0.90."""
        with torch.no_grad():
            self.gate_logits.weight.zero_()
            self.gate_logits.bias.zero_()
            nh = self.n_head
            self.gate_logits.bias[0:nh] = self.lam_init
            self.gate_logits.bias[nh:] = self.beta_init

    def _delta_scan(self, S, q, k, v, lam, beta):
        """Dispatch: fused Triton kernel on CUDA when enabled, else eager loop."""
        if self._use_fused and delta_scan_fused is not None and q.is_cuda:
            return delta_scan_fused(q, k, v, lam, beta)
        return self._delta_scan_eager(S, q, k, v, lam, beta)

    @torch.jit.ignore
    def _delta_scan_eager(self, S, q, k, v, lam, beta):
        """Sequential delta recurrence over T, vectorized over (B, nh).
        S:   (B, nh, hd, hd)  state (zeros at entry) — kept FP32 for accumulation.
        q/k/v: (B, T, nh, hd)
        lam/beta: (B, T, nh)
        returns o (B, T, nh, hd), final S (detached)
        """
        B, T, nh, hd = q.shape
        outs = torch.empty_like(q)
        # FP32 accumulation: projections may be fp16 but state is fp32 (audit requirement)
        S = S.float()
        q_f = q.float() if q.dtype != torch.float32 else q
        k_f = k.float() if k.dtype != torch.float32 else k
        v_f = v.float() if v.dtype != torch.float32 else v
        lam = lam.float().unsqueeze(-1).unsqueeze(-1)   # (B, T, nh, 1, 1)
        beta = beta.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, nh, 1, 1)
        for t in range(T):
            kt = k_f[:, t]                 # (B, nh, hd)
            qt = q_f[:, t]
            vt = v_f[:, t]
            # read-before-write: strictly past recall
            Sk = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)      # (B, nh, hd)
            o = torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1)       # (B, nh, hd)
            outs[:, t] = o.to(outs.dtype)
            # delta write: erase along kt, add v-prediction correction
            err = vt - Sk                                          # (B, nh, hd)
            S = lam[:, t] * S + beta[:, t] * kt.unsqueeze(-1) * err.unsqueeze(-2)
        return outs, S.detach()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        nh, hd, w = self.n_head, self.head_dim, self.window

        q, k, v = self.qkv(x).chunk(3, dim=-1)                 # (B, T, D)
        k_raw = k
        # normalize keys to unit-ish length per head (bounded interference, fp16-safe)
        k = self.k_norm(k.view(B, T, nh, hd)).view(B, T, nh, hd)
        qh = q.view(B, T, nh, hd)
        vh = v.view(B, T, nh, hd)

        g = torch.sigmoid(self.gate_logits(x))                 # (B, T, 2*nh)
        lam, beta = g[..., :nh], g[..., nh:]                   # (B, T, nh)

        S0 = q.new_zeros(B, nh, hd, hd)
        o_d, _ = self._delta_scan(S0, qh, k, vh, lam, beta)    # (B, T, nh, hd)

        if w > 0:
            # hybrid pathway: exact-recent window, fused with the associative
            # response by a learned per-head gate over [matrix, window].
            o_w = _causal_window(qh.transpose(1, 2), k_raw, v, w,
                                 self.rel_bias, self.per_head_bias)  # (B, nh, T, hd)

            o_d = o_d.transpose(1, 2)                          # (B, nh, T, hd)
            wts = torch.softmax(self.path_logits, dim=0)       # (2, nh)
            o = (wts[0].view(1, nh, 1, 1) * o_d
                 + wts[1].view(1, nh, 1, 1) * o_w)             # (B, nh, T, hd)
            o = o.transpose(1, 2).reshape(B, T, D)
        else:
            o = o_d.view(B, T, D)

        return self.ln(self.proj(o) + x), None
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
    # RetrievalBlock v2 upgrades (all off ⇒ exact v1 behavior).
    per_head_bias: bool = False       # per-head rel bias instead of shared
    retr_stride: int = 0              # >0: strided far window gap
    retr_stride_slots: int = 16       # how many strided slots per query
    retr_mem_slots: int = 0           # >0: content-derived segment memory (count)
    retr_mem_seg: int = 32            # bytes per memory segment
    retr_gated: bool = False          # learned per-head pathway fusion
    # Gated delta memory (off by default). Delta blocks also replace the LAST
    # n_delta SSM blocks (after any retrieval blocks); pure recurrence + O(hd^2).
    n_delta: int = 0
    delta_head: int = 4
    delta_window: int = 0             # >0: hybrid add exact-recent window pathway
    delta_lam_init: float = 2.2       # gate logit bias -> lam ~= sigmoid(2.2) ~ 0.90
    delta_beta_init: float = -2.2     # gate logit bias -> beta ~= 0.10 (low write-init, prevents early corruption)
    patch_factor: int = 0             # 0=off, 2 or 4: continuous byte patcher (causal, learned, O(n/patch))
    patch_residual: bool = True       # keep byte-rate residual route for spelling/code/Unicode
    activation_checkpointing: bool = False  # trade recompute for T4 VRAM


class BytePatcher(nn.Module):
    """Causal continuous patcher: groups P bytes into one latent, strictly causal.

    Training: (B,T,D) -> (B,T//P,D) via P*D -> D linear on non-overlapping patches.
    Inference causal lag: latent p (bytes p*P..p*P+P-1) is only visible to bytes
    >= (p+1)*P, i.e. shifted by P. First P bytes see zero latent context and rely
    on the byte-rate residual. This preserves byte causality — no future bytes
    leak into same-patch positions — while cutting SSM work by Px.
    Preserves byte residual route per ENGINEERING_AUDIT.md for spelling/code/Unicode.
    """
    def __init__(self, n_embd: int, patch_factor: int):
        super().__init__()
        assert patch_factor in (2, 4)
        self.patch_factor = patch_factor
        self.proj = nn.Linear(patch_factor * n_embd, n_embd, bias=False)
        # per-offset up-projection (latent -> P byte slots) is fused as repeat+residual;
        # a single latent value is broadcast, residual byte path carries fine grain.

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,T,D) byte embeddings, T divisible by P.
        Returns (x_latent (B,Tp,D), residual x (B,T,D) for later combine)."""
        B, T, D = x.shape
        P = self.patch_factor
        assert T % P == 0, f"T={T} must be divisible by patch_factor={P}"
        Tp = T // P
        x_p = x[:, : Tp * P, :].reshape(B, Tp, P * D)
        x_latent = self.proj(x_p)  # (B,Tp,D)
        return x_latent, x

    def upsample_causal(self, y_latent: torch.Tensor, T: int) -> torch.Tensor:
        """y_latent: (B,Tp,D) -> (B,T,D) causal-shifted broadcast."""
        B, Tp, D = y_latent.shape
        P = self.patch_factor
        y_rep = y_latent.repeat_interleave(P, dim=1)  # (B,T,D)
        # causal lag: first P bytes see zero
        y_shift = torch.zeros_like(y_rep)
        if T > P:
            y_shift[:, P:, :] = y_rep[:, :-P, :]
        return y_shift


@dataclass
class StreamState:
    """Constant-size inference state for a pure Stream stack."""
    blocks: List[SSMBlockState]
    patch_buf: Optional[torch.Tensor] = None  # (B, P-1, D) buffered bytes
    patch_ptr: int = 0
    last_latent: Optional[torch.Tensor] = None  # (B, D) last completed latent output


class Stream(nn.Module):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config

        self.byte_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.patcher: Optional[BytePatcher] = None
        if config.patch_factor in (2, 4):
            self.patcher = BytePatcher(config.n_embd, config.patch_factor)

        n_ssm = max(0, config.n_layer - config.n_retrieval - config.n_delta)
        self.blocks = nn.ModuleList(
            [SSMBlock(config.n_embd, ssm_d_state=config.ssm_d_state, bias=config.bias)
             for _ in range(n_ssm)]
            + [RetrievalBlock(config.n_embd, n_head=config.n_attn_head,
                              window=config.window_size, n_global=config.n_global,
                              bias=config.bias,
                              per_head_bias=config.per_head_bias,
                              stride=config.retr_stride,
                              stride_slots=config.retr_stride_slots,
                              mem_slots=config.retr_mem_slots,
                              mem_seg=config.retr_mem_seg,
                              gated=config.retr_gated)
               for _ in range(config.n_retrieval)]
            + [DeltaMemoryBlock(config.n_embd, n_head=config.delta_head,
                                window=config.delta_window,
                                bias=config.bias,
                                lam_init=config.delta_lam_init,
                                beta_init=config.delta_beta_init)
               for _ in range(config.n_delta)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.n_predict * config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)
        # self.apply re-initialized every nn.Linear above, which wipes the
        # pinned lambda/beta gate biases of DeltaMemoryBlock -- re-pin them.
        for _m in self.modules():
            if isinstance(_m, DeltaMemoryBlock):
                _m.reset_gate_bias()
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
                iter_num: int = 0,
                return_state: bool = False):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.byte_embed(idx)

        # Patch path: run SSM stack at T/P, then causal upsample + byte residual.
        # When patch_factor=0, this is identity (original path).
        if self.patcher is not None:
            P = self.config.patch_factor
            if T % P != 0:
                # pad to multiple of P for patching (causal pad on right, ignored in loss)
                pad = P - (T % P)
                # pad embeddings with zeros — head logits for padded positions are ignored
                x = F.pad(x, (0, 0, 0, pad))  # (B, T+pad, D)
                T_padded = T + pad
            else:
                T_padded = T
            x_latent, x_resid = self.patcher.forward_train(x)  # (B,Tp,D), (B,T_padded,D)
            # SSM stack now sees Tp = T/P positions -> Px less work/memory
            block_states_latent = []
            h = x_latent
            for block in self.blocks:
                if self.training and self.config.activation_checkpointing and not return_state:
                    h = checkpoint(lambda hh, layer=block: layer(hh)[0], h, use_reentrant=False)
                    bs = None
                else:
                    h, bs = block(h)
                block_states_latent.append(bs)
            # causal upsample + byte residual (preserves spelling/code per audit)
            y_up = self.patcher.upsample_causal(h, T_padded)  # (B,T_padded,D)
            if self.config.patch_residual:
                x = y_up + x_resid
            else:
                x = y_up
            if T_padded != T:
                x = x[:, :T, :]
                # block states remain latent-rate; for return_state we keep them as-is
            block_states = block_states_latent
        else:
            block_states = []
            for block in self.blocks:
                if self.training and self.config.activation_checkpointing and not return_state:
                    # Recompute each block in backward instead of retaining its
                    # activations. `use_reentrant=False` is robust with the custom
                    # SSM autograd function and does not change model numerics.
                    x = checkpoint(lambda h, layer=block: layer(h)[0], x,
                                   use_reentrant=False)
                    block_state = None
                else:
                    x, block_state = block(x)
                block_states.append(block_state)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = self._compute_loss(logits, targets)
        else:
            loss = None

        if return_state:
            if self.patcher is not None:
                Bp = self.patcher.patch_factor
                # h is latent SSM output when patcher active (captured above)
                try:
                    last_lat = h[:, -1, :]  # (B,D) last latent after SSM stack
                except Exception:
                    last_lat = None
                patch_buf = torch.zeros(B, Bp - 1, self.config.n_embd, device=x.device, dtype=x.dtype) if Bp > 1 else torch.zeros(B, 0, self.config.n_embd, device=x.device)
                state = StreamState(blocks=block_states, patch_buf=patch_buf, patch_ptr=0, last_latent=last_lat)
            else:
                state = StreamState(blocks=block_states)
            return logits, loss, state
        return logits, loss

    def _compute_loss(self, logits, targets):
        B, T, _ = logits.shape
        np = self.config.n_predict
        vs = self.config.vocab_size
        logits = logits.view(B, T, np, vs)

        # Decayed horizon weighting: near future matters more.
        # Equal weighting gives far horizons equal influence despite fewer labels
        # and weaker signal — see ENGINEERING_AUDIT.md. Weights [1.0, .35, .15, .05]
        # normalized, so k=0 dominates while far heads remain auxiliary.
        raw_w = [1.0, 0.35, 0.15, 0.05]
        w = raw_w[:np]
        w_sum = sum(w)
        w = [x / w_sum for x in w]
        loss = 0.0
        for k in range(np):
            loss = loss + w[k] * F.cross_entropy(
                logits[:, :T - k, k].reshape(-1, vs),
                targets[:, k:].reshape(-1),
                ignore_index=-1
            )
        return loss

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

    def _require_streaming_blocks(self):
        # Patcher is streamable (buffered) when combined with pure SSM.
        # Retrieval/Delta still need verified carried-state kernels — keep strict.
        unsupported = [type(block).__name__ for block in self.blocks
                       if not isinstance(block, SSMBlock)]
        if unsupported:
            joined = ', '.join(sorted(set(unsupported)))
            raise NotImplementedError(
                f"Stateful streaming is currently defined only for pure SSM Stream; "
                f"{joined} needs a verified carried-state kernel first.")

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor) -> Tuple[torch.Tensor, StreamState]:
        """Encode a non-empty byte prefix and return next-byte logits and state."""
        self._require_streaming_blocks()
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("prefill expects a non-empty (B, T) byte tensor")
        if idx.shape[1] > self.config.block_size:
            raise ValueError("prefix exceeds training block_size; chunk prefill explicitly")
        logits, _, state = self(idx, return_state=True)
        return logits[:, -1, :self.config.vocab_size], state

    @torch.no_grad()
    def step(self, idx: torch.Tensor, state: StreamState) -> Tuple[torch.Tensor, StreamState]:
        """Consume one byte per batch item and return logits for its successor.
        Pure SSM (+optional buffered patcher) only — O(1) per step.
        Retrieval/Delta need verified carried-state kernels first."""
        self._require_streaming_blocks()
        if idx.ndim == 2 and idx.shape[1] == 1:
            idx = idx[:, 0]
        if idx.ndim != 1:
            raise ValueError("step expects (B,) or (B, 1) byte ids")
        if len(state.blocks) != len(self.blocks):
            raise ValueError("StreamState belongs to a different model")
        B = idx.shape[0]
        x_byte = self.byte_embed(idx)  # (B,D)

        # Patcher buffering: accumulate P bytes into one latent step
        if self.patcher is not None:
            P = self.patcher.patch_factor
            # state.patch_buf: (B, P-1, D) ring, state.patch_ptr in [0,P-1)
            # We keep a flat buffer of buffered byte embeddings (excluding current)
            buf = state.patch_buf
            ptr = state.patch_ptr
            last_lat = state.last_latent  # (B,D) or None
            if buf is None:
                buf = torch.zeros(B, max(0, P - 1), x_byte.shape[-1], device=x_byte.device, dtype=x_byte.dtype)
                ptr = 0
                last_lat = None
            # need to decide if we have a full patch
            # We buffer P bytes: buf holds P-1 previous, plus current byte makes P
            if P == 1:
                # degenerate, no patching
                x_latent_in = x_byte
                # run SSM blocks at latent rate (here = byte rate)
                h = x_latent_in
                next_states = []
                for block, block_state in zip(self.blocks, state.blocks):
                    # block.step expects (B,D)
                    h, ns = block.step(h, block_state)
                    next_states.append(ns)
                last_lat = h
                y_up = h  # no shift needed
                x = y_up + x_byte if self.config.patch_residual else y_up
                logits = self.head(self.ln_f(x)).view(B, self.config.n_predict, self.config.vocab_size)
                return logits[:, 0], StreamState(blocks=next_states, patch_buf=buf, patch_ptr=0, last_latent=last_lat)
            else:
                # Build patch of P bytes: [buf[ptr], ..., buf[ptr+P-2], x_byte] circular
                # Simpler: keep linear buffer that fills sequentially, not ring, resetting after P
                # We maintain buf as (B,P-1,D) and ptr counts how many buffered
                # When ptr == P-1, current byte completes a patch
                if ptr == P - 1:
                    # complete patch: gather P bytes
                    # buf holds P-1 bytes in order, plus x_byte as last
                    flat = torch.cat([buf, x_byte.unsqueeze(1)], dim=1).reshape(B, P * x_byte.shape[-1])
                    x_latent_in = self.patcher.proj(flat)  # (B,D)
                    # run SSM stack one latent step
                    h = x_latent_in
                    next_states = []
                    for block, block_state in zip(self.blocks, state.blocks):
                        # Delta/Retrieval also have step; SSM does; handle all
                        if hasattr(block, 'step'):
                            h, ns = block.step(h, block_state)
                        else:
                            # fallback (should not happen after _require check)
                            h, ns = block(h.unsqueeze(1))[0].squeeze(1), block_state
                        next_states.append(ns)
                    last_lat = h
                    # reset buffer
                    buf = torch.zeros_like(buf)
                    ptr = 0
                else:
                    # not yet a full patch — SSM state unchanged
                    next_states = list(state.blocks)
                    # buffer current byte
                    buf[:, ptr, :] = x_byte
                    ptr += 1
                    last_lat = last_lat  # keep previous
                # causal upsample: byte sees last completed latent (shifted by P), not current patch
                # If we just completed a patch, its latent is NOT yet visible to current byte — visible next P bytes
                # So y_up for current byte is previous last_lat (before this step's patch completion)
                # We need to use the last_lat BEFORE the possible update for causality.
                # The code above updated last_lat before we compute y_up — fix by saving prev.
                # To keep causality, we recompute: y_up should be state.last_latent (prev), not new.
                y_up = state.last_latent if state.last_latent is not None else torch.zeros_like(x_byte)
                if self.config.patch_residual:
                    x = y_up + x_byte
                else:
                    x = y_up
                logits = self.head(self.ln_f(x)).view(B, self.config.n_predict, self.config.vocab_size)
                return logits[:, 0], StreamState(blocks=next_states, patch_buf=buf, patch_ptr=ptr, last_latent=last_lat)
        else:
            # Non-patched path: original O(1) per-byte SSM/Delta/Retrieval stepping
            x = x_byte
            next_states = []
            for block, block_state in zip(self.blocks, state.blocks):
                # Dispatch to block.step if available; fallback to full forward for verification
                if hasattr(block, 'step'):
                    x, ns = block.step(x, block_state)
                else:
                    # Should not happen, but keep for forward compat
                    x, ns = block(x.unsqueeze(1))
                    x = x.squeeze(1)
                    ns = block_state
                next_states.append(ns)
            logits = self.head(self.ln_f(x)).view(B, self.config.n_predict, self.config.vocab_size)
            return logits[:, 0], StreamState(blocks=next_states)

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = logits / temperature
        if top_k is not None:
            if not 1 <= top_k <= logits.shape[-1]:
                raise ValueError("top_k must be between 1 and vocab_size")
            threshold = torch.topk(logits, top_k, dim=-1).values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Correct one-byte autoregressive sampling with constant-size state."""
        was_training = self.training
        self.eval()
        try:
            logits, state = self.prefill(idx)
            for _ in range(max_new_tokens):
                idx_next = self._sample_next(logits, temperature, top_k)
                idx = torch.cat((idx, idx_next), dim=1)
                logits, state = self.step(idx_next, state)
            return idx
        finally:
            self.train(was_training)


# -----------------------------------------------------------------------------
# RetrievalBlock v2 verification (CPU, deterministic).
# -----------------------------------------------------------------------------
def check_retrieval_v2(seed: int = 1337) -> Tuple[bool, str]:
    """
    Verifies RetrievalBlock v2:
      (1) v2 with every feature OFF reproduces v1 exactly (byte-identical).
      (2) every pathway is causal: output at t is independent of inputs > t.
      (3) gradients flow to every new v2 parameter.
      (4) full Stream(StreamR) forward/backward with v2 enabled runs cleanly.
    """
    torch.manual_seed(seed)
    B, T, D_ = 2, 200, 64
    nh = 4
    torch.set_grad_enabled(True)

    msgs = []

    # (1) features-off block must equal the exact v1 formula (independent impl)
    block = RetrievalBlock(D_, n_head=nh, window=32, n_global=8,
                           per_head_bias=False, stride=0, mem_slots=0, gated=False)
    x = torch.randn(B, T, D_, requires_grad=True)
    y, _ = block(x)

    # reference: rebuild v1 math inline
    w, ng = 32, 8
    with torch.no_grad():
        q, k, v = block.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, nh, D_ // nh).transpose(1, 2)
        k_pad = F.pad(k, (0, 0, w - 1, 0))
        v_pad = F.pad(v, (0, 0, w - 1, 0))
        k_win = k_pad.unfold(1, w, 1).view(B, T, nh, D_ // nh, w).transpose(1, 2)
        v_win = v_pad.unfold(1, w, 1).view(B, T, nh, D_ // nh, w).transpose(1, 2)
        lw = torch.einsum('bhtd,bhtdw->bhtw', q, k_win) * ((D_ // nh) ** -0.5)
        dist = (w - 1) - torch.arange(w)
        lw = lw + block.rel_bias[dist].view(1, 1, 1, w)
        lg = torch.einsum('bhtd,ghd->bhtg', q, block.g_k) * ((D_ // nh) ** -0.5)
        att = torch.softmax(torch.cat([lg, lw], dim=-1), dim=-1)
        ow = torch.einsum('bhtw,bhtdw->bhtd', att[..., ng:], v_win)
        og = torch.einsum('bhtg,ghd->bhtd', att[..., :ng], block.g_v)
        y_ref = block.ln(block.proj((ow + og).transpose(1, 2).reshape(B, T, D_)) + x)
    d = (y - y_ref).abs().max().item()
    ok1 = d < 1e-6
    msgs.append(f"[1] v1-equivalence (features off): max|diff|={d:.2e} {'OK' if ok1 else 'FAIL'}")

    # (2) causality: y[t] must not depend on inputs at positions > t
    block_v2 = RetrievalBlock(D_, n_head=nh, window=32, n_global=8,
                              per_head_bias=True, stride=8, stride_slots=16,
                              mem_slots=6, mem_seg=16, gated=True)
    x2 = torch.randn(B, T, D_, requires_grad=True)
    y2, _ = block_v2(x2)
    t_cut = 50
    # gradient of output at early positions w.r.t. token at t_cut+1 (and beyond)
    loss = y2[:, :t_cut].sum()
    loss.backward(retain_graph=True)
    g_leak = x2.grad[:, t_cut + 1:].abs().max().item()
    # gradient w.r.t. an early token should be nonzero (block actually uses context)
    g_used = x2.grad[:, :t_cut].abs().max().item()
    ok2 = (g_leak < 1e-9) and (g_used > 0)
    msgs.append(f"[2] causality: leak_grad={g_leak:.2e} used_grad={g_used:.2e} "
                f"{'OK' if ok2 else 'FAIL'}")

    # (3) every new v2 param receives nonzero gradient
    block_v3 = RetrievalBlock(D_, n_head=nh, window=32, n_global=8,
                              per_head_bias=True, stride=8, stride_slots=16,
                              mem_slots=6, mem_seg=16, gated=True)
    x3 = torch.randn(B, T, D_, requires_grad=True)
    y3, _ = block_v3(x3)
    y3.pow(2).mean().backward()
    new_params = {
        'rel_bias': 'per-head bias', 'stride_bias': 'stride bias',
        'seg_q': 'segment pool query', 'mem_bias': 'memory bias',
        'path_logits': 'pathway gate',
    }
    ok3 = True
    for name, label in new_params.items():
        p = getattr(block_v3, name)
        gn = p.grad.abs().max().item() if p.grad is not None else 0.0
        ok3 &= gn > 0
        msgs.append(f"[3] grad[{name}] ({label}): {gn:.2e} {'OK' if gn > 0 else 'FAIL'}")
    if not ok3:
        msgs.append("[3] FAIL: some v2 params got zero gradient")
    else:
        msgs.append("[3] all v2 params receive gradients: OK")

    # (4) full model (StreamR) forward/backward with v2 enabled on GPU/CPU
    torch.manual_seed(seed)
    cfg = StreamConfig(n_embd=64, n_layer=2, n_predict=4, block_size=T,
                       ssm_d_state=4, n_retrieval=1, n_attn_head=nh,
                       window_size=32, n_global=8,
                       per_head_bias=True, retr_stride=8, retr_stride_slots=8,
                       retr_mem_slots=4, retr_mem_seg=16, retr_gated=True)
    stream = Stream(cfg)
    idx = torch.randint(0, 256, (B, T))
    tgt = torch.randint(0, 256, (B, T))
    logits, loss = stream(idx, targets=tgt)
    loss.backward()
    n_grad = sum(1 for p in stream.parameters() if p.grad is not None)
    n_params = sum(1 for p in stream.parameters())
    ok4 = loss is not None and torch.isfinite(loss) and n_grad == n_params
    n_tot = sum(p.numel() for p in stream.parameters())
    msgs.append(f"[4] StreamR v2 full model: loss={loss.item() if loss else float('nan'):.2f} "
                f"grads {n_grad}/{n_params} params={n_tot:,} {'OK' if ok4 else 'FAIL'}")

    # (5) chunked window pathway == full-unfold reference (multi-chunk forced)
    torch.manual_seed(seed)
    Bw, Tw, ww, nhw = 2, 200, 32, 4
    Dw = nhw * 16
    qw = torch.randn(Bw, nhw, Tw, 16)
    kw = torch.randn(Bw, Tw, Dw)
    vw = torch.randn(Bw, Tw, Dw)
    rb = torch.randn(nhw, 2 * ww - 1) * 0.1
    o_chunked = _causal_window(qw, kw, vw, ww, rb, per_head=True, chunk=48)
    k_pad = F.pad(kw, (0, 0, ww - 1, 0))
    v_pad = F.pad(vw, (0, 0, ww - 1, 0))
    k_full = k_pad.unfold(1, ww, 1).view(Bw, Tw, nhw, 16, ww).transpose(1, 2)
    v_full = v_pad.unfold(1, ww, 1).view(Bw, Tw, nhw, 16, ww).transpose(1, 2)
    lw = torch.einsum('bhtd,bhtdw->bhtw', qw, k_full) * (16 ** -0.5)
    dist_w = (ww - 1) - torch.arange(ww)
    lw = lw + rb[:, dist_w].unsqueeze(0).unsqueeze(2)
    att_f = torch.softmax(lw, dim=-1)
    o_full = torch.einsum('bhtw,bhtdw->bhtd', att_f, v_full)
    d5 = (o_chunked - o_full).abs().max().item()
    ok5 = d5 < 1e-5
    msgs.append(f"[5] chunked window vs full unfold (forced multi-chunk): "
                f"max|diff|={d5:.2e} {'OK' if ok5 else 'FAIL'}")

    ok = ok1 and ok2 and ok3 and ok4 and ok5
    summary = "\n".join(msgs) + f"\nRETRIEVAL V2 SUMMARY: {'ALL PASS' if ok else 'FAIL'}"
    return ok, summary


# -----------------------------------------------------------------------------
# DeltaMemoryBlock verification (CPU, deterministic).
# -----------------------------------------------------------------------------
def _ref_delta(block: DeltaMemoryBlock, x: torch.Tensor) -> torch.Tensor:
    """Independent re-implementation of the delta recurrence, unrolled in a
    different code structure (explicit per-step state list + gather) so the
    block's fused scan and the reference cannot share a bug."""
    B, T, D = x.shape
    nh, hd = block.n_head, block.head_dim
    with torch.no_grad():
        q, k, v = block.qkv(x).chunk(3, dim=-1)
        k_raw = k
        k = block.k_norm(k.view(B, T, nh, hd)).view(B, T, nh, hd)
        qh, vh = q.view(B, T, nh, hd), v.view(B, T, nh, hd)
        g = torch.sigmoid(block.gate_logits(x))
        lam, beta = g[..., :nh], g[..., nh:]
        S = [torch.zeros(B, nh, hd, hd)]
        preds = []
        for t in range(T):
            o = torch.einsum('bndm,bnm->bnd', S[t], qh[:, t])
            preds.append(o)
            err = vh[:, t] - torch.einsum('bndm,bnm->bnd', S[t], k[:, t])
            S.append(lam[:, t].unsqueeze(-1).unsqueeze(-1) * S[t]
                     + beta[:, t].unsqueeze(-1).unsqueeze(-1)
                     * torch.einsum('bnd,bne->bnde', k[:, t], err))
        o = torch.stack(preds, dim=1).reshape(B, T, D)
        if block.window > 0:
            # reference window pathway (mirror of block forward's layout)
            w = block.window
            k_pad = F.pad(k_raw, (0, 0, w - 1, 0))
            v_pad = F.pad(v, (0, 0, w - 1, 0))
            k_win = k_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)   # (B, nh, T, hd, w)
            v_win = v_pad.unfold(1, w, 1).view(B, T, nh, hd, w).transpose(1, 2)
            q_pj = qh.transpose(1, 2)                                             # (B, nh, T, hd)
            lw = torch.matmul(q_pj.unsqueeze(-2), k_win).squeeze(-2) * (hd ** -0.5)
            dist = (w - 1) - torch.arange(w)
            lw = lw + block.rel_bias[:, dist].unsqueeze(0).unsqueeze(2)
            att = torch.softmax(lw, dim=-1)
            o_w = torch.matmul(att.unsqueeze(-2), v_win.transpose(-1, -2)).squeeze(-2)   # (B, nh, T, hd)
            wts = torch.softmax(block.path_logits.detach(), dim=0)
            o_d = o.reshape(B, T, nh, hd).transpose(1, 2)          # (B, nh, T, hd)
            o = (wts[0].view(1, nh, 1, 1) * o_d
                 + wts[1].view(1, nh, 1, 1) * o_w)
            o = o.transpose(1, 2).reshape(B, T, D)
        return block.ln(block.proj(o) + x)


def check_delta(seed: int = 1337) -> Tuple[bool, str]:
    """
    Verifies DeltaMemoryBlock:
      (1) pure-recurrence scan equals an independently unrolled reference.
      (2) strict causality: output at t is independent of inputs > t.
      (3) gradients (incl. the lambda/beta gates) match finite differences.
      (4) hybrid (window fused) path equals its independent reference.
      (5) full Stream (Stream-D) forward/backward with n_delta=2 runs cleanly.
      (6) lambda/beta gate biases stay pinned after Stream's blanket re-init.
    """
    torch.manual_seed(seed)
    B, T, D = 2, 24, 32
    nh = 4
    msgs = []

    # (1) pure recurrence vs reference
    block = DeltaMemoryBlock(D, n_head=nh, window=0)
    x = torch.randn(B, T, D, requires_grad=True)
    y, _ = block(x)
    y_ref = _ref_delta(block, x.detach())
    d1 = (y - y_ref).abs().max().item()
    ok1 = d1 < 1e-6
    msgs.append(f"[1] pure-delta scan vs reference: max|diff|={d1:.2e} {'OK' if ok1 else 'FAIL'}")

    # (2) causality: early outputs must not depend on later inputs
    x2 = torch.randn(B, T, D, requires_grad=True)
    y2, _ = block(x2)
    t_cut = 8
    y2[:, :t_cut].sum().backward(retain_graph=True)
    leak = x2.grad[:, t_cut + 1:].abs().max().item()
    used = x2.grad[:, :t_cut].abs().max().item()
    ok2 = (leak < 1e-9) and (used > 0)
    msgs.append(f"[2] causality: leak_grad={leak:.2e} used_grad={used:.2e} "
                f"{'OK' if ok2 else 'FAIL'}")

    # (3) finite-difference grads on gate logits (lambda/beta) and a linear weight
    def make():
        b = DeltaMemoryBlock(D, n_head=nh, window=0)
        b = b.double()
        return b

    block3 = make()
    x3 = torch.randn(B, T, D, dtype=torch.double, requires_grad=True)
    y3 = block3(x3)[0]
    loss3 = y3.pow(2).mean()
    loss3.backward()
    ok3 = True
    # sample params: gate_logits.bias[0] (lambda init), gate_logits.weight[0,:3], qkv.weight[0,:3]
    checks = [('gate_logits.bias', block3.gate_logits.bias.data, 0),
              ('gate_logits.weight', block3.gate_logits.weight.data, 0),
              ('qkv.weight', block3.qkv.weight.data, 0)]
    for name, pdata, idx in checks:
        src = pdata.ravel()
        target = (block3.get_parameter(name) if name != 'gate_logits.bias'
                  else block3.gate_logits.bias)
        grad = target.grad.ravel()[idx]
        eps = 1e-4 * max(abs(src[idx].item()), 1e-3)
        p0 = src[idx].item()
        src[idx] = p0 + eps
        yp = block3(x3.detach())[0].pow(2).mean().item()
        src[idx] = p0 - eps
        ym = block3(x3.detach())[0].pow(2).mean().item()
        src[idx] = p0
        fd = (yp - ym) / (2 * eps)
        # For tiny gradients (<1e-6) relative error is meaningless — check absolute
        if max(abs(fd), abs(grad.item())) < 1e-6:
            ok_fd = abs(fd - grad.item()) < 1e-6
            rel = abs(fd - grad.item())
        else:
            rel = abs(fd - grad.item()) / (abs(fd) + abs(grad.item()) + 1e-12)
            ok_fd = rel < 5e-2
        ok3 &= ok_fd
        msgs.append(f"[3] FD[{name}[{idx}]] grad={grad.item():.4e} fd={fd:.4e} rel={rel:.2e} "
                    f"{'OK' if ok_fd else 'FAIL'}")

    # (4) hybrid window path vs reference
    torch.manual_seed(seed)
    bh = DeltaMemoryBlock(D, n_head=nh, window=6)
    x4 = torch.randn(B, T, D, requires_grad=True)
    y4, _ = bh(x4)
    y4_ref = _ref_delta(bh, x4.detach())
    d4 = (y4 - y4_ref).abs().max().item()
    ok4 = d4 < 1e-6
    msgs.append(f"[4] hybrid (window fused) vs reference: max|diff|={d4:.2e} {'OK' if ok4 else 'FAIL'}")

    # (5) full Stream-D model
    torch.manual_seed(seed)
    cfg = StreamConfig(n_embd=D, n_layer=2, n_predict=4, block_size=T,
                       ssm_d_state=4, n_delta=1,
                       delta_head=nh, delta_window=0)
    stream = Stream(cfg)
    idx = torch.randint(0, 256, (B, T))
    tgt = torch.randint(0, 256, (B, T))
    logits, loss = stream(idx, targets=tgt)
    loss.backward()
    n_grad = sum(1 for p in stream.parameters() if p.grad is not None)
    n_params = sum(1 for p in stream.parameters())
    ok5 = loss is not None and torch.isfinite(loss) and n_grad == n_params
    n_tot = sum(p.numel() for p in stream.parameters())
    msgs.append(f"[5] Stream-D full model: loss={loss.item() if loss else float('nan'):.2f} "
                f"grads {n_grad}/{n_params} params={n_tot:,} {'OK' if ok5 else 'FAIL'}")

    # (6) gate-bias pinning survives blanket re-initialization
    torch.manual_seed(seed)
    cfg6 = StreamConfig(n_embd=D, n_layer=1, n_predict=2, block_size=T,
                        ssm_d_state=4, n_delta=1, delta_head=nh)
    dblock = Stream(cfg6).blocks[-1]
    lam0 = torch.sigmoid(dblock.gate_logits.bias[:nh])
    bet0 = torch.sigmoid(dblock.gate_logits.bias[nh:])
    w0 = dblock.gate_logits.weight.abs().max().item()
    exp_lam = torch.sigmoid(torch.tensor(dblock.lam_init)).item()
    exp_bet = torch.sigmoid(torch.tensor(dblock.beta_init)).item()
    ok6 = (torch.allclose(lam0, torch.full_like(lam0, exp_lam), atol=1e-5)
           and torch.allclose(bet0, torch.full_like(bet0, exp_bet), atol=1e-5)
           and w0 == 0.0)
    msgs.append(f"[6] gate-bias pinned after _init_weights: "
                f"lam={lam0[0].item():.3f} (exp {exp_lam:.3f}) beta={bet0[0].item():.3f} "
                f"(exp {exp_bet:.3f}) |W|max={w0:.1e} {'OK' if ok6 else 'FAIL'}")

    ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
    summary = "\n".join(msgs) + f"\nDELTA SUMMARY: {'ALL PASS' if ok else 'FAIL'}"
    return ok, summary
