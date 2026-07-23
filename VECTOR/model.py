"""
Stream: Continuous Byte-Level SSM
Token-free, position-free, O(n) language model.
Predicts next N bytes directly from raw bytes — no tokenizer, no PE, no gate, no MoE.

Architecture:
- Byte embedding (256 → D) — the only "vocabulary"
- Stacked SSM blocks — recurrence = position by construction
- Multi-byte head: predict next N bytes per position
- Single loss: next-byte CE summed over N future predictions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


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
        B_s, T_s, H = u.shape
        N_state = self.ssm_d_state
        D_vec = self.D
        h = torch.zeros(B_s, H, N_state, device=u.device, dtype=u.dtype)
        a_t = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        b_t = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        out = torch.zeros_like(u)
        for t in range(T_s):
            h = h * a_t[:, t] + b_t[:, t]
            out[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D_vec * u[:, t, :]
        return out, (h.detach(), None)


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


class Stream(nn.Module):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config

        self.byte_embed = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([
            SSMBlock(config.n_embd, ssm_d_state=config.ssm_d_state, bias=config.bias)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.n_predict * config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('c_proj.weight'):
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
