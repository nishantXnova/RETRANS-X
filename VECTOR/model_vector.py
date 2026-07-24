"""
VECTOR: Versatile Efficient Concept Transformer Optimized Runtime
Gate + MoE + 5-loss architecture for byte-level language modeling.

Key components:
- FractalRouter: byte embedding + Fourier position features
- SaliencyGate: learned position pruning via STE (fixed bypass flag)
- SSMBlock: selective state space model (same core as Stream)
- AtomAttention: grouped-query attention with content-addressed keys
- MoELayer: sparse mixture of experts with load balancing
- DualLoss: L_pred + L_recon + L_budget + L_anchor + L_balance

Gate bug fix (vs original): bypass flag now correctly returns x unchanged with ones mask.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from model import parallel_ssm_scan


@dataclass
class VECTORConfig:
    vocab_size: int = 256
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 1
    c_dim: int = 16
    ssm_d_state: int = 4
    n_experts: int = 2
    block_size: int = 128
    dropout: float = 0.0
    bias: bool = False
    # Gate
    theta_init: float = 0.5
    gate_temperature: float = 1.0
    # Loss weights
    alpha_recon: float = 1.0
    beta_budget: float = 0.01
    gamma_anchor: float = 0.001
    # Pruning targets
    C_target: int = 48
    warmup_steps: int = 30
    T_min: int = 16
    T_max: int = 96


class FractalRouter(nn.Module):
    """Byte embedding + Fourier position features (learned freqs)."""
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.byte_embed = nn.Embedding(config.vocab_size, config.n_embd)
        half_dim = config.n_embd // 2
        self.freqs = nn.Parameter(torch.randn(half_dim) * 0.1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        emb = self.byte_embed(idx)
        pos = torch.arange(T, device=idx.device).float().unsqueeze(-1)
        angles = pos * self.freqs.unsqueeze(0)
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb + pos_enc.unsqueeze(0)


class SaliencyGate(nn.Module):
    """
    Learned position pruning with straight-through estimator.
    FIXED: bypass=True now correctly returns x unchanged with ones mask.
    """
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.theta = nn.Parameter(torch.full((config.n_embd,), config.theta_init))
        self.bypass = False

    def set_bypass(self, value: bool):
        self.bypass = value

    def forward(self, x: torch.Tensor, iter_num: int = 0, warmup_steps: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        iter_num, warmup_steps: during warmup, keep all atoms (no pruning).
        Returns: (masked_x, mask) where mask is (B, T) with STE hard mask.
        When bypass=True: mask = ones, x unchanged.
        """
        if self.bypass or (warmup_steps > 0 and iter_num < warmup_steps):
            return x, torch.ones(x.shape[0], x.shape[1], device=x.device)

        # Weighted saliency: learned per-dim importance
        theta_w = torch.sigmoid(self.theta)
        saliency = (x.abs() * theta_w.unsqueeze(0).unsqueeze(0)).mean(-1)
        saliency = saliency / (saliency.max(-1, keepdim=True)[0] + 1e-8)

        # Hard mask via STE
        mask_hard = (saliency > 0.5).float()
        mask_soft = saliency
        mask = mask_hard.detach() + mask_soft - mask_soft.detach()

        return x * mask.unsqueeze(-1), mask


class SSMBlock_VECTOR(nn.Module):
    """SSM block — same design as Stream's SSMBlock."""
    def __init__(self, n_embd: int, ssm_d_state: int = 16,
                 ssm_d_conv: int = 4, ssm_expand: int = 2, bias: bool = False):
        super().__init__()
        self.n_embd = n_embd
        self.ssm_d_state = ssm_d_state
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        y, _ = self._ssm_scan(x_conv, dt, A, B_param, C_param)
        y = y * gate
        out = self.out_proj(y)
        return self.ln(out + x)

    def _ssm_scan(self, u, dt, A, B, C):
        return parallel_ssm_scan(u, dt, A, B, C, self.D)


class AtomAttention(nn.Module):
    """
    Grouped-query attention with content addressing.
    GQA: n_head query heads, n_kv_head key/value heads.
    Masked positions are excluded from attention.
    """
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_head % config.n_kv_head == 0
        self.n_groups = config.n_head // config.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, gate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        att = att + causal.unsqueeze(0).unsqueeze(0)

        if gate_mask is not None:
            attn_mask = gate_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(attn_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class MoELayer(nn.Module):
    """Sparse mixture of experts with top-2 routing."""
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_embd = config.n_embd
        hidden = 4 * config.n_embd

        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, hidden, bias=config.bias),
                nn.GELU(),
                nn.Linear(hidden, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        S = B * T

        logits = self.gate(x_flat)
        weights = F.softmax(logits, dim=-1)

        top2_w, top2_idx = torch.topk(weights, 2, dim=-1)
        top2_w = top2_w / (top2_w.sum(-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(x_flat)
        expert_load = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            expert_mask = (top2_idx == i).any(-1)
            n_tokens = expert_mask.sum()
            if n_tokens > 0:
                w_mask = (top2_idx == i).float()
                w = (top2_w * w_mask).sum(-1)
                out[expert_mask] += w[expert_mask].unsqueeze(-1) * self.experts[i](x_flat[expert_mask])
            expert_load[i] = n_tokens.float()

        # Load balancing loss (Switch Transformer style)
        if expert_load.sum() > 0:
            load_frac = expert_load / expert_load.sum()
            importance_frac = weights.sum(0) / weights.sum(0).sum()
            balance_loss = self.n_experts * (load_frac * importance_frac).sum()
        else:
            balance_loss = torch.tensor(0.0, device=x.device)

        return out.view(B, T, D), balance_loss


class VECTORBlock(nn.Module):
    """One VECTOR group: Gate -> 3xSSM -> Attn -> MoE."""
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.gate = SaliencyGate(config)
        self.ssm_blocks = nn.ModuleList([
            SSMBlock_VECTOR(config.n_embd, ssm_d_state=config.ssm_d_state, bias=config.bias)
            for _ in range(3)
        ])
        self.attention = AtomAttention(config)
        self.moe = MoELayer(config)
        self.ln = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, iter_num: int = 0, warmup_steps: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gated_x, mask = self.gate(x, iter_num=iter_num, warmup_steps=warmup_steps)
        h = gated_x
        for ssm in self.ssm_blocks:
            h = ssm(h)
        h = self.attention(h, gate_mask=mask)
        h, moe_balance = self.moe(h)
        out = self.ln(h + x)
        return out, mask, moe_balance


class VECTORModel(nn.Module):
    """Full VECTOR model with 5-term dual loss."""
    def __init__(self, config: VECTORConfig):
        super().__init__()
        self.config = config
        self._loss_debug = {}

        self.router = FractalRouter(config)
        self.blocks = nn.ModuleList([
            VECTORBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self._anchor_reprs = []

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"VECTOR parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gate_bypass(self, value: bool):
        for block in self.blocks:
            block.gate.set_bypass(value)

    @property
    def loss_debug(self):
        return self._loss_debug

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                return_logits: bool = False,
                iter_num: int = 0):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.router(idx)

        masks = []
        moe_balances = []
        layer_outputs = [x]
        warmup_steps = self.config.warmup_steps

        for block in self.blocks:
            x, mask, moe_bal = block(x, iter_num=iter_num, warmup_steps=warmup_steps)
            masks.append(mask)
            moe_balances.append(moe_bal)
            layer_outputs.append(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = self._compute_loss(
                logits, targets, layer_outputs, masks, moe_balances
            )
        else:
            loss = None

        if return_logits:
            return logits, loss
        return logits, loss

    def _compute_loss(self, logits, targets, layer_outputs, masks, moe_balances):
        cfg = self.config
        B, T, V = logits.shape

        # 1. Prediction loss: standard CE
        loss_pred = F.cross_entropy(
            logits.view(-1, V), targets.view(-1), ignore_index=-1
        )

        # 2. Reconstruction loss: MSE at masked (pruned) positions
        avg_mask = sum(masks) / len(masks)
        recon_mask = (1.0 - avg_mask).unsqueeze(-1)
        router_out = layer_outputs[0]
        final_out = layer_outputs[-1]
        if recon_mask.sum() > 0:
            loss_recon = F.mse_loss(final_out * recon_mask, router_out * recon_mask)
        else:
            loss_recon = torch.tensor(0.0, device=logits.device)

        # 3. Budget loss: MSE toward C_target active count
        active_ratio = avg_mask.mean()
        target_ratio = cfg.C_target / cfg.block_size
        loss_budget = (active_ratio - target_ratio) ** 2

        # 4. Anchor loss: cosine distance between consecutive layers
        loss_anchor = torch.tensor(0.0, device=logits.device)
        if len(layer_outputs) >= 2:
            for i in range(len(layer_outputs) - 1):
                a = layer_outputs[i].view(-1, cfg.n_embd)
                b = layer_outputs[i + 1].view(-1, cfg.n_embd)
                a_norm = F.normalize(a, dim=-1)
                b_norm = F.normalize(b, dim=-1)
                loss_anchor = loss_anchor + (1.0 - (a_norm * b_norm).sum(-1)).mean()
            loss_anchor = loss_anchor / (len(layer_outputs) - 1)

        # 5. MoE balance loss
        loss_balance = sum(moe_balances) / len(moe_balances)

        total = (loss_pred
                 + cfg.alpha_recon * loss_recon
                 + cfg.beta_budget * loss_budget
                 + cfg.gamma_anchor * loss_anchor
                 + loss_balance)

        self._loss_debug = {
            'pred': loss_pred.item(),
            'recon': loss_recon.item(),
            'budget': loss_budget.item(),
            'anchor': loss_anchor.item(),
            'balance': loss_balance.item(),
            'active_ratio': active_ratio.item(),
        }

        return total

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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = 0.0
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
