"""
MoE-Stream: Stream + Sparse Mixture of Experts with expert lifecycle management.

Adds MoE FFN layers after each SSM block. Each token is routed to top-k experts.
Experts track their usage statistics to distinguish "rare but valuable" from
"truly useless" — enabling safe replacement of dead experts.

Key design:
- MoE FFN after each SSMBlock (Stream has no MLP — this adds specialized capacity)
- Loss-impact tracking via EMA of routing weight × output norm
- Expert value score = impact_when_used / frequency
- High value + low frequency = rare but valuable (keeper)
- Low value + low frequency = truly useless (replacement candidate)
- Probation period before any expert is eligible for replacement
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class MoEStreamConfig:
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    n_predict: int = 4
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False

    # MoE
    n_experts: int = 8
    top_k: int = 2
    expert_hidden_mult: int = 4

    # Expert lifecycle
    importance_momentum: float = 0.999
    replacement_threshold: float = 0.05
    expert_min_tokens: int = 5000
    moe_balance_coeff: float = 0.01


# ---------------------------------------------------------------------------
# SSM Block (copied from model.py for independence)
# ---------------------------------------------------------------------------
class SSMBlock(nn.Module):
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
        H = self.n_embd * (D // self.n_embd)

        x_proj = self.in_proj(x)
        x_main, gate = x_proj.chunk(2, dim=-1)
        x_main = self.act(x_main)
        gate = torch.sigmoid(gate)

        x_conv = self.conv1d(x_main.transpose(1, 2))[..., :T].transpose(1, 2)
        x_conv = self.act(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        B_param, C_param = self.x_proj(x_conv).chunk(2, dim=-1)
        A = -torch.exp(self.A_log.float())

        y = self._ssm_scan(x_conv, dt, A, B_param, C_param)
        y = y * gate
        out = self.out_proj(y)
        return self.ln(out + x)

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
        return out


# ---------------------------------------------------------------------------
# MoE FFN with expert lifecycle tracking
# ---------------------------------------------------------------------------
class MoEFFN(nn.Module):
    """
    Sparse mixture of experts FFN with gradient-aware lifecycle management.

    Expert value metric (gradient-based):
        value_score = impact_ema / (freq_ema + 1e-8)
        - impact_ema: EMA of -(∂L/∂y · y) per token when expert is selected
          (first-order approximation of how much this expert reduces the loss)
        - freq_ema: EMA of how often expert is selected
        - Birth-age boost protects recently-replaced experts
        - Adaptive per-layer threshold scales with the top expert

    Replacement:
        - Clones the best expert in the layer + small noise (not random init)
        - Sets an exploration bias on the router to encourage trying the new expert
        - Exploration bias decays exponentially

    Load balancing:
        - Uses batch-local token fractions (not cumulative history)
        - Only affects the router via detached counts
    """
    def __init__(self, config: MoEStreamConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.hidden_dim = config.n_embd * config.expert_hidden_mult
        self.momentum = config.importance_momentum
        self.min_tokens = config.expert_min_tokens
        self.replacement_threshold = config.replacement_threshold
        self._current_step = 0

        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.hidden_dim, config.n_embd, bias=config.bias),
            )
            for _ in range(config.n_experts)
        ])

        # Expert lifecycle buffers (not parameters — no gradient through these)
        self.register_buffer('_freq_ema', torch.zeros(config.n_experts))
        self.register_buffer('_impact_ema', torch.zeros(config.n_experts))
        self.register_buffer('_total_tokens', torch.zeros(config.n_experts, dtype=torch.long))
        self.register_buffer('_birth_step', torch.zeros(config.n_experts, dtype=torch.long))
        self.register_buffer('_n_replacements', torch.zeros(config.n_experts, dtype=torch.long))
        self.register_buffer('_exploration_bias', torch.zeros(config.n_experts))
        self.register_buffer('_grad_norm_ema', torch.zeros(1))
        # Optimizer reference (set externally for state clearing on replacement)
        self._optimizer: Optional[torch.optim.Optimizer] = None

        # Gradient-based impact tracking (backward hooks on each expert's last Linear)
        self._saved_outputs: List[Optional[torch.Tensor]] = [None] * config.n_experts
        self._pending_grad_impacts: List[List[torch.Tensor]] = [[] for _ in range(config.n_experts)]
        self._hook_handles: List = []
        self._register_impact_hooks()

        # Exploration bias decay per training step
        self._exploration_decay = 0.99

    def _register_impact_hooks(self):
        """Register backward hooks on each expert's last Linear to capture ∂L/∂y."""
        for i, expert in enumerate(self.experts):
            last_linear = expert[-1]
            def make_hook(ei):
                def bwd_hook(module, grad_input, grad_output):
                    saved = self._saved_outputs[ei]
                    if saved is not None and grad_output[0] is not None:
                        grad = grad_output[0].detach()
                        # Per-token impact: positive value = reduces loss
                        impact = -(grad * saved).sum(dim=-1)
                        self._pending_grad_impacts[ei].append(impact.cpu())
                return bwd_hook
            handle = last_linear.register_full_backward_hook(make_hook(i))
            self._hook_handles.append(handle)

    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        freq = self._freq_ema
        impact = self._grad_impact()
        value = self.get_value_scores(include_birth_boost=False)
        tokens = self._total_tokens.float()
        return {
            'freq_ema': freq,
            'impact_ema': impact,
            'value_score': value,
            'total_tokens': tokens,
            'birth_step': self._birth_step.float(),
            'n_replacements': self._n_replacements.float(),
            'exploration_bias': self._exploration_bias,
        }

    def _grad_impact(self) -> torch.Tensor:
        """Return current gradient-based impact EMA."""
        return self._impact_ema

    def flush_grad_impacts(self):
        """Called after backward to incorporate pending gradient impacts into EMA."""
        for i in range(self.n_experts):
            if self._pending_grad_impacts[i]:
                all_impacts = torch.cat(self._pending_grad_impacts[i])
                # Normalize by running estimate of grad*output scale so impact is ~unit
                norm = self._grad_norm_ema.clamp(min=1e-8)
                mean_impact = (all_impacts / norm).mean().to(self._impact_ema.device)
                self._impact_ema[i] = self.momentum * self._impact_ema[i] + (1 - self.momentum) * mean_impact
                self._pending_grad_impacts[i].clear()
                # Update gradient norm EMA from the last batch's impacts
                with torch.no_grad():
                    batch_norm = all_impacts.abs().mean()
                    self._grad_norm_ema[0] = self.momentum * self._grad_norm_ema[0] + (1 - self.momentum) * batch_norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        Returns: (output, balance_loss)
        """
        B, T, D = x.shape
        S = B * T
        x_flat = x.view(-1, D)

        # Gate logits with optional exploration bias for recently-replaced experts
        logits = self.gate(x_flat)
        if self._exploration_bias.any():
            logits = logits + self._exploration_bias.unsqueeze(0)

        weights = F.softmax(logits, dim=-1)

        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(x_flat)

        # Batch-local token counts for load balancing
        batch_counts = torch.zeros(self.n_experts, device=x.device)

        for i in range(self.n_experts):
            mask = (top_idx == i).any(-1)
            n_tokens = mask.sum()
            self._total_tokens[i] += n_tokens
            batch_counts[i] = n_tokens.float()

            if n_tokens > 0:
                w_mask = (top_idx == i).float()
                w = (top_w * w_mask).sum(-1)
                expert_out = self.experts[i](x_flat[mask])
                out[mask] += w[mask].unsqueeze(-1) * expert_out

                # Save output for gradient-based impact hook (only during training backward)
                if self.training:
                    self._saved_outputs[i] = expert_out.detach()

                # Update freq EMA (does not need gradient info)
                with torch.no_grad():
                    self._freq_ema[i] = self.momentum * self._freq_ema[i] + (1 - self.momentum) * n_tokens.float()

        # Load balancing loss — uses batch-local fractions, not cumulative history
        f_i = batch_counts / S
        P_i = weights.sum(0) / S
        balance_loss = self.n_experts * (f_i * P_i).sum()

        # Decay exploration bias
        if self.training:
            self._exploration_bias.mul_(self._exploration_decay)
            self._current_step += 1

        return out.view(B, T, D), balance_loss

    @torch.no_grad()
    def get_value_scores(self, include_birth_boost: bool = True) -> torch.Tensor:
        """Returns value_score per expert. High = valuable. Low = replacement candidate.

        Uses empirical-Bayes shrinkage to handle the low-frequency confidence problem:
        when an expert has few samples, its score is shrunk toward the population mean.
        Prevents "expert that got lucky twice" from being mistaken for "rare but valuable."

        If include_birth_boost is True, recently-replaced experts get a temporary
        score bonus to prevent immediate re-death.
        """
        freq = self._freq_ema
        impact = self._grad_impact()
        n = self._total_tokens.float()

        # Empirical-Bayes shrinkage: estimate confidence from sample count
        # Prior strength: at 5000 tokens, shrinkage ≈ 50%; at 50000, ≈ 10%
        prior_strength = self.min_tokens
        shrinkage = n / (n + prior_strength)
        # Population mean as the prior target
        population_mean = impact.sum() / (freq.sum() + 1e-8)
        # Shrunk estimate: less shrinkage when more data, more shrinkage when less
        impact_shrunk = shrinkage * (impact / (freq + 1e-8)) + (1 - shrinkage) * population_mean
        value = impact_shrunk.clamp(min=0.0)

        if include_birth_boost:
            age = self._total_tokens.float() / self.min_tokens
            survival_bonus = torch.clamp(2.0 - age, min=0.0) / 2.0 * 0.35
            value = value + survival_bonus

        max_val = value.max()
        if max_val > 0:
            value = value / max_val
        return value

    @torch.no_grad()
    def find_dead_experts(self, threshold: Optional[float] = None) -> List[int]:
        """
        Returns indices of experts eligible for replacement:
        - value_score below adaptive threshold
        - has seen enough tokens (past probation)
        - never kills the single best expert
        - caps at floor(n_experts / 2) per check
        """
        if threshold is None:
            threshold = self.replacement_threshold
        if threshold < 0:
            return []
        scores = self.get_value_scores(include_birth_boost=True)
        # Adaptive per-layer threshold: use max(global_threshold, max_score * 0.05)
        adaptive_th = max(threshold, scores.max().item() * 0.05)
        candidates = []
        for i in range(self.n_experts):
            if scores[i] < adaptive_th and self._total_tokens[i] >= self.min_tokens:
                candidates.append(i)
        # Never kill the best expert
        best_idx = torch.argmax(scores).item()
        candidates = [i for i in candidates if i != best_idx]
        # Cap at half the experts to keep the layer functional
        max_kill = max(1, self.n_experts // 2)
        if len(candidates) > max_kill:
            # Keep the ones with highest scores (least dead)
            candidates.sort(key=lambda i: scores[i].item(), reverse=True)
            candidates = candidates[:max_kill]
        return candidates

    @torch.no_grad()
    def replace_expert(self, idx: int) -> None:
        """
        Replace expert idx by cloning the best-performing expert + small noise.
        Falls back to scaled random init if no good source exists.
        Resets tracking stats, clears AdamW momentum buffers for the replaced
        parameters, and boosts routing probability for exploration.
        """
        # Find best expert to clone from
        scores = self.get_value_scores(include_birth_boost=False)
        best_idx = torch.argmax(scores).item()
        if best_idx == idx:
            best_idx = (best_idx + 1) % self.n_experts

        best_score = scores[best_idx].item()
        old_params = []
        if best_score > 0.05:
            noise_std = 0.01
            for target_layer, source_layer in zip(self.experts[idx], self.experts[best_idx]):
                if isinstance(target_layer, nn.Linear) and isinstance(source_layer, nn.Linear):
                    old_params.append(target_layer.weight)
                    noise = torch.randn_like(target_layer.weight) * noise_std
                    target_layer.weight.copy_(source_layer.weight + noise)
                    if target_layer.bias is not None and source_layer.bias is not None:
                        old_params.append(target_layer.bias)
                        noise_b = torch.randn_like(target_layer.bias) * noise_std
                        target_layer.bias.copy_(source_layer.bias + noise_b)
        else:
            for layer in self.experts[idx]:
                if isinstance(layer, nn.Linear):
                    old_params.append(layer.weight)
                    nn.init.normal_(layer.weight, mean=0.0, std=0.005)
                    if layer.bias is not None:
                        old_params.append(layer.bias)
                        nn.init.zeros_(layer.bias)

        # Reset tracking stats
        self._freq_ema[idx] = 0.0
        self._impact_ema[idx] = 0.0
        self._total_tokens[idx] = 0
        self._birth_step[idx] = self._current_step
        self._n_replacements[idx] += 1

        # Boost exploration: router will temporarily prefer this expert
        self._exploration_bias[idx] = 0.2

        # Clear AdamW optimizer state for replaced parameters.
        # Without this, stale momentum from the old expert corrupts the new expert's
        # first gradient steps, sending it far from its freshly-cloned initialization.
        if self._optimizer is not None:
            for group in self._optimizer.param_groups:
                for p in group['params']:
                    if p in old_params and id(p) in self._optimizer.state:
                        del self._optimizer.state[id(p)]

    @torch.no_grad()
    def log_utilization(self, prefix: str = "") -> str:
        """Returns a formatted string of expert utilization stats."""
        stats = self.get_expert_stats()
        scores = self.get_value_scores()
        lines = []
        for i in range(self.n_experts):
            lines.append(
                f"  {prefix}Expert {i}: score={scores[i]:.3f}, "
                f"freq={stats['freq_ema'][i]:.3f}, "
                f"impact={stats['impact_ema'][i]:.5f}, "
                f"tokens={int(stats['total_tokens'][i]):6d}, "
                f"bias={stats['exploration_bias'][i]:.3f}, "
                f"replaced={int(stats['n_replacements'][i])}x"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MoE-Stream Block: SSM + MoE FFN
# ---------------------------------------------------------------------------
class MoEStreamBlock(nn.Module):
    """SSMBlock followed by MoE FFN."""
    def __init__(self, config: MoEStreamConfig):
        super().__init__()
        self.ssm = SSMBlock(
            config.n_embd,
            ssm_d_state=config.ssm_d_state,
            ssm_d_conv=config.ssm_d_conv,
            ssm_expand=config.ssm_expand,
            bias=config.bias,
        )
        self.moe = MoEFFN(config)
        self.ln = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ssm(x)
        moe_out, moe_balance = self.moe(h)
        out = self.ln(h + moe_out)
        return out, moe_balance


# ---------------------------------------------------------------------------
# MoE-Stream Model
# ---------------------------------------------------------------------------
class MoEStream(nn.Module):
    """
    Stream + Sparse MoE with expert lifecycle management.

    Architecture:
        byte_embed → [SSMBlock → MoEFFN] × n_layer → LN → multi-byte head

    Expert lifecycle:
        - Each expert tracks its value score (impact / frequency)
        - Experts below threshold after probation are replaceable
        - Replacement re-initializes weights and resets tracking
    """
    def __init__(self, config: MoEStreamConfig):
        super().__init__()
        self.config = config

        self.byte_embed = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([
            MoEStreamBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.n_predict * config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"MoE-Stream parameters: {self.get_num_params() / 1e6:.2f}M")
        print(f"  {config.n_layer} layers, {config.n_experts} experts, top-{config.top_k}")

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

        total_moe_balance = 0.0
        for block in self.blocks:
            x, moe_bal = block(x)
            total_moe_balance = total_moe_balance + moe_bal

            # Update current step for lifecycle tracking (per block)
            block.moe._current_step = iter_num

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss, pred_loss = self._compute_loss(logits, targets, total_moe_balance)
            # Register post-backward hook to flush gradient-based impact EMAs
            if self.training:
                loss.register_hook(lambda _: self._post_backward_lifecycle())
        else:
            loss = None

        if return_logits:
            return logits, loss
        return logits, loss

    def _compute_loss(self, logits, targets, moe_balance):
        B, T, _ = logits.shape
        np = self.config.n_predict
        vs = self.config.vocab_size
        logits = logits.view(B, T, np, vs)

        pred_loss = 0.0
        for k in range(np):
            pred_loss = pred_loss + F.cross_entropy(
                logits[:, :T - k, k].reshape(-1, vs),
                targets[:, k:].reshape(-1),
                ignore_index=-1,
            )
        pred_loss = pred_loss / np

        total = pred_loss + self.config.moe_balance_coeff * moe_balance
        return total, pred_loss

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
        # Store optimizer reference in each MoE block for state clearing on expert replacement
        for block in self.blocks:
            block.moe._optimizer = optimizer
        return optimizer

    def _post_backward_lifecycle(self):
        """Called after loss.backward() to flush gradient impacts into EMAs."""
        for block in self.blocks:
            block.moe.flush_grad_impacts()

    @torch.no_grad()
    def log_expert_utilization(self) -> str:
        """Returns formatted expert utilization for all layers."""
        lines = []
        for i, block in enumerate(self.blocks):
            lines.append(f"Layer {i}:")
            lines.append(block.moe.log_utilization())
        return "\n".join(lines)

    @torch.no_grad()
    def get_expert_stats(self, block_idx: Optional[int] = None) -> Dict:
        """Returns expert stats for all blocks or a specific block."""
        if block_idx is not None:
            return self.blocks[block_idx].moe.get_expert_stats()
        return {f'block_{i}': self.blocks[i].moe.get_expert_stats()
                for i in range(len(self.blocks))}

    @torch.no_grad()
    def replace_dead_experts(self, threshold: Optional[float] = None):
        """
        Scans all MoE layers and replaces experts with low value scores.
        Returns summary of replacements made.
        """
        if threshold is None:
            threshold = self.config.replacement_threshold
        total_replaced = 0
        for bi, block in enumerate(self.blocks):
            dead = block.moe.find_dead_experts(threshold)
            for ei in dead:
                block.moe.replace_expert(ei)
                stats = block.moe.get_expert_stats()
                total_replaced += 1
            if dead:
                print(f"  Block {bi}: replaced {len(dead)} experts (indices {dead})")
        if total_replaced > 0:
            print(f"  Total: {total_replaced} experts replaced")
        return total_replaced

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
