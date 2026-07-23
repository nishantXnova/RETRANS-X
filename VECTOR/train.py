"""
Stream training script — byte-level SSM language model.
"""

import os, time, math, pickle, numpy as np
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 2
eval_only = False
always_save_checkpoint = True
dataset = 'bytes'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 256
n_embd = 256
n_layer = 6
ssm_d_state = 16
n_predict = 4
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
compile = False
# VECTOR-specific (defaults for Stream; overridden by VECTOR configs)
model_type = 'stream'
n_head = 4
n_kv_head = 2
c_dim = 32
n_experts = 4
theta_init = 0.5
gate_temperature = 1.0
alpha_recon = 1.0
beta_budget = 0.01
gamma_anchor = 0.001
C_target = 160
T_min = 32
T_max = 192
gate_bypass = False
# MoE-Stream specific
ssm_d_conv = 4
ssm_expand = 2
top_k = 2
expert_hidden_mult = 4
importance_momentum = 0.999
replacement_threshold = 0.05
expert_min_tokens = 5000
moe_balance_coeff = 0.01
replace_experts_interval = 0  # 0 = disabled

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = 256
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"found vocab_size = {vocab_size}")
else:
    print(f"defaulting vocab_size to 256")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint8, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint8, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Model
if model_type == 'vector':
    from model_vector import VECTORModel, VECTORConfig
    model_args = dict(
        vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, c_dim=c_dim,
        ssm_d_state=ssm_d_state, n_experts=n_experts,
        block_size=block_size, dropout=dropout, bias=bias,
        theta_init=theta_init, gate_temperature=gate_temperature,
        alpha_recon=alpha_recon, beta_budget=beta_budget,
        gamma_anchor=gamma_anchor, C_target=C_target,
        T_min=T_min, T_max=T_max,
    )
    gptconf = VECTORConfig(**model_args)
    model = VECTORModel(gptconf)
    if gate_bypass:
        model.set_gate_bypass(True)
        print("VECTOR gate BYPASSED — no pruning, SSM backbone only")
elif model_type == 'moe_stream':
    from moe_stream import MoEStream, MoEStreamConfig
    model_args = dict(
        vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
        ssm_d_state=ssm_d_state, ssm_d_conv=ssm_d_conv,
        ssm_expand=ssm_expand, n_predict=n_predict,
        block_size=block_size, dropout=dropout, bias=bias,
        n_experts=n_experts, top_k=top_k,
        expert_hidden_mult=expert_hidden_mult,
        importance_momentum=importance_momentum,
        replacement_threshold=replacement_threshold,
        expert_min_tokens=expert_min_tokens,
        moe_balance_coeff=moe_balance_coeff,
    )
    gptconf = MoEStreamConfig(**model_args)
    model = MoEStream(gptconf)
else:
    from model import Stream, StreamConfig
    model_args = dict(
        vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
        ssm_d_state=ssm_d_state, n_predict=n_predict,
        block_size=block_size, dropout=dropout, bias=bias,
    )
    gptconf = StreamConfig(**model_args)
    model = Stream(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16')) if device_type == 'cuda' else None
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Learning rate schedule
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Training loop
iter_num = 0
best_val_loss = 1e9

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if model_type == 'moe_stream':
            print("Expert utilization:")
            print(model.log_expert_utilization())
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Replace dead experts periodically (MoE-Stream only)
    if model_type == 'moe_stream' and replace_experts_interval > 0 \
       and iter_num > 0 and iter_num % replace_experts_interval == 0:
        n_replaced = model.replace_dead_experts(threshold=replacement_threshold)
        if n_replaced > 0 and master_process:
            print(f"step {iter_num}: replaced {n_replaced} dead experts")

    X, Y = get_batch('train')

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, targets=Y, iter_num=iter_num)
            loss = loss / gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if grad_clip != 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        extra = ""
        if model_type == 'vector' and hasattr(model, 'loss_debug') and model.loss_debug:
            ld = model.loss_debug
            extra = f" | pred={ld['pred']:.4f} recon={ld['recon']:.4f} budget={ld['budget']:.4f} active={ld['active_ratio']:.3f}"
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%{extra}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
