# VECTOR TinyStories Training Config
# Run with: python train.py
dataset = 'tinystories'
gradient_accumulation_steps = 4
batch_size = 4
block_size = 512

# Model
n_layer = 6
n_head = 6
n_kv_head = 2
c_dim = 64
ssm_d_state = 16
n_experts = 4
n_embd = 384
dropout = 0.0
bias = False

# Gate / Router
theta_init = 0.5
gate_temperature = 1.0

# Dual Loss weights
alpha_recon = 1.0
beta_budget = 0.01
gamma_anchor = 0.001
C_target = 128
warmup_steps = 500
T_min = 32
T_max = 512

# Optimizer
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 500
lr_decay_iters = 20000
min_lr = 6e-5

# System
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
