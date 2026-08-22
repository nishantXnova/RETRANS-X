# Stream-D (gated delta memory) — default long-context A/B leg, same socket as
# StreamR. 10M tier, CELL 13 protocol (T4): leg A T=4096 B=4, leg B T=16384 B=2.
# Pure-recurrence content-addressed retrieval: last n_delta SSM blocks are
# replaced by DeltaMemoryBlock (per-head associative matrices + gated delta rule,
# O(hd^2) state, no attention, no PE). n_delta=0 is plain Stream; the delta head
# count/init mirror config/streamr_long.py's retrieval head for a matched-budget
# comparison (Stream-D-10M ~ 10.4M params).
dataset = 'bytes'
batch_size = 4
block_size = 4096
n_embd = 256
n_layer = 16
n_delta = 2
delta_head = 4
delta_window = 0
delta_lam_init = 2.2
delta_beta_init = 0.0
ssm_d_state = 16
n_predict = 4
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 1200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1200
min_lr = 6e-5
dtype = 'float16'
compile = False
eval_interval = 250
log_interval = 10
always_save_checkpoint = True
out_dir = 'out_streamd_long'