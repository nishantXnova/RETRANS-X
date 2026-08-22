# StreamR (retrieval) v2 — default long-context configuration.
# 10M tier, CELL 13 A/B protocol (T4): leg A T=4096 B=4, leg B T=16384 B=2.
# RetrievalBlock v2 features enabled: per-head relative bias, strided far
# window, content-derived segment memory, learned pathway gating.
# To reproduce the v1 StreamR baseline exactly, set the six retr_*/per_head_*
# flags below to their off-states (per_head_bias=False, retr_stride=0,
# retr_mem_slots=0, retr_gated=False).
dataset = 'bytes'
batch_size = 4
block_size = 4096
n_embd = 256
n_layer = 16
n_retrieval = 2
n_attn_head = 4
window_size = 128
n_global = 16
per_head_bias = True
retr_stride = 8
retr_stride_slots = 32
retr_mem_slots = 16
retr_mem_seg = 64
retr_gated = True
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
out_dir = 'out_streamr_long'