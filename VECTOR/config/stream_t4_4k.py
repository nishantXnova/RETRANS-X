# T4 throughput configuration: pure Stream, no attention, fixed 4K contexts.
# 12L/256D is ~8M parameters and uses FP16 Tensor Cores with the verified
# shape-conditional Triton scan. Start here before changing model capacity.
dataset = 'bytes'
batch_size = 4
block_size = 4096
n_embd = 256
n_layer = 12
ssm_d_state = 8
n_predict = 4
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
device = 'cuda'
dtype = 'float16'          # T4 has fast FP16 Tensor Cores, not BF16 Tensor Cores
triton_scan = 'auto'       # verified fused/chunked selection by B*H and length
activation_checkpointing = False
compile = False             # custom Triton autograd is the optimized path
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True
out_dir = 'out_stream_t4_4k'
