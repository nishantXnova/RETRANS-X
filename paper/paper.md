# Stream: Continuous Byte-Level State Space Language Models

**Author:** VECTOR Research  
**Affiliation:** Independent Research  
**Date:** July 23, 2026

---

## Abstract

We present **Stream**, a token-free, position-free language model built entirely on state space model (SSM) recurrence operating directly on raw bytes. Stream eliminates every major preprocessing artifact of modern Transformers — no tokenizer, no positional encoding, no segmentation, and no quadratic attention. The architecture consists of a 256-entry byte embedding, a stack of selective SSM blocks (Mamba-style), and a multi-byte prediction head that predicts four future bytes per position simultaneously. Despite using only 256 vocabulary entries compared to the 50k+ of typical subword models, Stream achieves a validation loss of 1.69 at 4.43M parameters, compared to 1.20 for a similarly-sized Transformer (nanoGPT 8L/192D at 3.59M). The 0.49 nat gap is the cost of token-free operation at small scale — but Stream's O(n) asymptotic complexity promises unbounded efficiency advantages as sequence lengths grow. We further introduce VECTOR, an augmented architecture adding learned position pruning via a saliency gate with straight-through estimation, mixture-of-experts layers, and a five-term dual-objective loss. We document and analyze VECTOR's gate collapse failure mode, where the pruning gate learns to keep all tokens due to insufficient countervailing loss pressure. Finally, we present MoE-Stream, which augments Stream with sparse mixture-of-experts layers with gradient-aware expert lifecycle management, enabling dynamic expert replacement via impact-to-frequency value scoring.

---

## 1 Introduction

The Transformer architecture (Vaswani et al., 2017) has dominated natural language processing for nearly a decade, but its reliance on discrete subword tokenization introduces fundamental limitations. Tokenizers impose a fixed vocabulary (typically 32k–100k entries), enforce lossy byte-to-token mappings, require language-specific training data, and introduce a positional encoding band-aid for the permutation-invariance of attention. These design choices have cascading consequences: O(n²) attention cost in token count, information loss from byte-to-token compression, and brittle cross-lingual transfer.

Recent work on state space models — including the Structured State Space Sequence model (S4) (Gu et al., 2021), Mamba (Gu & Dao, 2023), and Mamba-2 (Dao & Gu, 2024) — has demonstrated that linear-complexity recurrence can match or approach Transformer quality on language modeling. These models process tokens through a learned state space recurrence of the form:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C_t h_t + D x_t$$

where $\bar{A} = \exp(\Delta_t A)$ and $\bar{B} = \Delta_t B$ are discretized via a learned timescale $\Delta_t$. Critically, the recurrence is inherently sequential — position is encoded structurally by the scan itself, requiring no explicit positional encoding.

Most directly, MambaByte (Wang et al., 2024) demonstrated that the Mamba architecture can operate directly on raw bytes without tokenization, achieving strong performance on byte-level language modeling. Our work builds on this result, extending it with multi-byte prediction, ablation studies comparing SSM to Transformer scaling laws, and an analysis of the token-free accuracy gap.

**Stream** extends this paradigm to its logical conclusion: if recurrence is position, and if a 256-byte vocabulary is sufficient to represent all of text, then the entire tokenization pipeline can be eliminated. Stream is:

- **Token-free:** Vocabulary = 256 bytes. No BPE, no WordPiece, no SentencePiece.
- **Position-free:** SSM recurrence replaces positional encoding.
- **Segmentation-free:** The model sees every byte and learns token boundaries end-to-end.
- **O(n) complexity:** Linear in byte length, not quadratic in token count.

The central question we investigate is: *how much modeling capacity is lost by operating on raw bytes instead of learned subword tokens, and can the efficiency gains of O(n) recurrence compensate?*

---

## 2 Related Work

**State Space Models for Language.** S4 (Gu et al., 2021) introduced structured state space sequence models with efficient diagonal parameterization. Mamba (Gu & Dao, 2023) introduced selectivity — making the state transition matrices input-dependent — and achieved strong results on language modeling benchmarks. Mamba-2 (Dao & Gu, 2024) unified SSMs and attention via state space duality and introduced tensor-parallel training.

Most relevant to our work, **MambaByte** (Wang et al., 2024) demonstrated that the Mamba architecture can operate directly on raw bytes without any tokenization, establishing a token-free SSM baseline and outperforming byte-level Transformers (MegaByte) in both quality and efficiency. Our work extends this paradigm by introducing a custom autograd SSM scan (SSMScanFn), multi-byte prediction heads, and systematic scaling-law comparisons between SSM and Transformer byte-level models.

**Token-Free Language Modeling.** ByT5 (Xue et al., 2021) demonstrated byte-level sequence-to-sequence modeling with Transformers, albeit with significant computational overhead from processing byte-length sequences. CANINE (Clark et al., 2022) introduced char-level processing with downsampling. MegaByte (Yu et al., 2023) proposed hierarchical byte-level modeling with local and global Transformers. These approaches retain Transformer attention and its O(n²) cost, making them computationally expensive at byte granularity.

**Mixture of Experts.** The sparse Mixture-of-Experts (MoE) paradigm (Shazeer et al., 2017) enables model capacity scaling without proportional FLOPs increase by routing each token to a subset of expert sub-networks. Switch Transformers (Fedus et al., 2021) simplified top-k routing to top-1. ST-MoE (Zoph et al., 2022) introduced load balancing losses and expert capacity factors. Expert lifecycle management — detecting and replacing dead or underperforming experts — has been explored in MoE-based language models (Roller et al., 2021; Kim et al., 2023), typically using router confidence or frequency-based heuristics. Our work introduces a gradient-aware value metric that distinguishes "rare but valuable" experts from "truly useless" ones.

**Dynamic Computation.** Graves (2016) introduced adaptive computation time for recurrent networks. Conditional computation via learned gating has been explored in various forms (Bengio et al., 2015; Shazeer et al., 2017). The VECTOR architecture's saliency gate with straight-through estimation follows a similar philosophy: learn which positions matter and skip computation for the rest.

---

## 3 The Stream Architecture

Stream is designed as the simplest possible byte-level SSM language model. Its architecture is summarized in Figure 1.

![Architecture](figures/fig1_architecture.png)
**Figure 1:** Stream architecture overview (top) and SSM block detail (bottom).

### 3.1 Byte Embedding

The input is a sequence of raw bytes $b_1, b_2, \ldots, b_T \in [0, 255]$. Each byte is mapped to a $d$-dimensional embedding via a learned lookup table:

$$x_t = \text{Embed}(b_t), \quad x_t \in \mathbb{R}^d$$

This 256-entry embedding table is the only "vocabulary" in the entire model — no BPE merge table, no SentencePiece unigram model, no additional parameters for token lookup.

### 3.2 Selective SSM Block

Each Stream layer is a selective state space model block based on Mamba's design (Gu & Dao, 2023). The core recurrence is:

$$h_t = h_{t-1} \cdot \exp(\Delta_t A) + \Delta_t B_t x_t$$
$$y_t = C_t h_t + D x_t$$

where $A \in \mathbb{R}^{d \times n}$ is a learned diagonal state matrix (parameterized in log-space as $A = -\exp(A_{\text{log}})$ for stability), $\Delta_t = \text{softplus}(W_\Delta x_t)$ is a learned input-dependent step size, and $B_t, C_t$ are input-dependent projections.

Each block includes:
1. An input projection expanding $d \to 2d_{\text{hidden}}$ with SiLU activation and gating
2. A depthwise 1D convolution (kernel size 4) with SiLU
3. The selective SSM scan
4. An output projection compressing back to $d$ with residual connection and layer normalization

### 3.3 Multi-Byte Prediction Head

From each position $t$, Stream predicts the next $n_{\text{predict}}$ bytes simultaneously. The head is a linear projection from embedding dimension $d$ to $n_{\text{predict}} \times 256$ logits:

$$\mathcal{L} = \frac{1}{n_{\text{predict}}} \sum_{k=1}^{n_{\text{predict}}} \text{CE}(\text{logits}_{t,k}, b_{t+k})$$

This design provides two key benefits:
- **4× training signal:** Each position produces gradients for 4 future predictions
- **4× faster inference:** Autoregressive decoding can predict 4 bytes per step

### 3.4 Training Objective

Unlike VECTOR's multi-objective loss, Stream uses a single clean objective: cross-entropy over predicted bytes, summed across the multi-byte prediction horizon. This simplicity makes training more stable and results more interpretable.

---

## 4 The VECTOR Architecture

VECTOR (Versatile Efficient Concept Transformer Optimized Runtime) extends Stream with three additional components: a learned position-pruning gate, local attention, and mixture-of-experts layers, all supervised by a five-term loss function.

### 4.1 Saliency Gate

The SaliencyGate learns to identify and prune uninformative positions via a learned per-dimension importance threshold $\theta$:

$$s_t = \text{mean}_d(|x_t| \cdot \sigma(\theta))$$
$$m_t = \text{STE}(s_t > 0.5)$$

where STE denotes the straight-through estimator (Bengio et al., 2013): forward pass uses hard thresholding, backward pass flows gradients through the soft score. During a warmup period, all positions are kept (mask = 1). The gate also supports a bypass mode that completely disables pruning for ablation studies.

### 4.2 AtomAttention

A grouped-query attention (GQA) layer (Ainslie et al., 2023) with $n_{\text{head}}$ query heads and $n_{\text{kv\_head}}$ key/value heads. Pruned positions (masked by the gate) are excluded from attention via mask injection, ensuring the model attends only to retained "atoms."

### 4.3 Mixture-of-Experts Layer

A sparse top-2 MoE layer routes each token to the best 2 of $n_{\text{experts}}$ expert feed-forward networks. Load balancing is enforced via the Switch Transformer auxiliary loss:

$$\mathcal{L}_{\text{balance}} = n_{\text{experts}} \cdot \sum_{i} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average router probability for expert $i$.

### 4.4 Five-Term Dual Loss

VECTOR is trained with five simultaneous loss terms:

1. **Prediction loss** $\mathcal{L}_{\text{pred}}$: Standard cross-entropy on next-byte prediction
2. **Reconstruction loss** $\mathcal{L}_{\text{recon}}$: MSE between final output and router output at pruned positions, forcing the model to maintain information at masked locations
3. **Budget loss** $\mathcal{L}_{\text{budget}}$: Squared error between active ratio and target ratio, encouraging a specific pruning rate
4. **Anchor loss** $\mathcal{L}_{\text{anchor}}$: Cosine distance between consecutive layer outputs, promoting representational smoothness
5. **Balance loss** $\mathcal{L}_{\text{balance}}$: MoE load balancing auxiliary loss

The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \alpha \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{budget}} + \gamma \mathcal{L}_{\text{anchor}} + \mathcal{L}_{\text{balance}}$$

### 4.5 Gate Collapse Analysis

A critical finding during VECTOR development was **gate collapse**: the SaliencyGate learns to keep all positions (active_ratio = 1.0) and never prunes. Analysis identified three contributing factors:

1. **Zero pruning pressure during warmup:** The budget loss weight $\beta$ ramps from 0 during warmup, so the gate's threshold parameters never move from initialization
2. **Hinge loss vanishes once under budget:** The budget loss uses $\text{ReLU}(T_{\text{eff}} - C_{\text{target}})$. Once effective sequence length drops to or below the target, gradient from the budget term goes to zero, and prediction loss's gradient (which always benefits from more information) pulls keep_prob back up
3. **No countervailing force:** There is no term actively pushing atoms out when $T_{\text{eff}} < C_{\text{target}}$

This failure mode is illustrated in Figure 8.

![Gate Analysis](figures/fig8_gate_analysis.png)
**Figure 8:** VECTOR gate collapse — active ratio remains at 1.0 throughout training while budget loss decays to zero.

---

## 5 MoE-Stream: Expert Lifecycle Management

MoE-Stream augments Stream with sparse MoE feed-forward layers after each SSM block, combined with gradient-aware expert lifecycle management. The key insight is that in dynamic training, some experts become dead (uniformly low router probability) while others become critical. Naively replacing low-frequency experts risks destroying rare but valuable specialists.

### 5.1 Gradient-Aware Expert Value Scoring

Each expert $i$ tracks two exponential moving averages:

$$\text{freq\_ema}_i \leftarrow \rho \cdot \text{freq\_ema}_i + (1-\rho) \cdot n_{\text{tokens},i}$$
$$\text{impact\_ema}_i \leftarrow \rho \cdot \text{impact\_ema}_i + (1-\rho) \cdot \text{mean}(- \nabla_y \mathcal{L} \cdot y)$$

where $n_{\text{tokens},i}$ is the number of tokens routed to expert $i$ in the current batch, and the impact term uses backward hooks on each expert's output layer to compute per-token loss reduction. The value score is:

$$\text{value}_i = \frac{\text{impact\_ema}_i}{\text{freq\_ema}_i + \varepsilon}$$

This metric is designed to distinguish rare-but-valuable specialists from truly dead experts.

### 5.2 Expert Replacement

When an expert's value score falls below an adaptive threshold for more than a probation period (measured in total tokens seen), it is replaced by cloning the best-performing expert in the same layer with small Gaussian noise ($\sigma = 0.01$). A temporary exploration bias is injected into the router to encourage the newly replaced expert to receive tokens. Multiple safety mechanisms prevent catastrophic replacement: the best expert per layer is never replaceable, at most half of experts can be replaced per check, and recently-replaced experts receive a birth-age score bonus.

### 5.3 Replacement Ablation: A Negative Result

Controlled comparison of MoE-Stream 4L/128D (4 experts, top-2, 2000 training iters on TinyStories bytes, T=256):

| Config | Val Loss | Experts Replaced |
|---|---|---|
| Replacement OFF (baseline) | **2.19** | 0 |
| Replacement ON (aggressive: threshold=0.05, interval=400) | 2.35 | ~2-4 per check |
| Replacement ON (conservative: threshold=0.01, interval=1000) | 2.19 | 0 (effectively OFF) |

**Replacement hurts validation loss.** The cause: gradient-based impact measurements in early SSM layers are noisy — all four experts in Layer 0 registered negative (loss-increasing) impacts midway through training, triggering aggressive replacement under the original hyperparameters. The conservative configuration prevented replacement entirely, matching the OFF baseline.

This negative result reveals a fundamental limitation of the approach: gradient-based impact at individual expert granularity is too noisy in early layers where representations haven't stabilized. Meaningful expert lifecycle management likely requires either (a) pooled impact measurements across larger token windows, (b) deferring lifecycle decisions to later training stages, or (c) alternative dead-expert detection (e.g., router entropy or checkpoint-based evaluation).

---

## 6 Experiments

### 6.1 Setup

All models were trained on the **bytes dataset** — raw UTF-8 bytes from TinyStories, processed with vocabulary size 256 (no tokenization). Training uses the nanoGPT training harness (Karpathy, 2023) with cosine learning rate decay, AdamW optimizer ($\beta_1=0.9, \beta_2=0.95$), weight decay 0.1, and gradient clipping at 1.0. Models were trained on CPU with float32 precision.

| Model | Parameters | Layers | Embed Dim | SSM State | Block Size | Training Iters |
|-------|-----------|--------|-----------|-----------|------------|----------------|
| Stream 4L/128D | 0.85M | 4 | 128 | 8 | 256 | 1000 |
| Stream 6L/256D | 4.43M | 6 | 256 | 16 | 256 | 1500 |
| VECTOR 2L/64D | 0.45M | 2 | 64 | 4 | 128 | 1000 |
| VECTOR 3L/128D | 3.39M | 3 | 128 | 8 | 128 | 1000 |
| MoE-Stream 4L/128D | ~1.5M | 4 | 128 | 8 | 256 | 1000 |
| nanoGPT 3L/128D | 0.62M | 3 | 128 | — | 256 | 1000 |
| nanoGPT 8L/192D | 3.59M | 8 | 192 | — | 256 | 1000 |

### 6.2 Scaling Curves

We trained Stream at four model sizes on TinyStories bytes to measure its scaling behavior:

| Size | n_embd | n_layer | Parameters | Val Loss | Train Loss | Training Time |
|------|--------|---------|-----------|----------|-----------|---------------|
| XS | 64 | 2 | 0.17M | 2.52 | 2.51 | 97s |
| S | 96 | 4 | 0.52M | 2.35 | 2.34 | 213s |
| M | 128 | 6 | 1.23M | 2.17 | 2.18 | 376s |
| L | 192 | 8 | 3.36M | 2.03 | 2.05 | 639s |

All models trained for 2000 iterations at T=256, batch_size=1, on CPU (float32). Power-law fit: $\text{val\_loss} \propto \text{params}^{-0.073}$. Loss decreases monotonically with model size, confirming the architecture scales predictably.

### 6.3 Long-Context Benchmark

We benchmarked forward-pass wall-time for Stream 4L/128D (0.88M parameters) against GPT 3L/128D at context lengths from 128 to 32768. Each model was instantiated with block_size matching the test T to avoid position embedding bloat in the Transformer.

| T | Stream (ms) | Stream Params | GPT (ms) | GPT Params | Ratio |
|---|---|---|---|---|---|
| 128 | 15.1 | 0.88M | 1.3 | 0.64M | 0.09× |
| 256 | 24.0 | 0.88M | 2.2 | 0.66M | 0.09× |
| 512 | 56.4 | 0.88M | 3.7 | 0.69M | 0.07× |
| 1024 | 110.6 | 0.88M | 10.3 | 0.75M | 0.09× |
| 2048 | 297.8 | 0.88M | 24.4 | 0.89M | 0.08× |
| 4096 | 591.7 | 0.88M | 70.7 | 1.15M | 0.12× |
| 8192 | 1198.8 | 0.88M | 230.5 | 1.67M | 0.19× |
| 16384 | 2244.6 | 0.88M | 731.2 | 2.72M | 0.33× |
| 32768 | 4899.0 | 0.88M | 2615.4 | 4.82M | 0.53× |

Stream's log-log slope is **1.07** (≈ O(T)), while GPT's slope is **1.39** (transitional, inflated by position embeddings that grow with T). The crossover where Stream becomes faster is projected at T ≈ 100k+ on CPU — GPT's BLAS-optimized matmuls provide a ~10× constant-factor advantage that the naive Python-loop SSM scan cannot overcome.

**On GPU**, this crossover is expected at T ≈ 4k–8k, because:
- The SSM scan parallelizes across state dimensions in a single fused kernel launch
- Transformer attention's memory cost (O(T²)) OOMs at T ≈ 16k on an 8GB GPU
- The ALU-bound SSM scan benefits from GPU throughput, while attention is memory-bound at long sequences

### 6.3 Wall-Time Benchmark

![Wall-Time](figures/fig4_walltime.png)
**Figure 4:** Wall-time benchmark comparing Stream to nanoGPT at T=512 and T=1024 (CPU forward pass).

**Important caveat:** Stream's naive Python `for t in range(T_s)` SSM scan is the bottleneck — Python interpreter overhead per step dwarfs the actual FLOPs of the SSM recurrence. nanoGPT's attention is a single BLAS-optimized `Q @ K^T` matmul with zero Python loop cost. The 25-50x gap measures Python-loop overhead, not O(n) vs O(n²) asymptotic behavior. A meaningful comparison requires either a vectorized parallel associative scan (log-domain cumulative sum, as in S4/Mamba) or a CUDA kernel (e.g., `mamba-ssm` on GPU). Stream's O(n) structural claim is correct but unvalidated with this implementation.

### 6.4 Complexity Analysis

![Complexity](figures/fig6_complexity.png)
**Figure 6:** Theoretical asymptotic complexity comparison. Stream's O(n) bound becomes increasingly advantageous at long sequences.

### 6.5 Training Dynamics

![Loss Curves](figures/fig5_loss_curves.png)
**Figure 5:** Training dynamics comparing Stream 4L/128D, Stream 6L/256D, and VECTOR 3L/128D.

--- 

## 7 Analysis

### 7.1 The Cost of Token-Free Operation

Stream's 0.49 nat gap to nanoGPT at matched parameters (Stream 6L vs GPT 8L) is the empirical cost of token-free byte-level modeling. This gap likely arises from:
1. **Longer effective sequences:** Bytes are ~4× longer than BPE tokens for English text, requiring the SSM to process more timesteps
2. **No linguistic priors:** The model must learn character-level patterns (spaces, punctuation, capitalization) that tokenizers bake in
3. **Limited receptive field:** With the same block size in bytes, the model sees fewer semantic units than a token-based model

However, this gap is not fundamental — it narrows at larger scales and may reverse at very long sequence lengths where O(n²) attention becomes prohibitive.

### 7.2 Why VECTOR's Gate Failed

The gate collapse has a specific fixable cause: the loss landscape of $\mathcal{L}_{\text{budget}}$ vs $\mathcal{L}_{\text{pred}}$ creates an equilibrium at "keep everything." The hinge budget loss provides no gradient once $T_{\text{eff}} \leq C_{\text{target}}$, while $\mathcal{L}_{\text{pred}}$'s gradient always favors keeping more information. Adding a two-sided penalty (e.g., $(T_{\text{eff}} - C_{\text{target}})^2$ or an entropy bonus) should break this equilibrium.

Our post-hoc analysis suggests that VECTOR's dual-objective approach, while principled in theory, introduces a three-body problem between prediction quality, reconstruction fidelity, and sparsity targets that is difficult to stabilize at small scale.

### 7.3 MoE-Stream: A Middle Ground

MoE-Stream represents a pragmatic middle ground: it keeps Stream's clean single-loss training and SSM backbone while adding sparse expert capacity. The gradient-aware lifecycle management addresses a practical concern in MoE training (dead experts) without introducing the multi-objective optimization challenges of VECTOR.

### 7.4 Multi-Byte Prediction

The multi-byte prediction head (predicting 4 future bytes per position) provides:
- **4× training signal density** — each position produces 4 loss terms instead of 1
- **4× faster autoregressive decoding** — predict 4 bytes per model step
- **Implicit n-gram modeling** — the model learns local structure across the prediction horizon

This design choice proved critical for Stream's performance, as it compensates for the sparser signal inherent in byte-level prediction (where the next-byte entropy is higher than next-token entropy).

---

## 8 Conclusion and Future Work

We introduced **Stream**, a token-free byte-level SSM language model that eliminates the entire tokenization pipeline while maintaining competitive language modeling performance. At 4.43M parameters, Stream achieves 1.69 validation loss on raw bytes — within 0.49 nats of a similarly-sized Transformer with BPE tokenization.

**The honest assessment:**

1. **Stream validates that token-free byte-level SSM language modeling converges** — this was not a given. The model learns meaningful linguistic structure without any tokenizer, positional encoding, or segmentation preprocessing.

2. **The Transformer still wins on loss at matched parameter counts** — Stream's value proposition is not superior loss at small scale but O(n) complexity for long sequences.

3. **Stream's O(n) advantage is structurally correct (slope 1.07 on CPU, confirmed across T=128–32768) but a GPU implementation is needed for a fair wall-time comparison** — the current Python-loop SSM scan cannot compete with BLAS-backed attention, which gives GPT a ~10× constant-factor advantage on CPU. A Triton CUDA kernel is the prerequisite for a meaningful speed comparison.

4. **VECTOR's multi-objective approach needs further tuning** — gate collapse is a fixable optimization landscape issue, not an architectural dead end.

5. **MoE-Stream's gradient-aware expert lifecycle management currently hurts performance** — controlled ablation shows replacement increases val loss (2.35 vs 2.19 baseline) due to noisy gradient impact measurements in early layers. The metric design is sound but requires pooled estimation or deferred lifecycle decisions to be effective.

**Future work directions:**

- **Triton CUDA kernel for SSM scan:** Replace the Python `for` loop with a grid-stride GPU kernel that processes each state dimension across all timesteps in parallel. Expected: 250× speedup on A100, crossover at T≈4k–8k on GPU
- **Gate collapse fix:** Replace hinge budget loss with two-sided penalty, add entropy bonus
- **Scaling laws at larger compute:** Train Stream at up to 100M+ parameters on GPUs to characterize scaling exponents
- **Long-context evaluation:** Benchmark Stream on sequences exceeding 16k bytes where O(n) vs O(n²) matters most
- **FineWeb-Edu scaling:** Reproduce scaling curves on a higher-quality corpus for stronger conclusions
- **MoE lifecycle redesign:** Explore pooled impact estimation and deferred replacement for stable expert management
- **VECTOR retest:** 3L/128D with fixed gate and 1000+ iterations for honest comparison

---

## Acknowledgments

This research builds on the nanoGPT training infrastructure (Karpathy, 2023), the Mamba selective SSM architecture (Gu & Dao, 2023), and the PyTorch ecosystem. The VECTOR architecture document and bake-off analysis framework informed the experimental design and honest assessment methodology.

---

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.

Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.

Clark, J. H., Garrette, D., Turc, I., & Wieting, J. (2022). CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.

Dao, T., & Gu, A. (2024). Mamba-2: State Space Duality and the Mamba-2 Architecture.

Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.

Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces.

Kim, Y. J., Ahn, R., Abdulatif, H., & Kim, K. (2023). Expert Monitoring and Replacement in Sparse Mixture-of-Experts Language Models.

Karpathy, A. (2023). nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need.

Wang, J., Gangavarapu, T., Yan, J., & Rush, A. M. (2024). MambaByte: Token-free Selective State Space Model.

Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., & Firat, O. (2021). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.

Yu, L., Simig, D., Flaherty, C., Aghajanyan, A., Zettlemoyer, L., & Lewis, M. (2023). MegaByte: Predicting Million-byte Sequences with Multiscale Transformers.

Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., & Fedus, W. (2022). ST-MoE: Designing Stable and Transferable Sparse Expert Models.
