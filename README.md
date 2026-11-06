# Sift & SiftX: Selective State Spaces with Bounded Associative Memory

A clean, from-scratch PyTorch implementation of the **Mamba-1** selective state space architecture, trained as a language model on GPU, compared against a Transformer baseline, and extended with **SiftX**: a bounded, gated associative memory mechanism solving the fading-memory bottleneck.

> **Architectural Credit:**  
> The core selective state space architecture is based entirely on:  
> **Albert Gu and Tri Dao (2023)**, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).  
> Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba).  
> *"Sift"* is this repository's standalone implementation; all architectural credit for Mamba belongs to the original authors.

---

## Key Experimental Findings (RTX 3050 Laptop GPU)

1. **WikiText-2 Language Modeling:** Sift SSM outperforms the Transformer baseline by **42.6% lower perplexity** (3.497 vs 6.098) at matched ~3.7M parameter scale.
2. **Inference Memory Scaling:** Sift recurrent state is strictly constant at **140.0 KiB** from sequence length 256 to 4,096 tokens. The Transformer KV cache scales linearly from 1,024 KiB to **16,384 KiB** (**117× larger at 4k tokens**).
3. **Flat Generation Latency:** Autoregressive token generation remains flat at **29.4 ms/token** across all context lengths.
4. **Selective Scan Hardware Acceleration:** Custom associative parallel scan executes in **82.1 ms fwd+bwd** on GPU (**9.0× faster** than the 737.4 ms sequential autograd loop).
5. **Where Sift Loses (Honest Reporting):**
   - **Shakespeare Character Modeling:** Transformer achieved lower perplexity (4.693 vs 6.792) and trained faster (74.6s vs 2917.5s) due to cuBLAS tensor core saturation versus pure PyTorch scan overhead.
   - **Multi-Query Associative Recall (MQAR):** Pure Sift collapses to near-chance (~2.05%) due to the fundamental fading memory bottleneck of fixed-state SSMs. SiftX introduces a 64-slot bounded memory bank and learned gate to restore retrieval.

---

## Measured Benchmark Results

All benchmarks measured directly on an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**, PyTorch 2.11.0+cu128.

### 1. Task Performance Comparison

| Task | Model | Params | Steps | Train Loss | Val Loss | Val Metric / PPL | Peak VRAM | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Copy Task** | Sift | 66.1K | 3,000 | 0.0013 | 0.0007 | **100.0% accuracy** | 222.8 MB | 173.4s |
| | Transformer | 52.9K | 3,000 | 2.7e-8 | 3.2e-8 | **100.0% accuracy** | 29.6 MB | 34.3s |
| **Adding Task** | Sift | 65.7K | 3,000 | 0.0005 | 0.0004 | **94.9% accuracy** | 1264.1 MB | 1231.4s |
| | Transformer | 63.2K | 3,000 | 7.2e-6 | 4.9e-6 | **100.0% accuracy** | 86.7 MB | 61.7s |
| **WikiText-2** | Sift | 3.79M | 2,000 | 1.2074 | **1.2519** | **3.497 perplexity** | 1993.9 MB | 3536.4s |
| | Transformer | 3.58M | 2,000 | 1.8328 | 1.8080 | 6.098 perplexity | 275.4 MB | 214.3s |
| **Shakespeare** | Sift | 0.82M | 5,000 | 0.8300 | 1.9158 | 6.793 perplexity | 957.3 MB | 2917.5s |
| | Transformer | 0.83M | 5,000 | 1.2795 | **1.5461** | **4.693 perplexity** | 115.6 MB | 74.6s |

### 2. Inference State Memory vs Sequence Length

| Tokens | Sift Recurrent State | Transformer KV Cache | Memory Ratio (TF / Sift) |
|---|---|---|---|
| 256 | **140.0 KiB** | 1,024.0 KiB | 7.3× |
| 512 | **140.0 KiB** | 2,048.0 KiB | 14.6× |
| 1,024 | **140.0 KiB** | 4,096.0 KiB | 29.3× |
| 2,048 | **140.0 KiB** | 8,192.0 KiB | 58.5× |
| 4,096 | **140.0 KiB** | **16,384.0 KiB** | **117.0×** |

---

## SiftX: Bounded Gated Associative Memory

While continuous selective state spaces excel at smooth sequential compression, theoretical works (Arora et al., 2023; B'MOJO, 2024) prove they cannot solve exact multi-query associative recall (MQAR) over long sequences.

**SiftX** addresses this limitation:
- **Bounded Slot Bank:** Maintains a compact $M=64$ slot memory bank (sub-quadratic $O(M \cdot L)$ cost).
- **Learned Straight-Through Gate:** Binary retrieval gate $g_t \in \{0, 1\}$ trained via straight-through estimation with budget regularization.
- **Negative Results & Critical Fixes:**
  - *Gate Poisoning:* Gradients from the budget penalty backpropagating into hidden representations corrupted semantic features. Fixed by detaching inputs before the gating layer (`x.detach()`).
  - *Curriculum Schedule:* Training write policy and read gate concurrently from step 0 led to degenerate closure ($g_t \to 0$). Fixed via a 3-phase curriculum (60% open warmup, 25% budget anneal, 15% enforcement).

For full theoretical derivation, literature review (Taipan, Griffin, Titans, Hydra arXiv:2508.15099, AMOR arXiv:2602.13215), and pilot diagnostics, see [docs/SIFTX_REPORT.md](file:///docs/SIFTX_REPORT.md) and [docs/SIFTX_PILOTS.md](file:///docs/SIFTX_PILOTS.md).

---

## Architectural Deviations from Official Mamba

1. **Pure PyTorch Parallel Scan:** Implements an associative parallel scan algorithm directly in PyTorch rather than using fused CUDA/C++ kernels. Maximizes portability and readability across Windows/Linux/macOS.
2. **Real-Only Parameterization:** Transition parameter matrix $A \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}}$ is initialized purely along real diagonal components without complex-valued state representations.
3. **Explicit Multi-Timescale Channels:** SiftX introduces partitioned decay rates across channels to capture both rapid local and slow long-range dependencies.

---

## Reproduction & Quickstart

### 1. Installation & Unit Tests

```bash
git clone https://github.com/bobby/sift.git
cd sift
python -m venv .venv
.venv\Scripts\activate  # On Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run complete test suite (98 tests passed)
pytest tests/ -v
```

### 2. Training Baseline Models

```bash
# WikiText-2 Language Model (Sift)
python -m training.train configs/sift_wikitext2_gpu.yaml --device cuda

# Transformer Baseline
python -m training.train configs/transformer_wikitext2_gpu.yaml --device cuda
```

### 3. Benchmarking Memory & Latency

```bash
python -m benchmarks.bench_memory --device cuda
python -m benchmarks.bench_latency --device cuda
```

### 4. Interactive 3D Visualizer & Results Dashboard

Open `frontend/index.html` in any browser to inspect the interactive 3D state space walkthrough, memory comparison slider, and measured GPU benchmark cards.

---

## Limitations

- **Model Scale:** Tested on small-scale architectures (~0.8M to 3.8M parameters).
- **Training Throughput:** Pure PyTorch implementation is memory-bandwidth bound on GPU scans; production scale requires fused Triton kernels.
- **Single-Seed Ablations:** SiftX GPU suite was evaluated on single deterministic seeds ($s=0$) due to workstation GPU time constraints.
