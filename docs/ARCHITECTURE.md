# How Sift works: the Mamba-1 architecture, explained

This document walks through every equation and tensor shape in Sift, in the order the
code runs. File references point at the implementation.

Notation used everywhere:

| Symbol | Meaning | Sift (Shakespeare) |
|---|---|---|
| `b` | batch size | 16 |
| `L` | sequence length | 256 |
| `d_model` | width of the residual stream | 128 |
| `D = d_inner` | channels inside a block, `expand × d_model` | 256 |
| `N = d_state` | size of each channel's hidden state ("notebook") | 16 |
| `R = dt_rank` | rank of the Δ projection, `ceil(d_model / 16)` | 8 |
| `K = d_conv` | causal convolution width | 4 |

---

## 1. Why: attention vs. recurrence

A Transformer lets each token attend to every earlier token. Training costs O(L²), and at
generation time it keeps a **KV cache** that grows by one entry per token per layer.

A recurrent model keeps a **fixed-size state** `h` and updates it once per token, so it costs
O(L) to train and O(1) per generated token. Classical RNNs were hard to train in parallel and
used fixed update rules. Linear state space models (S4) fixed the training problem but kept
fixed rules: every token was processed the same way, so the model could not decide to ignore
a filler token or latch onto an important one.

**Mamba's idea: make the SSM parameters functions of the input.** That is *selectivity*.

---

## 2. The state space model

Continuous time (`models/ssm.py`):

```
h'(t) = A h(t) + B x(t)          h ∈ R^N
y(t)  = C h(t)
```

Mamba runs one independent SSM per channel (`D` of them), each with an `N`-dimensional
state, and makes `A` **diagonal**. Every matrix operation below is therefore elementwise.

### Discretization

To process tokens we need a discrete update with step size Δ. Assuming the input is held
constant over each step (zero-order hold, ZOH):

```
A_bar = exp(Δ A)
B_bar = (Δ A)^-1 (exp(Δ A) − I) Δ B     =  (exp(Δ A) − 1) / A · B     (diagonal A)

h_t = A_bar · h_{t-1} + B_bar · x_t
y_t = C · h_t
```

**What Sift uses (same as the official code):** `A_bar = exp(ΔA)` exactly, but
`B_bar ≈ Δ B`. Since `(e^z − 1)/z = 1 + z/2 + …`, the two agree when `|ΔA|` is small, and the
simpler form is cheaper. Both versions are in `models/ssm.py`. `tests/test_ssm.py` checks the
exact one against a real matrix exponential, and checks that the two agree for small Δ.

### Parameterization of A (stability)

`A` is stored as `A_log` and used as `A = −exp(A_log)`, so it is **always negative**, whatever
the optimizer does. With `Δ > 0`, `A_bar = exp(ΔA) ∈ (0, 1)`: the state can only decay, never
explode. Initialization is S4D-Real: `A_n = −(n + 1)` for `n = 0..N−1`, which gives each
channel a range of memory timescales.

---

## 3. Selectivity: Δ, B, C from the input (`SiftBlock.ssm_params`)

```
x_dbl = x_proj(x)                      (b, L, D) → (b, L, R + 2N)
dt, B, C = split(x_dbl, [R, N, N])
Δ = softplus(dt_proj(dt))              (b, L, R) → (b, L, D),   Δ > 0
```

- **Δ** (one per channel per token) is the "focus" knob. Large Δ makes `A_bar → 0`, so the
  state resets onto the current token. Small Δ makes `A_bar → 1`, so the token is mostly
  ignored and memory persists. Δ comes from a **low-rank** projection (rank `R`) to save
  parameters.
- **B** `(b, L, N)` decides what gets written into the state. **C** `(b, L, N)` decides what
  gets read out. Both are shared across the `D` channels.
- **Δ initialization** (`init_dt_proj`): the bias is set to `softplus⁻¹(dt)` with `dt` sampled
  log-uniformly in `[0.001, 0.1]`, so channels start out covering a range of timescales.

`tests/test_sift_block.py::test_selective_params_depend_on_input` verifies that Δ, B and C
really change with the input and across time.

---

## 4. The selective scan (`models/selective_scan.py`)

Given `u, Δ: (b,L,D)`, `A: (D,N)`, `B, C: (b,L,N)`, `D_skip: (D,)`:

```
A_bar[t] = exp(Δ_t ⊗ A)                        (b, D, N)
Bx[t]    = Δ_t ⊗ B_t ⊙ u_t                      (b, D, N)
h_t      = A_bar[t] ⊙ h_{t-1} + Bx[t]           (b, D, N)
y_t      = Σ_n h_t[:, :, n] · C_t[:, n] + D_skip ⊙ u_t     (b, D)
```

Because Δ, B and C change at every step, the old trick of computing an SSM as one long
convolution no longer works. That trick needs time-invariant parameters. Sift has four
implementations of the scan, and all of them are tested to agree.

| Implementation | Idea | Notes |
|---|---|---|
| `selective_scan_ref` | Python loop, plain autograd | Ground truth. Stores everything. |
| `parallel_scan` + `_SelectiveScanFn` (`mode="parallel"`) | **Associative scan** | log₂L parallel rounds; default on GPU |
| `_SelectiveScanFn` (`mode="sequential"`) | Same custom backward, loop scan | |
| `_FusedScanFn` (`mode="fused"`) | Per-step tiles + chunked recompute | Default on CPU; fastest here |

### Why the scan can be parallel

Each step is an affine map `h ↦ a·h + x`. Composing two of them gives another affine map:

```
(a₁, x₁) ∘ then (a₂, x₂)  =  (a₁·a₂,  a₂·x₁ + x₂)
```

This combine operator is **associative**, so prefixes can be grouped in any order. The
Hillis–Steele scan in `parallel_scan` combines position `t` with `t − 2^k` in round `k`.
After `log₂ L` rounds, every position holds its full prefix.

### The backward pass is a reverse scan

With `g_t = ∂Loss/∂h_t`, the chain rule gives

```
g_t = C_t ⊙ ∂Loss/∂y_t  +  A_bar[t+1] ⊙ g_{t+1}
```

This is the same linear recurrence run backwards in time. So the backward pass reuses the
scan, and from `g` we get `∂/∂Bx = g` and `∂/∂A_bar = g ⊙ h_{t-1}`, then chain back to Δ, A,
B, C and u. All of these gradients are written out by hand and checked with
`torch.autograd.gradcheck` in float64.

### Memory: recompute instead of store

The official CUDA kernel is fast for two reasons:

1. **It never writes the `(b, L, D, N)` state tensors to GPU main memory.** It loads Δ, A, B,
   C and u into on-chip SRAM, discretizes and scans there, and writes back only `y` of shape
   `(b, L, D)`.
2. **It recomputes the states in the backward pass** instead of saving them.

Sift reproduces the *algorithmic* part of this in pure PyTorch:

- `_SelectiveScanFn` saves only the inputs and recomputes `A_bar`, `Bx` and `h` in backward.
- `_FusedScanFn` goes further. It never builds any `(b, L, D, N)` tensor: each time step works
  on a `(b, D, N)` tile that fits in CPU cache, forward keeps only one state per 16 steps, and
  backward recomputes one chunk at a time. On CPU this is about 3× faster than materializing.

What pure PyTorch **cannot** do is fuse all those small operations into one kernel. Each step
still launches several separate ops. That is why the custom kernel matters, and why Sift is
slower than a Transformer on this hardware (see the README benchmarks). A Triton or CUDA
kernel would be the next step, kept separate and tested against `selective_scan_ref`.

---

## 5. The Sift block (`models/sift_block.py`)

```
hidden (b, L, d_model)
  │ in_proj: Linear(d_model → 2D, no bias)
  ├────────────── z (b, L, D)   gate branch
  ▼
  x (b, L, D)
  │ Conv1d(D, D, K=4, groups=D, padding=3)[..., :L]   causal depthwise conv
  │ SiLU
  │ x_proj → Δ, B, C;  A = −exp(A_log)
  │ selective scan (+ D_skip · x)
  ▼
  y (b, L, D)  ⊙ SiLU(z)          gating
  │ out_proj: Linear(D → d_model, no bias)
  ▼
out (b, L, d_model)
```

**Causal convolution.** Padding by `K−1` on both sides and keeping the first `L` outputs means
output `t` sees only inputs `t−3 … t`. It mixes in very recent context before the SSM.
`test_conv_alone_is_causal` sends an impulse through it to check this.

**Recurrent mode (`step`).** For generation, the block keeps two fixed-size states: the last
`K` conv inputs `(b, D, K)` and the SSM state `(b, D, N)`. One token goes in and one comes out,
in constant time and memory. `test_step_matches_forward` checks that running token by token
gives exactly the same output as the parallel forward pass.

---

## 6. The language model (`models/sift_model.py`)

```
x = Embedding(tokens)                                 (b, L, d_model)
for each of n_layers:  x = x + SiftBlock(RMSNorm(x))  pre-norm residual
logits = LM_head(RMSNorm(x))                          LM_head tied to Embedding
```

Unlike a Transformer, there is **no separate MLP block**: the Sift block is the only layer
type, and the in/out projections play the MLP's role. A Sift block has about `6·d_model²`
parameters, versus about `12·d_model²` for a Transformer layer. That is why the comparison
uses 7 Sift blocks against 4 Transformer layers (825K vs 834K parameters).

Initialization follows the official code: zero linear biases (except Δ's), embeddings
`N(0, 0.02)`, and each block's `out_proj` scaled by `1/√n_layers` so the residual stream
doesn't grow with depth. Weight decay is skipped for `A_log`, `D_skip`, biases and norms.

---

## 7. Deviations from the original, stated explicitly

| Item | Original Mamba-1 | Sift | Why |
|---|---|---|---|
| Scan kernel | Fused CUDA kernel (SRAM, recompute) | Pure PyTorch (associative or fused-loop, recompute) | Educational; a CUDA kernel is a separate, optional step |
| `B_bar` | `Δ B` | `Δ B` | Same. Exact ZOH is also implemented, in `ssm.py` |
| Residual stream | fp32 residual, optional fused add+norm | Plain residual | Same math, no fusion |
| Vocab padding | Padded to a multiple of 8 | Not padded | Speed-only trick |
| Complex A | Supported (for S4D-Lin) | Real A only | Mamba-1 LMs use real A |
| Tokenizer | GPT-NeoX BPE | Character-level | Small datasets, small vocab |

Everything else (projections, conv, Δ/B/C parameterization, A and Δ initialization, D skip,
gating, pre-norm RMSNorm residual, tied head) follows the paper and the official
`mamba_simple.py` / `mixer_seq_simple.py`.
