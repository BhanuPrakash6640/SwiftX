"""
Selective scan: the "notebook update" at the heart of Sift (Mamba-1).

Three implementations, all computing exactly the same function:

1. selective_scan_ref
   Slow, simple, token-by-token loop. The ground truth every other version is
   tested against. Uses plain autograd, so it stores every intermediate.

2. parallel_scan  (the associative scan)
   Computes all prefix states h_1..h_L of  h_t = a_t * h_{t-1} + x_t  in
   log2(L) parallel steps instead of L sequential ones (Hillis-Steele scan).

3. selective_scan_fast
   A custom autograd.Function that mirrors the *ideas* of the official
   hardware-aware kernel in pure PyTorch:
     - it never stores the big (b, L, D, N) state tensors for backward;
       it saves only the small inputs and RECOMPUTES the states in backward;
     - the backward pass is itself a (reverse) scan.
   What it does NOT do (see docs/ARCHITECTURE.md): kernel fusion and keeping
   the state in on-chip SRAM. Those need a CUDA/Triton kernel.

Shapes (letters used everywhere in this project):
    b = batch size        L = sequence length
    D = d_inner channels  N = d_state (notebook size per channel)
"""

import torch


# ---------------------------------------------------------------------------
# 1. Reference implementation
# ---------------------------------------------------------------------------
def selective_scan_ref(u, delta, A, B, C, D=None):
    """
    u:     (b, L, D)  input to the scan
    delta: (b, L, D)  step size Δ (> 0)
    A:     (D, N)     decay rates (< 0)
    B:     (b, L, N)  write vector per token (shared across channels)
    C:     (b, L, N)  read vector per token (shared across channels)
    D:     (D,)       optional skip term
    returns y: (b, L, D)
    """
    b, L, d_inner = u.shape
    N = A.shape[1]

    # Zero-order hold for A:  A_bar = exp(Δ A), always in (0, 1).
    A_bar = torch.exp(delta.unsqueeze(-1) * A)                     # (b, L, D, N)
    # Simplified discretization of B used by the official code: B_bar = Δ B.
    Bu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)    # (b, L, D, N)

    h = torch.zeros(b, d_inner, N, dtype=u.dtype, device=u.device)
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + Bu[:, t]                             # update notebook
        ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))          # read notebook
    y = torch.stack(ys, dim=1)                                     # (b, L, D)
    if D is not None:
        y = y + u * D
    return y


# ---------------------------------------------------------------------------
# 2. Associative (parallel) scan
# ---------------------------------------------------------------------------
def parallel_scan(a, x):
    """
    Solve h_t = a_t * h_{t-1} + x_t (h_0 = 0) for every t, along dim 1.

    Why this can be parallel: each step is an affine map h -> a*h + x, and
    composing two such maps gives another one:
        (a1, x1) then (a2, x2)  =  (a1*a2,  a2*x1 + x2)
    This "combine" is associative, so prefixes can be grouped any way we like.
    Hillis-Steele: at round k, every position t combines with t - 2^k.
    After log2(L) rounds, position t holds the composition of steps 1..t.

    Work O(L log L), depth O(log L). a, x: (b, L, ...). Returns h like x.
    """
    a = a.clone()
    x = x.clone()
    L = a.shape[1]
    offset = 1
    while offset < L:
        # Right-hand sides are new tensors, so reading old values is safe.
        x[:, offset:] = x[:, offset:] + a[:, offset:] * x[:, :-offset]
        a[:, offset:] = a[:, offset:] * a[:, :-offset]
        offset *= 2
    return x


def sequential_scan(a, x):
    """Same result as parallel_scan, computed with a plain loop (no autograd needed)."""
    h = torch.zeros_like(x[:, 0])
    out = torch.empty_like(x)
    for t in range(x.shape[1]):
        h = a[:, t] * h + x[:, t]
        out[:, t] = h
    return out


def _scan(a, x, mode):
    if mode == "auto":
        # A GPU runs the log-depth version well. On a CPU, the loop does
        # ~log2(L) times less work and is usually faster (see benchmarks).
        mode = "parallel" if a.is_cuda else "sequential"
    return parallel_scan(a, x) if mode == "parallel" else sequential_scan(a, x)


# ---------------------------------------------------------------------------
# 3. Memory-efficient selective scan with a hand-written backward pass
# ---------------------------------------------------------------------------
class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, mode):
        with torch.no_grad():
            dA = torch.exp(delta.unsqueeze(-1) * A)                       # (b,L,D,N)
            dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b,L,D,N)
            h = _scan(dA, dBu, mode)                                      # (b,L,D,N)
            y = torch.einsum("bldn,bln->bld", h, C)
            if D is not None:
                y = y + u * D
        # Save only the small inputs. The big tensors (dA, dBu, h) are freed now
        # and recomputed in backward: the same trade the official kernel makes.
        ctx.save_for_backward(u, delta, A, B, C, D)
        ctx.mode = mode
        ctx.has_D = D is not None
        return y

    @staticmethod
    def backward(ctx, gy):
        u, delta, A, B, C, D = ctx.saved_tensors
        mode = ctx.mode

        # Recompute the forward intermediates.
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        h = _scan(dA, dBu, mode)

        # y_t = <h_t, C_t>  ->  dL/dC and the "direct" gradient into each h_t.
        gC = torch.einsum("bld,bldn->bln", gy, h)
        gh_direct = gy.unsqueeze(-1) * C.unsqueeze(2)                 # (b,L,D,N)

        # h_t feeds h_{t+1} through a_{t+1}, so the total gradient obeys
        #   g_t = gh_direct_t + a_{t+1} * g_{t+1}
        # which is the SAME scan run backwards in time.
        a_next = torch.zeros_like(dA)
        a_next[:, :-1] = dA[:, 1:]
        g = _scan(a_next.flip(1), gh_direct.flip(1), mode).flip(1)   # dL/dh_t

        # h_t = dA_t * h_{t-1} + dBu_t
        g_dBu = g
        h_prev = torch.zeros_like(h)
        h_prev[:, 1:] = h[:, :-1]
        g_dA = g * h_prev

        # Chain rule back to the original inputs.
        g_dA_pre = g_dA * dA                      # d exp(z)/dz = exp(z), z = delta*A
        dt = delta.unsqueeze(-1)
        gdelta = (g_dA_pre * A).sum(-1) + (g_dBu * B.unsqueeze(2)).sum(-1) * u
        gA = (g_dA_pre * dt).sum(dim=(0, 1))
        gB = torch.einsum("bldn,bld->bln", g_dBu, delta * u)
        gu = (g_dBu * B.unsqueeze(2)).sum(-1) * delta
        gD = None
        if ctx.has_D:
            gu = gu + gy * D
            gD = (gy * u).sum(dim=(0, 1))
        return gu, gdelta, gA, gB, gC, gD, None


# ---------------------------------------------------------------------------
# 4. Time-fused scan with chunked recomputation (fastest on CPU)
# ---------------------------------------------------------------------------
class _FusedScanFn(torch.autograd.Function):
    """
    The closest pure-PyTorch analogue of the official kernel's memory strategy:
      - The big (b, L, D, N) tensors are NEVER built. Each time step works on a
        small (b, D, N) tile that stays in CPU cache (the kernel keeps it in SRAM).
      - Forward stores only the state at every chunk_size-th step.
      - Backward walks chunks from last to first, recomputes the states inside
        one chunk from its checkpoint, then runs the reverse recurrence.
    Memory saved for backward: O(b * L/chunk * D * N) instead of O(b * L * D * N).
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, chunk):
        b, L, d_inner = u.shape
        h = u.new_zeros(b, d_inner, A.shape[1])
        y = torch.empty_like(u)
        ckpts = []
        with torch.no_grad():
            for t in range(L):
                if t % chunk == 0:
                    ckpts.append(h)
                dt = delta[:, t].unsqueeze(-1)                                  # (b, D, 1)
                h = torch.exp(dt * A) * h + (dt * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
                y[:, t] = torch.einsum("bdn,bn->bd", h, C[:, t])
            if D is not None:
                y = y + u * D
        ctx.save_for_backward(u, delta, A, B, C, D, torch.stack(ckpts, 1))
        ctx.chunk = chunk
        ctx.has_D = D is not None
        return y

    @staticmethod
    def backward(ctx, gy):
        u, delta, A, B, C, D, ckpts = ctx.saved_tensors
        chunk = ctx.chunk
        L = u.shape[1]
        gu = torch.zeros_like(u)
        gdelta = torch.empty_like(delta)
        gB = torch.empty_like(B)
        gC = torch.empty_like(C)
        gA = torch.zeros_like(A)
        g = torch.zeros_like(ckpts[:, 0])                  # dL/dh carried backwards in time
        for c in reversed(range(ckpts.shape[1])):
            s, e = c * chunk, min(L, (c + 1) * chunk)
            # Recompute the states of this chunk from its checkpoint.
            hs, dAs = [ckpts[:, c]], []
            for t in range(s, e):
                dt = delta[:, t].unsqueeze(-1)
                dA = torch.exp(dt * A)
                dAs.append(dA)
                hs.append(dA * hs[-1] + (dt * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1))
            # Reverse recurrence through the chunk.
            for i in reversed(range(e - s)):
                t = s + i
                h_t, h_prev, dA = hs[i + 1], hs[i], dAs[i]
                gyt = gy[:, t]
                gC[:, t] = torch.einsum("bd,bdn->bn", gyt, h_t)
                g = g + gyt.unsqueeze(-1) * C[:, t].unsqueeze(1)                # total dL/dh_t
                g_pre = g * h_prev * dA                                         # through exp(Δ A)
                dt = delta[:, t]
                gBu = (g * B[:, t].unsqueeze(1)).sum(-1)                        # (b, D)
                gdelta[:, t] = (g_pre * A).sum(-1) + gBu * u[:, t]
                gA += (g_pre * dt.unsqueeze(-1)).sum(0)
                gB[:, t] = torch.einsum("bdn,bd->bn", g, dt * u[:, t])
                gu[:, t] = gBu * dt
                g = g * dA                                                      # hand to h_{t-1}
        gD = None
        if ctx.has_D:
            gu = gu + gy * D
            gD = (gy * u).sum(dim=(0, 1))
        return gu, gdelta, gA, gB, gC, gD, None


def selective_scan_fast(u, delta, A, B, C, D=None, mode="auto", chunk=16):
    """
    Drop-in replacement for selective_scan_ref.
    mode: 'auto'       -> 'parallel' on GPU, 'fused' on CPU
          'parallel'   -> associative scan over materialized tensors, recompute in backward
          'sequential' -> same, but with a plain loop for the scan
          'fused'      -> per-time-step loop, nothing (b,L,D,N) ever built, chunked recompute
          'triton'     -> fused Triton GPU kernel (models/triton_scan.py); falls back to
                          'parallel' if Triton is missing or the tensors are on CPU
    """
    if mode == "triton":
        from .triton_scan import HAS_TRITON, selective_scan_triton
        if HAS_TRITON and u.is_cuda:
            return selective_scan_triton(u, delta, A, B, C, D)
        mode = "parallel" if u.is_cuda else "fused"
    if mode == "auto":
        mode = "parallel" if u.is_cuda else "fused"
    if mode == "fused":
        return _FusedScanFn.apply(u, delta, A, B, C, D, chunk)
    return _SelectiveScanFn.apply(u, delta, A, B, C, D, mode)
