"""
Fused selective scan in Triton (forward + backward), the GPU analogue of the official Mamba kernel.

    y_t = C_t · h_t + D ⊙ u_t,      h_t = exp(Δ_t A) ⊙ h_{t-1} + (Δ_t u_t) B_t

One program owns one batch element and a block of BD channels. Its (BD, N) state lives in
registers for the whole sequence, so the (b, L, D, N) tensor of all states is never written
to GPU memory. That is the whole point of the original CUDA kernel.

Backward uses chunked recomputation: the forward pass saves the state at the start of every
K-step chunk. The backward walks chunks from last to first, recomputes the K states of one chunk
into a small scratch buffer, and runs the reverse recurrence
    g_t = C_t gy_t + exp(Δ_{t+1} A) ⊙ g_{t+1}
Memory saved for backward: O(b · L/K · D · N).

Usage: models/selective_scan.py routes mode="triton" here; if Triton is not installed or the
tensors are not on a GPU, callers fall back to the PyTorch scan. Correctness is tested against
the reference loop in tests/test_triton_scan.py (works on CPU via TRITON_INTERPRET=1).
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:          # pragma: no cover - Triton missing (e.g. plain Windows CPU install)
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fwd_kernel(u, dl, A, B, C, Dp, y, ck, L, D, N, K, NCK,
                    HAS_D: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        pb = tl.program_id(0)
        pd = tl.program_id(1)
        d = pd * BD + tl.arange(0, BD)
        n = tl.arange(0, BN)
        md = d < D
        mn = n < N
        m2 = md[:, None] & mn[None, :]
        a = tl.load(A + d[:, None] * N + n[None, :], mask=m2, other=0.0)
        dd = tl.load(Dp + d, mask=md & HAS_D, other=0.0)
        h = tl.zeros((BD, BN), dtype=tl.float32)
        for t in range(0, L):
            if t % K == 0:                                    # checkpoint: state BEFORE this chunk
                tl.store(ck + ((pb * NCK + t // K) * D + d[:, None]) * N + n[None, :], h, mask=m2)
            off = (pb * L + t) * D + d
            ut = tl.load(u + off, mask=md, other=0.0)
            dt = tl.load(dl + off, mask=md, other=0.0)
            bt = tl.load(B + (pb * L + t) * N + n, mask=mn, other=0.0)
            ct = tl.load(C + (pb * L + t) * N + n, mask=mn, other=0.0)
            h = tl.exp(dt[:, None] * a) * h + (dt * ut)[:, None] * bt[None, :]
            yt = tl.sum(h * ct[None, :], axis=1) + dd * ut
            tl.store(y + off, yt, mask=md)

    @triton.jit
    def _bwd_kernel(u, dl, A, B, C, Dp, gy, ck, hs,
                    du, ddl, dA, dB, dC, dDp, L, D, N, K, NCK,
                    HAS_D: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        pb = tl.program_id(0)
        pd = tl.program_id(1)
        d = pd * BD + tl.arange(0, BD)
        n = tl.arange(0, BN)
        md = d < D
        mn = n < N
        m2 = md[:, None] & mn[None, :]
        a = tl.load(A + d[:, None] * N + n[None, :], mask=m2, other=0.0)
        dd = tl.load(Dp + d, mask=md & HAS_D, other=0.0)
        g = tl.zeros((BD, BN), dtype=tl.float32)              # dL/dh_t, carried backwards
        nxt = tl.zeros((BD, BN), dtype=tl.float32)            # exp(Δ_{t+1} A); 0 after the last step
        gA = tl.zeros((BD, BN), dtype=tl.float32)
        gD = tl.zeros((BD,), dtype=tl.float32)
        for ci in range(0, NCK):
            c = NCK - 1 - ci
            t0 = c * K
            t1 = tl.minimum(t0 + K, L)
            h0 = tl.load(ck + ((pb * NCK + c) * D + d[:, None]) * N + n[None, :], mask=m2, other=0.0)
            h = h0
            for t in range(t0, t1):                           # recompute this chunk's states
                off = (pb * L + t) * D + d
                ut = tl.load(u + off, mask=md, other=0.0)
                dt = tl.load(dl + off, mask=md, other=0.0)
                bt = tl.load(B + (pb * L + t) * N + n, mask=mn, other=0.0)
                h = tl.exp(dt[:, None] * a) * h + (dt * ut)[:, None] * bt[None, :]
                tl.store(hs + ((pb * K + (t - t0)) * D + d[:, None]) * N + n[None, :], h, mask=m2)
            for s in range(0, t1 - t0):                       # reverse recurrence inside the chunk
                t = t1 - 1 - s
                off = (pb * L + t) * D + d
                ut = tl.load(u + off, mask=md, other=0.0)
                dt = tl.load(dl + off, mask=md, other=0.0)
                gyt = tl.load(gy + off, mask=md, other=0.0)
                bt = tl.load(B + (pb * L + t) * N + n, mask=mn, other=0.0)
                ct = tl.load(C + (pb * L + t) * N + n, mask=mn, other=0.0)
                ht = tl.load(hs + ((pb * K + (t - t0)) * D + d[:, None]) * N + n[None, :], mask=m2, other=0.0)
                ip = tl.maximum(t - t0 - 1, 0)
                hp_s = tl.load(hs + ((pb * K + ip) * D + d[:, None]) * N + n[None, :], mask=m2, other=0.0)
                hprev = tl.where(t > t0, hp_s, h0)
                dAt = tl.exp(dt[:, None] * a)
                g = gyt[:, None] * ct[None, :] + nxt * g
                tl.atomic_add(dC + (pb * L + t) * N + n, tl.sum(gyt[:, None] * ht, axis=0), mask=mn)
                tl.atomic_add(dB + (pb * L + t) * N + n, tl.sum(g * (dt * ut)[:, None], axis=0), mask=mn)
                tl.store(du + off, tl.sum(g * bt[None, :], axis=1) * dt + dd * gyt, mask=md)
                tl.store(ddl + off, tl.sum(g * (ut[:, None] * bt[None, :] + a * hprev * dAt), axis=1), mask=md)
                gA += g * hprev * dAt * dt[:, None]
                gD += gyt * ut
                nxt = dAt
        tl.atomic_add(dA + d[:, None] * N + n[None, :], gA, mask=m2)
        tl.atomic_add(dDp + d, gD, mask=md & HAS_D)


def _launch_cfg(D, N):
    BN = triton.next_power_of_2(N)
    BD = 32 if BN <= 16 else 16
    return BD, BN


class TritonScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, K):
        dtype = u.dtype
        u, delta, A, B, C = (t.float().contiguous() for t in (u, delta, A, B, C))
        has_d = D is not None
        Dv = D.float().contiguous() if has_d else torch.zeros(u.shape[-1], device=u.device)
        b, L, Dm = u.shape
        N = A.shape[1]
        NCK = triton.cdiv(L, K)
        BD, BN = _launch_cfg(Dm, N)
        y = torch.empty_like(u)
        ck = torch.empty(b, NCK, Dm, N, device=u.device, dtype=torch.float32)
        grid = (b, triton.cdiv(Dm, BD))
        _fwd_kernel[grid](u, delta, A, B, C, Dv, y, ck, L, Dm, N, K, NCK, HAS_D=has_d, BD=BD, BN=BN)
        ctx.save_for_backward(u, delta, A, B, C, Dv, ck)
        ctx.has_d, ctx.K, ctx.dtype = has_d, K, dtype
        return y.to(dtype)

    @staticmethod
    def backward(ctx, gy):
        u, delta, A, B, C, Dv, ck = ctx.saved_tensors
        K = ctx.K
        b, L, Dm = u.shape
        N = A.shape[1]
        NCK = ck.shape[1]
        BD, BN = _launch_cfg(Dm, N)
        gy = gy.float().contiguous()
        hs = torch.empty(b, K, Dm, N, device=u.device, dtype=torch.float32)
        du, ddl = torch.empty_like(u), torch.empty_like(delta)
        dA, dDv = torch.zeros_like(A), torch.zeros_like(Dv)
        dB, dC = torch.zeros_like(B), torch.zeros_like(C)
        grid = (b, triton.cdiv(Dm, BD))
        _bwd_kernel[grid](u, delta, A, B, C, Dv, gy, ck, hs, du, ddl, dA, dB, dC, dDv,
                          L, Dm, N, K, NCK, HAS_D=ctx.has_d, BD=BD, BN=BN)
        dt = ctx.dtype
        return (du.to(dt), ddl.to(dt), dA, dB.to(dt), dC.to(dt),
                dDv if ctx.has_d else None, None)


def selective_scan_triton(u, delta, A, B, C, D=None, chunk=64):
    """Same contract as selective_scan_ref: u, delta (b,L,D); A (D,N); B, C (b,L,N); D (D,)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    return TritonScanFn.apply(u, delta, A, B, C, D, chunk)
