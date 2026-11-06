"""
How cost grows with sequence length, plus a comparison of the scan implementations.

1. Training step (forward + backward) time vs L, batch 4, for both models.
2. Selective-scan implementations on one layer-sized problem:
   reference (autograd loop) vs parallel vs sequential vs fused.

    python -m benchmarks.sequence_scaling
"""

import torch

from benchmarks.common import device, models, save, timeit
from models.selective_scan import selective_scan_fast, selective_scan_ref


def train_step_scaling(dev, lengths, batch=4):
    ms = models(dev)
    out = {k: {} for k in ms}
    for L in lengths:
        x = torch.randint(0, 65, (batch, L), device=dev)
        for name, m in ms.items():
            def step():
                m.zero_grad(set_to_none=True)
                _, loss = m(x, x)
                loss.backward()
            out[name][L] = timeit(step, dev, warmup=1, repeat=2) * 1000
            print(f"train step L={L:5d} {name:12s} {out[name][L]:9.1f} ms")
    return out


def scan_impls(dev, b=8, L=256, D=256, N=16):
    torch.manual_seed(0)
    mk = lambda *s: torch.randn(*s, device=dev)  # noqa: E731
    u, B, C = mk(b, L, D), mk(b, L, N), mk(b, L, N)
    delta = torch.rand(b, L, D, device=dev) * 0.1 + 1e-3
    A = -torch.rand(D, N, device=dev) - 0.1
    Dp = mk(D)
    impls = {
        "reference (autograd loop)": lambda *t: selective_scan_ref(*t),
        "parallel (associative scan)": lambda *t: selective_scan_fast(*t, mode="parallel"),
        "sequential (custom backward)": lambda *t: selective_scan_fast(*t, mode="sequential"),
        "fused (chunked recompute)": lambda *t: selective_scan_fast(*t, mode="fused"),
    }
    out = {}
    for name, fn in impls.items():
        ts = [t.clone().requires_grad_() for t in (u, delta, A, B, C, Dp)]
        fwd = timeit(lambda: fn(*ts), dev) * 1000
        def fb():
            fn(*ts).sum().backward()
        out[name] = {"forward_ms": fwd, "fwd_bwd_ms": timeit(fb, dev) * 1000}
        print(f"{name:32s} fwd {out[name]['forward_ms']:8.1f} ms   fwd+bwd {out[name]['fwd_bwd_ms']:8.1f} ms")
    return {"shape": dict(b=b, L=L, D=D, N=N), "results": out}


def main():
    dev = device()
    scaling = train_step_scaling(dev, [128, 256, 512, 1024, 2048, 4096])
    scans = scan_impls(dev)
    save("sequence_scaling", {"train_step_ms_batch4": scaling, "scan_implementations": scans})


if __name__ == "__main__":
    main()
