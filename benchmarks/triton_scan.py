"""
Selective-scan kernel benchmark on the GPU: PyTorch parallel scan vs the fused Triton kernel.

    python -m benchmarks.triton_scan

Times forward and forward+backward at growing sequence lengths (batch 8, D=256, N=16, the
shape of one Sift block) and records peak GPU memory. Also re-checks Triton against the
reference on the GPU first; if Triton is missing or wrong, it records why and exits cleanly.
"""

import json
import os
import traceback

import torch

from benchmarks.common import RESULTS, timeit
from models.selective_scan import selective_scan_fast, selective_scan_ref


def inputs(b, L, D, N, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    u = torch.randn(b, L, D, device=dev, generator=g, requires_grad=True)
    dl = (torch.rand(b, L, D, device=dev, generator=g) * 0.1 + 0.001).requires_grad_()
    A = (-torch.rand(D, N, device=dev, generator=g) - 0.5).requires_grad_()
    B = torch.randn(b, L, N, device=dev, generator=g, requires_grad=True)
    C = torch.randn(b, L, N, device=dev, generator=g, requires_grad=True)
    Dp = torch.randn(D, device=dev, generator=g, requires_grad=True)
    return u, dl, A, B, C, Dp


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "rows": []}
    path = os.path.join(RESULTS, "triton_scan_cuda.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU")
        from models.triton_scan import HAS_TRITON
        if not HAS_TRITON:
            raise RuntimeError("Triton not installed (pip install triton-windows on Windows)")
        x = inputs(2, 300, 64, 16, "cuda")
        y1 = selective_scan_ref(*x)
        y2 = selective_scan_fast(*x, mode="triton")
        g1 = torch.autograd.grad(y1.sum(), x)
        g2 = torch.autograd.grad(y2.sum(), x)
        err = max([(y1 - y2).abs().max().item()] + [(a - c).abs().max().item() for a, c in zip(g1, g2)])
        out["max_abs_error_vs_reference"] = err
        print(f"triton vs reference: max abs error {err:.2e}")
        for L in (256, 512, 1024, 2048, 4096, 8192):
            for mode in ("parallel", "triton"):
                x = inputs(8, L, 256, 16, "cuda")
                fwd = lambda: selective_scan_fast(*x, mode=mode)          # noqa: E731
                def fb():
                    y = selective_scan_fast(*x, mode=mode)
                    torch.autograd.grad(y.sum(), x)
                torch.cuda.reset_peak_memory_stats()
                tf = timeit(lambda: torch.no_grad()(fwd)(), "cuda")
                tb = timeit(fb, "cuda")
                mem = torch.cuda.max_memory_allocated() / 2**20
                row = {"L": L, "mode": mode, "fwd_ms": tf * 1000, "fwd_bwd_ms": tb * 1000, "peak_mb": mem}
                out["rows"].append(row)
                print(row)
                del x
                torch.cuda.empty_cache()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()[-2000:]
        print("triton benchmark skipped:", out["error"])
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
