"""Triton scan vs the reference loop. On CPU this runs the kernels in Triton's interpreter."""

import os
import subprocess
import sys

import pytest

CHECK = r'''
import torch
from models.selective_scan import selective_scan_ref
from models.triton_scan import selective_scan_triton
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for (b, L, D, N, K, hasD) in [(2, 37, 40, 16, 8, True), (1, 20, 8, 5, 64, False)]:
    u = torch.randn(b, L, D, device=dev, requires_grad=True)
    dl = (torch.rand(b, L, D, device=dev) * 0.2 + 0.01).requires_grad_()
    A = (-torch.rand(D, N, device=dev) - 0.5).requires_grad_()
    B = torch.randn(b, L, N, device=dev, requires_grad=True)
    C = torch.randn(b, L, N, device=dev, requires_grad=True)
    Dp = torch.randn(D, device=dev, requires_grad=True) if hasD else None
    ins = [u, dl, A, B, C] + ([Dp] if hasD else [])
    gy = torch.randn(b, L, D, device=dev)
    y1 = selective_scan_ref(u, dl, A, B, C, Dp)
    g1 = torch.autograd.grad(y1, ins, gy)
    y2 = selective_scan_triton(u, dl, A, B, C, Dp, chunk=K)
    g2 = torch.autograd.grad(y2, ins, gy)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4), (y1 - y2).abs().max()
    for name, a, c in zip("u delta A B C D".split(), g1, g2):
        assert torch.allclose(a, c, atol=2e-3, rtol=2e-3), (name, (a - c).abs().max())
print("OK")
'''


def test_triton_scan_matches_reference():
    pytest.importorskip("triton")
    env = dict(os.environ)
    import torch
    if not torch.cuda.is_available():
        env["TRITON_INTERPRET"] = "1"          # run the kernels on CPU, slowly but exactly
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", CHECK], cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-3000:]
