import math

import pytest
import torch

from models.selective_scan import (
    parallel_scan,
    selective_scan_fast,
    selective_scan_ref,
    sequential_scan,
)


def make_inputs(b=2, L=6, D=3, N=4, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, L, D, generator=g)
    delta = torch.rand(b, L, D, generator=g) * 0.5 + 0.01   # positive
    A = -torch.rand(D, N, generator=g) - 0.1                # negative
    B = torch.randn(b, L, N, generator=g)
    C = torch.randn(b, L, N, generator=g)
    Dskip = torch.randn(D, generator=g)
    return [t.to(dtype) for t in (u, delta, A, B, C, Dskip)]


# ---------------- reference scan ----------------
def test_output_shape():
    u, delta, A, B, C, Dskip = make_inputs()
    assert selective_scan_ref(u, delta, A, B, C, Dskip).shape == u.shape


def test_hand_computed_tiny_case():
    # Δ=1, A=-ln2 -> A_bar=0.5; B=C=1; u=[2,4,6]
    # h1=2, h2=0.5*2+4=5, h3=0.5*5+6=8.5
    u = torch.tensor([[[2.0], [4.0], [6.0]]])
    delta = torch.ones(1, 3, 1)
    A = torch.tensor([[-math.log(2.0)]])
    B = torch.ones(1, 3, 1)
    C = torch.ones(1, 3, 1)
    y = selective_scan_ref(u, delta, A, B, C)
    assert torch.allclose(y.flatten(), torch.tensor([2.0, 5.0, 8.5]), atol=1e-6)


def test_causal():
    u, delta, A, B, C, Dskip = make_inputs()
    y1 = selective_scan_ref(u, delta, A, B, C, Dskip)
    u2 = u.clone()
    u2[:, 4:] += 10.0
    y2 = selective_scan_ref(u2, delta, A, B, C, Dskip)
    assert torch.allclose(y1[:, :4], y2[:, :4])
    assert not torch.allclose(y1[:, 4:], y2[:, 4:])


def test_reference_gradcheck():
    tensors = [t.double().requires_grad_() for t in make_inputs()]
    assert torch.autograd.gradcheck(selective_scan_ref, tensors)


# ---------------- associative scan ----------------
@pytest.mark.parametrize("L", [1, 2, 5, 8, 33, 128])
def test_parallel_scan_matches_loop(L):
    g = torch.Generator().manual_seed(L)
    a = torch.rand(3, L, 4, 2, generator=g)
    x = torch.randn(3, L, 4, 2, generator=g)
    assert torch.allclose(parallel_scan(a, x), sequential_scan(a, x), atol=1e-5)


# ---------------- fast scan vs reference ----------------
@pytest.mark.parametrize("mode", ["parallel", "sequential", "fused"])
@pytest.mark.parametrize("L", [1, 7, 64])
def test_fast_forward_matches_reference(mode, L):
    u, delta, A, B, C, Dskip = make_inputs(b=2, L=L, D=8, N=16, seed=1)
    y_ref = selective_scan_ref(u, delta, A, B, C, Dskip)
    y_fast = selective_scan_fast(u, delta, A, B, C, Dskip, mode=mode)
    assert torch.allclose(y_ref, y_fast, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["parallel", "sequential", "fused"])
def test_fast_backward_matches_reference(mode):
    inputs = make_inputs(b=2, L=20, D=5, N=6, seed=2, dtype=torch.float64)
    a = [t.clone().requires_grad_() for t in inputs]
    b = [t.clone().requires_grad_() for t in inputs]
    gy = torch.randn(2, 20, 5, dtype=torch.float64)
    (selective_scan_ref(*a) * gy).sum().backward()
    (selective_scan_fast(*b, mode=mode) * gy).sum().backward()
    for ta, tb in zip(a, b):
        assert torch.allclose(ta.grad, tb.grad, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("mode", ["parallel", "sequential", "fused"])
def test_fast_gradcheck(mode):
    tensors = [t.double().requires_grad_() for t in make_inputs()]
    fn = lambda *t: selective_scan_fast(*t, mode=mode)  # noqa: E731
    assert torch.autograd.gradcheck(fn, tensors)


def test_fast_without_skip():
    u, delta, A, B, C, _ = make_inputs()
    assert torch.allclose(
        selective_scan_ref(u, delta, A, B, C), selective_scan_fast(u, delta, A, B, C), atol=1e-5
    )
