import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssm import discretize_simplified, discretize_zoh, init_A_log, init_dt_proj


def test_zoh_matches_matrix_formula():
    # For diagonal A, (ΔA)^-1 (exp(ΔA) - I) ΔB computed with real matrices
    # must equal our elementwise formula.
    torch.manual_seed(0)
    N = 5
    a = -torch.rand(N).double() - 0.1
    dt = torch.tensor(0.3).double()
    Bv = torch.randn(N).double()
    dA = torch.diag(dt * a)
    B_bar_matrix = torch.linalg.inv(dA) @ (torch.matrix_exp(dA) - torch.eye(N).double()) @ (dt * Bv)
    _, B_bar = discretize_zoh(dt.view(1, 1, 1), a.view(1, N), Bv.view(1, 1, N))
    assert torch.allclose(B_bar.flatten(), B_bar_matrix, atol=1e-12)


def test_simplified_close_to_zoh_for_small_delta():
    A = -torch.arange(1, 17).float().view(1, 16)
    B = torch.randn(1, 1, 16)
    small = torch.full((1, 1, 1), 1e-3)
    _, b_zoh = discretize_zoh(small, A, B)
    _, b_simple = discretize_simplified(small, A, B)
    assert torch.allclose(b_zoh, b_simple, rtol=1e-2, atol=1e-6)


def test_A_negative_and_stable():
    A = -torch.exp(init_A_log(8, 16))
    assert (A < 0).all()
    assert torch.allclose(A[0], -torch.arange(1, 17).float())
    dA = torch.exp(torch.rand(8, 1) * A)
    assert ((dA > 0) & (dA < 1)).all()


def test_dt_init_range():
    torch.manual_seed(0)
    proj = nn.Linear(8, 512)
    init_dt_proj(proj, 8, dt_min=1e-3, dt_max=1e-1)
    dt = F.softplus(proj.bias)
    assert dt.min() >= 1e-3 * 0.99 and dt.max() <= 1e-1 * 1.01
