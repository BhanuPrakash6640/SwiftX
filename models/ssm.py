"""
State space model math: discretization and initialization.

Continuous SSM:   h'(t) = A h(t) + B x(t),   y(t) = C h(t)
Discrete SSM:     h_t = A_bar h_{t-1} + B_bar x_t,   y_t = C h_t

Mamba uses a DIAGONAL A, so every formula below is elementwise.
"""

import math

import torch
import torch.nn.functional as F


def discretize_zoh(delta, A, B):
    """
    Exact zero-order hold (input held constant over each step of length Δ):
        A_bar = exp(ΔA)
        B_bar = (ΔA)^-1 (exp(ΔA) - I) ΔB  =  (exp(ΔA) - 1) / A * B   (diagonal A)

    delta: (b, L, D)  A: (D, N)  B: (b, L, N)
    returns A_bar, B_bar: (b, L, D, N)
    """
    dA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(dA)
    B_bar = torch.expm1(dA) / A * B.unsqueeze(2)   # expm1 is accurate for small dA
    return A_bar, B_bar


def discretize_simplified(delta, A, B):
    """
    What the official Mamba code (and Sift) actually uses:
        A_bar = exp(ΔA)      (exact)
        B_bar = Δ B          (first-order approximation of the ZOH B_bar)
    Since (e^z - 1)/z = 1 + z/2 + ..., the two agree when |ΔA| is small.
    """
    A_bar = torch.exp(delta.unsqueeze(-1) * A)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
    return A_bar, B_bar


def init_A_log(d_inner, d_state):
    """
    S4D-Real initialization: A_n = -(n + 1) for n = 0..N-1, same for every channel.
    We store log(-A) and use A = -exp(A_log), so A stays negative during training
    no matter what the optimizer does. Negative A makes exp(ΔA) < 1: a stable,
    fading memory instead of one that explodes.
    """
    A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)  # (D, N)
    return torch.log(A)


def init_dt_proj(dt_proj, dt_rank, dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, dt_scale=1.0):
    """
    Initialize the Δ projection so that, at the start of training,
    softplus(bias) is spread log-uniformly over [dt_min, dt_max].
    Small Δ means "remember for long"; large Δ means "focus on the current token".
    Starting with a mix lets different channels specialize in different timescales.
    """
    std = dt_rank ** -0.5 * dt_scale
    torch.nn.init.uniform_(dt_proj.weight, -std, std)
    d_inner = dt_proj.out_features
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: if softplus(x) = dt then x = dt + log(1 - exp(-dt)).
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    dt_proj.bias._no_reinit = True


def softplus(x):
    return F.softplus(x)
