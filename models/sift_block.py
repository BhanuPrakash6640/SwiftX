"""
The Sift block: a from-scratch implementation of the Mamba-1 "mixer".

    x (b, L, d_model)
      │ in_proj
      ├──────────────── z (gate)  (b, L, d_inner)
      ▼
      x (b, L, d_inner)
      │ causal depthwise conv1d (kernel d_conv) → SiLU
      │ x_proj → [ Δ_low (dt_rank) | B (d_state) | C (d_state) ]
      │ dt_proj + softplus → Δ (b, L, d_inner)
      │ selective scan (A, B, C, Δ, D)
      ▼
      y * SiLU(z)
      │ out_proj
      ▼
    out (b, L, d_model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_scan import selective_scan_fast, selective_scan_ref
from .ssm import init_A_log, init_dt_proj


class SiftBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=1e-3,
        dt_max=1e-1,
        conv_bias=True,
        bias=False,
        scan_impl="fast",       # "fast" (custom backward) or "ref" (plain loop)
        scan_mode="auto",       # for "fast": "auto" | "parallel" | "sequential"
        timescales=None,        # SiftX: list of (dt_min, dt_max), one per channel group
        pathway_gate=False,     # SiftX: learned per-token mix of the timescale groups
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_impl = scan_impl
        self.scan_mode = scan_mode

        # A. Input projection into two branches (x and gate z).
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # B. Causal depthwise conv: each channel has its own 4-tap filter.
        #    padding = d_conv - 1 on both sides; we keep only the first L outputs,
        #    so output t only sees inputs t-3..t (never the future).
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=conv_bias,
        )

        # D. Selective parameters: Δ (low rank), B, C all computed from x.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        init_dt_proj(self.dt_proj, self.dt_rank, dt_min, dt_max)
        # Multi-timescale pathways (SiftX extension, not in Mamba-1): split the channels into
        # groups whose step sizes start in different ranges: large Δ = fast, short memory;
        # small Δ = slow, long memory. Training can still move each channel anywhere.
        self.timescales = [tuple(t) for t in timescales] if timescales else None
        self.n_groups = len(self.timescales) if self.timescales else 1
        # channel -> group index (groups may differ in size by one channel)
        self.register_buffer("group_of", torch.zeros(self.d_inner, dtype=torch.long), persistent=False)
        if self.timescales:
            bounds = torch.linspace(0, self.d_inner, self.n_groups + 1).round().long().tolist()
            for i, (lo, hi) in enumerate(self.timescales):
                a, b = bounds[i], bounds[i + 1]
                self.group_of[a:b] = i
                tmp = nn.Linear(self.dt_rank, b - a, bias=True)
                init_dt_proj(tmp, self.dt_rank, lo, hi)
                with torch.no_grad():
                    self.dt_proj.bias[a:b] = tmp.bias
        self.pathway = None
        if pathway_gate and self.n_groups > 1:
            self.pathway = nn.Linear(self.d_inner, self.n_groups)
            nn.init.zeros_(self.pathway.weight)
            nn.init.zeros_(self.pathway.bias)
            self.pathway.bias._no_reinit = True

        # E. SSM parameters that do NOT depend on the input.
        self.A_log = nn.Parameter(init_A_log(self.d_inner, d_state))   # (D, N)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))                # skip
        self.D._no_weight_decay = True

        # H. Output projection.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ------------------------------------------------------------------
    def ssm_params(self, x):
        """x: (b, L, d_inner) after conv+SiLU -> Δ (b,L,D), B (b,L,N), C (b,L,N)."""
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))      # Δ > 0, one value per channel
        return delta, B, C

    def forward(self, hidden):
        """hidden: (b, L, d_model) -> (b, L, d_model)"""
        L = hidden.shape[1]
        xz = self.in_proj(hidden)                                  # (b, L, 2D)
        x, z = xz.chunk(2, dim=-1)                                 # (b, L, D) each

        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)  # causal conv
        x = F.silu(x)

        delta, B, C = self.ssm_params(x)
        A = -torch.exp(self.A_log.float())                         # (D, N), < 0

        # The scan always runs in float32, even under mixed precision: the state
        # is a long product of decays, and low precision drifts.
        dtype = x.dtype
        args = (x.float(), delta.float(), A, B.float(), C.float(), self.D.float())
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.scan_impl == "ref":
                y = selective_scan_ref(*args)
            else:
                y = selective_scan_fast(*args, mode=self.scan_mode)
        y = self._mix_pathways(y.to(dtype), x)

        y = y * F.silu(z)                                          # G. gating
        return self.out_proj(y)                                    # H.

    def _mix_pathways(self, y, x):
        """SiftX: reweight fast/medium/slow channel groups per token (softmax × G, so 1 at init)."""
        if self.pathway is None:
            return y
        w = torch.softmax(self.pathway(x), dim=-1) * self.n_groups            # (..., G)
        return y * w[..., self.group_of]                                      # (..., D)

    # ------------------------------------------------------------------
    # Recurrent mode for generation: one token in, one token out, with a
    # FIXED-SIZE state (conv window + SSM notebook). No growing cache.
    # ------------------------------------------------------------------
    def allocate_state(self, batch, device=None, dtype=torch.float32):
        conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(self, hidden, state):
        """hidden: (b, d_model) one token. state: (conv_state, ssm_state). Updated in place."""
        conv_state, ssm_state = state
        x, z = self.in_proj(hidden).chunk(2, dim=-1)               # (b, D)

        # Shift the conv window left and append the new input.
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        w = self.conv1d.weight.squeeze(1)                          # (D, d_conv)
        x = (conv_state * w).sum(-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = F.silu(x)

        delta, B, C = self.ssm_params(x)                           # (b,D), (b,N), (b,N)
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(delta.unsqueeze(-1) * A)                    # (b, D, N)
        dBx = delta.unsqueeze(-1) * B.unsqueeze(1) * x.unsqueeze(-1)
        ssm_state.copy_(ssm_state * dA + dBx)
        y = (ssm_state * C.unsqueeze(1)).sum(-1) + self.D * x
        y = self._mix_pathways(y, x)
        y = y * F.silu(z)
        return self.out_proj(y)
