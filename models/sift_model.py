"""
Full Sift language model (Mamba-1 backbone).

    tokens → Embedding → [ x = x + SiftBlock(RMSNorm(x)) ] × n_layers → RMSNorm → LM head
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sift_block import SiftBlock


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned per-feature gain."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass
class SiftConfig:
    vocab_size: int = 65
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: object = "auto"
    scan_impl: str = "fast"
    scan_mode: str = "auto"
    tie_embeddings: bool = True
    timescales: object = None
    pathway_gate: bool = False


class ResidualBlock(nn.Module):
    """Pre-norm residual wrapper: x + mixer(norm(x)). (No separate MLP in Mamba.)"""

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = SiftBlock(
            cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
            dt_rank=cfg.dt_rank, scan_impl=cfg.scan_impl, scan_mode=cfg.scan_mode,
            timescales=cfg.timescales, pathway_gate=cfg.pathway_gate,
        )

    def forward(self, x):
        return x + self.mixer(self.norm(x))

    def step(self, x, state):
        return x + self.mixer.step(self.norm(x), state)


class SiftBackbone(nn.Module):
    """Stack of residual Sift blocks on continuous inputs (b, L, d_model)."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self._init_weights(cfg.n_layers)

    def _init_weights(self, n_layers):
        # Follows the official initialization: zero linear biases (except Δ's),
        # and shrink each block's output projection by 1/sqrt(n_layers) so the
        # residual stream doesn't grow with depth.
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.zeros_(m.bias)
        for layer in self.layers:
            w = layer.mixer.out_proj.weight
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            with torch.no_grad():
                w /= math.sqrt(n_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class SiftLM(nn.Module):
    def __init__(self, cfg: SiftConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.backbone = SiftBackbone(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, idx, targets=None):
        """idx: (b, L) token ids -> logits (b, L, vocab), optional loss."""
        logits = self.lm_head(self.backbone(self.embedding(idx)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---------------- recurrent generation ----------------
    def allocate_state(self, batch, device=None):
        return [l.mixer.allocate_state(batch, device) for l in self.backbone.layers]

    def step(self, token, state):
        """token: (b,) -> logits (b, vocab). Constant time and memory per token."""
        x = self.embedding(token)
        for layer, s in zip(self.backbone.layers, state):
            x = layer.step(x, s)
        return self.lm_head(self.backbone.norm_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Feed the prompt token by token, then sample. idx: (b, L) -> (b, L + new)."""
        state = self.allocate_state(idx.shape[0], idx.device)
        for t in range(idx.shape[1]):
            logits = self.step(idx[:, t], state)
        out = [idx]
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_k)
            out.append(nxt[:, None])
            logits = self.step(nxt, state)
        return torch.cat(out, dim=1)


def sample(logits, temperature=1.0, top_k=None):
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    return torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
