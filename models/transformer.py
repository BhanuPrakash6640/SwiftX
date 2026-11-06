"""
Transformer baseline: a GPT-style decoder with pre-norm blocks.

    tokens → Embedding + learned positions → [ x += Attn(LN(x)); x += MLP(LN(x)) ] × n → LN → head

Uses PyTorch's fused scaled_dot_product_attention (a strong, optimized baseline)
and a KV cache for generation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 65
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 1024
    mlp_ratio: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def _heads(self, t):
        b, L, d = t.shape
        return t.view(b, L, self.n_heads, d // self.n_heads).transpose(1, 2)

    def forward(self, x, cache=None):
        b, L, d = x.shape
        q, k, v = map(self._heads, self.qkv(x).split(d, dim=-1))   # (b, h, L, hd)
        if cache is not None:
            # Generation: append new keys/values; the cache GROWS with every token.
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        is_causal = cache is None or L > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(b, L, d)
        return self.proj(y)


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class TransformerBackbone(nn.Module):
    """Continuous-input backbone (b, L, d_model) -> (b, L, d_model), with positions."""

    def __init__(self, cfg):
        super().__init__()
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)
        for layer in self.layers:  # GPT-2 style scaled init on residual projections
            for w in (layer.attn.proj.weight, layer.mlp[2].weight):
                nn.init.normal_(w, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, caches=None, start_pos=0):
        L = x.shape[1]
        pos = torch.arange(start_pos, start_pos + L, device=x.device)
        x = x + self.pos(pos)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i])
        return self.ln_f(x)


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.backbone = TransformerBackbone(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.backbone(self.embedding(idx)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---------------- KV-cache generation ----------------
    def allocate_state(self, batch=None, device=None):
        return [dict() for _ in self.backbone.layers]

    def step(self, token, caches, pos):
        x = self.embedding(token[:, None])
        return self.lm_head(self.backbone(x, caches, start_pos=pos))[:, -1]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        from .sift_model import sample
        caches = self.allocate_state()
        x = self.embedding(idx)
        logits = self.lm_head(self.backbone(x, caches))[:, -1]
        out, pos = [idx], idx.shape[1]
        window = self.cfg.max_seq_len
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_k)
            out.append(nxt[:, None])
            if pos < window:
                logits = self.step(nxt, caches, pos)
                pos += 1
            else:
                # Past the learned positions: slide the window, recompute without cache.
                ctx = torch.cat(out, dim=1)[:, -window:]
                logits = self(ctx)[0][:, -1]
        return torch.cat(out, dim=1)
