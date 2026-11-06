"""
SiftX: an adaptive multi-memory sequence model (research prototype, working name).

    tokens → Embedding → [ Sift blocks (multi-timescale) … RecallLayer … Sift blocks ] → RMSNorm → head

Three kinds of memory:
    short/compressed  - selective SSM state in every Sift block, split into fast / medium / slow
                        timescale pathways with a learned per-token mix   (models/sift_block.py)
    exact retrieval   - RecallLayer: attention over the current chunk + a bounded memory bank,
                        used only when a learned read gate opens            (models/recall.py)
    persistent bank   - the memory bank itself: M slots per chunk, filled by a write policy

Recall layers go in the MIDDLE of the stack (never first): hybrid-architecture studies find
attention at the front hurts (arXiv:2510.04800).

Baselines built from the same class (so every comparison changes one thing):
    "sift"          recall_at=[]                                      (pure Mamba-1)
    "hybrid"        policy="full", read_gate="always", no timescales  (simple Jamba-style hybrid)
    "siftx"         policy="learned", read_gate="learned", timescales (full proposal)
    ablations       swap one of: policy, read_gate, timescales
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recall import RecallLayer
from .sift_model import ResidualBlock, RMSNorm, SiftConfig

DEFAULT_TIMESCALES = [(1e-2, 1e-1), (1e-3, 1e-2), (1e-4, 1e-3)]   # fast, medium, slow Δ ranges


@dataclass
class SiftXConfig:
    vocab_size: int = 65
    d_model: int = 128
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    timescales: object = field(default_factory=lambda: list(DEFAULT_TIMESCALES))
    pathway_gate: bool = True
    recall_at: object = "middle"        # list of block indices AFTER which a RecallLayer is inserted
    n_heads: int = 4
    chunk: int = 32
    mem_slots: int = 16
    policy: str = "learned"
    read_gate: str = "learned"
    gate_penalty: float = 1.0
    gate_budget: float = 0.3
    gate_warmup: int = 1000
    gate_anneal: int = 1000
    explore: bool = True
    tie_embeddings: bool = True

    def recall_positions(self):
        if self.recall_at == "middle":
            return [self.n_layers // 2 - 1]
        return list(self.recall_at or [])


class SiftXBackbone(nn.Module):
    def __init__(self, cfg: SiftXConfig):
        super().__init__()
        self.cfg = cfg
        scfg = SiftConfig(d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
                          d_conv=cfg.d_conv, expand=cfg.expand,
                          timescales=cfg.timescales or None, pathway_gate=cfg.pathway_gate)
        pos = set(cfg.recall_positions())
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            self.layers.append(ResidualBlock(scfg))
            if i in pos:
                self.layers.append(RecallLayer(cfg.d_model, cfg.n_heads, cfg.chunk, cfg.mem_slots,
                                               cfg.policy, cfg.read_gate, cfg.gate_penalty, gate_budget=cfg.gate_budget,
                                               gate_warmup=cfg.gate_warmup, gate_anneal=cfg.gate_anneal, explore=cfg.explore))
        self.norm_f = RMSNorm(cfg.d_model)
        # same residual-projection rescaling as the plain Sift backbone
        import math
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                w = layer.mixer.out_proj.weight
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                with torch.no_grad():
                    w /= math.sqrt(cfg.n_layers)

    @property
    def recall_layers(self):
        return [l for l in self.layers if isinstance(l, RecallLayer)]

    def aux_loss(self):
        rl = self.recall_layers
        return sum(l.aux_loss for l in rl) if rl else None

    def post_backward(self):
        for l in self.recall_layers:
            l.post_backward()

    def stats(self):
        rl = self.recall_layers
        if not rl:
            return {}
        return {"retrieval_rate": sum(l.stats.get("retrieval_rate", 0) for l in rl) / len(rl)}

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class SiftXLM(nn.Module):
    def __init__(self, cfg: SiftXConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.backbone = SiftXBackbone(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.backbone(self.embedding(idx)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def aux_loss(self):
        return self.backbone.aux_loss()

    # ---- streaming generation: Sift blocks keep a fixed SSM state, RecallLayers keep M + C tokens
    def allocate_state(self, batch, device=None):
        return [l.allocate_state(batch, device) if isinstance(l, RecallLayer)
                else l.mixer.allocate_state(batch, device) for l in self.backbone.layers]

    @torch.no_grad()
    def step(self, token, state):
        x = self.embedding(token)
        for layer, s in zip(self.backbone.layers, state):
            x = layer.step(x, s)
        return self.lm_head(self.backbone.norm_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        from .sift_model import sample
        state = self.allocate_state(idx.shape[0], idx.device)
        for t in range(idx.shape[1]):
            logits = self.step(idx[:, t], state)
        out = [idx]
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_k)
            out.append(nxt[:, None])
            logits = self.step(nxt, state)
        return torch.cat(out, dim=1)

    def post_backward(self):
        self.backbone.post_backward()

    def stats(self):
        return self.backbone.stats()


# Named variants used by the experiments: each changes exactly one thing.
VARIANTS = {
    "sift":            dict(recall_at=[], timescales=None, pathway_gate=False),
    "sift_mt":         dict(recall_at=[]),                                            # + multi-timescale only
    "hybrid":          dict(policy="full", read_gate="always", timescales=None, pathway_gate=False),
    "siftx":           dict(),                                                        # full proposal
    "siftx_no_mt":     dict(timescales=None, pathway_gate=False),
    "siftx_surprise":  dict(policy="surprise"),
    "siftx_recent":    dict(policy="recent"),
    "siftx_random":    dict(policy="random"),
    "siftx_always":    dict(read_gate="always"),
    "siftx_cf":        dict(read_gate="counterfactual"),                              # counterfactual gate
}


def build_siftx(variant="siftx", **kw):
    cfg = dict(VARIANTS[variant])
    cfg.update(kw)
    return SiftXConfig(**cfg)
