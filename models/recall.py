"""
Recall layer: gated sparse retrieval over a causal, fixed-size memory bank.

This is the research component of SiftX. For every chunk of C tokens:

    memory  = the M highest-scoring tokens from all EARLIER chunks
              (score = "write score", chosen by a memory policy)
    keys    = memory (M)  +  tokens of the current chunk up to now (causal, ≤ C)
    out_t   = r_t · Attention(q_t, keys)        r_t = learned read gate in [0, 1]

So each query looks at M + C keys instead of all L previous tokens:
cost O(L·(M + C)) instead of O(L²), and memory is bounded by M.

Memory policies (what gets written), compared in the ablations:
    learned   - a linear "write head" scores each token (trained end to end)
    surprise  - fixed rule inspired by B'MOJO: how badly a short moving average of the
                past predicts this token's hidden state (no parameters)
    random    - random scores (control)
    recent    - most recent tokens (≈ sliding-window attention, Taipan/SWA-like control)
    full      - ordinary causal attention over everything (the "simple hybrid" baseline)

Read gate: 'learned' (sigmoid head, trained toward a compute budget: the average open rate is
pulled toward gate_budget, default 30% of tokens) or 'always' (100%).
The gate is HARD in training and at test time: tokens with p_t < 0.5 skip retrieval entirely
(straight-through gradient), so the reported retrieval rate is real skipped work.
History: a plain penalty on the mean (λ·mean p) made the gate shut itself off in the CPU
pilots (retrieval rate 0% by step 200), because early on the retrieved values are ~0 and the
penalty is the only signal. The budget form fixes that.

Gradient to the write head: a stored token's attention logit is shifted by log σ(write score),
so a token that helps when stored learns to score higher (the top-k itself is not differentiable).
During training the learned policy samples its M slots with Gumbel-top-k (exploration), so useful
tokens get stored by chance before the write head knows about them; at test time it is plain top-k.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, chunk=32, mem_slots=16, policy="learned",
                 read_gate="learned", gate_penalty=1.0, gate_init=3.0, gate_budget=0.3,
                 gate_warmup=1000, gate_anneal=1000, explore=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert policy in ("learned", "surprise", "random", "recent", "full")
        assert read_gate in ("learned", "always", "counterfactual")
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.chunk, self.mem_slots, self.policy, self.read_gate = chunk, mem_slots, policy, read_gate
        self.gate_penalty, self.gate_budget, self.gate_warmup = gate_penalty, gate_budget, gate_warmup
        self.gate_anneal = gate_anneal
        self.explore = explore
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))   # for the budget warm-up
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out.weight)          # starts as an identity residual: never hurts at init
        self.write_head = nn.Linear(d_model, 1)
        self.gate_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_init)   # sigmoid(3) ≈ 0.95: retrieval on at start
        self.aux_loss = torch.zeros(())
        self.stats = {}

    # ---------------------------------------------------------------
    def write_scores(self, x):
        """x: (b, L, d) normalized hidden states -> (b, L) scores, causal by construction."""
        b, L, _ = x.shape
        if self.policy == "learned":
            return self.write_head(x).squeeze(-1)
        if self.policy == "surprise":
            # error of a fixed predictor: mean of the previous 4 hidden states
            pad = F.pad(x.transpose(1, 2), (4, 0))                          # (b, d, L+4)
            pred = F.avg_pool1d(pad, kernel_size=4, stride=1)[..., :L]      # mean of x[t-4..t-1]
            return (x - pred.transpose(1, 2)).pow(2).mean(-1)
        if self.policy == "random":
            return torch.rand(b, L, device=x.device)
        if self.policy == "recent":
            return torch.arange(L, device=x.device, dtype=x.dtype).expand(b, L).clone()
        raise ValueError(self.policy)

    def gate(self, x):
        if self.read_gate == "always":
            return torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        # The gate READS the hidden state but does not write gradient back into it: in the CPU
        # diagnostics the budget penalty, flowing through gate_head into the residual stream,
        # stalled learning even while every gate was still open (diag_gate_only). Detaching
        # keeps the gate a pure controller; it still learns from the task loss via its own weights.
        p = torch.sigmoid(self.gate_head(x.detach()).squeeze(-1))           # (b, L) soft open-probability
        self._p = p
        if self.training and self.train_steps.item() < self.gate_warmup:
            # Curriculum phase 1: gate forced open, so the memory learns WHAT to store and the
            # model learns to use it, before the gate has to decide WHEN to skip. (Learning both
            # at once stalled at chance in the MQAR pilots: neither signal is useful without the other.)
            return torch.ones_like(p)
        hard = (p > 0.5).to(p.dtype)                                         # 1 = retrieve, 0 = skip
        if self.read_gate == "counterfactual":
            return hard.detach()           # trained by its own labels in post_backward(), not by STE
        # Same hard decision in training and at test time. Straight-through gradient:
        # forward uses 0/1, backward flows through p, so a closed gate can learn to reopen.
        return hard + p - p.detach()

    # ---------------------------------------------------------------
    def forward(self, h):
        b, L, d = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).split(d, dim=-1)
        heads = lambda t: t.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # noqa: E731
        r = self.gate(x)

        if self.policy == "full":
            y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
            y = y.transpose(1, 2).reshape(b, L, d)
        else:
            y = self._chunked(x, heads(q), heads(k), heads(v))

        if self.read_gate == "counterfactual":
            a = self.out(y)                                       # what retrieval would add, every token
            res = h + a * r.unsqueeze(-1)
            self.aux_loss = torch.zeros((), device=h.device)
            if self.training and res.requires_grad:
                self._x_gate = x.detach()
                def hook(g, a=a):
                    # First-order counterfactual: removing a_t changes the loss by about -g_t·a_t,
                    # so  benefit_t = -g_t·a_t  > 0  means "retrieval lowered the loss here".
                    self._benefit = -(g.float() * a.detach().float()).sum(-1)
                res.register_hook(hook)
                self.train_steps += 1
            self.stats = {"retrieval_rate": (r > 0.5).float().mean().item()}
            return res

        out = self.out(y * r.unsqueeze(-1))
        if self.read_gate == "learned" and self.training:
            # Compute budget on the HARD on/off rate (straight-through). Warm-up: the target starts
            # at 100% (learn WHAT to retrieve first) and tightens linearly to gate_budget
            # (then learn WHEN retrieval can be skipped).
            # Phase 2: the target tightens linearly from 100% to gate_budget over gate_anneal steps.
            t = self.train_steps.item() - self.gate_warmup
            if t < 0:
                self.aux_loss = torch.zeros((), device=h.device)
            else:
                frac = min(1.0, t / max(1, self.gate_anneal))
                target = 1.0 - (1.0 - self.gate_budget) * frac
                self.aux_loss = self.gate_penalty * (r.mean() - target) ** 2
            self.train_steps += 1
        else:
            self.aux_loss = torch.zeros((), device=h.device)
        self.stats = {"retrieval_rate": (r > 0.5).float().mean().item()}
        return h + out

    def budget_target(self):
        """Fraction of tokens allowed to retrieve now: 100% during warm-up, then tightening."""
        t = self.train_steps.item() - self.gate_warmup
        if t < 0:
            return 1.0
        return 1.0 - (1.0 - self.gate_budget) * min(1.0, t / max(1, self.gate_anneal))

    def post_backward(self):
        """Counterfactual gate update, called by the trainer after loss.backward().

        Label = 1 for tokens whose retrieval helped (benefit > 0) AND that rank inside the current
        budget; the gate head is trained on these labels with binary cross-entropy. Unlike the
        budget penalty, every token gets a direct "should you have looked?" signal.
        """
        b = getattr(self, "_benefit", None)
        if b is None:
            return
        keep = self.budget_target()
        thr = torch.quantile(b.flatten(), 1.0 - keep) if keep < 1.0 else b.min() - 1
        target = ((b > thr) & (b > 0)).float()
        logits = self.gate_head(self._x_gate).squeeze(-1).float()
        loss = self.gate_penalty * F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        with torch.no_grad():
            self.stats["gate_target_rate"] = target.mean().item()
            self.stats["gate_agreement"] = ((logits > 0).float() == target).float().mean().item()
        self._benefit = self._x_gate = None

    def _chunked(self, x, q, k, v):
        """q, k, v: (b, H, L, dh). Local causal attention per chunk + attention to M memory slots."""
        b, H, L, dh = q.shape
        C, M = self.chunk, self.mem_slots
        pad = (-L) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        Lp, nc = L + pad, (L + pad) // C
        qc, kc, vc = (t.view(b, H, nc, C, dh) for t in (q, k, v))

        # Local part: causal attention inside the chunk.
        local = torch.einsum("bhncd,bhnkd->bhnck", qc, kc) / math.sqrt(dh)          # (b,H,nc,C,C)
        causal = torch.ones(C, C, dtype=torch.bool, device=x.device).tril()
        local = local.masked_fill(~causal, float("-inf"))

        # Memory part: for chunk c, top-M tokens (by write score) among positions < c*C.
        s = self.write_scores(x)                                                     # (b, L)
        if pad:
            s = F.pad(s, (0, pad), value=float("-inf"))
        pos = torch.arange(Lp, device=x.device)
        eligible = pos.view(1, Lp) < (torch.arange(nc, device=x.device) * C).view(nc, 1)   # (nc, Lp)
        masked = s.unsqueeze(1).masked_fill(~eligible.unsqueeze(0), float("-inf"))   # (b, nc, Lp)
        Mk = min(M, Lp)
        pick = masked
        if self.policy == "learned" and self.training and self.explore:
            # Gumbel-top-k: sample M tokens without replacement with probability ∝ softmax(score).
            # Early on (scores ≈ equal) this explores, so useful tokens get stored by chance and
            # can earn a higher score; as scores separate, sampling becomes near-deterministic.
            u = torch.rand_like(masked).clamp_(1e-9, 1 - 1e-9)
            pick = masked.detach() - torch.log(-torch.log(u))
        _, top_i = pick.topk(Mk, dim=-1)                                             # (b, nc, M)
        top_s = torch.gather(masked, 2, top_i)
        valid = torch.isfinite(top_s)                                                # early chunks: fewer
        idx = top_i.unsqueeze(1).unsqueeze(-1).expand(b, H, nc, Mk, dh)
        km = torch.gather(k.unsqueeze(2).expand(b, H, nc, Lp, dh), 3, idx)          # (b,H,nc,M,dh)
        vm = torch.gather(v.unsqueeze(2).expand(b, H, nc, Lp, dh), 3, idx)
        mem = torch.einsum("bhncd,bhnmd->bhncm", qc, km) / math.sqrt(dh)            # (b,H,nc,C,M)
        if self.policy == "learned":
            # gradient path into the write head: a stored token's attention logit is shifted by
            # log σ(score), so tokens that get used (and help) learn to score higher
            bias = F.logsigmoid(top_s.masked_fill(~valid, 0.0))                     # (b, nc, M)
            mem = mem + bias.unsqueeze(1).unsqueeze(3)
        mem = mem.masked_fill(~valid.unsqueeze(1).unsqueeze(3), float("-inf"))

        att = torch.cat([mem, local], dim=-1).softmax(-1)                            # (b,H,nc,C,M+C)
        y = torch.einsum("bhncm,bhnmd->bhncd", att[..., :Mk], vm) + \
            torch.einsum("bhnck,bhnkd->bhncd", att[..., Mk:], vc)
        y = y.reshape(b, H, Lp, dh)[:, :, :L]
        if not self.training:
            self.last_memory = top_i.detach()                                       # for analysis
        return y.transpose(1, 2).reshape(b, L, H * dh)

    # ---------------------------------------------------------------
    # Streaming (one token at a time) with BOUNDED state.
    #
    # Exact equivalence with forward(): the memory for chunk c is top-M over all positions < c*C,
    # and top-M(S ∪ new) = top-M(top-M(S) ∪ new), so we only ever keep M memory slots plus the
    # current chunk (≤ C tokens). State size is O(M + C) no matter how long the stream gets.
    # (policy="full" is the plain-attention baseline: its cache grows with every token.)
    # ---------------------------------------------------------------
    def allocate_state(self, batch, device=None):
        return {"t": 0, "mem_k": None, "mem_v": None, "mem_s": None,
                "loc_k": None, "loc_v": None, "loc_s": None, "hist": []}

    def _score_one(self, x, st):
        """Write score for the newest token x: (b, d) -> (b,)."""
        if self.policy == "learned":
            return self.write_head(x).squeeze(-1)
        if self.policy == "surprise":
            hist = st["hist"]
            prev = torch.stack(hist[-4:], 0) if hist else None
            pred = (prev.sum(0) / 4) if prev is not None else torch.zeros_like(x)   # zero-padded mean
            st["hist"] = (hist + [x])[-4:]
            return (x - pred).pow(2).mean(-1)
        if self.policy == "random":
            return torch.rand(x.shape[0], device=x.device)
        if self.policy == "recent":
            return torch.full((x.shape[0],), float(st["t"]), device=x.device, dtype=x.dtype)
        raise ValueError(self.policy)

    @staticmethod
    def _cat(a, b, dim):
        return b if a is None else torch.cat([a, b], dim)

    def state_tokens(self, st):
        """How many tokens this layer is currently holding (for the memory benchmark)."""
        n = lambda t: 0 if t is None else t.shape[2]   # noqa: E731
        return n(st["mem_k"]) + n(st["loc_k"])

    @torch.no_grad()
    def step(self, h, st):
        """h: (b, d) one token -> (b, d)."""
        b, d = h.shape
        H, dh = self.n_heads, self.d_head
        x = self.norm(h)
        q, k, v = (t.view(b, H, 1, dh) for t in self.qkv(x).split(d, dim=-1))
        r = self.gate(x[:, None])[:, 0]                                         # (b,)

        if self.policy == "full":
            st["loc_k"], st["loc_v"] = self._cat(st["loc_k"], k, 2), self._cat(st["loc_v"], v, 2)
            keys, vals, bias = st["loc_k"], st["loc_v"], None
        else:
            C, M = self.chunk, self.mem_slots
            if st["t"] > 0 and st["t"] % C == 0:                                 # chunk boundary:
                ks = self._cat(st["mem_k"], st["loc_k"], 2)                       # merge the finished chunk
                vs = self._cat(st["mem_v"], st["loc_v"], 2)                       # into memory, keep top-M
                ss = self._cat(st["mem_s"], st["loc_s"], 1)
                m = min(M, ss.shape[1])
                top_s, top_i = ss.topk(m, dim=-1)
                idx = top_i[:, None, :, None].expand(b, H, m, dh)
                st["mem_k"], st["mem_v"], st["mem_s"] = ks.gather(2, idx), vs.gather(2, idx), top_s
                st["loc_k"] = st["loc_v"] = st["loc_s"] = None
            s = self._score_one(x, st)
            st["loc_k"], st["loc_v"] = self._cat(st["loc_k"], k, 2), self._cat(st["loc_v"], v, 2)
            st["loc_s"] = self._cat(st["loc_s"], s[:, None], 1)
            if st["mem_k"] is not None:
                keys = torch.cat([st["mem_k"], st["loc_k"]], 2)
                vals = torch.cat([st["mem_v"], st["loc_v"]], 2)
                nm = st["mem_k"].shape[2]
                bias = torch.zeros(b, keys.shape[2], device=h.device, dtype=q.dtype)
                if self.policy == "learned":
                    bias[:, :nm] = F.logsigmoid(st["mem_s"])
            else:
                keys, vals, bias = st["loc_k"], st["loc_v"], None
        st["t"] += 1

        if not bool((r > 0.5).any()):                    # every gate closed: retrieval really skipped
            return h
        att = (q @ keys.transpose(-1, -2)) / math.sqrt(dh)                     # (b, H, 1, n)
        if bias is not None:
            att = att + bias[:, None, None, :]
        y = (att.softmax(-1) @ vals).reshape(b, d)
        return h + self.out(y * r[:, None])

