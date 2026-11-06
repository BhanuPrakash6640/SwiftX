import math

import pytest
import torch
import torch.nn.functional as F

from data.dataset import IGNORE, MQARTask, NeedleTask
from models import build_lm, count_params
from models.recall import RecallLayer
from models.sift_block import SiftBlock
from models.siftx import VARIANTS, SiftXBackbone, build_siftx


def lm(variant, **kw):
    torch.manual_seed(0)
    return build_lm("siftx", variant=variant, vocab_size=20, d_model=32, n_layers=4, d_state=8,
                    chunk=8, mem_slots=4, **kw)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_is_causal(variant):
    m = lm(variant).eval()
    idx = torch.randint(0, 20, (2, 40))
    idx2 = idx.clone()
    idx2[:, 25:] = torch.randint(0, 20, (2, 15))
    a, b = m(idx)[0], m(idx2)[0]
    assert torch.allclose(a[:, :25], b[:, :25], atol=1e-5), variant


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_trains(variant):
    m = lm(variant)
    idx = torch.randint(0, 20, (2, 24))
    _, loss = m(idx, idx)
    aux = m.aux_loss()
    (loss + (aux if aux is not None else 0)).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_recall_layer_is_identity_at_init():
    layer = RecallLayer(32, n_heads=4, chunk=8, mem_slots=4)
    h = torch.randn(2, 20, 32)
    assert torch.allclose(layer(h), h)            # zero-initialized output projection


def test_memory_only_contains_earlier_chunks():
    torch.manual_seed(0)
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, policy="learned").eval()
    layer(torch.randn(1, 40, 16))
    mem = layer.last_memory[0]                    # (n_chunks, M)
    for c in range(mem.shape[0]):
        if c > 0:
            assert (mem[c] < c * 8).all()


def test_recent_policy_picks_most_recent():
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, policy="recent").eval()
    layer(torch.randn(1, 32, 16))
    assert sorted(layer.last_memory[0, 2].tolist()) == [12, 13, 14, 15]


def test_chunked_equals_full_attention_when_memory_holds_everything():
    # With enough slots to store every earlier token, chunked attention == full causal attention.
    torch.manual_seed(0)
    a = RecallLayer(16, n_heads=2, chunk=4, mem_slots=64, policy="recent", read_gate="always").eval()
    b = RecallLayer(16, n_heads=2, chunk=4, mem_slots=64, policy="full", read_gate="always").eval()
    b.load_state_dict(a.state_dict())
    for l in (a, b):
        torch.nn.init.normal_(l.out.weight, std=0.2)
    b.out.weight.data.copy_(a.out.weight.data)
    h = torch.randn(2, 18, 16)
    assert torch.allclose(a(h), b(h), atol=1e-5)


def test_hard_gate_skips_retrieval_at_eval():
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, gate_warmup=0)
    torch.nn.init.normal_(layer.out.weight, std=0.2)
    torch.nn.init.constant_(layer.gate_head.bias, -5.0)     # gate closed everywhere
    h = torch.randn(1, 16, 16)
    layer.eval()
    assert torch.allclose(layer(h), h) and layer.stats["retrieval_rate"] == 0.0
    layer.train()
    layer(h).sum().backward()
    assert layer.aux_loss.item() > 0
    assert layer.gate_head.bias.grad is not None and layer.gate_head.bias.grad.abs() > 0   # closed gate still learns


def test_gate_curriculum_forced_open_then_budgeted():
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, gate_warmup=2, gate_anneal=2)
    torch.nn.init.constant_(layer.gate_head.bias, -5.0)     # gate would be closed
    h = torch.randn(1, 16, 16)
    layer.train()
    for _ in range(2):                                      # phase 1: forced open, no budget
        layer(h)
        assert layer.stats["retrieval_rate"] == 1.0 and layer.aux_loss.item() == 0.0
    layer(h)                                                # phase 2: real gate + budget
    assert layer.stats["retrieval_rate"] == 0.0 and layer.aux_loss.item() > 0


def test_gate_does_not_send_gradient_into_hidden_state():
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, gate_warmup=0)
    torch.nn.init.normal_(layer.gate_head.weight)
    h = torch.randn(1, 16, 16, requires_grad=True)
    layer.train()
    layer(h)
    layer.aux_loss.backward()
    assert h.grad is None or h.grad.abs().max() == 0       # budget loss only trains the gate head
    assert layer.gate_head.weight.grad.abs().max() > 0


def test_timescale_groups_initialized_in_their_ranges():
    torch.manual_seed(0)
    blk = SiftBlock(32, d_state=8, expand=3, timescales=[(1e-2, 1e-1), (1e-3, 1e-2), (1e-4, 1e-3)], pathway_gate=True)
    dt = F.softplus(blk.dt_proj.bias)
    g = blk.group_of
    assert dt[g == 0].min() >= 0.99e-2 and dt[g == 1].max() <= 1.01e-2 and dt[g == 2].max() <= 1.01e-3


def test_pathway_mix_neutral_at_init_and_step_matches_forward():
    torch.manual_seed(0)
    blk = SiftBlock(16, d_state=4, expand=3, timescales=[(1e-2, 1e-1), (1e-3, 1e-2), (1e-4, 1e-3)], pathway_gate=True)
    x = torch.randn(2, 10, 16)
    y = blk(x)
    st = blk.allocate_state(2)
    assert torch.allclose(y, torch.stack([blk.step(x[:, t], st) for t in range(10)], 1), atol=1e-5)


def test_mqar_layout():
    t = MQARTask(seq_len=64, n_pairs=8)
    x, y = t.get_batch("train", 3, torch.Generator().manual_seed(0))
    for b in range(3):
        pairs = {x[b, 2 * i].item(): x[b, 2 * i + 1].item() for i in range(8)}
        qs = (y[b] != IGNORE).nonzero().flatten()
        assert len(qs) == 8 and (qs >= 16).all()
        for p in qs:
            assert pairs[x[b, p].item()] == y[b, p].item()


def test_needle_layout():
    t = NeedleTask(seq_len=64, val_seq_len=128)
    x, y = t.get_batch("val", 4, torch.Generator().manual_seed(0))
    assert x.shape == (4, 128) and (y[:, :-1] == IGNORE).all()
    for b in range(4):
        p = (x[b, :-2] == t.MARK).nonzero().flatten()[0]
        assert x[b, p + 1] == x[b, -1] and x[b, p + 2] == y[b, -1]


def test_counterfactual_benefit_matches_real_loss_change():
    torch.manual_seed(0)
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, read_gate="counterfactual", gate_warmup=10)
    torch.nn.init.normal_(layer.out.weight, std=1e-3)        # small retrieval -> first order is accurate
    h = torch.randn(1, 16, 16)
    w = torch.randn(16)
    layer.train()
    res = layer(h)
    loss_fn = lambda z: (z @ w).sin().pow(2)                 # nonlinear per-token loss  # noqa: E731
    loss_fn(res).sum().backward()
    benefit = layer._benefit[0]
    with torch.no_grad():                                      # real: loss without minus loss with
        a = res - h
        real = loss_fn(h)[0] - loss_fn(h + a)[0]
    assert torch.allclose(benefit, real, rtol=0.05, atol=5e-5)        # first-order estimate
    assert (benefit.sign() == real.sign()).float().mean() > 0.9


def test_counterfactual_gate_trains_only_through_its_labels():
    from models.siftx import SiftXLM, build_siftx
    torch.manual_seed(0)
    cfg = build_siftx("siftx_cf", vocab_size=20, d_model=32, n_layers=2, d_state=4, chunk=8, mem_slots=4,
                      recall_at=[0], gate_warmup=0)
    m = SiftXLM(cfg).train()
    torch.nn.init.normal_(m.backbone.recall_layers[0].out.weight, std=0.1)
    x = torch.randint(0, 20, (2, 32))
    _, loss = m(x, x)
    loss.backward()
    gh = m.backbone.recall_layers[0].gate_head
    assert gh.weight.grad is None                                # task loss does not reach the gate
    m.post_backward()
    assert gh.weight.grad is not None and gh.weight.grad.abs().sum() > 0
    s = m.backbone.recall_layers[0].stats
    assert 0.0 <= s["gate_target_rate"] <= 1.0 and "gate_agreement" in s


import pytest


@pytest.mark.parametrize("policy", ["learned", "surprise", "recent", "full"])
@pytest.mark.parametrize("read_gate", ["learned", "always", "counterfactual"])
def test_recall_streaming_matches_forward(policy, read_gate):
    torch.manual_seed(0)
    layer = RecallLayer(16, n_heads=2, chunk=8, mem_slots=4, policy=policy, read_gate=read_gate).eval()
    for p in (layer.out.weight, layer.write_head.weight, layer.gate_head.weight):
        torch.nn.init.normal_(p, std=0.5)                    # some gates open, some closed
    torch.nn.init.zeros_(layer.gate_head.bias)
    h = torch.randn(2, 29, 16)                               # not a multiple of the chunk
    full = layer(h)
    st = layer.allocate_state(2)
    steps = torch.stack([layer.step(h[:, t], st) for t in range(29)], 1)
    assert torch.allclose(full, steps, atol=1e-5)
    if policy != "full":
        assert layer.state_tokens(st) <= 4 + 8               # bounded: M + C tokens, forever


def test_siftx_lm_step_matches_forward_and_generates():
    from models.siftx import SiftXLM, build_siftx
    torch.manual_seed(0)
    cfg = build_siftx("siftx", vocab_size=20, d_model=32, n_layers=2, d_state=4, chunk=8, mem_slots=4,
                      recall_at=[0])
    m = SiftXLM(cfg).eval()
    torch.nn.init.normal_(m.backbone.recall_layers[0].out.weight, std=0.3)
    x = torch.randint(0, 20, (2, 21))
    logits, _ = m(x)
    st = m.allocate_state(2)
    steps = torch.stack([m.step(x[:, t], st) for t in range(21)], 1)
    assert torch.allclose(logits, steps, atol=1e-4)
    assert m.generate(x[:, :3], 10).shape == (2, 13)
