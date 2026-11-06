import torch

from models.sift_model import SiftConfig, SiftLM, count_params


def tiny_cfg(**kw):
    return SiftConfig(vocab_size=11, d_model=32, n_layers=2, d_state=8, **kw)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = SiftLM(tiny_cfg())
    idx = torch.randint(0, 11, (3, 15))
    targets = torch.randint(0, 11, (3, 15))
    logits, loss = m(idx, targets)
    assert logits.shape == (3, 15, 11)
    # random init should be near uniform: loss ≈ ln(11) ≈ 2.40
    assert abs(loss.item() - torch.log(torch.tensor(11.0)).item()) < 0.3


def test_tied_embeddings():
    m = SiftLM(tiny_cfg())
    assert m.lm_head.weight.data_ptr() == m.embedding.weight.data_ptr()
    assert count_params(m) > 0


def test_model_is_causal():
    torch.manual_seed(0)
    m = SiftLM(tiny_cfg())
    idx = torch.randint(0, 11, (1, 20))
    idx2 = idx.clone()
    idx2[0, 10:] = (idx2[0, 10:] + 1) % 11
    l1, _ = m(idx)
    l2, _ = m(idx2)
    assert torch.allclose(l1[:, :10], l2[:, :10], atol=1e-5)


def test_recurrent_step_matches_parallel_forward():
    torch.manual_seed(0)
    m = SiftLM(tiny_cfg()).eval()
    idx = torch.randint(0, 11, (2, 12))
    logits, _ = m(idx)
    state = m.allocate_state(2)
    steps = torch.stack([m.step(idx[:, t], state) for t in range(12)], 1)
    assert torch.allclose(logits, steps, atol=1e-4)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = SiftLM(tiny_cfg())
    idx = torch.randint(0, 11, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = m(idx[:, :-1], idx[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_generate_length():
    m = SiftLM(tiny_cfg())
    out = m.generate(torch.zeros(2, 3, dtype=torch.long), 5, temperature=0)
    assert out.shape == (2, 8)
