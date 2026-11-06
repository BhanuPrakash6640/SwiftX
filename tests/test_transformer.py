import torch

from models.transformer import TransformerConfig, TransformerLM


def tiny():
    torch.manual_seed(0)
    return TransformerLM(TransformerConfig(vocab_size=11, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)).eval()


def test_shapes():
    logits, loss = tiny()(torch.randint(0, 11, (2, 10)), torch.randint(0, 11, (2, 10)))
    assert logits.shape == (2, 10, 11) and loss.ndim == 0


def test_causal():
    m = tiny()
    idx = torch.randint(0, 11, (1, 20))
    idx2 = idx.clone()
    idx2[0, 12:] = (idx2[0, 12:] + 3) % 11
    assert torch.allclose(m(idx)[0][:, :12], m(idx2)[0][:, :12], atol=1e-5)


def test_kv_cache_matches_full_forward():
    m = tiny()
    idx = torch.randint(0, 11, (2, 12))
    full, _ = m(idx)
    caches = m.allocate_state()
    steps = torch.stack([m.step(idx[:, t], caches, t) for t in range(12)], 1)
    assert torch.allclose(full, steps, atol=1e-5)
    assert caches[0]["k"].shape[2] == 12          # the cache grew with every token
