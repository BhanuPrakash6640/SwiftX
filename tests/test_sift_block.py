import torch

from models.sift_block import SiftBlock


def make_block(**kw):
    torch.manual_seed(0)
    return SiftBlock(d_model=16, d_state=8, d_conv=4, expand=2, **kw)


def test_shapes_and_dims():
    blk = make_block()
    assert blk.d_inner == 32 and blk.dt_rank == 1
    x = torch.randn(2, 10, 16)
    assert blk(x).shape == (2, 10, 16)


def test_fast_matches_reference_scan():
    blk = make_block()
    x = torch.randn(2, 12, 16)
    y_fast = blk(x)
    blk.scan_impl = "ref"
    y_ref = blk(x)
    assert torch.allclose(y_fast, y_ref, atol=1e-5)


def test_causal_no_future_leak():
    blk = make_block()
    x = torch.randn(1, 20, 16)
    y1 = blk(x)
    x2 = x.clone()
    x2[:, 13:] = torch.randn(1, 7, 16)
    y2 = blk(x2)
    assert torch.allclose(y1[:, :13], y2[:, :13], atol=1e-6)
    assert not torch.allclose(y1[:, 13:], y2[:, 13:])


def test_conv_alone_is_causal():
    blk = make_block()
    x = torch.zeros(1, 32, 10)
    x[:, :, 5] = 1.0                                  # impulse at t=5
    out = blk.conv1d(x)[..., :10] - blk.conv1d.bias.view(1, -1, 1)
    assert torch.all(out[..., :5].abs() < 1e-7)       # nothing before t=5
    assert torch.all(out[..., 9:].abs() < 1e-7)       # kernel 4: reaches t=5..8 only


def test_selective_params_depend_on_input():
    blk = make_block()
    x1, x2 = torch.randn(1, 6, 32), torch.randn(1, 6, 32)
    d1, B1, C1 = blk.ssm_params(x1)
    d2, B2, C2 = blk.ssm_params(x2)
    assert (d1 > 0).all()
    assert not torch.allclose(d1, d2)
    assert not torch.allclose(B1, B2)
    assert not torch.allclose(C1, C2)
    # and they vary across time steps within one sequence
    assert not torch.allclose(d1[:, 0], d1[:, 1])


def test_step_matches_forward():
    blk = make_block()
    x = torch.randn(2, 9, 16)
    y_full = blk(x)
    state = blk.allocate_state(2)
    ys = [blk.step(x[:, t], state) for t in range(9)]
    assert torch.allclose(y_full, torch.stack(ys, 1), atol=1e-5)


def test_gradients_reach_all_parameters():
    blk = make_block()
    blk(torch.randn(2, 8, 16)).pow(2).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
