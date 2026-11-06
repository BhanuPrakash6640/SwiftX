import copy

import torch

from data.dataset import IGNORE, AddingTask, CopyTask
from data.tokenizer import CharTokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.train import build_model, lr_at, train


def test_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("low row")) == "low row"
    tok.save(tmp_path / "t.json")
    assert CharTokenizer.load(tmp_path / "t.json").chars == tok.chars


def test_copy_task_layout():
    x, y = CopyTask(n_copy=5, n_symbols=4).get_batch("train", 3)
    assert x.shape == y.shape == (3, 11)
    assert (x[:, 5] == 1).all() and (x[:, 6:] == 0).all()
    assert torch.equal(y[:, 5:10], x[:, :5])          # predict the stored tokens in order
    assert (y[:, :5] == IGNORE).all() and (y[:, 10] == IGNORE).all()


def test_adding_task_targets():
    x, y = AddingTask(seq_len=20).get_batch("train", 4)
    assert x.shape == (4, 20, 2)
    assert (x[..., 1].sum(1) == 2).all()
    assert torch.allclose((x[..., 0] * x[..., 1]).sum(1), y)


def test_lr_schedule():
    t = dict(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, t) == 0.1 and abs(lr_at(9, t) - 1.0) < 1e-9
    assert abs(lr_at(110, t) - 0.1) < 1e-9


def test_checkpoint_roundtrip(tmp_path):
    task = CopyTask(n_copy=4)
    m = build_model(dict(type="sift", d_model=16, n_layers=1), task)
    opt = torch.optim.AdamW(m.parameters())
    m(*task.get_batch("train", 2))[1].backward()
    opt.step()
    save_checkpoint(tmp_path / "c.pt", m, opt, 7, {"a": 1})
    m2 = build_model(dict(type="sift", d_model=16, n_layers=1), task)
    opt2 = torch.optim.AdamW(m2.parameters())
    ck = load_checkpoint(tmp_path / "c.pt", m2, opt2)
    assert ck["step"] == 7
    for p, q in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(p, q)


def test_training_loss_decreases(tmp_path, monkeypatch):
    import training.train as tr
    monkeypatch.setattr(tr, "ROOT", str(tmp_path))
    for mtype in ("sift", "transformer"):
        cfg = {
            "name": f"t_{mtype}",
            "model": {"type": mtype, "d_model": 32, "n_layers": 1},
            "task": {"name": "copy", "n_copy": 2, "n_symbols": 4},
            "train": dict(batch_size=32, max_steps=100, lr=3e-3, min_lr=3e-4, warmup_steps=5,
                          weight_decay=0.1, grad_clip=1.0, eval_every=50, eval_batches=2, seed=0),
        }
        _, s = train(copy.deepcopy(cfg), device="cpu")
        assert s["final_val_loss"] < 0.7     # chance level is ln(4) ≈ 1.39
