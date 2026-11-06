"""
Evaluate a trained language-model checkpoint and print a text sample.

    python -m training.evaluate experiments/results/sift_shakespeare/ckpt.pt --prompt "ROMEO:"
"""

import argparse
import math

import torch

from data.dataset import build_task
from data.tokenizer import CharTokenizer
from training.train import build_model


@torch.no_grad()
def full_val_loss(model, task, seq_len, device, batch_size=16):
    """Average loss over the whole validation split in non-overlapping windows."""
    data = task.data["val"]
    n = (len(data) - 1) // seq_len
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        starts = range(i * seq_len, min(n, i + batch_size) * seq_len, seq_len)
        x = torch.stack([data[s: s + seq_len] for s in starts]).to(device)
        y = torch.stack([data[s + 1: s + 1 + seq_len] for s in starts]).to(device)
        _, loss = model(x, y)
        total += loss.item() * x.numel()
        count += x.numel()
    return total / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    task_cfg = dict(cfg["task"])
    task = build_task(task_cfg.pop("name"), **task_cfg)
    model = build_model(cfg["model"], task).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss = full_val_loss(model, task, task.seq_len, args.device)
    print(f"full validation loss {loss:.4f} | perplexity {math.exp(loss):.3f} | bits/char {loss / math.log(2):.3f}")

    tok = CharTokenizer(ckpt["extra"]["tokenizer"]["chars"])
    torch.manual_seed(0)
    idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
    out = model.generate(idx, args.tokens, args.temperature, args.top_k)
    print("-" * 60)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
