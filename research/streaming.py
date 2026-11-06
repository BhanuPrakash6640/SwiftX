"""
Streaming generation benchmark: memory held per sequence while generating token by token.

    python -m research.streaming                 # GPU if available, 8192 tokens
    python -m research.streaming --tokens 1024   # quick CPU run

For each model it feeds tokens one at a time and records, every `--every` tokens:
    state_kib   bytes of everything the model keeps between tokens (SSM states, conv buffers,
                memory slots, local chunk, or KV cache), per sequence
    ms_per_tok  average wall time per token so far
The models are untrained: memory and speed are properties of the architecture, not the weights.

Expected shape (and what the plot shows): Transformer and the full-attention hybrid grow
linearly (KV cache); Sift is flat; SiftX is flat too: its memory is capped at M + C tokens.
"""

import argparse
import json
import os
import time

import torch

from models import build_lm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results", "research")
VOCAB = 65


def nbytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def build(name, max_len):
    torch.manual_seed(0)
    if name == "transformer":
        return build_lm("transformer", vocab_size=VOCAB, d_model=128, n_layers=4, n_heads=4, max_seq_len=max_len)
    if name == "sift":
        return build_lm("sift", vocab_size=VOCAB, d_model=128, n_layers=7, d_state=16)
    variant = {"siftx": "siftx", "hybrid": "hybrid"}[name]
    return build_lm("siftx", variant=variant, vocab_size=VOCAB, d_model=128, n_layers=6, d_state=16,
                    chunk=64, mem_slots=64)


@torch.no_grad()
def run(name, tokens, every, dev):
    model = build(name, tokens + 1).to(dev).eval()
    tok = torch.zeros(1, dtype=torch.long, device=dev)
    warm = model.allocate_state(1, dev)                      # warm-up: first calls pay one-off costs
    for t in range(16):
        model.step(tok, warm, t) if name == "transformer" else model.step(tok, warm)
    state = model.allocate_state(1, dev)
    rows, t0 = [], time.time()
    for t in range(tokens):
        if name == "transformer":
            logits = model.step(tok, state, t)
        else:
            logits = model.step(tok, state)
        tok = logits.argmax(-1)
        if (t + 1) % every == 0:
            if dev == "cuda":
                torch.cuda.synchronize()
            rows.append({"tokens": t + 1, "state_kib": nbytes(state) / 1024,
                         "ms_per_tok": 1000 * (time.time() - t0) / (t + 1)})
    return rows


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sift": "#2a78d6", "siftx": "#1b9e77", "hybrid": "#7570b3", "transformer": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, rows in res.items():
        x = [r["tokens"] for r in rows]
        axes[0].plot(x, [r["state_kib"] for r in rows], label=name, color=colors.get(name), lw=2)
        axes[1].plot(x, [r["ms_per_tok"] for r in rows], label=name, color=colors.get(name), lw=2)
    axes[0].set(xlabel="tokens generated", ylabel="memory kept per sequence (KiB)", title="Memory while streaming")
    axes[1].set(xlabel="tokens generated", ylabel="ms per token (running mean)", title="Speed while streaming")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--every", type=int, default=128)
    p.add_argument("--models", default="sift,siftx,hybrid,transformer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name in a.models.split(","):
        res[name] = run(name, a.tokens, a.every, a.device)
        last = res[name][-1]
        print(f"{name:12s} {last['tokens']} tokens  state {last['state_kib']:.1f} KiB  {last['ms_per_tok']:.2f} ms/tok")
    base = os.path.join(OUT, f"streaming_{a.device}")
    with open(base + ".json", "w") as f:
        json.dump({"device": a.device, "results": res}, f, indent=2)
    plot(res, base + ".png")
    print("saved", base + ".json/.png")


if __name__ == "__main__":
    main()
