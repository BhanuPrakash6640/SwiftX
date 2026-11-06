"""
Efficiency of the hybrids: forward time and peak memory vs sequence length.

Compares, at the same width and depth: Sift, simple hybrid (full attention layer),
SiftX (bounded memory bank + gated retrieval), and the Transformer.

    python -m research.bench
"""

import json
import os

import torch

from benchmarks.common import ROOT, device, timeit
from models import build_lm, count_params

OUT = os.path.join(ROOT, "experiments", "results", "research")


def build(name, dev):
    torch.manual_seed(0)
    kw = dict(vocab_size=128, d_model=128, n_layers=6, d_state=16, chunk=64, mem_slots=64)
    if name == "transformer":
        return build_lm("transformer", vocab_size=128, d_model=128, n_layers=3, n_heads=4, max_seq_len=16384).to(dev)
    return build_lm("siftx", variant=name, **kw).to(dev)


@torch.no_grad()
def main():
    dev = device()
    names = ["sift", "hybrid", "siftx", "transformer"]
    lengths = [512, 1024, 2048, 4096, 8192]
    res = {"forward_ms_batch2": {n: {} for n in names}, "peak_mb_batch2": {n: {} for n in names}, "params": {}}
    for n in names:
        m = build(n, dev).eval()
        res["params"][n] = count_params(m)
        for L in lengths:
            x = torch.randint(0, 128, (2, L), device=dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t = timeit(lambda: m(x), dev, warmup=1, repeat=2) * 1000
            res["forward_ms_batch2"][n][L] = t
            if dev == "cuda":
                res["peak_mb_batch2"][n][L] = torch.cuda.max_memory_allocated() / 2**20 if t == t else float("nan")
            print(f"{n:12s} L={L:5d} {t:9.1f} ms")
        del m
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"bench_{dev}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
