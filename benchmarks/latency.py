"""
Inference latency.

1. Prompt processing: one forward pass over a batch-1 prompt of length L.
2. Generation: time per new token at increasing positions.
   Sift uses its recurrent step (fixed-size state);
   the Transformer uses a KV cache (attention over all previous tokens).

    python -m benchmarks.latency
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from benchmarks.common import device, models, save, sync, timeit


@torch.no_grad()
def prompt_latency(ms, dev, lengths):
    out = {k: {} for k in ms}
    for L in lengths:
        x = torch.randint(0, 65, (1, L), device=dev)
        for name, m in ms.items():
            out[name][L] = timeit(lambda: m(x), dev) * 1000
            print(f"prompt L={L:5d} {name:12s} {out[name][L]:8.1f} ms")
    return out


@torch.no_grad()
def generation_latency(ms, dev, n_tokens, report_every):
    out = {}
    for name, m in ms.items():
        state = m.allocate_state(1, dev)
        tok = torch.zeros(1, dtype=torch.long, device=dev)
        per_pos, t_block = {}, time.perf_counter()
        for pos in range(n_tokens):
            if name == "sift":
                m.step(tok, state)
            else:
                m.step(tok, state, pos)
            if (pos + 1) % report_every == 0:
                sync(dev)
                per_pos[pos + 1] = (time.perf_counter() - t_block) / report_every * 1000
                t_block = time.perf_counter()
        out[name] = per_pos
        print(f"generation {name:12s} " + "  ".join(f"@{p}:{v:.2f}ms" for p, v in per_pos.items()))
    return out


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--device", default=None, help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    dev = args.device or device()
    ms = {k: m.eval() for k, m in models(dev).items()}
    prompt = prompt_latency(ms, dev, [128, 256, 512, 1024, 2048, 4096])
    gen = generation_latency(ms, dev, n_tokens=4096, report_every=256)
    save("latency", {"prompt_ms": prompt, "generation_ms_per_token": gen})


if __name__ == "__main__":
    main()
