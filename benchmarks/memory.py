"""
Memory.

1. Inference state: bytes the model must keep between generated tokens.
   Sift: conv window + SSM notebook per layer (constant).
   Transformer: KV cache (grows linearly with tokens generated).
   Measured from the real tensors after generating, not from a formula.

2. Peak training memory for one forward+backward step vs sequence length.
   GPU: torch.cuda.max_memory_allocated. CPU: peak RSS of a fresh process
   (Linux/macOS) minus the RSS right after the model is built.

    python -m benchmarks.memory
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from benchmarks.common import device, models, save


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


@torch.no_grad()
def inference_state(dev, checkpoints):
    ms = {k: m.eval() for k, m in models(dev).items()}
    out = {k: {} for k in ms}
    for name, m in ms.items():
        state = m.allocate_state(1, dev)
        tok = torch.zeros(1, dtype=torch.long, device=dev)
        for pos in range(max(checkpoints)):
            m.step(tok, state) if name == "sift" else m.step(tok, state, pos)
            if pos + 1 in checkpoints:
                out[name][pos + 1] = tensor_bytes(state) / 2**10   # KiB
    for name in out:
        print(f"state {name:12s} " + "  ".join(f"@{p}:{v:.0f}KiB" for p, v in out[name].items()))
    return out


CHILD = r"""
import json, resource, sys, torch
sys.path.insert(0, {root!r})
from benchmarks.common import models
name, L, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = models("cpu")[name]
x = torch.randint(0, 65, (batch, L))
_, loss = m(x[:, :8], x[:, :8]); loss.backward(); m.zero_grad()   # warm allocator
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
_, loss = m(x, x); loss.backward()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024   # ru_maxrss: bytes on macOS, KiB on Linux
print(json.dumps({{"mb": max(0, peak - base) * scale / 2**20}}))
"""


def training_memory(dev, lengths, batch=4):
    out = {"sift": {}, "transformer": {}}
    for name in out:
        for L in lengths:
            if dev == "cuda":
                m = models(dev)[name]
                x = torch.randint(0, 65, (batch, L), device=dev)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                try:
                    _, loss = m(x, x)
                    loss.backward()
                    mb = (torch.cuda.max_memory_allocated() - base) / 2**20
                    del loss
                except torch.cuda.OutOfMemoryError:
                    mb = float("nan")
                del m, x
                torch.cuda.empty_cache()
            else:
                res = subprocess.run([sys.executable, "-c", CHILD.format(root=ROOT), name, str(L), str(batch)],
                                     capture_output=True, text=True, check=True)
                mb = json.loads(res.stdout.strip().splitlines()[-1])["mb"]
            out[name][L] = mb
            print(f"train-step memory L={L:5d} {name:12s} {mb:8.1f} MB")
    return out


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark")
    parser.add_argument("--device", default=None, help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    dev = args.device or device()
    if dev == "cpu" and sys.platform.startswith("win"):
        print("CPU peak-memory measurement needs Linux/macOS (resource module); skipping part 2.")
        train = None
    else:
        train = training_memory(dev, [256, 512, 1024, 2048, 4096])
    state = inference_state(dev, [256, 512, 1024, 2048, 4096])
    save("memory", {"inference_state_kib": state, "train_step_peak_mb_batch4": train})


if __name__ == "__main__":
    main()
