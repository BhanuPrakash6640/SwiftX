"""Shared helpers: the two benchmark models and a timer."""

import json
import os
import platform
import time

import torch

from models import build_lm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results", "benchmarks")
VOCAB = 65

# Same sizes as the Tiny Shakespeare experiment (~0.83M parameters each).
SIFT_KW = dict(vocab_size=VOCAB, d_model=128, n_layers=7, d_state=16, d_conv=4, expand=2)
TF_KW = dict(vocab_size=VOCAB, d_model=128, n_layers=4, n_heads=4, max_seq_len=16384)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def models(dev=None):
    dev = dev or device()
    torch.manual_seed(0)
    return {"sift": build_lm("sift", **SIFT_KW).to(dev), "transformer": build_lm("transformer", **TF_KW).to(dev)}


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, warmup=1, repeat=3):
    """Best-of-N wall time in seconds; NaN if the GPU runs out of memory."""
    try:
        for _ in range(warmup):
            fn()
        sync(dev)
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            sync(dev)
            times.append(time.perf_counter() - t)
        return min(times)
    except torch.cuda.OutOfMemoryError:
        print("  (out of GPU memory at this size, recorded as NaN)")
        torch.cuda.empty_cache()
        return float("nan")


def hardware():
    dev = device()
    info = {"device": dev, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "platform": platform.platform(), "cpu": platform.processor() or platform.machine()}
    if dev == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    data = {"hardware": hardware(), **data}
    with open(os.path.join(RESULTS, f"{name}_{device()}.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
