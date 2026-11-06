"""
Final extras, run automatically at the start of `python -m research.plots` (the last step of
scripts/run_everything.bat), so a run that is already in progress picks them up.

    python -m research.extras         # or run directly

Each step is skipped if its output already exists, and a failing step never stops the others:
  1. streaming benchmark        experiments/results/research/streaming_<device>.json/.png
  2. Triton kernel benchmark    experiments/results/benchmarks/triton_scan_cuda.json
                                (on Windows it first tries `pip install triton-windows`)
  3. analysis of the counterfactual-gate runs, merged into research/analysis.json
  4. memory-viewer trace from the best SiftX MQAR model -> frontend/memory_trace.json,
     embedded into frontend/index.html
"""

import json
import os
import subprocess
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def step(name, fn):
    print(f"[extras] {name} ...", flush=True)
    try:
        fn()
        print(f"[extras] {name}: ok", flush=True)
    except Exception:
        print(f"[extras] {name}: FAILED\n{traceback.format_exc()}", flush=True)


def py(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def streaming():
    if os.path.exists(os.path.join(RES, "research", f"streaming_{DEV}.json")):
        return
    py("-m", "research.streaming", "--tokens", "8192" if DEV == "cuda" else "1024")


def triton_bench():
    if os.path.exists(os.path.join(RES, "benchmarks", "triton_scan_cuda.json")) or DEV != "cuda":
        return
    try:
        import triton  # noqa: F401
    except Exception:
        if sys.platform == "win32":
            subprocess.run([sys.executable, "-m", "pip", "install", "triton-windows"], cwd=ROOT)
    py("-m", "benchmarks.triton_scan")


def analyze_cf():
    out = os.path.join(RES, "research", "analysis.json")
    ck = [os.path.join(RES, n, "ckpt.pt") for n in ("mqar_siftx_cf_s0", "mqar_siftx_cf_s1", "lm_siftx_cf",
                                                     "mqar_siftx_always_s0")]
    ck = [c for c in ck if os.path.exists(c)]
    if not ck:
        return
    tmp = os.path.join(RES, "research", "analysis_cf.json")
    py("-m", "research.analyze", *ck, "--out", tmp)
    merged = json.load(open(out)) if os.path.exists(out) else {}
    merged.update(json.load(open(tmp)))
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)


def viewer():
    from research.export_viewer import export
    best, best_acc = None, -1.0
    for n in ("mqar_siftx_cf_s0", "mqar_siftx_cf_s1", "mqar_siftx_s0", "mqar_siftx_s1"):
        s = os.path.join(RES, n, "summary.json")
        if os.path.exists(s):
            acc = json.load(open(s)).get("final_val_metric", 0)
            if acc > best_acc:
                best, best_acc = n, acc
    if best is None:
        return
    d = export(os.path.join(RES, best, "ckpt.pt"), DEV)
    path = os.path.join(ROOT, "frontend", "memory_trace.json")
    with open(path, "w") as f:
        json.dump(d, f)
    print("[extras] viewer trace from", best, d["summary"])
    py(os.path.join(ROOT, "research", "add_viewer.py"), os.path.join(ROOT, "frontend", "index.html"), path)


def main():
    step("streaming benchmark", streaming)
    step("triton kernel benchmark", triton_bench)
    step("counterfactual-gate analysis", analyze_cf)
    step("memory viewer trace", viewer)


if __name__ == "__main__":
    main()
