"""
Fast-track runner for SiftX Research Suite.
Runs all core models, ablations, needle, and LM at 1000 steps,
then executes research.analyze, research.bench, and research.plots.
"""

import glob
import json
import os
import subprocess
import sys
import time
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train import train

RES_DIR = os.path.join(ROOT, "experiments", "results")
LOG_DIR = os.path.join(ROOT, "experiments", "logs")
RESEARCH_DIR = os.path.join(ROOT, "configs", "research")

RUNS = [
    # 1. MQAR Core & Ablations
    "mqar_transformer_s0",
    "mqar_hybrid_s0",
    "mqar_siftx_s0",
    "mqar_siftx_always_s0",
    "mqar_siftx_surprise_s0",
    "mqar_siftx_recent_s0",
    "mqar_siftx_random_s0",
    "mqar_siftx_no_mt_s0",
    "mqar_sift_big_s0",
    "mqar_sift_mt_s0",
    
    # 2. Needle in a Haystack
    "needle_sift",
    "needle_transformer",
    "needle_hybrid",
    "needle_siftx",
    
    # 3. Shakespeare Language Modeling
    "lm_transformer",
    "lm_hybrid",
    "lm_siftx",
    "lm_sift"
]


def gate_curriculum(model, steps):
    if model.get("type") == "siftx":
        model["gate_warmup"] = int(0.6 * steps)
        model["gate_anneal"] = int(0.25 * steps)
    return model


def run_model(name, max_steps=1000):
    summary_path = os.path.join(RES_DIR, name, "summary.json")
    if os.path.exists(summary_path):
        print(f"[{name}] already done, skipping.")
        return

    cfg_path = os.path.join(RESEARCH_DIR, f"{name}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["max_steps"] = max_steps
    cfg["model"] = gate_curriculum(cfg["model"], max_steps)

    print(f"\n=======================================================")
    print(f"  STARTING: {name} (max_steps={max_steps})")
    print(f"=======================================================", flush=True)

    t0 = time.time()
    ckpt_path = os.path.join(RES_DIR, name, "ckpt.pt")
    resume = ckpt_path if os.path.exists(ckpt_path) else None

    train(cfg, device="cuda", resume=resume)
    print(f"[{name}] Finished in {time.time() - t0:.1f}s", flush=True)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    print(f"[Fast-Track] Starting {len(RUNS)} research runs on CUDA...")
    t_start = time.time()

    for r in RUNS:
        try:
            run_model(r, max_steps=1000)
        except Exception as e:
            print(f"[ERROR in {r}]: {e}", flush=True)

    print(f"\n[Fast-Track] All models trained in {(time.time() - t_start)/60:.1f} minutes!")

    # Analysis
    print("\n--- Running research.analyze ---", flush=True)
    ckpts = [
        os.path.join(RES_DIR, "mqar_siftx_s0", "ckpt.pt"),
        os.path.join(RES_DIR, "mqar_siftx_surprise_s0", "ckpt.pt"),
        os.path.join(RES_DIR, "mqar_siftx_recent_s0", "ckpt.pt"),
        os.path.join(RES_DIR, "mqar_siftx_random_s0", "ckpt.pt"),
        os.path.join(RES_DIR, "mqar_siftx_no_mt_s0", "ckpt.pt"),
        os.path.join(RES_DIR, "needle_sift", "ckpt.pt"),
        os.path.join(RES_DIR, "needle_transformer", "ckpt.pt"),
        os.path.join(RES_DIR, "needle_hybrid", "ckpt.pt"),
        os.path.join(RES_DIR, "needle_siftx", "ckpt.pt"),
        os.path.join(RES_DIR, "lm_siftx", "ckpt.pt"),
    ]
    valid_ckpts = [c for c in ckpts if os.path.exists(c)]
    analysis_out = os.path.join(RES_DIR, "research", "analysis.json")
    subprocess.run([sys.executable, "-m", "research.analyze", *valid_ckpts, "--out", analysis_out], cwd=ROOT)

    # Benchmarks
    print("\n--- Running research.bench ---", flush=True)
    subprocess.run([sys.executable, "-m", "research.bench"], cwd=ROOT)

    # Plots
    print("\n--- Running research.plots ---", flush=True)
    subprocess.run([sys.executable, "-m", "research.plots"], cwd=ROOT)

    print(f"\n[Fast-Track Complete!] Everything finished in {(time.time() - t_start)/60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
