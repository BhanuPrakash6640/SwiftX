"""
Generate every research config in configs/research/ (so the suite is reproducible).

    python -m research.make_configs

Experiments
  A. MQAR (associative recall)  - main models x 2 seeds + one-change ablations
  B. Needle (train 512, test longer) - length generalization of exact retrieval
  C. Tiny Shakespeare LM        - does the hybrid hurt or help ordinary language modelling?
"""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "research")

TRAIN = dict(weight_decay=0.1, grad_clip=1.0, warmup_steps=100, amp=True, ckpt_every=500)

# ---- model families at matched width -------------------------------------------------
def siftx(variant, n_layers=4, **kw):
    return dict(type="siftx", variant=variant, d_model=128, n_layers=n_layers, d_state=16, **kw)

def transformer(n_layers):
    return dict(type="transformer", d_model=128, n_layers=n_layers, n_heads=4)

MQAR_MODELS = {
    "sift":         siftx("sift"),
    "sift_big":     siftx("sift", n_layers=5),          # more parameters than SiftX: controls for size
    "transformer":  transformer(3),
    "hybrid":       siftx("hybrid", chunk=32, mem_slots=64),
    "siftx":        siftx("siftx", chunk=32, mem_slots=64),
    "siftx_cf":     siftx("siftx_cf", chunk=32, mem_slots=64),     # counterfactual read gate
}
MQAR_ABLATIONS = {
    "sift_mt":         siftx("sift_mt"),
    "siftx_no_mt":     siftx("siftx_no_mt", chunk=32, mem_slots=64),
    "siftx_surprise":  siftx("siftx_surprise", chunk=32, mem_slots=64),
    "siftx_recent":    siftx("siftx_recent", chunk=32, mem_slots=64),
    "siftx_random":    siftx("siftx_random", chunk=32, mem_slots=64),
    "siftx_always":    siftx("siftx_always", chunk=32, mem_slots=64),
}


def gate_curriculum(model, steps):
    """Read-gate curriculum (see models/recall.py): forced open for the first 60% of training,
    budget tightened to 30% over the next 25%, held at budget for the last 15%. The CPU pilots
    showed that tightening before the memory has learned the task stalls learning."""
    if model.get("type") == "siftx":
        model = dict(model, gate_warmup=int(0.6 * steps), gate_anneal=int(0.25 * steps))
    return model


def write(name, model, task, train):
    model = gate_curriculum(model, train["max_steps"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.yaml"), "w") as f:
        yaml.safe_dump({"name": name, "model": model, "task": task, "train": train}, f, sort_keys=False)
    return name


def main():
    names = []
    # A. MQAR: 32 key-value pairs in 256 tokens. Memory budget (64 slots) = exactly the kv block.
    mqar = dict(name="mqar", seq_len=256, n_pairs=32, n_keys=64, n_values=64, n_noise=127)
    tr = dict(TRAIN, batch_size=32, max_steps=3000, lr=3e-3, min_lr=3e-4, eval_every=250, eval_batches=8)
    for seed in (0, 1):
        for m, cfg in MQAR_MODELS.items():
            names.append(write(f"mqar_{m}_s{seed}", cfg, mqar, dict(tr, seed=seed)))
    for m, cfg in MQAR_ABLATIONS.items():
        names.append(write(f"mqar_{m}_s0", cfg, mqar, dict(tr, seed=0)))

    # B. Needle: train at 512, validate at 2048 (4x longer than anything seen in training).
    needle = dict(name="needle", seq_len=512, val_seq_len=2048)
    trn = dict(TRAIN, batch_size=32, max_steps=2500, lr=3e-3, min_lr=3e-4, eval_every=250, eval_batches=4, seed=0)
    for m in ("sift", "transformer", "hybrid", "siftx"):
        cfg = dict(MQAR_MODELS[m])
        if m in ("hybrid", "siftx"):
            cfg.update(chunk=64, mem_slots=32)
        names.append(write(f"needle_{m}", cfg, needle, trn))

    # C. Tiny Shakespeare at ~0.83M parameters, same budget for all four.
    lm = dict(name="shakespeare", seq_len=256)
    trl = dict(TRAIN, batch_size=16, max_steps=3000, lr=2e-3, min_lr=2e-4, eval_every=250, eval_batches=8, seed=0)
    lm_models = {
        "sift": siftx("sift", n_layers=7),
        "transformer": dict(transformer(4)),
        "hybrid": siftx("hybrid", n_layers=6, chunk=64, mem_slots=64),
        "siftx": siftx("siftx", n_layers=6, chunk=64, mem_slots=64),
        "siftx_cf": siftx("siftx_cf", n_layers=6, chunk=64, mem_slots=64),
    }
    for m, cfg in lm_models.items():
        t =dict(trl, lr=1e-3, min_lr=1e-4) if m == "transformer" else trl
        names.append(write(f"lm_{m}", cfg, lm, t))
    with open(os.path.join(OUT, "ORDER.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} configs to {OUT}")


if __name__ == "__main__":
    main()
