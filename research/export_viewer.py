"""
Export one MQAR example from a trained SiftX checkpoint for the website's memory viewer.

    python -m research.export_viewer experiments/results/mqar_siftx_s0/ckpt.pt --out frontend/memory_trace.json

For a single validation sequence it records, per token:
    kind       key / value / noise / query           (what the token is)
    gate       did the read gate open (1) or skip (0)
    p          the gate's open-probability
    correct    for query tokens: did the model answer right
and per chunk: which earlier positions the memory bank held (M slots).
"""

import argparse
import json
import os

import torch

from data.dataset import IGNORE
from models.recall import RecallLayer
from research.analyze import load


@torch.no_grad()
def export(path, device="cpu", seed=3):
    model, task, name, cfg = load(path, device)
    assert name == "mqar", "the viewer is built for MQAR checkpoints"
    layer = [m for m in model.modules() if isinstance(m, RecallLayer)][0]
    gen = torch.Generator().manual_seed(seed)
    x, y = task.get_batch("val", 1, gen)
    x, y = x.to(device), y.to(device)
    captured = {}
    orig = layer.gate

    def g(h):
        r = orig(h)
        captured["r"] = r.detach()
        captured["p"] = torch.sigmoid(layer.gate_head(h.detach()).squeeze(-1)) if layer.read_gate != "always" else torch.ones_like(r)
        return r
    layer.gate = g
    logits, _ = model(x)
    layer.gate = RecallLayer.gate.__get__(layer, RecallLayer)

    tok, tgt, pred = x[0].tolist(), y[0].tolist(), logits[0].argmax(-1).tolist()
    n_kv = 2 * task.K
    kinds = []
    for i, t in enumerate(tok):
        if tgt[i] != IGNORE:
            kinds.append("query")
        elif i < n_kv:
            kinds.append("key" if i % 2 == 0 else "value")
        else:
            kinds.append("noise")
    r = captured["r"][0].tolist()
    p = captured["p"][0].tolist()
    mem = getattr(layer, "last_memory", None)
    chunks = mem[0].tolist() if mem is not None else []
    tokens = [{"i": i, "kind": kinds[i], "gate": int(r[i] > 0.5), "p": round(p[i], 3),
               "correct": (int(pred[i] == tgt[i]) if tgt[i] != IGNORE else None)} for i in range(len(tok))]
    q = [t for t in tokens if t["kind"] == "query"]
    return {"run": cfg["name"], "chunk": layer.chunk, "mem_slots": layer.mem_slots, "policy": layer.policy,
            "read_gate": layer.read_gate, "seq_len": len(tok), "tokens": tokens,
            "memory_by_chunk": chunks,
            "summary": {"retrieval_rate": sum(t["gate"] for t in tokens) / len(tokens),
                        "gate_on_queries": sum(t["gate"] for t in q) / max(1, len(q)),
                        "query_accuracy": sum(t["correct"] for t in q) / max(1, len(q))}}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("ckpt")
    a.add_argument("--out", default="frontend/memory_trace.json")
    a.add_argument("--device", default="cpu")
    args = a.parse_args()
    d = export(args.ckpt, args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(d, f)
    print(json.dumps(d["summary"]), "->", args.out)


if __name__ == "__main__":
    main()
