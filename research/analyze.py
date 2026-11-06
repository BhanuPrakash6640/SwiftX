"""
Look inside trained SiftX models: does the memory controller do what we claim?

For an MQAR checkpoint it measures:
  memory_kv_fraction   share of memory slots holding a key or value token (vs noise)
  gate_at_queries      read-gate open rate on query tokens (where a fact is needed)
  gate_elsewhere       read-gate open rate on all other tokens
  acc_hard_gate        accuracy with the real hard gate (retrieval skipped when r < 0.5)
  acc_gate_forced_on   accuracy if every token were allowed to retrieve
For a language-model checkpoint it measures the correlation between the read gate
and the model's own per-token loss (is retrieval "uncertainty-driven"?).

    python -m research.analyze experiments/results/<run>/ckpt.pt
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from data.dataset import IGNORE, build_task
from models.recall import RecallLayer
from training.train import build_model


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    tcfg = dict(cfg["task"])
    name = tcfg.pop("name")
    task = build_task(name, **tcfg)
    model = build_model(cfg["model"], task).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), task, name, cfg


def recall_layers(model):
    return [m for m in model.modules() if isinstance(m, RecallLayer)]


def capture_gates(model):
    """Hook each RecallLayer's gate so we can read r_t per token."""
    store = []
    for layer in recall_layers(model):
        orig = layer.gate
        def g(x, orig=orig):
            r = orig(x)
            store.append(r.detach())
            return r
        layer.gate = g
    return store


@torch.no_grad()
def analyze_mqar(model, task, device, batches=8, bs=32):
    gen = torch.Generator().manual_seed(123)
    out = {"memory_kv_fraction": [], "gate_at_queries": [], "gate_elsewhere": [],
           "acc_hard_gate": [], "acc_gate_forced_on": []}
    layers = recall_layers(model)
    for _ in range(batches):
        x, y = task.get_batch("val", bs, gen)
        x, y = x.to(device), y.to(device)
        gates = capture_gates(model)
        logits, _ = model(x)
        mask = y != IGNORE
        out["acc_hard_gate"].append((logits.argmax(-1)[mask] == y[mask]).float().mean().item())
        if gates:
            r = torch.stack(gates).mean(0)                      # (b, L) averaged over recall layers
            out["gate_at_queries"].append((r[mask] > 0).float().mean().item())
            out["gate_elsewhere"].append((r[~mask] > 0).float().mean().item())
        for layer in layers:
            mem = layer.last_memory                             # (b, n_chunks, M) token positions
            tok = torch.gather(x.unsqueeze(1).expand(-1, mem.shape[1], -1), 2, mem)
            kv = (tok >= task.k0) & (tok < task.n0)
            out["memory_kv_fraction"].append(kv[:, 2:].float().mean().item())   # chunks ≥ 2 see the kv block
        # same batch, gate forced open
        saved = [(l, l.read_gate) for l in layers if l.read_gate in ("learned", "counterfactual")]
        for l, _ in saved:
            l.gate = (lambda x, l=l: torch.ones(x.shape[:2], device=x.device))
        if saved:
            logits2, _ = model(x)
            out["acc_gate_forced_on"].append((logits2.argmax(-1)[mask] == y[mask]).float().mean().item())
        for l in layers:                                        # restore
            l.gate = RecallLayer.gate.__get__(l, RecallLayer)
    return {k: (sum(v) / len(v) if v else None) for k, v in out.items()}


@torch.no_grad()
def analyze_lm(model, task, device, batches=8, bs=8):
    gen = torch.Generator().manual_seed(123)
    rs, ls = [], []
    for _ in range(batches):
        x, y = task.get_batch("val", bs, gen)
        x, y = x.to(device), y.to(device)
        for l in recall_layers(model):
            l.gate = (lambda x, l=l: torch.sigmoid(l.gate_head(x).squeeze(-1)))   # soft value for analysis
        gates = capture_gates(model)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")        # (b, L)
        if gates:
            rs.append(torch.stack(gates).mean(0).flatten())
            ls.append(loss.flatten())
        for l in recall_layers(model):
            l.gate = RecallLayer.gate.__get__(l, RecallLayer)
    if not rs:
        return {}
    r, loss = torch.cat(rs), torch.cat(ls)
    rr = r.argsort().argsort().float()
    lr = loss.argsort().argsort().float()
    spearman = torch.corrcoef(torch.stack([rr, lr]))[0, 1].item()
    hi = loss > loss.median()
    return {"gate_loss_spearman": spearman, "gate_open_rate": (r > 0.5).float().mean().item(),
            "gate_on_hard_tokens": (r[hi] > 0.5).float().mean().item(),
            "gate_on_easy_tokens": (r[~hi] > 0.5).float().mean().item()}


@torch.no_grad()
def analyze_needle(model, task, device, lengths=(512, 1024, 2048, 4096), batches=4, bs=16):
    """Exact-retrieval accuracy as the haystack grows past the training length."""
    out = {}
    for L in lengths:
        if getattr(getattr(model, "cfg", None), "max_seq_len", 10**9) < L:
            out[str(L)] = None                                   # learned positions end here
            continue
        task.val_seq_len = L
        gen = torch.Generator().manual_seed(7)
        accs = []
        try:
            for _ in range(batches):
                x, y = task.get_batch("val", bs, gen)
                logits, _ = model(x.to(device))
                accs.append((logits[:, -1].argmax(-1).cpu() == y[:, -1]).float().mean().item())
            out[str(L)] = sum(accs) / len(accs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out[str(L)] = float("nan")
    return {"needle_accuracy_by_length": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    results = {}
    for path in args.ckpts:
        model, task, name, cfg = load(path, args.device)
        if name == "mqar":
            res = analyze_mqar(model, task, args.device)
        elif name == "needle":
            res = analyze_needle(model, task, args.device)
        else:
            res = analyze_lm(model, task, args.device)
        results[cfg["name"]] = res
        print(cfg["name"], json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
