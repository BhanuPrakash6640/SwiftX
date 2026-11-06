"""
Train Sift or the Transformer baseline from a YAML config.

    python -m training.train configs/sift_shakespeare.yaml
    python -m training.train configs/transformer_shakespeare.yaml --max_steps 200

Writes to experiments/results/<name>/:
    metrics.jsonl   one line per eval (step, train/val loss, time)
    summary.json    final numbers (params, losses, perplexity, speed, memory)
    ckpt.pt         final checkpoint (model + optimizer + config)
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.dataset import IGNORE, build_task
from models import build_backbone, build_lm, count_params
from training.checkpoint import load_checkpoint, save_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Regressor(nn.Module):
    """Backbone + linear in/out heads, for continuous tasks like adding."""

    def __init__(self, model_type, in_dim, **model_kw):
        super().__init__()
        model_kw.pop("vocab_size", None)
        d = model_kw["d_model"]
        self.inp = nn.Linear(in_dim, d)
        self.backbone = build_backbone(model_type, **model_kw)
        self.out = nn.Linear(d, 1)

    def aux_loss(self):
        return self.backbone.aux_loss() if hasattr(self.backbone, "aux_loss") else None

    def stats(self):
        return self.backbone.stats() if hasattr(self.backbone, "stats") else {}

    def post_backward(self):
        if hasattr(self.backbone, "post_backward"):
            self.backbone.post_backward()

    def forward(self, x, targets=None):
        pred = self.out(self.backbone(self.inp(x))[:, -1]).squeeze(-1)
        loss = F.mse_loss(pred, targets) if targets is not None else None
        return pred, loss


def build_model(mcfg, task):
    mcfg = dict(mcfg)
    mtype = mcfg.pop("type")
    if hasattr(task, "in_dim"):
        if mtype == "transformer":
            mcfg.setdefault("max_seq_len", task.seq_len)
        return Regressor(mtype, task.in_dim, **mcfg)
    mcfg["vocab_size"] = task.vocab_size
    if mtype == "transformer":
        mcfg.setdefault("max_seq_len", max(task.seq_len, getattr(task, "val_seq_len", 0)))
    return build_lm(mtype, **mcfg)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, embeddings' 1-D params, A_log and D."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, tcfg):
    """Linear warmup, then cosine decay to min_lr."""
    warm, total = tcfg["warmup_steps"], tcfg["max_steps"]
    if step < warm:
        return tcfg["lr"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return tcfg["min_lr"] + 0.5 * (tcfg["lr"] - tcfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def metric(task_name, out, y):
    """Task-specific accuracy, in addition to the loss."""
    if task_name in ("copy", "mqar", "needle"):
        mask = y != IGNORE
        return (out.argmax(-1)[mask] == y[mask]).float().mean().item()
    if task_name == "adding":
        return ((out - y).abs() < 0.04).float().mean().item()
    return None


@torch.no_grad()
def evaluate(model, task, task_name, tcfg, device, generator):
    model.eval()
    losses, metrics, rates = [], [], []
    for _ in range(tcfg["eval_batches"]):
        x, y = task.get_batch("val", tcfg["batch_size"], generator)
        x, y = x.to(device), y.to(device)
        out, loss = model(x, y)
        losses.append(loss.item())
        m = metric(task_name, out, y)
        if m is not None:
            metrics.append(m)
        st = model.stats() if hasattr(model, "stats") else {}
        if "retrieval_rate" in st:
            rates.append(st["retrieval_rate"])
    model.train()
    evaluate.last_retrieval_rate = sum(rates) / len(rates) if rates else None
    return sum(losses) / len(losses), (sum(metrics) / len(metrics) if metrics else None)


def train(cfg, device=None, resume=None):
    tcfg = cfg["train"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg.get("seed", 0))
    task_cfg = dict(cfg["task"])
    task_name = task_cfg.pop("name")
    task = build_task(task_name, **task_cfg)
    model = build_model(cfg["model"], task).to(device)
    n_params = count_params(model)

    opt = torch.optim.AdamW(
        param_groups(model, tcfg["weight_decay"]), lr=tcfg["lr"], betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )
    out_dir = os.path.join(ROOT, "experiments", "results", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, opt, device)["step"]

    # bf16 autocast for matmuls (the scan itself always runs in fp32). Needs an
    # RTX 30-series or newer; older GPUs silently fall back to fp32.
    use_amp = tcfg.get("amp", False) and device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16
    train_gen = torch.Generator().manual_seed(tcfg.get("seed", 0))
    val_seed = tcfg.get("seed", 0) + 1000
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"[{cfg['name']}] {cfg['model']['type']}  params={n_params:,}  device={device}")
    log_path = os.path.join(out_dir, "metrics.jsonl")
    log = open(log_path, "a" if resume else "w")
    tokens_per_step = tcfg["batch_size"] * task.seq_len
    train_time, t_last, recent = 0.0, time.time(), []

    for step in range(start_step, tcfg["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        x, y = task.get_batch("train", tcfg["batch_size"], train_gen)
        x, y = x.to(device), y.to(device)
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
            aux = model.aux_loss() if hasattr(model, "aux_loss") else None
        opt.zero_grad(set_to_none=True)
        (loss + aux if aux is not None else loss).backward()
        if hasattr(model, "post_backward"):
            model.post_backward()               # e.g. counterfactual read-gate update (SiftX)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        recent.append(loss.item())

        last = step == tcfg["max_steps"] - 1
        if (step + 1) % tcfg["eval_every"] == 0 or last:
            gen = torch.Generator().manual_seed(val_seed)   # same val batches every time
            val_loss, val_metric = evaluate(model, task, task_name, tcfg, device, gen)
            rec = {
                "step": step + 1,
                "train_loss": sum(recent) / len(recent),
                "val_loss": val_loss,
                "val_metric": val_metric,
                "lr": opt.param_groups[0]["lr"],
                "retrieval_rate": getattr(evaluate, "last_retrieval_rate", None),
                "train_time_s": train_time,
            }
            recent = []
            log.write(json.dumps(rec) + "\n")
            log.flush()
            extra = f" metric={val_metric:.3f}" if val_metric is not None else ""
            if rec["retrieval_rate"] is not None:
                extra += f" retrieve={rec['retrieval_rate']:.2f}"
            print(f"step {step + 1:5d} | train {rec['train_loss']:.4f} | val {val_loss:.4f}{extra} "
                  f"| {time.time() - t_last:.1f}s", flush=True)
            t_last = time.time()
        if tcfg.get("ckpt_every") and (step + 1) % tcfg["ckpt_every"] == 0:
            save_checkpoint(os.path.join(out_dir, "ckpt.pt"), model, opt, step + 1, cfg)
    log.close()
    if start_step >= tcfg["max_steps"]:          # resumed from a checkpoint taken at the very end
        gen = torch.Generator().manual_seed(val_seed)
        val_loss, val_metric = evaluate(model, task, task_name, tcfg, device, gen)
        rec = {"train_loss": float("nan"), "val_loss": val_loss, "val_metric": val_metric,
               "retrieval_rate": getattr(evaluate, "last_retrieval_rate", None)}

    extra = {"tokenizer": task.tokenizer.to_dict()} if hasattr(task, "tokenizer") else {}
    save_checkpoint(os.path.join(out_dir, "ckpt.pt"), model, opt, tcfg["max_steps"], cfg, extra)
    summary = {
        "name": cfg["name"],
        "model": cfg["model"]["type"],
        "task": task_name,
        "params": n_params,
        "device": device,
        "steps": tcfg["max_steps"],
        "final_train_loss": rec["train_loss"],
        "final_val_loss": rec["val_loss"],
        "final_val_metric": rec["val_metric"],
        "final_retrieval_rate": rec.get("retrieval_rate"),
        "train_time_s": train_time,
        "tokens_per_s": tokens_per_step * (tcfg["max_steps"] - start_step) / max(train_time, 1e-9),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    }
    if task_name in ("shakespeare", "wikitext2"):
        summary["val_perplexity"] = math.exp(rec["val_loss"])
        summary["val_bits_per_char"] = rec["val_loss"] / math.log(2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return model, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--max_steps", type=int)
    p.add_argument("--device")
    p.add_argument("--resume")
    p.add_argument("--name")
    p.add_argument("--lr", type=float, help="peak learning rate (min_lr becomes lr/10)")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.name:
        cfg["name"] = args.name
    if args.lr:
        cfg["train"]["lr"], cfg["train"]["min_lr"] = args.lr, args.lr / 10
    train(cfg, args.device, args.resume)


if __name__ == "__main__":
    main()
