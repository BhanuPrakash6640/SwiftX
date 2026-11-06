"""
Turn experiment and benchmark results into PNG charts in experiments/results/plots/.

    python -m benchmarks.plots
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.common import ROOT  # noqa: E402

RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "plots")
COLORS = {"sift": "#2a78d6", "transformer": "#eb6834"}
LABELS = {"sift": "Sift (Mamba-1)", "transformer": "Transformer"}
DEV, SUFFIX = "cpu", ""
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def read_metrics(name):
    path = os.path.join(RES, name, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f]


def read_bench(name):
    path = os.path.join(RES, "benchmarks", f"{name}_{DEV}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def label_end(ax, xs, ys, model, fmt="{:.2f}"):
    ax.annotate(f"{LABELS[model]}  " + fmt.format(ys[-1]), (xs[-1], ys[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    name = name.replace(".png", f"_{DEV}.png")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def plot_curves(task, key, ylabel, title, fname, logy=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    found = False
    for m in ("sift", "transformer"):
        rows = read_metrics(f"{m}_{task}{SUFFIX}")
        if not rows:
            continue
        found = True
        xs = [r["step"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color=COLORS[m], label=LABELS[m], marker="o", markersize=3)
        label_end(ax, xs, ys, m)
    if not found:
        plt.close(fig)
        return
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right")
    ax.margins(x=0.22)
    save(fig, fname)


def plot_xy(data, title, xlabel, ylabel, fname, logx=True, logy=True, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in ("sift", "transformer"):
        if m not in data:
            continue
        pts = [(int(k), v) for k, v in data[m].items() if v is not None and v == v]  # drop NaN (OOM)
        if not pts:
            continue
        xs, ys = map(list, zip(*pts))
        ax.plot(xs, ys, color=COLORS[m], label=LABELS[m], marker="o", markersize=4)
        label_end(ax, xs, ys, m, fmt)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    ax.margins(x=0.25)
    save(fig, fname)


def main():
    global DEV, SUFFIX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    DEV = ap.parse_args().device
    SUFFIX = "_gpu" if DEV == "cuda" else ""
    if DEV == "cuda":
        plot_curves("wikitext2", "val_loss", "validation loss (nats/char)",
                    "WikiText-2 (char-level): validation loss", "wikitext2_val_loss.png")
    plot_curves("shakespeare", "val_loss", "validation loss (nats/char)",
                "Tiny Shakespeare: validation loss", "shakespeare_val_loss.png")
    plot_curves("copy", "val_metric", "token accuracy", "Copy task: recall accuracy", "copy_accuracy.png")
    plot_curves("adding", "val_loss", "MSE (log scale)", "Adding task: validation MSE",
                "adding_mse.png", logy=True)

    lat = read_bench("latency")
    if lat:
        plot_xy(lat["generation_ms_per_token"], "Generation: time per new token",
                "tokens already generated", "ms per token", "generation_latency.png", logy=False, fmt="{:.2f} ms")
        plot_xy(lat["prompt_ms"], "Prompt processing (batch 1, one forward pass)",
                "prompt length", "ms", "prompt_latency.png", fmt="{:.0f} ms")
    mem = read_bench("memory")
    if mem:
        plot_xy(mem["inference_state_kib"], "Inference state kept between tokens",
                "tokens generated", "KiB", "inference_state.png", fmt="{:.0f} KiB")
        if mem.get("train_step_peak_mb_batch4"):
            plot_xy(mem["train_step_peak_mb_batch4"], "Peak training-step memory (batch 4)",
                    "sequence length", "MB", "train_memory.png", fmt="{:.0f} MB")
    sc = read_bench("sequence_scaling")
    if sc:
        plot_xy(sc["train_step_ms_batch4"], "Training step time (forward + backward, batch 4)",
                "sequence length", "ms", "train_step_scaling.png", fmt="{:.0f} ms")
        res = sc["scan_implementations"]["results"]
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        names = list(res)[::-1]
        vals = [res[n]["fwd_bwd_ms"] for n in names]
        ax.barh(names, vals, color=COLORS["sift"], height=0.55)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f} ms", (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
        shape = sc["scan_implementations"]["shape"]
        ax.set_title(f"Selective scan implementations, forward + backward "
                     f"(b={shape['b']}, L={shape['L']}, D={shape['D']}, N={shape['N']})", loc="left", fontsize=10)
        ax.set_xlabel("ms (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
        save(fig, "scan_implementations.png")


if __name__ == "__main__":
    main()
